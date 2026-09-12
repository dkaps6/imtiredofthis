#!/usr/bin/env python3
"""Persist historical MC arrays with fold-safe TE-R5P / WR-R15 upstream of simulation.

Frozen by Issue #535 checkpoint 18. Research only.

2024: TE-R5P + WR-R15.
2025: TE-R5P only; WR-R15 2025 confirmation is forbidden by its own contract.

The emitted hybrid shards use specialist arrays only on authorized WR/TE receiving
rows and canonical legacy arrays everywhere else. Before specialists are applied,
the explicit-entitlement baseline must be elementwise identical to legacy MC.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.component_predictions import build_mc_predictions
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement
from scripts.modeling.te_r5p_entitlement_adapter_v1 import (
    _load_snaps as _load_participation_snaps,
    _strict_prior_features as _te_strict_prior_features,
)
from scripts.modeling.wr_r15_entitlement_adapter_v1 import (
    WR_POS,
    _strict_prior_features as _wr_strict_prior_features,
)
from scripts.research.persist_historical_simulated_outcomes_v1 import (
    KEYS,
    _canon_keys,
    _exact_week,
    _historical_outcomes,
    _parse_weeks,
    _read,
    _read_optional,
)
from scripts.simulation_explicit_entitlement_v1 import simulate as explicit_simulate
from scripts.simulation_v2 import simulate as legacy_simulate
from scripts.utils.canonical_names import canon_team

RECEIVING_MARKETS = {"rec_yards", "receptions", "rush_rec_yards"}
TE_FEATURES = [
    "b0_te_room_share", "log_b0_te_pool", "pool_ratio", "room_size",
    "prior1_same_team_offense_pct", "prior1_same_team_offense_snaps",
    "prior1_anyteam_offense_pct", "prior3_anyteam_offense_pct",
    "prior1_anyteam_offense_snaps", "prior3_anyteam_offense_snaps",
    "log1p_prior_count_same_team", "log1p_prior_count_anyteam",
    "prior1_same_team_available", "prior3_same_team_available",
    "snap_share_prior1_same_team", "snap_share_prior3_anyteam",
]
WR_FEATURES = [
    "b0_secondary_room_share", "log_b0_secondary_pool", "secondary_room_size",
    "prior1_same_team_offense_pct", "prior1_same_team_offense_snaps",
    "prior1_anyteam_offense_pct", "prior3_anyteam_offense_pct",
    "prior1_anyteam_offense_snaps", "prior3_anyteam_offense_snaps",
    "log1p_prior_count_same_team", "log1p_prior_count_anyteam",
    "prior1_same_team_available", "prior3_same_team_available",
    "secondary_snap_share_prior1_same_team", "secondary_snap_share_prior3_anyteam",
]


def _pos_family(value) -> str:
    p = "" if value is None or pd.isna(value) else str(value).upper().strip()
    if p in WR_POS or p.startswith("WR"):
        return "WR"
    if p.startswith("TE"):
        return "TE"
    if p in {"HB", "TB"} or p.startswith("RB"):
        return "RB"
    if p.startswith("FB"):
        return "FB"
    if p.startswith("QB"):
        return "QB"
    return p


def _load_fold_params(path: Path, *, test_season: int, features: list[str], label: str) -> dict:
    x = _read(path, f"{label} fold coefficients")
    x.columns = [str(c).strip().lower() for c in x.columns]
    need = {"test_season", "feature", "standardized_coefficient", "scaler_mean", "scaler_scale", "ridge_intercept"}
    missing = sorted(need - set(x.columns))
    if missing:
        raise RuntimeError(f"{label} coefficient artifact missing columns: {missing}")
    x["test_season"] = pd.to_numeric(x["test_season"], errors="raise").astype(int)
    g = x.loc[x["test_season"].eq(int(test_season))].copy()
    if len(g) != len(features):
        raise RuntimeError(f"{label} expected {len(features)} coefficients for test {test_season}, found {len(g)}")
    if set(g["feature"].astype(str)) != set(features):
        raise RuntimeError(
            f"{label} feature contract mismatch test={test_season}: "
            f"missing={sorted(set(features)-set(g['feature']))} extra={sorted(set(g['feature'])-set(features))}"
        )
    if label == "WR-R15" and int(test_season) == 2024:
        if "train_season" not in g.columns:
            raise RuntimeError("WR-R15 fold artifact missing train_season")
        train = sorted(pd.to_numeric(g["train_season"], errors="raise").astype(int).unique().tolist())
        if train != [2023]:
            raise RuntimeError(f"WR-R15 2024 fold lineage drifted: train_season={train}")
    g = g.set_index("feature").loc[features]
    scale = pd.to_numeric(g["scaler_scale"], errors="raise").to_numpy(float)
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise RuntimeError(f"{label} invalid scaler scale")
    intercepts = pd.to_numeric(g["ridge_intercept"], errors="raise").unique()
    if len(intercepts) != 1 or not np.isfinite(float(intercepts[0])):
        raise RuntimeError(f"{label} invalid fold intercept")
    return {
        "mean": pd.to_numeric(g["scaler_mean"], errors="raise").to_numpy(float),
        "scale": scale,
        "coef": pd.to_numeric(g["standardized_coefficient"], errors="raise").to_numpy(float),
        "intercept": float(intercepts[0]),
        "test_season": int(test_season),
        "features": list(features),
    }


def _allocate_softmax(pool: float, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    score = np.asarray(scores, dtype=float)
    if len(score) == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    if not np.isfinite(score).all():
        raise RuntimeError("specialist allocation received non-finite scores")
    if pool <= 0:
        return np.zeros(len(score), dtype=float), np.zeros(len(score), dtype=float)
    stable = score - float(np.max(score))
    weights = np.exp(stable)
    room = weights / float(weights.sum())
    candidate = float(pool) * room
    gap = float(pool) - float(candidate.sum())
    candidate[int(np.argmax(room))] += gap
    return room, candidate


def apply_te_fold(metrics: pd.DataFrame, *, snaps: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    out = metrics.copy()
    if not out.index.is_unique:
        raise RuntimeError("TE historical fold requires unique player-row index")
    baseline = pd.to_numeric(out["entitlement_tgt_share"], errors="raise").astype(float)
    out["wrte_pre_te_entitlement"] = baseline
    out["te_r5p_oos_applied"] = False
    out["te_r5p_oos_test_season"] = int(params["test_season"])
    pos = out["position"].map(_pos_family)
    te = out.loc[pos.eq("TE")].copy()
    if te.empty:
        raise RuntimeError("TE historical fold found zero TEs")
    feat = _te_strict_prior_features(te, snaps)
    feat["baseline_entitlement_tgt_share"] = pd.to_numeric(feat["entitlement_tgt_share"], errors="raise").astype(float)
    feat["b0_te_pool"] = feat.groupby(["event_id", "team"])["baseline_entitlement_tgt_share"].transform("sum")
    feat["b0_te_room_share"] = np.where(feat["b0_te_pool"].gt(0), feat["baseline_entitlement_tgt_share"] / feat["b0_te_pool"], 0.0)
    feat["log_b0_te_pool"] = np.log1p(feat["b0_te_pool"].clip(lower=0.0))
    feat["pool_ratio"] = 1.0
    feat["room_size"] = feat.groupby(["event_id", "team"])["player_clean_key"].transform("count").astype(float)
    feat["prior1_same_team_available"] = feat["prior1_same_team"].fillna(False).astype(float)
    feat["prior3_same_team_available"] = feat["prior3_same_team"].fillna(False).astype(float)
    feat["log1p_prior_count_same_team"] = np.log1p(pd.to_numeric(feat["prior_count_same_team"], errors="coerce").fillna(0).clip(lower=0))
    feat["log1p_prior_count_anyteam"] = np.log1p(pd.to_numeric(feat["prior_count_anyteam"], errors="coerce").fillna(0).clip(lower=0))
    for src, dst in (("prior1_same_team_offense_pct", "snap_share_prior1_same_team"), ("prior3_anyteam_offense_pct", "snap_share_prior3_anyteam")):
        z = pd.to_numeric(feat[src], errors="coerce").fillna(0.0).clip(lower=0.0)
        den = z.groupby([feat["event_id"], feat["team"]]).transform("sum")
        feat[dst] = np.where(den.gt(0), z / den, 0.0)
    for c in TE_FEATURES:
        feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)
    x = feat[TE_FEATURES].to_numpy(float)
    residual = ((x - params["mean"]) / params["scale"]) @ params["coef"] + params["intercept"]
    residual = np.clip(residual, -1.0, 1.0)
    feat["te_r5p_oos_residual"] = residual
    feat["te_r5p_oos_score"] = np.log(feat["b0_te_room_share"].clip(lower=0.0).to_numpy(float) + 0.02) + residual
    feat["te_r5p_oos_room_share"] = 0.0
    feat["te_r5p_oos_entitlement_tgt_share"] = 0.0
    for _, idx in feat.groupby(["event_id", "team"], sort=False).groups.items():
        pool = float(feat.loc[idx, "b0_te_pool"].iloc[0])
        room, candidate = _allocate_softmax(pool, feat.loc[idx, "te_r5p_oos_score"].to_numpy(float))
        feat.loc[idx, "te_r5p_oos_room_share"] = room
        feat.loc[idx, "te_r5p_oos_entitlement_tgt_share"] = candidate
    final = feat.set_index("_row_index")["te_r5p_oos_entitlement_tgt_share"]
    out.loc[final.index, "entitlement_tgt_share"] = final.astype(float)
    out.loc[final.index, "te_r5p_oos_applied"] = True
    before = te.groupby(["event_id", "team"])["entitlement_tgt_share"].sum().rename("before")
    after = out.loc[pos.eq("TE")].groupby(["event_id", "team"])["entitlement_tgt_share"].sum().rename("after")
    pool = pd.concat([before, after], axis=1)
    pool["gap"] = (pool["after"] - pool["before"]).abs()
    non_te_gap = (pd.to_numeric(out.loc[~pos.eq("TE"), "entitlement_tgt_share"], errors="raise") - baseline.loc[~pos.eq("TE")]).abs()
    audit = {
        "model": "TE_R5P_PRODUCTION_CONTRACT_OOS_FOLD", "test_season": int(params["test_season"]),
        "applied_players": int(len(final)),
        "team_te_pool_max_abs_gap": float(pool["gap"].max()) if len(pool) else 0.0,
        "non_te_max_abs_gap": float(non_te_gap.max()) if len(non_te_gap) else 0.0,
    }
    if audit["team_te_pool_max_abs_gap"] > 1e-10 or audit["non_te_max_abs_gap"] > 1e-10:
        raise RuntimeError(f"TE fold conservation failed: {audit}")
    return out, feat, audit


def apply_wr_fold(metrics: pd.DataFrame, *, snaps: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    out = metrics.copy()
    if not out.index.is_unique:
        raise RuntimeError("WR historical fold requires unique player-row index")
    baseline = pd.to_numeric(out["entitlement_tgt_share"], errors="raise").astype(float)
    out["wr_r15_oos_baseline_entitlement"] = baseline
    out["wr_r15_oos_applied"] = False
    out["wr_r15_oos_anchor"] = False
    out["wr_r15_oos_test_season"] = int(params["test_season"])
    pos = out["position"].map(_pos_family)
    wr = out.loc[pos.eq("WR")].copy()
    if wr.empty:
        raise RuntimeError("WR historical fold found zero WRs")
    anchors, secondary = [], []
    for _, g in wr.groupby(["event_id", "team"], sort=False):
        anchor = pd.to_numeric(g["entitlement_tgt_share"], errors="raise").idxmax()
        anchors.append(anchor)
        secondary.extend([i for i in g.index if i != anchor])
    if not secondary:
        raise RuntimeError("WR historical fold found zero WR2+ rows")
    out.loc[anchors, "wr_r15_oos_anchor"] = True
    sec = out.loc[secondary].copy()
    sec["_source_row_index"] = sec.index
    sec["b0_secondary_pool"] = sec.groupby(["event_id", "team"])["entitlement_tgt_share"].transform("sum")
    sec["b0_secondary_room_share"] = np.where(sec["b0_secondary_pool"].gt(0), sec["entitlement_tgt_share"] / sec["b0_secondary_pool"], 0.0)
    sec["log_b0_secondary_pool"] = np.log1p(sec["b0_secondary_pool"].clip(lower=0.0))
    sec["secondary_room_size"] = sec.groupby(["event_id", "team"])["player_clean_key"].transform("count").astype(float)
    feat, future_violations = _wr_strict_prior_features(sec, snaps)
    if int(future_violations) != 0:
        raise RuntimeError(f"WR historical fold used same/future participation rows: {future_violations}")
    feat["prior1_same_team_available"] = feat["prior1_same_team"].fillna(False).astype(float)
    feat["prior3_same_team_available"] = feat["prior3_same_team"].fillna(False).astype(float)
    feat["log1p_prior_count_same_team"] = np.log1p(pd.to_numeric(feat["prior_count_same_team"], errors="coerce").fillna(0).clip(lower=0))
    feat["log1p_prior_count_anyteam"] = np.log1p(pd.to_numeric(feat["prior_count_anyteam"], errors="coerce").fillna(0).clip(lower=0))
    for src, dst in (("prior1_same_team_offense_pct", "secondary_snap_share_prior1_same_team"), ("prior3_anyteam_offense_pct", "secondary_snap_share_prior3_anyteam")):
        z = pd.to_numeric(feat[src], errors="coerce").fillna(0.0).clip(lower=0.0)
        den = z.groupby([feat["event_id"], feat["team"]]).transform("sum")
        feat[dst] = np.where(den.gt(0), z / den, 0.0)
    for c in WR_FEATURES:
        feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)
    x = feat[WR_FEATURES].to_numpy(float)
    residual = ((x - params["mean"]) / params["scale"]) @ params["coef"] + params["intercept"]
    residual = np.clip(residual, -1.0, 1.0)
    feat["wr_r15_oos_residual"] = residual
    feat["wr_r15_oos_score"] = np.log(feat["b0_secondary_room_share"].clip(lower=0.0).to_numpy(float) + 0.02) + residual
    feat["wr_r15_oos_secondary_room_share"] = 0.0
    feat["wr_r15_oos_entitlement_tgt_share"] = pd.to_numeric(feat["entitlement_tgt_share"], errors="raise").astype(float)
    for _, idx in feat.groupby(["event_id", "team"], sort=False).groups.items():
        pool = float(feat.loc[idx, "b0_secondary_pool"].iloc[0])
        room, candidate = _allocate_softmax(pool, feat.loc[idx, "wr_r15_oos_score"].to_numpy(float))
        feat.loc[idx, "wr_r15_oos_secondary_room_share"] = room
        feat.loc[idx, "wr_r15_oos_entitlement_tgt_share"] = candidate
    final = feat.set_index("_source_row_index")["wr_r15_oos_entitlement_tgt_share"]
    out.loc[final.index, "entitlement_tgt_share"] = final.astype(float)
    out.loc[final.index, "wr_r15_oos_applied"] = True
    anchor_gap = (pd.to_numeric(out.loc[anchors, "entitlement_tgt_share"], errors="raise") - baseline.loc[anchors]).abs()
    non_wr_gap = (pd.to_numeric(out.loc[~pos.eq("WR"), "entitlement_tgt_share"], errors="raise") - baseline.loc[~pos.eq("WR")]).abs()
    before_secondary = wr.loc[secondary].groupby(["event_id", "team"])["entitlement_tgt_share"].sum()
    after_secondary = out.loc[secondary].groupby(["event_id", "team"])["entitlement_tgt_share"].sum()
    sec_gap = (after_secondary - before_secondary).abs()
    before_room = wr.groupby(["event_id", "team"])["entitlement_tgt_share"].sum()
    after_room = out.loc[pos.eq("WR")].groupby(["event_id", "team"])["entitlement_tgt_share"].sum()
    room_gap = (after_room - before_room).abs()
    audit = {
        "model": "WR_R15_OOS_FOLD", "test_season": int(params["test_season"]),
        "applied_secondary_players": int(len(final)), "anchors": int(len(anchors)),
        "m38_wr1_anchor_max_abs_gap": float(anchor_gap.max()) if len(anchor_gap) else 0.0,
        "wr2plus_pool_max_abs_gap": float(sec_gap.max()) if len(sec_gap) else 0.0,
        "wr_room_mass_max_abs_gap": float(room_gap.max()) if len(room_gap) else 0.0,
        "non_wr_max_abs_gap": float(non_wr_gap.max()) if len(non_wr_gap) else 0.0,
        "same_future_participation": int(future_violations),
    }
    if max(audit["m38_wr1_anchor_max_abs_gap"], audit["wr2plus_pool_max_abs_gap"], audit["wr_room_mass_max_abs_gap"], audit["non_wr_max_abs_gap"]) > 1e-10:
        raise RuntimeError(f"WR fold conservation failed: {audit}")
    return out, feat, audit


def _compare_results_exact(left, right, *, label: str) -> dict:
    if set(left.values) != set(right.values):
        missing = sorted(set(left.values) - set(right.values))[:10]
        extra = sorted(set(right.values) - set(left.values))[:10]
        raise RuntimeError(f"{label} key universe changed missing={missing} extra={extra}")
    max_mean = max_element = 0.0
    changed = 0
    for key in left.values:
        a = np.asarray(left.values[key], dtype=float)
        b = np.asarray(right.values[key], dtype=float)
        if a.shape != b.shape:
            raise RuntimeError(f"{label} shape mismatch key={key}: {a.shape} vs {b.shape}")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise RuntimeError(f"{label} non-finite simulation array key={key}")
        mg = abs(float(a.mean()) - float(b.mean())) if len(a) else 0.0
        eg = float(np.max(np.abs(a - b))) if len(a) else 0.0
        max_mean, max_element = max(max_mean, mg), max(max_element, eg)
        if eg > 1e-12:
            changed += 1
    if max_mean > 1e-12 or max_element > 1e-12 or changed:
        raise RuntimeError(f"{label} is not exact: changed={changed} max_mean={max_mean} max_element={max_element}")
    return {"keys": int(len(left.values)), "changed_arrays": int(changed), "max_mean_gap": float(max_mean), "max_element_gap": float(max_element)}


def _authorized(row: pd.Series, season: int) -> tuple[bool, str]:
    market, pos = str(row.get("market", "")).lower(), _pos_family(row.get("position"))
    if market not in RECEIVING_MARKETS:
        return False, "BASELINE_UNTREATED"
    if pos == "TE" and int(season) in {2024, 2025}:
        return True, "TE_R5P_OOS_PRODUCTION_ORDER"
    if pos == "WR" and int(season) == 2024:
        return True, "WR_R15_OOS_PRODUCTION_ORDER"
    if pos == "WR" and int(season) == 2025:
        return False, "NO_WR_R15_OOS_AUTHORITY"
    return False, "BASELINE_UNTREATED"


def persist_season(*, player_logs_path: Path, team_weekly_path: Path, schedule_path: Path, universe_dir: Path,
                   component_file: Path, component_out: Path, season: int, prior_season: int, weeks: list[int],
                   out_dir: Path, te_coefficients: Path, wr_coefficients: Path,
                   injuries_path: Path | None, weather_path: Path | None, iterations: int) -> pd.DataFrame:
    if int(season) not in {2024, 2025}:
        raise RuntimeError(f"frozen WR/TE replay supports only 2024/2025, got {season}")
    player_logs = _read(player_logs_path, "player logs")
    team_weekly = _read(team_weekly_path, "historical team-week features")
    schedule = _read(schedule_path, "historical schedule")
    component = _canon_keys(_read(component_file, "component predictions"))
    injuries_history, weather_history = _read_optional(injuries_path), _read_optional(weather_path)
    te_params = _load_fold_params(te_coefficients, test_season=int(season), features=TE_FEATURES, label="TE-R5P")
    wr_params = _load_fold_params(wr_coefficients, test_season=2024, features=WR_FEATURES, label="WR-R15") if int(season) == 2024 else None
    snaps, snap_dup_rate, snap_source_seasons = _load_participation_snaps()
    if snap_dup_rate > 0.01:
        raise RuntimeError(f"participation snap duplicate rate too high: {snap_dup_rate}")
    out_dir.mkdir(parents=True, exist_ok=True)
    updates, audits = [], []
    for week in weeks:
        universe = _read(universe_dir / f"{season}_week_{week:02d}.csv", f"pregame universe {season} W{week:02d}")
        injuries, weather = _exact_week(injuries_history, season, week), _exact_week(weather_history, season, week)
        seed = 42 + int(week)
        bundle = build_historical_context_bundle(
            player_logs=player_logs, team_weekly=team_weekly, pregame_universe=universe,
            schedule=schedule, season=int(season), week=int(week), prior_season=int(prior_season),
            injuries=injuries, weather=weather,
        )
        metrics = build_mc_predictions(bundle, iterations=int(iterations), seed=seed)
        legacy = legacy_simulate(metrics, iterations=int(iterations), seed=seed)
        player_cols = ["event_id", "team", "player_clean_key"]
        players = metrics.sort_values(player_cols).drop_duplicates(player_cols, keep="last").copy()
        if players.duplicated(player_cols).any():
            raise RuntimeError(f"{season} W{week:02d}: player-level state is not unique")
        explicit_base, _ = materialize_target_entitlement(players)
        explicit_baseline = explicit_simulate(explicit_base, iterations=int(iterations), seed=seed)
        parity = _compare_results_exact(legacy, explicit_baseline, label=f"{season} W{week:02d} explicit entitlement baseline")
        te_final, _, te_audit = apply_te_fold(explicit_base, snaps=snaps, params=te_params)
        if int(season) == 2024:
            final, _, wr_audit = apply_wr_fold(te_final, snaps=snaps, params=wr_params)
        else:
            final = te_final
            wr_audit = {
                "m38_wr1_anchor_max_abs_gap": 0.0, "wr2plus_pool_max_abs_gap": 0.0,
                "wr_room_mass_max_abs_gap": 0.0, "non_wr_max_abs_gap": 0.0,
                "same_future_participation": 0,
            }
        specialist = explicit_simulate(final, iterations=int(iterations), seed=seed)
        base_team = explicit_base.groupby(["event_id", "team"])["entitlement_tgt_share"].sum()
        final_team = final.groupby(["event_id", "team"])["entitlement_tgt_share"].sum()
        team_gap = float((final_team - base_team).abs().max()) if len(base_team) else 0.0
        if team_gap > 1e-10:
            raise RuntimeError(f"{season} W{week:02d}: team modeled target mass changed: {team_gap}")
        arrays, meta_rows = {}, []
        authorized_rows = wr_2025_apps = 0
        for i, (_, row) in enumerate(metrics.iterrows()):
            base_arr = _historical_outcomes(legacy, row)
            if base_arr is None:
                continue
            treatment, route = _authorized(row, int(season))
            arr = base_arr
            if treatment:
                spec_arr = _historical_outcomes(specialist, row)
                if spec_arr is None:
                    raise RuntimeError(f"{season} W{week:02d}: specialist distribution missing {row.get('player_clean_key')} {row.get('market')}")
                arr, authorized_rows = spec_arr, authorized_rows + 1
            if int(season) == 2025 and _pos_family(row.get("position")) == "WR" and route.startswith("WR_R15"):
                wr_2025_apps += 1
            if len(arr) != int(iterations):
                raise RuntimeError(f"{season} W{week:02d}: draw-count mismatch")
            key = f"a{i:06d}"
            arrays[key] = np.asarray(arr, dtype=float)
            meta_rows.append({
                "season": int(season), "week": int(week), "team": canon_team(row.get("team")),
                "opponent": canon_team(row.get("opponent")), "player": row.get("player"),
                "player_clean_key": str(row.get("player_clean_key")), "position": row.get("position"),
                "market": str(row.get("market")).lower(), "event_id": row.get("event_id"),
                "array_key": key, "draws": int(len(arr)), "mc_mean": float(np.mean(arr)),
                "mc_sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "wrte_authorized_treatment": bool(treatment), "wrte_route": route,
                "baseline_mc_mean": float(np.mean(base_arr)),
            })
            if treatment:
                updates.append({
                    "season": int(season), "week": int(week), "team": canon_team(row.get("team")),
                    "opponent": canon_team(row.get("opponent")), "player_clean_key": str(row.get("player_clean_key")),
                    "market": str(row.get("market")).lower(), "specialist_mc_proj": float(np.mean(arr)),
                    "baseline_mc_proj_rebuilt": float(np.mean(base_arr)), "wrte_route": route,
                })
        if wr_2025_apps:
            raise RuntimeError(f"2025 WR-R15 applications forbidden, found {wr_2025_apps}")
        meta = _canon_keys(pd.DataFrame(meta_rows))
        if meta.duplicated(KEYS).any():
            bad = meta.loc[meta.duplicated(KEYS, keep=False), KEYS].head(10).to_dict("records")
            raise RuntimeError(f"{season} W{week:02d}: duplicate hybrid identities: {bad}")
        shard = f"{season}_week_{week:02d}.npz"
        meta["npz_file"] = shard
        comp_week = component.loc[component["season"].eq(int(season)) & component["week"].eq(int(week))].copy()
        chk_base = comp_week[KEYS + ["mc_proj"]].merge(meta[KEYS + ["baseline_mc_mean"]], on=KEYS, how="left", validate="one_to_one")
        if chk_base["baseline_mc_mean"].isna().any():
            raise RuntimeError(f"{season} W{week:02d}: baseline distribution lineage incomplete")
        base_delta = (pd.to_numeric(chk_base["mc_proj"], errors="coerce") - pd.to_numeric(chk_base["baseline_mc_mean"], errors="coerce")).abs()
        max_base_delta = float(base_delta.max()) if len(base_delta) else 0.0
        if not np.isfinite(max_base_delta) or max_base_delta > 1e-8:
            raise RuntimeError(f"{season} W{week:02d}: canonical baseline MC mismatch {max_base_delta}")
        np.savez_compressed(out_dir / shard, **arrays)
        meta.to_csv(out_dir / f"{season}_week_{week:02d}_metadata.csv", index=False)
        audits.append({
            "season": int(season), "week": int(week), "seed": seed, "iterations": int(iterations),
            "hybrid_distribution_rows": int(len(meta)), "authorized_rows": int(authorized_rows),
            "baseline_max_abs_mc_mean_delta": max_base_delta,
            "explicit_baseline_keys": parity["keys"], "explicit_baseline_changed_arrays": parity["changed_arrays"],
            "explicit_baseline_max_mean_gap": parity["max_mean_gap"], "explicit_baseline_max_element_gap": parity["max_element_gap"],
            "team_modeled_mass_max_abs_gap": team_gap,
            "te_pool_max_abs_gap": te_audit["team_te_pool_max_abs_gap"], "te_non_te_max_abs_gap": te_audit["non_te_max_abs_gap"],
            "wr1_anchor_max_abs_gap": wr_audit["m38_wr1_anchor_max_abs_gap"], "wr2plus_pool_max_abs_gap": wr_audit["wr2plus_pool_max_abs_gap"],
            "wr_room_mass_max_abs_gap": wr_audit["wr_room_mass_max_abs_gap"], "wr_non_wr_max_abs_gap": wr_audit["non_wr_max_abs_gap"],
            "wr_same_future_participation": wr_audit["same_future_participation"], "wr_2025_applications": int(wr_2025_apps),
            "snap_duplicate_rate": float(snap_dup_rate), "snap_source_seasons": ",".join(str(v) for v in snap_source_seasons), "status": "PASS",
        })
        print(f"[wrte-historical] {season} W{week:02d} rows={len(meta)} authorized={authorized_rows} base_delta={max_base_delta:.3g} explicit_gap={parity['max_element_gap']:.3g}")

    updates_df = _canon_keys(pd.DataFrame(updates)) if updates else pd.DataFrame(columns=KEYS + ["specialist_mc_proj"])
    if len(updates_df) and updates_df.duplicated(KEYS).any():
        raise RuntimeError("specialist component updates contain duplicate identities")
    comp_out = component.copy()
    if len(updates_df):
        joined = comp_out.merge(updates_df[KEYS + ["specialist_mc_proj", "wrte_route"]], on=KEYS, how="left", validate="one_to_one")
        mask = joined["specialist_mc_proj"].notna()
        joined["wrte_baseline_mc_proj"] = pd.to_numeric(joined["mc_proj"], errors="coerce")
        joined["wrte_authorized_treatment"] = mask
        joined["wrte_route"] = joined["wrte_route"].fillna("BASELINE_UNTREATED")
        joined.loc[mask, "mc_proj"] = pd.to_numeric(joined.loc[mask, "specialist_mc_proj"], errors="raise")
        comp_out = joined.drop(columns=["specialist_mc_proj"])
    else:
        comp_out["wrte_baseline_mc_proj"] = pd.to_numeric(comp_out["mc_proj"], errors="coerce")
        comp_out["wrte_authorized_treatment"] = False
        comp_out["wrte_route"] = "BASELINE_UNTREATED"
    all_meta = pd.concat([pd.read_csv(p) for p in sorted(out_dir.glob(f"{season}_week_*_metadata.csv"))], ignore_index=True)
    all_meta = _canon_keys(all_meta)
    chk = comp_out[KEYS + ["mc_proj"]].merge(all_meta[KEYS + ["mc_mean"]], on=KEYS, how="left", validate="one_to_one")
    if chk["mc_mean"].isna().any():
        raise RuntimeError(f"{season}: component rows missing hybrid distribution lineage")
    delta = (pd.to_numeric(chk["mc_proj"], errors="coerce") - pd.to_numeric(chk["mc_mean"], errors="coerce")).abs()
    max_delta = float(delta.max()) if len(delta) else 0.0
    if not np.isfinite(max_delta) or max_delta > 1e-8:
        raise RuntimeError(f"{season}: specialist component/distribution mean mismatch {max_delta}")
    component_out.parent.mkdir(parents=True, exist_ok=True)
    comp_out.to_csv(component_out, index=False)
    pd.DataFrame(audits).to_csv(out_dir / f"{season}_wrte_replay_audit.csv", index=False)
    updates_df.to_csv(out_dir / f"{season}_wrte_component_updates.csv", index=False)
    return pd.DataFrame(audits)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--prior-season", type=int, required=True)
    ap.add_argument("--weeks", default="1-18")
    ap.add_argument("--player-logs", type=Path, required=True)
    ap.add_argument("--team-weekly", type=Path, required=True)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--universe-dir", type=Path, required=True)
    ap.add_argument("--component-file", type=Path, required=True)
    ap.add_argument("--component-out", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--te-coefficients", type=Path, required=True)
    ap.add_argument("--wr-coefficients", type=Path, required=True)
    ap.add_argument("--injuries", type=Path, default=Path("data/backtests/injuries_history.csv"))
    ap.add_argument("--weather", type=Path, default=Path("data/backtests/weather_history.csv"))
    ap.add_argument("--iterations", type=int, default=2000)
    a = ap.parse_args()
    persist_season(
        player_logs_path=a.player_logs, team_weekly_path=a.team_weekly, schedule_path=a.schedule,
        universe_dir=a.universe_dir, component_file=a.component_file, component_out=a.component_out,
        season=a.season, prior_season=a.prior_season, weeks=_parse_weeks(a.weeks), out_dir=a.out_dir,
        te_coefficients=a.te_coefficients, wr_coefficients=a.wr_coefficients,
        injuries_path=a.injuries, weather_path=a.weather, iterations=a.iterations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
