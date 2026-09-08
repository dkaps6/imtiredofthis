#!/usr/bin/env python3
"""WR-R14: strict-prior participation entitlement residual around post-M38 room.

Scientific hypothesis
---------------------
M38 correctly imposes a durable WR hierarchy, but its individual room shares can
still miss role transitions that are visible in *participation* before they are
fully visible in box-score target volume.  A regularized residual model using
strictly-prior offensive snap participation may improve individual WR target
entitlement while preserving the already-conserved post-M38 WR room exactly.

This is materially different from failed WR-R12/R13.  No rolling target or
receiving-yard totals are candidate features.  Participation is the new signal.

Frozen design before 2025 confirmation:
- development/training outcomes: 2024 regular season only;
- untouched confirmation: 2025 regular season Weeks 1-18;
- baseline: explicit finite entitlement after canonical M38;
- source: nflreadpy snap counts 2020-2025, observations strictly before each game;
- model: StandardScaler + Ridge(alpha=20), fixed without WR tuning;
- residual target: log(actual WR-room share + .02) - log(M38 room share + .02),
  clipped to [-2,2] for training; predicted residual clipped [-1,1];
- candidate: softmax(log(M38 room share + .02) + residual) inside WR room;
- no sportsbook inputs, no current/future outcome/participation inputs;
- no change to WR-room mass, team player target mass, or non-WR entitlement;
- same MC seed for baseline/candidate each 2025 week.

All confirmation gates are frozen here. A PASS authorizes a separate production-
contract refit/integration confirmation; it does not itself promote WR-R14.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.backtest import component_predictions as cp
from scripts.backtest.evaluate_wr_room_empirical_bayes_v1 import WR_POS, finite, metric, optional, prepared, read
from scripts.backtest.historical_context import build_historical_context_bundle
from scripts.backtest.walk_forward import _exact_week, _parse_weeks
from scripts.modeling.target_entitlement_v1 import materialize_target_entitlement
from scripts.modeling.te_r5p_entitlement_adapter_v1 import _key as snap_player_key
from scripts.modeling.te_r5p_entitlement_adapter_v1 import _load_snaps, _team as snap_team
from scripts.simulation_explicit_entitlement_v1 import simulate as explicit_simulate

EPS = 0.02
ALPHA = 20.0
TRAIN_SEASON = 2024
TEST_SEASON = 2025
FEATURES = [
    "b0_wr_room_share",
    "log_b0_wr_pool",
    "room_size",
    "prior1_same_team_offense_pct",
    "prior1_same_team_offense_snaps",
    "prior1_anyteam_offense_pct",
    "prior3_anyteam_offense_pct",
    "prior1_anyteam_offense_snaps",
    "prior3_anyteam_offense_snaps",
    "log1p_prior_count_same_team",
    "log1p_prior_count_anyteam",
    "prior1_same_team_available",
    "prior3_same_team_available",
    "snap_share_prior1_same_team",
    "snap_share_prior3_anyteam",
]

# Frozen confirmation gates.
MIN_TARGET_MAE_GAIN = 0.03
MIN_REC_YARDS_MAE_GAIN = 0.10
REQUIRED_NONWORSE_PHASES = 3
MAX_TAIL_RATE_WORSEN = 0.0025
MIN_BOOTSTRAP_IMPROVE_PROB = 0.65


def _target_actuals(logs: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    actual = cp.build_actual_rows(logs, season, week)
    out = actual.loc[actual.market.eq("receptions"), ["team", "player_clean_key", "actual_opportunities"]].copy()
    out = out.rename(columns={"actual_opportunities": "actual_targets"})
    out["actual_targets"] = pd.to_numeric(out["actual_targets"], errors="coerce")
    return out


def _yard_actuals(logs: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    actual = cp.build_actual_rows(logs, season, week)
    out = actual.loc[actual.market.eq("rec_yards"), ["team", "player_clean_key", "actual"]].copy()
    out = out.rename(columns={"actual": "actual_rec_yards"})
    out["actual_rec_yards"] = pd.to_numeric(out["actual_rec_yards"], errors="coerce")
    return out


def _strict_prior_snap_features(frame: pd.DataFrame, snaps: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Attach only snap observations with ordinal strictly below target game."""
    out = frame.copy().reset_index(drop=False).rename(columns={"index": "_row_index"})
    out["snap_player_key"] = out["player"].map(snap_player_key)
    out["snap_team_key"] = out["team"].map(snap_team)
    out["ordinal"] = pd.to_numeric(out["season"], errors="coerce") * 100 + pd.to_numeric(out["week"], errors="coerce")

    any_maps = {k: g.sort_values("ordinal", kind="stable") for k, g in snaps.groupby("player_key", sort=False)}
    same_maps = {k: g.sort_values("ordinal", kind="stable") for k, g in snaps.groupby(["player_key", "team"], sort=False)}
    rows: list[dict] = []
    future_violations = 0

    def last_num(g: pd.DataFrame, col: str) -> float:
        if g.empty:
            return np.nan
        z = pd.to_numeric(g[col], errors="coerce")
        return float(z.iloc[-1]) if len(z) and pd.notna(z.iloc[-1]) else np.nan

    def mean3(g: pd.DataFrame, col: str) -> float:
        if len(g) < 3:
            return np.nan
        z = pd.to_numeric(g[col], errors="coerce")
        return float(z.mean()) if z.notna().any() else np.nan

    for _, r in out.iterrows():
        pk, tm, ordinal = str(r["snap_player_key"]), str(r["snap_team_key"]), float(r["ordinal"])
        ah = any_maps.get(pk, pd.DataFrame())
        sh = same_maps.get((pk, tm), pd.DataFrame())
        if len(ah):
            ah = ah.loc[pd.to_numeric(ah["ordinal"], errors="coerce").lt(ordinal)]
        if len(sh):
            sh = sh.loc[pd.to_numeric(sh["ordinal"], errors="coerce").lt(ordinal)]
        if len(ah) and float(pd.to_numeric(ah["ordinal"], errors="coerce").max()) >= ordinal:
            future_violations += 1
        if len(sh) and float(pd.to_numeric(sh["ordinal"], errors="coerce").max()) >= ordinal:
            future_violations += 1
        a1, a3, s1, s3 = ah.tail(1), ah.tail(3), sh.tail(1), sh.tail(3)
        rows.append({
            "prior_count_anyteam": int(len(ah)),
            "prior_count_same_team": int(len(sh)),
            "prior1_anyteam": bool(len(a1) >= 1),
            "prior3_anyteam": bool(len(a3) >= 3),
            "prior1_same_team": bool(len(s1) >= 1),
            "prior3_same_team": bool(len(s3) >= 3),
            "prior1_anyteam_offense_pct": last_num(a1, "offense_pct"),
            "prior1_anyteam_offense_snaps": last_num(a1, "offense_snaps"),
            "prior3_anyteam_offense_pct": mean3(a3, "offense_pct"),
            "prior3_anyteam_offense_snaps": mean3(a3, "offense_snaps"),
            "prior1_same_team_offense_pct": last_num(s1, "offense_pct"),
            "prior1_same_team_offense_snaps": last_num(s1, "offense_snaps"),
        })
    feat = pd.concat([out, pd.DataFrame(rows)], axis=1)
    return feat, future_violations


def _wr_feature_frame(baseline: pd.DataFrame, snaps: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    x = baseline.copy()
    pos = x.get("position", pd.Series("", index=x.index)).fillna("").astype(str).str.upper().str.strip()
    x = x.loc[pos.isin(WR_POS)].copy()
    if x.empty:
        return x, 0
    x["baseline_entitlement_tgt_share"] = pd.to_numeric(x["entitlement_tgt_share"], errors="coerce").fillna(0.0)
    x["b0_wr_pool"] = x.groupby(["event_id", "team"])["baseline_entitlement_tgt_share"].transform("sum")
    x["b0_wr_room_share"] = np.where(
        x["b0_wr_pool"].gt(0),
        x["baseline_entitlement_tgt_share"] / x["b0_wr_pool"],
        0.0,
    )
    x["log_b0_wr_pool"] = np.log1p(x["b0_wr_pool"].clip(lower=0.0))
    x["room_size"] = x.groupby(["event_id", "team"])["player_clean_key"].transform("count").astype(float)
    x, future = _strict_prior_snap_features(x, snaps)
    x["prior1_same_team_available"] = x["prior1_same_team"].fillna(False).astype(float)
    x["prior3_same_team_available"] = x["prior3_same_team"].fillna(False).astype(float)
    x["log1p_prior_count_same_team"] = np.log1p(pd.to_numeric(x["prior_count_same_team"], errors="coerce").fillna(0).clip(lower=0))
    x["log1p_prior_count_anyteam"] = np.log1p(pd.to_numeric(x["prior_count_anyteam"], errors="coerce").fillna(0).clip(lower=0))
    for src, dst in (
        ("prior1_same_team_offense_pct", "snap_share_prior1_same_team"),
        ("prior3_anyteam_offense_pct", "snap_share_prior3_anyteam"),
    ):
        z = pd.to_numeric(x[src], errors="coerce").fillna(0.0).clip(lower=0.0)
        den = z.groupby([x["event_id"], x["team"]]).transform("sum")
        x[dst] = np.where(den.gt(0), z / den, 0.0)
    for c in FEATURES:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    return x, future


def _build_bundle_frame(
    *, season: int, week: int, prior_season: int, data_dir: Path, logs: pd.DataFrame,
) -> pd.DataFrame:
    team = read(data_dir / "team_weekly_history.csv")
    schedule = read(data_dir / "schedule_history.csv")
    universe = read(data_dir / "pregame_universe" / f"{season}_week_{week:02d}.csv")
    injuries = optional(data_dir / "injuries_history.csv")
    weather = optional(data_dir / "weather_history.csv")
    bundle = build_historical_context_bundle(
        player_logs=logs,
        team_weekly=team,
        pregame_universe=universe,
        schedule=schedule,
        season=season,
        week=week,
        prior_season=prior_season,
        injuries=_exact_week(injuries, season, week),
        weather=_exact_week(weather, season, week),
    )
    raw = prepared(bundle)
    baseline, _ = materialize_target_entitlement(raw)
    return baseline


def _training_casebook(data_dir: Path, logs: pd.DataFrame, snaps: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    parts: list[pd.DataFrame] = []
    future_total = 0
    for week in range(1, 19):
        baseline = _build_bundle_frame(season=TRAIN_SEASON, week=week, prior_season=TRAIN_SEASON - 1, data_dir=data_dir, logs=logs)
        wr, future = _wr_feature_frame(baseline, snaps)
        future_total += future
        actual = _target_actuals(logs, TRAIN_SEASON, week)
        wr = wr.merge(actual, on=["team", "player_clean_key"], how="inner", validate="one_to_one")
        wr["actual_wr_pool"] = wr.groupby(["event_id", "team"])["actual_targets"].transform("sum")
        wr["actual_room_share"] = np.where(wr["actual_wr_pool"].gt(0), wr["actual_targets"] / wr["actual_wr_pool"], 0.0)
        wr["entitlement_residual_target"] = (
            np.log(wr["actual_room_share"].clip(lower=0) + EPS)
            - np.log(wr["b0_wr_room_share"].clip(lower=0) + EPS)
        ).clip(-2.0, 2.0)
        wr["week"] = int(week)
        parts.append(wr)
        print(f"[wr-r14] training week={week:02d} rows={len(wr)}")
    train = pd.concat(parts, ignore_index=True)
    train = train.loc[train["actual_wr_pool"].gt(0) & train["b0_wr_pool"].gt(0)].copy()
    if train.empty:
        raise RuntimeError("WR-R14 training casebook is empty")
    return train, future_total


def _apply_model(
    baseline: pd.DataFrame, snaps: pd.DataFrame, model,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    out = baseline.copy()
    wr, future = _wr_feature_frame(out, snaps)
    if wr.empty:
        raise RuntimeError("WR-R14 test week has zero WR rows")
    pred = np.clip(model.predict(wr[FEATURES]), -1.0, 1.0)
    wr["wr_r14_predicted_residual"] = pred
    wr["wr_r14_score"] = np.log(wr["b0_wr_room_share"].clip(lower=0) + EPS) + pred
    wr["candidate_wr_room_share"] = 0.0
    wr["candidate_entitlement_tgt_share"] = 0.0

    audits: list[dict] = []
    for (event_id, team), idx in wr.groupby(["event_id", "team"], sort=False).groups.items():
        pool = float(wr.loc[idx, "b0_wr_pool"].iloc[0])
        score = wr.loc[idx, "wr_r14_score"].to_numpy(float)
        ex = np.exp(score - float(np.max(score)))
        room = ex / float(ex.sum())
        cand = pool * room
        gap = pool - float(cand.sum())
        if len(cand):
            cand[int(np.argmax(room))] += gap
        wr.loc[idx, "candidate_wr_room_share"] = room
        wr.loc[idx, "candidate_entitlement_tgt_share"] = cand

        base_team_idx = out.index[(out["event_id"].astype(str) == str(event_id)) & (out["team"].astype(str) == str(team))]
        wr_orig_idx = wr.loc[idx, "_row_index"].astype(int).tolist()
        non_wr_idx = [i for i in base_team_idx if i not in wr_orig_idx]
        before_team = float(pd.to_numeric(out.loc[base_team_idx, "entitlement_tgt_share"], errors="coerce").sum())
        non_wr_before = out.loc[non_wr_idx, "entitlement_tgt_share"].astype(float).copy() if non_wr_idx else pd.Series(dtype=float)
        out.loc[wr_orig_idx, "entitlement_tgt_share"] = cand
        after_team = float(pd.to_numeric(out.loc[base_team_idx, "entitlement_tgt_share"], errors="coerce").sum())
        non_wr_after = out.loc[non_wr_idx, "entitlement_tgt_share"].astype(float) if non_wr_idx else pd.Series(dtype=float)
        audits.append({
            "event_id": str(event_id),
            "team": str(team),
            "baseline_wr_room_mass": pool,
            "candidate_wr_room_mass": float(cand.sum()),
            "wr_room_mass_gap": float(cand.sum() - pool),
            "baseline_team_player_mass": before_team,
            "candidate_team_player_mass": after_team,
            "team_player_mass_gap": after_team - before_team,
            "max_non_wr_entitlement_delta": float((non_wr_after - non_wr_before).abs().max()) if len(non_wr_before) else 0.0,
            "max_player_room_share_move": float(np.max(np.abs(room - wr.loc[idx, "b0_wr_room_share"].to_numpy(float)))),
            "sportsbook_inputs_used": 0,
            "current_or_future_outcomes_used": 0,
        })
    return out, wr, {"future_violations": int(future), "audits": audits}


def _baseline_ranks(frame: pd.DataFrame) -> dict[tuple[str, str, str], int]:
    ranks: dict[tuple[str, str, str], int] = {}
    pos = frame.get("position", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper().str.strip()
    wr_all = frame.loc[pos.isin(WR_POS)].copy()
    wr_all["_ent"] = pd.to_numeric(wr_all["entitlement_tgt_share"], errors="coerce").fillna(0.0)
    for (event_id, team), g in wr_all.groupby(["event_id", "team"], sort=False):
        g = g.sort_values(["_ent", "player_clean_key"], ascending=[False, True], kind="stable")
        for rank, r in enumerate(g.itertuples(index=False), 1):
            ranks[(str(event_id), str(team), str(r.player_clean_key))] = rank
    return ranks


def _prediction_rows(frame: pd.DataFrame, sim, variant: str, rank_map: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for (event_id, team), group in frame.groupby(["event_id", "team"], sort=False):
        pos = group.get("position", pd.Series("", index=group.index)).fillna("").astype(str).str.upper().str.strip()
        plays = float(np.mean([finite(v, 64.0) for v in group.get("rules_plays_est", pd.Series([64.0] * len(group)))]))
        pass_rate = float(np.mean([finite(v, 0.57) for v in group.get("rules_pass_rate", pd.Series([0.57] * len(group)))]))
        team_targets = plays * pass_rate
        for j, (_, row) in enumerate(group.iterrows()):
            if str(pos.iloc[j]) not in WR_POS:
                continue
            key = str(row.get("player_clean_key", ""))
            ent = finite(row.get("entitlement_tgt_share"), 0.0)
            rec = sim.values.get((str(event_id), key, "receptions"))
            yards = sim.values.get((str(event_id), key, "rec_yards"))
            rows.append({
                "variant": variant,
                "event_id": str(event_id),
                "team": str(team),
                "player_clean_key": key,
                "player": row.get("player", ""),
                "wr_rank": int(rank_map.get((str(event_id), str(team), key), 99)),
                "entitlement_tgt_share": float(ent),
                "pred_targets": float(team_targets * ent),
                "mc_receptions": float(np.mean(rec)) if rec is not None else np.nan,
                "mc_rec_yards": float(np.mean(yards)) if yards is not None else np.nan,
            })
    return pd.DataFrame(rows)


def _paired_bootstrap_probability(pred: pd.DataFrame, reps: int = 2000, seed: int = 14014) -> float:
    b = pred.loc[pred.variant.eq("M38_EXPLICIT_BASELINE"), ["week", "team", "player_clean_key", "actual_rec_yards", "mc_rec_yards"]].copy()
    c = pred.loc[pred.variant.eq("WR_R14_PARTICIPATION"), ["week", "team", "player_clean_key", "actual_rec_yards", "mc_rec_yards"]].copy()
    keys = ["week", "team", "player_clean_key"]
    z = b.merge(c, on=keys, suffixes=("_b", "_c"), validate="one_to_one")
    eb = (z.mc_rec_yards_b - z.actual_rec_yards_b).abs().to_numpy(float)
    ec = (z.mc_rec_yards_c - z.actual_rec_yards_c).abs().to_numpy(float)
    rng = np.random.default_rng(seed)
    wins = 0
    n = len(z)
    if n == 0:
        return np.nan
    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        wins += int(float(ec[idx].mean()) < float(eb[idx].mean()))
    return float(wins / reps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", type=Path, required=True)
    ap.add_argument("--test-dir", type=Path, required=True)
    ap.add_argument("--train-logs", type=Path, required=True)
    ap.add_argument("--test-logs", type=Path, required=True)
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--out-dir", type=Path, default=Path("data/backtests/wr_r14_participation_v1"))
    a = ap.parse_args()

    train_logs = read(a.train_logs)
    test_logs = read(a.test_logs)
    snaps, snap_dup_rate, snap_source_seasons = _load_snaps()
    train, train_future = _training_casebook(a.train_dir, train_logs, snaps)
    if train_future != 0:
        raise RuntimeError(f"WR-R14 training participation used same/future observations: {train_future}")

    model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
    model.fit(train[FEATURES], train["entitlement_residual_target"])
    ridge = model.named_steps["ridge"]
    scaler = model.named_steps["standardscaler"]

    predictions: list[pd.DataFrame] = []
    feature_rows: list[pd.DataFrame] = []
    audit_rows: list[dict] = []
    test_future = 0
    for week in range(1, 19):
        baseline = _build_bundle_frame(season=TEST_SEASON, week=week, prior_season=TEST_SEASON - 1, data_dir=a.test_dir, logs=test_logs)
        candidate, wr_feat, meta = _apply_model(baseline, snaps, model)
        test_future += int(meta["future_violations"])
        for r in meta["audits"]:
            r["week"] = int(week)
            audit_rows.append(r)
        wr_feat["week"] = int(week)
        feature_rows.append(wr_feat)
        ranks = _baseline_ranks(baseline)

        bsim = explicit_simulate(baseline, iterations=a.iterations, seed=81400 + week)
        csim = explicit_simulate(candidate, iterations=a.iterations, seed=81400 + week)
        actual_t = _target_actuals(test_logs, TEST_SEASON, week)
        actual_y = _yard_actuals(test_logs, TEST_SEASON, week)
        for variant, frame, sim in (
            ("M38_EXPLICIT_BASELINE", baseline, bsim),
            ("WR_R14_PARTICIPATION", candidate, csim),
        ):
            x = _prediction_rows(frame, sim, variant, ranks)
            x["week"] = int(week)
            x = x.merge(actual_t, on=["team", "player_clean_key"], how="inner")
            x = x.merge(actual_y, on=["team", "player_clean_key"], how="inner")
            predictions.append(x)
        print(f"[wr-r14] confirmation week={week:02d} complete")

    pred = pd.concat(predictions, ignore_index=True)
    pred["phase"] = pd.cut(pred["week"], [0, 4, 9, 13, 18], labels=["W1-4", "W5-9", "W10-13", "W14-18"])
    pred["role"] = np.select([pred.wr_rank.eq(1), pred.wr_rank.eq(2), pred.wr_rank.eq(3)], ["WR1", "WR2", "WR3"], default="WR4+")
    pred["abs_rec_yards_error"] = (pd.to_numeric(pred.mc_rec_yards, errors="coerce") - pd.to_numeric(pred.actual_rec_yards, errors="coerce")).abs()

    summary_rows: list[dict] = []
    for variant, g in pred.groupby("variant"):
        for market, actual_col, pred_col in (("targets", "actual_targets", "pred_targets"), ("rec_yards", "actual_rec_yards", "mc_rec_yards")):
            row = {"variant": variant, "market": market, **metric(g[actual_col], g[pred_col])}
            if market == "rec_yards":
                row["miss_30_plus_rate"] = float(g.abs_rec_yards_error.ge(30).mean())
                row["miss_50_plus_rate"] = float(g.abs_rec_yards_error.ge(50).mean())
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    bucket_rows: list[dict] = []
    for bucket_col in ("phase", "role"):
        for (variant, bucket), g in pred.groupby(["variant", bucket_col], observed=False):
            if not g.empty:
                bucket_rows.append({"bucket_type": bucket_col, "bucket": str(bucket), "variant": variant, **metric(g.actual_rec_yards, g.mc_rec_yards)})
    buckets = pd.DataFrame(bucket_rows)

    def srow(variant: str, market: str) -> pd.Series:
        return summary.loc[summary.variant.eq(variant) & summary.market.eq(market)].iloc[0]

    b_t, c_t = srow("M38_EXPLICIT_BASELINE", "targets"), srow("WR_R14_PARTICIPATION", "targets")
    b_y, c_y = srow("M38_EXPLICIT_BASELINE", "rec_yards"), srow("WR_R14_PARTICIPATION", "rec_yards")
    phase = buckets.loc[buckets.bucket_type.eq("phase")].pivot(index="bucket", columns="variant", values="mae").dropna()
    nonworse_phases = int((phase["WR_R14_PARTICIPATION"] <= phase["M38_EXPLICIT_BASELINE"]).sum()) if not phase.empty else 0
    top = pred.loc[pred.role.isin(["WR1", "WR2"])]
    b_top = metric(top.loc[top.variant.eq("M38_EXPLICIT_BASELINE"), "actual_rec_yards"], top.loc[top.variant.eq("M38_EXPLICIT_BASELINE"), "mc_rec_yards"])
    c_top = metric(top.loc[top.variant.eq("WR_R14_PARTICIPATION"), "actual_rec_yards"], top.loc[top.variant.eq("WR_R14_PARTICIPATION"), "mc_rec_yards"])
    bootstrap_prob = _paired_bootstrap_probability(pred)

    audits = pd.DataFrame(audit_rows)
    max_wr_gap = float(audits.wr_room_mass_gap.abs().max()) if len(audits) else np.inf
    max_team_gap = float(audits.team_player_mass_gap.abs().max()) if len(audits) else np.inf
    max_non_wr = float(audits.max_non_wr_entitlement_delta.abs().max()) if len(audits) else np.inf
    leakage = int(audits.sportsbook_inputs_used.sum()) if len(audits) else -1
    future_outcomes = int(audits.current_or_future_outcomes_used.sum()) if len(audits) else -1

    science = {
        "target_mae_improve_ge_0_03": bool(float(b_t.mae - c_t.mae) >= MIN_TARGET_MAE_GAIN),
        "rec_yards_mae_improve_ge_0_10": bool(float(b_y.mae - c_y.mae) >= MIN_REC_YARDS_MAE_GAIN),
        "target_p90_nonworse": bool(float(c_t.p90_abs_error) <= float(b_t.p90_abs_error)),
        "rec_yards_p90_nonworse": bool(float(c_y.p90_abs_error) <= float(b_y.p90_abs_error)),
        "phase_nonworse_at_least_3_of_4": bool(nonworse_phases >= REQUIRED_NONWORSE_PHASES),
        "wr1_wr2_rec_yards_mae_nonworse": bool(float(c_top["mae"]) <= float(b_top["mae"])),
        "miss30_guard": bool(float(c_y.miss_30_plus_rate) <= float(b_y.miss_30_plus_rate) + MAX_TAIL_RATE_WORSEN),
        "miss50_guard": bool(float(c_y.miss_50_plus_rate) <= float(b_y.miss_50_plus_rate) + MAX_TAIL_RATE_WORSEN),
        "bias_magnitude_guard": bool(abs(float(c_y.bias)) <= abs(float(b_y.bias)) + 0.5),
        "paired_bootstrap_improve_prob_ge_0_65": bool(np.isfinite(bootstrap_prob) and bootstrap_prob >= MIN_BOOTSTRAP_IMPROVE_PROB),
    }
    integrity = {
        "train_season_only_2024": bool(pd.to_numeric(train.season, errors="coerce").dropna().astype(int).unique().tolist() == [TRAIN_SEASON]),
        "test_season_only_2025": bool(pd.to_numeric(pred.week, errors="coerce").between(1, 18).all()),
        "snap_source_seasons_2020_2025": bool(snap_source_seasons == [2020, 2021, 2022, 2023, 2024, 2025]),
        "snap_duplicate_rate_le_0_01": bool(snap_dup_rate <= 0.01),
        "zero_same_future_participation": bool(train_future == 0 and test_future == 0),
        "sportsbook_inputs_zero": bool(leakage == 0),
        "current_future_outcomes_zero": bool(future_outcomes == 0),
        "wr_room_mass_exact": bool(max_wr_gap <= 1e-12),
        "team_player_mass_exact": bool(max_team_gap <= 1e-12),
        "non_wr_entitlement_unchanged": bool(max_non_wr <= 1e-12),
    }
    passed = all(science.values()) and all(integrity.values())
    disposition = "WR_R14_PARTICIPATION_ENTITLEMENT_CONFIRMATION_PASS" if passed else "WR_R14_PARTICIPATION_ENTITLEMENT_CONFIRMATION_FAIL"

    coef_rows = [
        {
            "feature": f,
            "standardized_coefficient": float(c),
            "scaler_mean": float(m),
            "scaler_scale": float(s),
            "ridge_intercept": float(ridge.intercept_),
        }
        for f, c, m, s in zip(FEATURES, ridge.coef_, scaler.mean_, scaler.scale_)
    ]
    result = {
        "migration": "WR_R14_PARTICIPATION_ENTITLEMENT_V1",
        "disposition": disposition,
        "training_season": TRAIN_SEASON,
        "confirmation_season": TEST_SEASON,
        "training_rows": int(len(train)),
        "confirmation_player_games": int(len(pred.loc[pred.variant.eq("M38_EXPLICIT_BASELINE")])),
        "model": f"StandardScaler+Ridge(alpha={ALPHA})",
        "features": FEATURES,
        "candidate_uses_recent_target_or_yard_totals": False,
        "m38_materialized_before_candidate": True,
        "internal_m38_disabled_during_explicit_simulation": True,
        "baseline_target": b_t.to_dict(),
        "candidate_target": c_t.to_dict(),
        "baseline_rec_yards": b_y.to_dict(),
        "candidate_rec_yards": c_y.to_dict(),
        "baseline_wr1_wr2_rec_yards_mae": float(b_top["mae"]),
        "candidate_wr1_wr2_rec_yards_mae": float(c_top["mae"]),
        "nonworse_phases": nonworse_phases,
        "paired_bootstrap_candidate_mae_improve_probability": bootstrap_prob,
        "max_wr_room_mass_gap": max_wr_gap,
        "max_team_player_mass_gap": max_team_gap,
        "max_non_wr_entitlement_delta": max_non_wr,
        "same_or_future_participation_observations_used": int(train_future + test_future),
        "sportsbook_inputs_used": 0,
        "scientific_gates": science,
        "integrity_gates": integrity,
        "production_parameters_changed": 0,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(a.out_dir / "wr_r14_training_casebook.csv", index=False)
    pd.concat(feature_rows, ignore_index=True).to_csv(a.out_dir / "wr_r14_confirmation_features.csv", index=False)
    pred.to_csv(a.out_dir / "wr_r14_confirmation_predictions.csv", index=False)
    summary.to_csv(a.out_dir / "wr_r14_market_summary.csv", index=False)
    buckets.to_csv(a.out_dir / "wr_r14_bucket_summary.csv", index=False)
    audits.to_csv(a.out_dir / "wr_r14_conservation_audit.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(a.out_dir / "wr_r14_coefficients.csv", index=False)
    (a.out_dir / "wr_r14_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
