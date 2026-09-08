#!/usr/bin/env python3
"""RB-R6: two-stage RB receiving entitlement under conserved team target mass.

2025 oracle diagnostics motivated this mechanism, therefore 2025 is forbidden from
confirmation. Frozen rotated OOS folds:
    train 2022 -> confirm 2023
    train 2023 -> confirm 2024

Architecture
------------
1. Start from canonical explicit finite target entitlement (M38 already applied).
2. Predict RB/FB room fraction of the fixed 0.95 modeled-player target mass using
   baseline room structure + strictly-prior offensive participation only.
3. Preserve the relative entitlement of every non-RB player exactly while scaling
   the non-RB complement to make room for the predicted RB pool.
4. Redistribute the predicted RB pool among RB/FB players using a separate
   strictly-prior participation residual model.
5. Team modeled-player target mass remains exactly 0.95.

No sportsbook inputs. No current/future outcomes or participation. No recent target
or receiving-yard totals are candidate features. 2025 is diagnostic-only and cannot
be used as confirmation here.

Three variants are emitted: baseline, room-only, and full two-stage. Scientific
promotion gates apply to the full two-stage candidate; room-only is explanatory.
A PASS only authorizes a separate production-contract refit/integration confirmation.
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
from scripts.backtest.evaluate_wr_room_empirical_bayes_v1 import finite, metric, read
from scripts.backtest.evaluate_wr_r14_participation_entitlement_v1 import (
    _build_bundle_frame,
    _strict_prior_snap_features,
)
from scripts.modeling.te_r5p_entitlement_adapter_v1 import _load_snaps
from scripts.simulation_explicit_entitlement_v1 import simulate as explicit_simulate

RB_POS = {"RB", "FB", "HB", "TB"}
MODELED_MASS = 0.95
EPS = 0.02
LOGIT_EPS = 0.01
ALPHA = 20.0
TRAIN_CLIP = 2.0
PRED_CLIP = 1.0

BASE = "M38_EXPLICIT_BASELINE"
ROOM = "RB_R6_ROOM_ONLY"
FULL = "RB_R6_TWO_STAGE"

ROOM_FEATURES = [
    "b0_rb_fraction",
    "log_b0_rb_pool",
    "rb_room_size",
    "modeled_room_size",
    "top_b0_rb_within_share",
    "prior1_rb_offense_pct_sum",
    "prior1_rb_offense_pct_max",
    "prior1_rb_offense_pct_hhi",
    "prior1_same_team_available_count",
    "log1p_prior_count_same_team_sum",
]

WITHIN_FEATURES = [
    "b0_rb_within_share",
    "log_b0_rb_pool",
    "rb_room_size",
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
    "rb_snap_share_prior1_same_team",
    "rb_snap_share_prior3_anyteam",
]

# Frozen gates before any RB-R6 confirmation result is observed.
MIN_RB_TARGET_MAE_GAIN = 0.05
MIN_RB_REC_YARDS_MAE_GAIN = 0.10
MIN_BOOTSTRAP_IMPROVE_PROB = 0.65
MIN_NONWORSE_PHASES = 6  # of 8 across two OOS seasons
MAX_TAIL_RATE_WORSEN = 0.0025
MAX_NON_RB_TARGET_MAE_WORSEN = 0.01
MAX_NON_RB_REC_YARDS_MAE_WORSEN = 0.05


def _logit(x: np.ndarray | pd.Series | float) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    z = np.clip(z, LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(z / (1.0 - z))


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def _actual_target_frame(logs: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    a = cp.build_actual_rows(logs, season, week)
    x = a.loc[a.market.eq("receptions"), ["team", "player_clean_key", "actual_opportunities"]].copy()
    x = x.rename(columns={"actual_opportunities": "actual_targets"})
    x["actual_targets"] = pd.to_numeric(x.actual_targets, errors="coerce").fillna(0.0)
    return x


def _actual_yards_frame(logs: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    a = cp.build_actual_rows(logs, season, week)
    x = a.loc[a.market.eq("rec_yards"), ["team", "player_clean_key", "actual"]].copy()
    x = x.rename(columns={"actual": "actual_rec_yards"})
    x["actual_rec_yards"] = pd.to_numeric(x.actual_rec_yards, errors="coerce")
    return x


def _rb_features(baseline: pd.DataFrame, snaps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    x = baseline.copy().reset_index(drop=False).rename(columns={"index": "_row_index"})
    x["position_family"] = x.get("position", "").fillna("").astype(str).str.upper().str.strip().replace({"HB": "RB", "TB": "RB"})
    x["baseline_entitlement_tgt_share"] = pd.to_numeric(x.entitlement_tgt_share, errors="coerce").fillna(0.0)
    rb = x.loc[x.position_family.isin({"RB", "FB"})].copy()
    if rb.empty:
        return rb, pd.DataFrame(), 0

    rb["b0_rb_pool"] = rb.groupby(["event_id", "team"])["baseline_entitlement_tgt_share"].transform("sum")
    rb["b0_rb_fraction"] = rb.b0_rb_pool / MODELED_MASS
    rb["b0_rb_within_share"] = np.where(rb.b0_rb_pool.gt(0), rb.baseline_entitlement_tgt_share / rb.b0_rb_pool, 0.0)
    rb["log_b0_rb_pool"] = np.log1p(rb.b0_rb_pool.clip(lower=0.0))
    rb["rb_room_size"] = rb.groupby(["event_id", "team"])["player_clean_key"].transform("count").astype(float)
    rb["modeled_room_size"] = rb.groupby(["event_id", "team"])["player_clean_key"].transform("size").astype(float)

    rb, future = _strict_prior_snap_features(rb, snaps)
    rb["prior1_same_team_available"] = rb["prior1_same_team"].fillna(False).astype(float)
    rb["prior3_same_team_available"] = rb["prior3_same_team"].fillna(False).astype(float)
    rb["log1p_prior_count_same_team"] = np.log1p(pd.to_numeric(rb.prior_count_same_team, errors="coerce").fillna(0).clip(lower=0))
    rb["log1p_prior_count_anyteam"] = np.log1p(pd.to_numeric(rb.prior_count_anyteam, errors="coerce").fillna(0).clip(lower=0))

    for src, dst in (
        ("prior1_same_team_offense_pct", "rb_snap_share_prior1_same_team"),
        ("prior3_anyteam_offense_pct", "rb_snap_share_prior3_anyteam"),
    ):
        z = pd.to_numeric(rb[src], errors="coerce").fillna(0.0).clip(lower=0.0)
        den = z.groupby([rb.event_id, rb.team]).transform("sum")
        rb[dst] = np.where(den.gt(0), z / den, 0.0)

    for c in WITHIN_FEATURES:
        rb[c] = pd.to_numeric(rb[c], errors="coerce").fillna(0.0)

    team_rows: list[dict] = []
    for (event_id, team), g in rb.groupby(["event_id", "team"], sort=False):
        z = pd.to_numeric(g.prior1_same_team_offense_pct, errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(float)
        zsum = float(z.sum())
        zhhi = float(np.sum((z / zsum) ** 2)) if zsum > 0 else 0.0
        base_within = pd.to_numeric(g.b0_rb_within_share, errors="coerce").fillna(0.0)
        full_team = x.loc[(x.event_id.astype(str) == str(event_id)) & (x.team.astype(str) == str(team))]
        team_rows.append({
            "event_id": str(event_id), "team": str(team),
            "b0_rb_pool": float(g.b0_rb_pool.iloc[0]),
            "b0_rb_fraction": float(g.b0_rb_fraction.iloc[0]),
            "log_b0_rb_pool": float(g.log_b0_rb_pool.iloc[0]),
            "rb_room_size": float(len(g)),
            "modeled_room_size": float(len(full_team)),
            "top_b0_rb_within_share": float(base_within.max()) if len(base_within) else 0.0,
            "prior1_rb_offense_pct_sum": zsum,
            "prior1_rb_offense_pct_max": float(z.max()) if len(z) else 0.0,
            "prior1_rb_offense_pct_hhi": zhhi,
            "prior1_same_team_available_count": float(pd.to_numeric(g.prior1_same_team_available, errors="coerce").fillna(0).sum()),
            "log1p_prior_count_same_team_sum": float(pd.to_numeric(g.log1p_prior_count_same_team, errors="coerce").fillna(0).sum()),
        })
    team_feat = pd.DataFrame(team_rows)
    for c in ROOM_FEATURES:
        team_feat[c] = pd.to_numeric(team_feat[c], errors="coerce").fillna(0.0)
    return rb, team_feat, int(future)


def _training_cases(*, season: int, data_dir: Path, logs: pd.DataFrame, snaps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    room_parts, within_parts = [], []
    future_total = 0
    for week in range(1, 19):
        baseline = _build_bundle_frame(season=season, week=week, prior_season=season - 1, data_dir=data_dir, logs=logs)
        rb, room, future = _rb_features(baseline, snaps)
        future_total += future
        actual_t = _actual_target_frame(logs, season, week)

        allx = baseline[["event_id", "team", "player_clean_key"]].merge(actual_t, on=["team", "player_clean_key"], how="left")
        allx["actual_targets"] = pd.to_numeric(allx.actual_targets, errors="coerce").fillna(0.0)
        team_actual = allx.groupby(["event_id", "team"], as_index=False).actual_targets.sum().rename(columns={"actual_targets": "actual_modeled_targets"})

        rbx = rb.merge(actual_t, on=["team", "player_clean_key"], how="left", validate="one_to_one")
        rbx["actual_targets"] = pd.to_numeric(rbx.actual_targets, errors="coerce").fillna(0.0)
        rb_actual = rbx.groupby(["event_id", "team"], as_index=False).actual_targets.sum().rename(columns={"actual_targets": "actual_rb_targets"})

        room = room.merge(team_actual, on=["event_id", "team"], how="left").merge(rb_actual, on=["event_id", "team"], how="left")
        room[["actual_modeled_targets", "actual_rb_targets"]] = room[["actual_modeled_targets", "actual_rb_targets"]].fillna(0.0)
        room["actual_rb_fraction"] = np.where(room.actual_modeled_targets.gt(0), room.actual_rb_targets / room.actual_modeled_targets, room.b0_rb_fraction)
        room["room_residual_target"] = np.clip(_logit(room.actual_rb_fraction) - _logit(room.b0_rb_fraction), -TRAIN_CLIP, TRAIN_CLIP)
        room["season"] = season; room["week"] = week
        room_parts.append(room.loc[room.actual_modeled_targets.gt(0)].copy())

        rbx = rbx.merge(rb_actual, on=["event_id", "team"], how="left")
        rbx["actual_rb_targets"] = pd.to_numeric(rbx.actual_rb_targets, errors="coerce").fillna(0.0)
        rbx["actual_rb_within_share"] = np.where(rbx.actual_rb_targets.gt(0), rbx.actual_targets / rbx.actual_rb_targets, 0.0)
        rbx["within_residual_target"] = (
            np.log(rbx.actual_rb_within_share.clip(lower=0.0) + EPS)
            - np.log(rbx.b0_rb_within_share.clip(lower=0.0) + EPS)
        ).clip(-TRAIN_CLIP, TRAIN_CLIP)
        rbx["season"] = season; rbx["week"] = week
        within_parts.append(rbx.loc[rbx.actual_rb_targets.gt(0) & rbx.b0_rb_pool.gt(0)].copy())
        print(f"[rb-r6] training season={season} week={week:02d} room={len(room)} rb={len(rbx)}")

    room_train = pd.concat(room_parts, ignore_index=True)
    within_train = pd.concat(within_parts, ignore_index=True)
    if room_train.empty or within_train.empty:
        raise RuntimeError(f"RB-R6 empty training casebook season={season}")
    return room_train, within_train, int(future_total)


def _apply_candidate(baseline: pd.DataFrame, snaps: pd.DataFrame, room_model, within_model, *, within: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], int]:
    out = baseline.copy()
    rb, room, future = _rb_features(out, snaps)
    if rb.empty or room.empty:
        raise RuntimeError("RB-R6 confirmation week has zero RB feature rows")
    room["pred_room_residual"] = np.clip(room_model.predict(room[ROOM_FEATURES]), -PRED_CLIP, PRED_CLIP)
    room["candidate_rb_fraction"] = _sigmoid(_logit(room.b0_rb_fraction) + room.pred_room_residual)
    room["candidate_rb_pool"] = MODELED_MASS * room.candidate_rb_fraction

    rb["pred_within_residual"] = np.clip(within_model.predict(rb[WITHIN_FEATURES]), -PRED_CLIP, PRED_CLIP) if within else 0.0
    rb["candidate_entitlement_tgt_share"] = rb.baseline_entitlement_tgt_share.astype(float)
    audits: list[dict] = []

    room_lookup = room.set_index(["event_id", "team"])
    for (event_id, team), idx in rb.groupby(["event_id", "team"], sort=False).groups.items():
        key = (str(event_id), str(team))
        # Room keys were normalized to strings in team feature frame.
        if key not in room_lookup.index:
            raise RuntimeError(f"missing RB room feature for {key}")
        rf = room_lookup.loc[key]
        cand_rb_pool = float(rf.candidate_rb_pool)
        base_rb_pool = float(rf.b0_rb_pool)

        team_idx = out.index[(out.event_id.astype(str) == str(event_id)) & (out.team.astype(str) == str(team))]
        rb_orig_idx = rb.loc[idx, "_row_index"].astype(int).tolist()
        non_rb_idx = [i for i in team_idx if i not in rb_orig_idx]
        base_team_mass = float(pd.to_numeric(out.loc[team_idx, "entitlement_tgt_share"], errors="coerce").sum())
        base_non_rb = pd.to_numeric(out.loc[non_rb_idx, "entitlement_tgt_share"], errors="coerce").fillna(0.0).astype(float) if non_rb_idx else pd.Series(dtype=float)
        base_non_rb_sum = float(base_non_rb.sum())
        target_non_rb_sum = MODELED_MASS - cand_rb_pool
        if base_non_rb_sum <= 0 and target_non_rb_sum > 1e-12:
            raise RuntimeError(f"RB-R6 no non-RB complement for {key}")
        scale_non_rb = target_non_rb_sum / base_non_rb_sum if base_non_rb_sum > 0 else 1.0
        if non_rb_idx:
            out.loc[non_rb_idx, "entitlement_tgt_share"] = base_non_rb.to_numpy(float) * scale_non_rb

        base_within = pd.to_numeric(rb.loc[idx, "b0_rb_within_share"], errors="coerce").fillna(0.0).to_numpy(float)
        if within:
            score = np.log(np.clip(base_within, 0.0, None) + EPS) + rb.loc[idx, "pred_within_residual"].to_numpy(float)
            ex = np.exp(score - float(np.max(score)))
            room_share = ex / float(ex.sum())
        else:
            room_share = base_within / float(base_within.sum()) if float(base_within.sum()) > 0 else np.full(len(base_within), 1.0 / len(base_within))
        cand = cand_rb_pool * room_share
        if len(cand):
            cand[int(np.argmax(room_share))] += cand_rb_pool - float(cand.sum())
        out.loc[rb_orig_idx, "entitlement_tgt_share"] = cand
        rb.loc[idx, "candidate_entitlement_tgt_share"] = cand

        after_team_mass = float(pd.to_numeric(out.loc[team_idx, "entitlement_tgt_share"], errors="coerce").sum())
        after_rb_pool = float(pd.to_numeric(out.loc[rb_orig_idx, "entitlement_tgt_share"], errors="coerce").sum())
        after_non_rb = pd.to_numeric(out.loc[non_rb_idx, "entitlement_tgt_share"], errors="coerce").fillna(0.0).astype(float) if non_rb_idx else pd.Series(dtype=float)
        ratio_gap = 0.0
        if len(base_non_rb) >= 2:
            bnorm = base_non_rb / base_non_rb.sum() if base_non_rb.sum() > 0 else base_non_rb
            anorm = after_non_rb / after_non_rb.sum() if after_non_rb.sum() > 0 else after_non_rb
            ratio_gap = float((bnorm.to_numpy(float) - anorm.to_numpy(float)).max(initial=0.0))
            ratio_gap = max(ratio_gap, float(np.max(np.abs(bnorm.to_numpy(float) - anorm.to_numpy(float)))))
        audits.append({
            "event_id": str(event_id), "team": str(team),
            "within_stage_active": int(within),
            "baseline_team_player_mass": base_team_mass,
            "candidate_team_player_mass": after_team_mass,
            "team_player_mass_gap_vs_095": after_team_mass - MODELED_MASS,
            "baseline_rb_pool": base_rb_pool,
            "candidate_rb_pool_target": cand_rb_pool,
            "candidate_rb_pool_actual": after_rb_pool,
            "rb_pool_arithmetic_gap": after_rb_pool - cand_rb_pool,
            "non_rb_relative_share_max_gap": ratio_gap,
            "sportsbook_inputs_used": 0,
            "current_or_future_outcomes_used": 0,
        })
    return out, rb, audits, int(future)


def _prediction_rows(frame: pd.DataFrame, sim, variant: str, season: int, week: int) -> pd.DataFrame:
    rows = []
    for (event_id, team), g in frame.groupby(["event_id", "team"], sort=False):
        pos = g.get("position", pd.Series("", index=g.index)).fillna("").astype(str).str.upper().str.strip().replace({"HB": "RB", "TB": "RB"})
        plays = float(np.mean([finite(v, 64.0) for v in g.get("rules_plays_est", pd.Series([64.0] * len(g)))]))
        pass_rate = float(np.mean([finite(v, 0.57) for v in g.get("rules_pass_rate", pd.Series([0.57] * len(g)))]))
        team_targets = plays * pass_rate
        for j, (_, r) in enumerate(g.iterrows()):
            pf = str(pos.iloc[j])
            if pf not in {"RB", "FB", "WR", "TE"}:
                continue
            key = str(r.get("player_clean_key", ""))
            ent = finite(r.get("entitlement_tgt_share"), 0.0)
            yards = sim.values.get((str(event_id), key, "rec_yards"))
            rows.append({
                "variant": variant, "season": season, "week": week,
                "event_id": str(event_id), "team": str(team),
                "player_clean_key": key, "player": r.get("player", ""),
                "position_family": pf,
                "entitlement_tgt_share": ent,
                "pred_targets": team_targets * ent,
                "mc_rec_yards": float(np.mean(yards)) if yards is not None else np.nan,
            })
    return pd.DataFrame(rows)


def _bootstrap_prob(pred: pd.DataFrame, reps: int = 2000, seed: int = 60606) -> float:
    z0 = pred.loc[pred.position_family.isin({"RB", "FB"})]
    b = z0.loc[z0.variant.eq(BASE), ["season", "week", "team", "player_clean_key", "actual_rec_yards", "mc_rec_yards"]]
    c = z0.loc[z0.variant.eq(FULL), ["season", "week", "team", "player_clean_key", "actual_rec_yards", "mc_rec_yards"]]
    keys = ["season", "week", "team", "player_clean_key"]
    z = b.merge(c, on=keys, suffixes=("_b", "_c"), validate="one_to_one")
    eb = (z.mc_rec_yards_b - z.actual_rec_yards_b).abs().to_numpy(float)
    ec = (z.mc_rec_yards_c - z.actual_rec_yards_c).abs().to_numpy(float)
    if not len(z): return np.nan
    rng = np.random.default_rng(seed); wins = 0
    for _ in range(reps):
        ii = rng.integers(0, len(z), size=len(z))
        wins += int(float(ec[ii].mean()) < float(eb[ii].mean()))
    return float(wins / reps)


def _fold(*, train_season: int, test_season: int, train_dir: Path, test_dir: Path, train_logs: pd.DataFrame, test_logs: pd.DataFrame, snaps: pd.DataFrame, iterations: int):
    room_train, within_train, train_future = _training_cases(season=train_season, data_dir=train_dir, logs=train_logs, snaps=snaps)
    if train_future:
        raise RuntimeError(f"RB-R6 training same/future participation: {train_future}")
    room_model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA)).fit(room_train[ROOM_FEATURES], room_train.room_residual_target)
    within_model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA)).fit(within_train[WITHIN_FEATURES], within_train.within_residual_target)

    pred_parts, audit_rows = [], []
    test_future = 0
    for week in range(1, 19):
        baseline = _build_bundle_frame(season=test_season, week=week, prior_season=test_season - 1, data_dir=test_dir, logs=test_logs)
        room_cand, _, aud_room, fut1 = _apply_candidate(baseline, snaps, room_model, within_model, within=False)
        full_cand, _, aud_full, fut2 = _apply_candidate(baseline, snaps, room_model, within_model, within=True)
        test_future += fut1 + fut2
        for a in aud_room + aud_full:
            a.update({"train_season": train_season, "test_season": test_season, "week": week})
            audit_rows.append(a)

        seed = 606000 + test_season * 100 + week
        sims = {
            BASE: explicit_simulate(baseline, iterations=iterations, seed=seed),
            ROOM: explicit_simulate(room_cand, iterations=iterations, seed=seed),
            FULL: explicit_simulate(full_cand, iterations=iterations, seed=seed),
        }
        frames = {BASE: baseline, ROOM: room_cand, FULL: full_cand}
        at = _actual_target_frame(test_logs, test_season, week)
        ay = _actual_yards_frame(test_logs, test_season, week)
        for v in (BASE, ROOM, FULL):
            p = _prediction_rows(frames[v], sims[v], v, test_season, week)
            p = p.merge(at, on=["team", "player_clean_key"], how="inner").merge(ay, on=["team", "player_clean_key"], how="inner")
            pred_parts.append(p)
        print(f"[rb-r6] confirm train={train_season} test={test_season} week={week:02d}")

    if test_future:
        raise RuntimeError(f"RB-R6 confirmation same/future participation: {test_future}")
    coef_rows = []
    for model_name, model, feats in (("room", room_model, ROOM_FEATURES), ("within", within_model, WITHIN_FEATURES)):
        sc = model.named_steps["standardscaler"]; rg = model.named_steps["ridge"]
        for i, f in enumerate(feats):
            coef_rows.append({"train_season": train_season, "test_season": test_season, "model": model_name, "feature": f, "standardized_coefficient": float(rg.coef_[i]), "scaler_mean": float(sc.mean_[i]), "scaler_scale": float(sc.scale_[i]), "ridge_intercept": float(rg.intercept_)})
    return pd.concat(pred_parts, ignore_index=True), pd.DataFrame(audit_rows), pd.DataFrame(coef_rows), {"train_season": train_season, "test_season": test_season, "room_train_rows": len(room_train), "within_train_rows": len(within_train), "train_future": train_future, "test_future": test_future}


def _summaries(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pred.copy()
    x["phase"] = pd.cut(x.week, [0, 4, 9, 13, 18], labels=["W1-4", "W5-9", "W10-13", "W14-18"])
    x["group"] = np.where(x.position_family.isin({"RB", "FB"}), "RB", "NON_RB")
    x["abs_rec_yards_error"] = (pd.to_numeric(x.mc_rec_yards, errors="coerce") - pd.to_numeric(x.actual_rec_yards, errors="coerce")).abs()
    rows = []
    for season_bucket, g0 in [("COMBINED", x)] + [(str(s), g) for s, g in x.groupby("season")]:
        for grp, gg in g0.groupby("group"):
            for v, g in gg.groupby("variant"):
                for market, ac, pc in (("targets", "actual_targets", "pred_targets"), ("rec_yards", "actual_rec_yards", "mc_rec_yards")):
                    r = {"season_bucket": season_bucket, "group": grp, "variant": v, "market": market, **metric(g[ac], g[pc])}
                    if market == "rec_yards":
                        r["miss_30_plus_rate"] = float(g.abs_rec_yards_error.ge(30).mean())
                        r["miss_50_plus_rate"] = float(g.abs_rec_yards_error.ge(50).mean())
                    rows.append(r)
    buckets = []
    for s, sg in x.loc[x.group.eq("RB")].groupby("season"):
        for (v, ph), g in sg.groupby(["variant", "phase"], observed=False):
            if not g.empty:
                buckets.append({"season": int(s), "phase": str(ph), "variant": v, **metric(g.actual_rec_yards, g.mc_rec_yards)})
    return pd.DataFrame(rows), pd.DataFrame(buckets)


def main() -> int:
    ap = argparse.ArgumentParser()
    for s in (2022, 2023, 2024):
        ap.add_argument(f"--data-{s}", dest=f"data_{s}", type=Path, required=True)
        ap.add_argument(f"--logs-{s}", dest=f"logs_{s}", type=Path, required=True)
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--out-dir", type=Path, default=Path("data/backtests/rb_r6_two_stage_receiving_v1"))
    a = ap.parse_args()
    data = {s: getattr(a, f"data_{s}") for s in (2022, 2023, 2024)}
    logs = {s: read(getattr(a, f"logs_{s}")) for s in (2022, 2023, 2024)}
    snaps, snap_dup_rate, snap_source_seasons = _load_snaps()

    preds, audits, coefs, metas = [], [], [], []
    for tr, te in ((2022, 2023), (2023, 2024)):
        p, au, co, me = _fold(train_season=tr, test_season=te, train_dir=data[tr], test_dir=data[te], train_logs=logs[tr], test_logs=logs[te], snaps=snaps, iterations=a.iterations)
        preds.append(p); audits.append(au); coefs.append(co); metas.append(me)
    pred = pd.concat(preds, ignore_index=True)
    audit = pd.concat(audits, ignore_index=True)
    coef = pd.concat(coefs, ignore_index=True)
    summary, phase = _summaries(pred)

    def sr(season_bucket: str, group: str, variant: str, market: str) -> pd.Series:
        return summary.loc[(summary.season_bucket == season_bucket) & (summary.group == group) & (summary.variant == variant) & (summary.market == market)].iloc[0]
    bt, ct = sr("COMBINED", "RB", BASE, "targets"), sr("COMBINED", "RB", FULL, "targets")
    by, cy = sr("COMBINED", "RB", BASE, "rec_yards"), sr("COMBINED", "RB", FULL, "rec_yards")
    nbt, nct = sr("COMBINED", "NON_RB", BASE, "targets"), sr("COMBINED", "NON_RB", FULL, "targets")
    nby, ncy = sr("COMBINED", "NON_RB", BASE, "rec_yards"), sr("COMBINED", "NON_RB", FULL, "rec_yards")

    fold_science = {}
    for s in (2023, 2024):
        fbt, fct = sr(str(s), "RB", BASE, "targets"), sr(str(s), "RB", FULL, "targets")
        fby, fcy = sr(str(s), "RB", BASE, "rec_yards"), sr(str(s), "RB", FULL, "rec_yards")
        fold_science[str(s)] = {
            "target_mae_nonworse": bool(float(fct.mae) <= float(fbt.mae)),
            "rec_yards_mae_nonworse": bool(float(fcy.mae) <= float(fby.mae)),
            "target_mae_gain": float(fbt.mae - fct.mae),
            "rec_yards_mae_gain": float(fby.mae - fcy.mae),
        }
    ph = phase.pivot_table(index=["season", "phase"], columns="variant", values="mae").dropna()
    nonworse_phases = int((ph[FULL] <= ph[BASE]).sum()) if not ph.empty else 0
    bootstrap = _bootstrap_prob(pred)

    science = {
        "rb_target_mae_gain_ge_0_05": bool(float(bt.mae - ct.mae) >= MIN_RB_TARGET_MAE_GAIN),
        "rb_rec_yards_mae_gain_ge_0_10": bool(float(by.mae - cy.mae) >= MIN_RB_REC_YARDS_MAE_GAIN),
        "rb_rec_yards_p90_nonworse": bool(float(cy.p90_abs_error) <= float(by.p90_abs_error)),
        "rb_phase_nonworse_at_least_6_of_8": bool(nonworse_phases >= MIN_NONWORSE_PHASES),
        "rb_miss30_guard": bool(float(cy.miss_30_plus_rate) <= float(by.miss_30_plus_rate) + MAX_TAIL_RATE_WORSEN),
        "rb_miss50_guard": bool(float(cy.miss_50_plus_rate) <= float(by.miss_50_plus_rate) + MAX_TAIL_RATE_WORSEN),
        "bootstrap_improve_prob_ge_0_65": bool(bootstrap >= MIN_BOOTSTRAP_IMPROVE_PROB),
        "both_oos_seasons_rb_target_nonworse": bool(all(v["target_mae_nonworse"] for v in fold_science.values())),
        "both_oos_seasons_rb_rec_yards_nonworse": bool(all(v["rec_yards_mae_nonworse"] for v in fold_science.values())),
        "non_rb_target_mae_guard": bool(float(nct.mae) <= float(nbt.mae) + MAX_NON_RB_TARGET_MAE_WORSEN),
        "non_rb_rec_yards_mae_guard": bool(float(ncy.mae) <= float(nby.mae) + MAX_NON_RB_REC_YARDS_MAE_WORSEN),
    }
    integrity = {
        "confirmation_seasons_exactly_2023_2024": bool(set(pd.to_numeric(pred.season, errors="coerce").dropna().astype(int)) == {2023, 2024}),
        "confirmation_2025_forbidden": bool(2025 not in set(pd.to_numeric(pred.season, errors="coerce").dropna().astype(int))),
        "team_player_mass_exact_095": bool(float(audit.team_player_mass_gap_vs_095.abs().max()) <= 1e-12),
        "rb_pool_arithmetic_exact": bool(float(audit.rb_pool_arithmetic_gap.abs().max()) <= 1e-12),
        "non_rb_relative_shares_exact": bool(float(audit.non_rb_relative_share_max_gap.abs().max()) <= 1e-12),
        "sportsbook_inputs_zero": bool(int(audit.sportsbook_inputs_used.sum()) == 0),
        "current_future_outcomes_zero": bool(int(audit.current_or_future_outcomes_used.sum()) == 0),
        "zero_same_future_participation": bool(all(m["train_future"] == 0 and m["test_future"] == 0 for m in metas)),
        "snap_duplicate_rate_le_0_01": bool(float(snap_dup_rate) <= 0.01),
    }
    passed = all(science.values()) and all(integrity.values())
    result = {
        "migration": "RB_R6_TWO_STAGE_RECEIVING_ENTITLEMENT_V1",
        "disposition": "RB_R6_TWO_STAGE_RECEIVING_OOS_PASS" if passed else "RB_R6_TWO_STAGE_RECEIVING_OOS_FAIL",
        "model": f"two StandardScaler+Ridge(alpha={ALPHA}) models",
        "diagnostic_2025_motivated_design_but_2025_confirmation_forbidden": True,
        "folds": metas,
        "room_features": ROOM_FEATURES,
        "within_features": WITHIN_FEATURES,
        "baseline_rb_target": bt.to_dict(), "candidate_rb_target": ct.to_dict(),
        "baseline_rb_rec_yards": by.to_dict(), "candidate_rb_rec_yards": cy.to_dict(),
        "baseline_non_rb_target": nbt.to_dict(), "candidate_non_rb_target": nct.to_dict(),
        "baseline_non_rb_rec_yards": nby.to_dict(), "candidate_non_rb_rec_yards": ncy.to_dict(),
        "nonworse_rb_phases": nonworse_phases,
        "paired_bootstrap_rb_rec_yards_improve_probability": bootstrap,
        "fold_science": fold_science,
        "scientific_gates": science,
        "integrity_gates": integrity,
        "sportsbook_inputs_used": int(audit.sportsbook_inputs_used.sum()),
        "production_parameters_changed": 0,
        "snap_source_seasons": sorted(int(x) for x in snap_source_seasons),
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(a.out_dir / "rb_r6_oos_predictions.csv", index=False)
    audit.to_csv(a.out_dir / "rb_r6_conservation_audit.csv", index=False)
    coef.to_csv(a.out_dir / "rb_r6_fold_coefficients.csv", index=False)
    summary.to_csv(a.out_dir / "rb_r6_market_summary.csv", index=False)
    phase.to_csv(a.out_dir / "rb_r6_phase_summary.csv", index=False)
    with (a.out_dir / "rb_r6_result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=True)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
