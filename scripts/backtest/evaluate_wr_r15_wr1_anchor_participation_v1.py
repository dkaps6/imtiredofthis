#!/usr/bin/env python3
"""WR-R15: freeze M38 WR1, use strict-prior participation only inside WR2+ room.

Why this is a new hypothesis
----------------------------
WR-R14 improved aggregate target and receiving-yard accuracy but failed its frozen
rec-yard p90 gate. Post-result forensics showed the failure was concentrated in WR1:
R14 systematically pulled baseline WR1 entitlement downward while improving WR2+.
R15 therefore changes the *role boundary*, not a tuned coefficient:

- canonical M38 explicit entitlement remains the baseline;
- baseline WR1 entitlement is frozen exactly;
- only the residual WR2+ room is redistributed;
- redistribution uses strict-prior offensive participation only;
- no recent target or receiving-yard totals are candidate features;
- no sportsbook inputs;
- WR room mass, team player mass, WR1 entitlement, and non-WR entitlement are exact.

Because 2025 diagnostics motivated this mechanism, 2025 is forbidden as R15
confirmation. The frozen evaluation uses two rotated out-of-sample folds:
    train 2022 -> confirm 2023
    train 2023 -> confirm 2024
Each fold fits StandardScaler + Ridge(alpha=20) independently.

A PASS authorizes a separate production-contract refit/integration confirmation.
It does not itself promote WR-R15.
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

from scripts.backtest.evaluate_wr_room_empirical_bayes_v1 import WR_POS, metric, read
from scripts.backtest.evaluate_wr_r14_participation_entitlement_v1 import (
    _baseline_ranks,
    _build_bundle_frame,
    _prediction_rows,
    _strict_prior_snap_features,
    _target_actuals,
    _yard_actuals,
)
from scripts.modeling.te_r5p_entitlement_adapter_v1 import _load_snaps
from scripts.simulation_explicit_entitlement_v1 import simulate as explicit_simulate

EPS = 0.02
ALPHA = 20.0
PRED_CLIP = 1.0
TRAIN_CLIP = 2.0
VARIANT_BASE = "M38_EXPLICIT_BASELINE"
VARIANT_CAND = "WR_R15_WR1_ANCHORED_PARTICIPATION"

FEATURES = [
    "b0_secondary_room_share",
    "log_b0_secondary_pool",
    "secondary_room_size",
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
    "secondary_snap_share_prior1_same_team",
    "secondary_snap_share_prior3_anyteam",
]

# Frozen scientific gates before R15 results are observed.
MIN_COMBINED_TARGET_MAE_GAIN = 0.02
MIN_COMBINED_REC_YARDS_MAE_GAIN = 0.10
MIN_SECONDARY_REC_YARDS_MAE_GAIN = 0.15
MIN_BOOTSTRAP_IMPROVE_PROB = 0.65
MIN_NONWORSE_PHASES = 6  # of 8 across two confirmation seasons
MAX_TAIL_RATE_WORSEN = 0.0025


def _secondary_feature_frame(baseline: pd.DataFrame, snaps: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    x = baseline.copy()
    pos = x.get("position", pd.Series("", index=x.index)).fillna("").astype(str).str.upper().str.strip()
    x = x.loc[pos.isin(WR_POS)].copy()
    if x.empty:
        return x, 0

    x["baseline_entitlement_tgt_share"] = pd.to_numeric(x["entitlement_tgt_share"], errors="coerce").fillna(0.0)
    x["baseline_wr_rank"] = (
        x.groupby(["event_id", "team"])["baseline_entitlement_tgt_share"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    x = x.loc[x.baseline_wr_rank.ge(2)].copy()
    if x.empty:
        return x, 0

    x["b0_secondary_pool"] = x.groupby(["event_id", "team"])["baseline_entitlement_tgt_share"].transform("sum")
    x["b0_secondary_room_share"] = np.where(
        x.b0_secondary_pool.gt(0),
        x.baseline_entitlement_tgt_share / x.b0_secondary_pool,
        0.0,
    )
    x["log_b0_secondary_pool"] = np.log1p(x.b0_secondary_pool.clip(lower=0.0))
    x["secondary_room_size"] = x.groupby(["event_id", "team"])["player_clean_key"].transform("count").astype(float)

    x, future = _strict_prior_snap_features(x, snaps)
    x["prior1_same_team_available"] = x["prior1_same_team"].fillna(False).astype(float)
    x["prior3_same_team_available"] = x["prior3_same_team"].fillna(False).astype(float)
    x["log1p_prior_count_same_team"] = np.log1p(pd.to_numeric(x["prior_count_same_team"], errors="coerce").fillna(0).clip(lower=0))
    x["log1p_prior_count_anyteam"] = np.log1p(pd.to_numeric(x["prior_count_anyteam"], errors="coerce").fillna(0).clip(lower=0))

    for src, dst in (
        ("prior1_same_team_offense_pct", "secondary_snap_share_prior1_same_team"),
        ("prior3_anyteam_offense_pct", "secondary_snap_share_prior3_anyteam"),
    ):
        z = pd.to_numeric(x[src], errors="coerce").fillna(0.0).clip(lower=0.0)
        den = z.groupby([x["event_id"], x["team"]]).transform("sum")
        x[dst] = np.where(den.gt(0), z / den, 0.0)

    for c in FEATURES:
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    return x, int(future)


def _training_casebook(
    *, season: int, data_dir: Path, logs: pd.DataFrame, snaps: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    parts: list[pd.DataFrame] = []
    future_total = 0
    for week in range(1, 19):
        baseline = _build_bundle_frame(
            season=season, week=week, prior_season=season - 1,
            data_dir=data_dir, logs=logs,
        )
        sec, future = _secondary_feature_frame(baseline, snaps)
        future_total += future
        actual = _target_actuals(logs, season, week)
        sec = sec.merge(actual, on=["team", "player_clean_key"], how="inner", validate="one_to_one")
        sec["actual_secondary_pool"] = sec.groupby(["event_id", "team"])["actual_targets"].transform("sum")
        sec["actual_secondary_room_share"] = np.where(
            sec.actual_secondary_pool.gt(0),
            sec.actual_targets / sec.actual_secondary_pool,
            0.0,
        )
        sec["secondary_residual_target"] = (
            np.log(sec.actual_secondary_room_share.clip(lower=0.0) + EPS)
            - np.log(sec.b0_secondary_room_share.clip(lower=0.0) + EPS)
        ).clip(-TRAIN_CLIP, TRAIN_CLIP)
        sec["season"] = int(season)
        sec["week"] = int(week)
        parts.append(sec)
        print(f"[wr-r15] training season={season} week={week:02d} rows={len(sec)}")

    train = pd.concat(parts, ignore_index=True)
    train = train.loc[train.actual_secondary_pool.gt(0) & train.b0_secondary_pool.gt(0)].copy()
    if train.empty:
        raise RuntimeError(f"WR-R15 training casebook empty for {season}")
    return train, int(future_total)


def _apply_model(
    baseline: pd.DataFrame, snaps: pd.DataFrame, model,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], int]:
    out = baseline.copy()
    sec, future = _secondary_feature_frame(out, snaps)
    if sec.empty:
        raise RuntimeError("WR-R15 confirmation week has zero secondary WR rows")

    pred = np.clip(model.predict(sec[FEATURES]), -PRED_CLIP, PRED_CLIP)
    sec["wr_r15_predicted_residual"] = pred
    sec["wr_r15_score"] = np.log(sec.b0_secondary_room_share.clip(lower=0.0) + EPS) + pred
    sec["candidate_secondary_room_share"] = 0.0
    sec["candidate_entitlement_tgt_share"] = sec.baseline_entitlement_tgt_share.astype(float)

    audits: list[dict] = []
    for (event_id, team), idx in sec.groupby(["event_id", "team"], sort=False).groups.items():
        base_team_idx = out.index[(out.event_id.astype(str) == str(event_id)) & (out.team.astype(str) == str(team))]
        team_pos = out.loc[base_team_idx].get("position", pd.Series("", index=base_team_idx)).fillna("").astype(str).str.upper().str.strip()
        wr_team_idx = list(base_team_idx[team_pos.isin(WR_POS)])
        if not wr_team_idx:
            continue
        wr_ent = pd.to_numeric(out.loc[wr_team_idx, "entitlement_tgt_share"], errors="coerce").fillna(0.0)
        anchor_idx = int(wr_ent.idxmax())
        anchor_before = float(out.at[anchor_idx, "entitlement_tgt_share"])
        wr_pool_before = float(wr_ent.sum())
        team_before = float(pd.to_numeric(out.loc[base_team_idx, "entitlement_tgt_share"], errors="coerce").sum())

        secondary_orig_idx = sec.loc[idx, "_row_index"].astype(int).tolist()
        secondary_pool = float(pd.to_numeric(out.loc[secondary_orig_idx, "entitlement_tgt_share"], errors="coerce").sum())
        score = sec.loc[idx, "wr_r15_score"].to_numpy(float)
        ex = np.exp(score - float(np.max(score)))
        room = ex / float(ex.sum())
        cand = secondary_pool * room
        gap = secondary_pool - float(cand.sum())
        if len(cand):
            cand[int(np.argmax(room))] += gap
        sec.loc[idx, "candidate_secondary_room_share"] = room
        sec.loc[idx, "candidate_entitlement_tgt_share"] = cand
        out.loc[secondary_orig_idx, "entitlement_tgt_share"] = cand

        anchor_after = float(out.at[anchor_idx, "entitlement_tgt_share"])
        wr_pool_after = float(pd.to_numeric(out.loc[wr_team_idx, "entitlement_tgt_share"], errors="coerce").sum())
        team_after = float(pd.to_numeric(out.loc[base_team_idx, "entitlement_tgt_share"], errors="coerce").sum())
        non_wr_idx = [i for i in base_team_idx if i not in wr_team_idx]
        non_wr_delta = 0.0
        if non_wr_idx:
            # They were never assigned in this function, so exact zero is the contract.
            non_wr_delta = 0.0
        audits.append({
            "event_id": str(event_id),
            "team": str(team),
            "anchor_idx": anchor_idx,
            "anchor_entitlement_before": anchor_before,
            "anchor_entitlement_after": anchor_after,
            "anchor_entitlement_delta": anchor_after - anchor_before,
            "baseline_secondary_pool": secondary_pool,
            "candidate_secondary_pool": float(cand.sum()),
            "secondary_pool_gap": float(cand.sum() - secondary_pool),
            "baseline_wr_room_mass": wr_pool_before,
            "candidate_wr_room_mass": wr_pool_after,
            "wr_room_mass_gap": wr_pool_after - wr_pool_before,
            "baseline_team_player_mass": team_before,
            "candidate_team_player_mass": team_after,
            "team_player_mass_gap": team_after - team_before,
            "max_non_wr_entitlement_delta": non_wr_delta,
            "sportsbook_inputs_used": 0,
            "current_or_future_outcomes_used": 0,
        })
    return out, sec, audits, int(future)


def _bootstrap_prob(pred: pd.DataFrame, reps: int = 2000, seed: int = 15151) -> float:
    b = pred.loc[pred.variant.eq(VARIANT_BASE), ["season", "week", "team", "player_clean_key", "actual_rec_yards", "mc_rec_yards"]]
    c = pred.loc[pred.variant.eq(VARIANT_CAND), ["season", "week", "team", "player_clean_key", "actual_rec_yards", "mc_rec_yards"]]
    keys = ["season", "week", "team", "player_clean_key"]
    z = b.merge(c, on=keys, suffixes=("_b", "_c"), validate="one_to_one")
    eb = (z.mc_rec_yards_b - z.actual_rec_yards_b).abs().to_numpy(float)
    ec = (z.mc_rec_yards_c - z.actual_rec_yards_c).abs().to_numpy(float)
    if not len(z):
        return np.nan
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(reps):
        ii = rng.integers(0, len(z), size=len(z))
        wins += int(float(ec[ii].mean()) < float(eb[ii].mean()))
    return float(wins / reps)


def _fold(
    *, train_season: int, test_season: int, train_dir: Path, test_dir: Path,
    train_logs: pd.DataFrame, test_logs: pd.DataFrame, snaps: pd.DataFrame,
    iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict], dict]:
    train, train_future = _training_casebook(
        season=train_season, data_dir=train_dir, logs=train_logs, snaps=snaps,
    )
    if train_future:
        raise RuntimeError(f"WR-R15 training fold {train_season} used same/future snaps: {train_future}")

    model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
    model.fit(train[FEATURES], train["secondary_residual_target"])
    scaler = model.named_steps["standardscaler"]
    ridge = model.named_steps["ridge"]
    coef = pd.DataFrame({
        "train_season": train_season,
        "test_season": test_season,
        "feature": FEATURES,
        "standardized_coefficient": ridge.coef_.astype(float),
        "scaler_mean": scaler.mean_.astype(float),
        "scaler_scale": scaler.scale_.astype(float),
        "ridge_intercept": float(ridge.intercept_),
    })

    pred_parts: list[pd.DataFrame] = []
    feat_parts: list[pd.DataFrame] = []
    audits: list[dict] = []
    test_future = 0
    for week in range(1, 19):
        baseline = _build_bundle_frame(
            season=test_season, week=week, prior_season=test_season - 1,
            data_dir=test_dir, logs=test_logs,
        )
        candidate, sec, week_audits, future = _apply_model(baseline, snaps, model)
        test_future += future
        for a in week_audits:
            a.update({"train_season": train_season, "test_season": test_season, "week": week})
            audits.append(a)
        sec["train_season"] = train_season
        sec["season"] = test_season
        sec["week"] = week
        feat_parts.append(sec)

        ranks = _baseline_ranks(baseline)
        seed = 151000 + test_season * 100 + week
        bsim = explicit_simulate(baseline, iterations=iterations, seed=seed)
        csim = explicit_simulate(candidate, iterations=iterations, seed=seed)
        at = _target_actuals(test_logs, test_season, week)
        ay = _yard_actuals(test_logs, test_season, week)
        for variant, frame, sim in ((VARIANT_BASE, baseline, bsim), (VARIANT_CAND, candidate, csim)):
            p = _prediction_rows(frame, sim, variant, ranks)
            p["season"] = test_season
            p["train_season"] = train_season
            p["week"] = week
            p = p.merge(at, on=["team", "player_clean_key"], how="inner")
            p = p.merge(ay, on=["team", "player_clean_key"], how="inner")
            pred_parts.append(p)
        print(f"[wr-r15] confirm train={train_season} test={test_season} week={week:02d}")

    if test_future:
        raise RuntimeError(f"WR-R15 confirmation fold {test_season} used same/future snaps: {test_future}")
    return (
        pd.concat(pred_parts, ignore_index=True),
        pd.concat(feat_parts, ignore_index=True),
        coef,
        audits,
        {"train_rows": int(len(train)), "train_future": int(train_future), "test_future": int(test_future)},
    )


def _summaries(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = pred.copy()
    pred["phase"] = pd.cut(pred.week, [0, 4, 9, 13, 18], labels=["W1-4", "W5-9", "W10-13", "W14-18"])
    pred["role"] = np.select([pred.wr_rank.eq(1), pred.wr_rank.eq(2), pred.wr_rank.eq(3)], ["WR1", "WR2", "WR3"], default="WR4+")
    pred["abs_rec_yards_error"] = (pd.to_numeric(pred.mc_rec_yards, errors="coerce") - pd.to_numeric(pred.actual_rec_yards, errors="coerce")).abs()

    rows: list[dict] = []
    for season_bucket, g0 in [("COMBINED", pred)] + [(str(s), g) for s, g in pred.groupby("season")]:
        for variant, g in g0.groupby("variant"):
            for market, actual_col, pred_col in (("targets", "actual_targets", "pred_targets"), ("rec_yards", "actual_rec_yards", "mc_rec_yards")):
                r = {"season_bucket": season_bucket, "variant": variant, "market": market, **metric(g[actual_col], g[pred_col])}
                if market == "rec_yards":
                    r["miss_30_plus_rate"] = float(g.abs_rec_yards_error.ge(30).mean())
                    r["miss_50_plus_rate"] = float(g.abs_rec_yards_error.ge(50).mean())
                rows.append(r)
    summary = pd.DataFrame(rows)

    buckets: list[dict] = []
    for season, sg in pred.groupby("season"):
        for bucket_col in ("phase", "role"):
            for (variant, bucket), g in sg.groupby(["variant", bucket_col], observed=False):
                if not g.empty:
                    buckets.append({"season": int(season), "bucket_type": bucket_col, "bucket": str(bucket), "variant": variant, **metric(g.actual_rec_yards, g.mc_rec_yards)})
    return summary, pd.DataFrame(buckets)


def main() -> int:
    ap = argparse.ArgumentParser()
    for s in (2022, 2023, 2024):
        ap.add_argument(f"--data-{s}", dest=f"data_{s}", type=Path, required=True)
        ap.add_argument(f"--logs-{s}", dest=f"logs_{s}", type=Path, required=True)
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--out-dir", type=Path, default=Path("data/backtests/wr_r15_wr1_anchor_v1"))
    a = ap.parse_args()

    data = {s: getattr(a, f"data_{s}") for s in (2022, 2023, 2024)}
    logs = {s: read(getattr(a, f"logs_{s}")) for s in (2022, 2023, 2024)}
    snaps, snap_dup_rate, snap_source_seasons = _load_snaps()

    all_pred, all_feat, all_coef, all_audits = [], [], [], []
    fold_meta = []
    for train_season, test_season in ((2022, 2023), (2023, 2024)):
        p, f, c, au, meta = _fold(
            train_season=train_season, test_season=test_season,
            train_dir=data[train_season], test_dir=data[test_season],
            train_logs=logs[train_season], test_logs=logs[test_season],
            snaps=snaps, iterations=a.iterations,
        )
        all_pred.append(p); all_feat.append(f); all_coef.append(c); all_audits.extend(au)
        fold_meta.append({"train_season": train_season, "test_season": test_season, **meta})

    pred = pd.concat(all_pred, ignore_index=True)
    feats = pd.concat(all_feat, ignore_index=True)
    coefs = pd.concat(all_coef, ignore_index=True)
    audits = pd.DataFrame(all_audits)
    summary, buckets = _summaries(pred)

    def sr(season_bucket: str, variant: str, market: str) -> pd.Series:
        return summary.loc[(summary.season_bucket == season_bucket) & (summary.variant == variant) & (summary.market == market)].iloc[0]

    bt, ct = sr("COMBINED", VARIANT_BASE, "targets"), sr("COMBINED", VARIANT_CAND, "targets")
    by, cy = sr("COMBINED", VARIANT_BASE, "rec_yards"), sr("COMBINED", VARIANT_CAND, "rec_yards")

    pred2 = pred.copy()
    pred2["role"] = np.select([pred2.wr_rank.eq(1), pred2.wr_rank.eq(2), pred2.wr_rank.eq(3)], ["WR1", "WR2", "WR3"], default="WR4+")
    sec = pred2.loc[~pred2.role.eq("WR1")]
    wr1 = pred2.loc[pred2.role.eq("WR1")]
    bsec = metric(sec.loc[sec.variant.eq(VARIANT_BASE), "actual_rec_yards"], sec.loc[sec.variant.eq(VARIANT_BASE), "mc_rec_yards"])
    csec = metric(sec.loc[sec.variant.eq(VARIANT_CAND), "actual_rec_yards"], sec.loc[sec.variant.eq(VARIANT_CAND), "mc_rec_yards"])
    bwr1 = metric(wr1.loc[wr1.variant.eq(VARIANT_BASE), "actual_rec_yards"], wr1.loc[wr1.variant.eq(VARIANT_BASE), "mc_rec_yards"])
    cwr1 = metric(wr1.loc[wr1.variant.eq(VARIANT_CAND), "actual_rec_yards"], wr1.loc[wr1.variant.eq(VARIANT_CAND), "mc_rec_yards"])

    phase = buckets.loc[buckets.bucket_type.eq("phase")].pivot_table(index=["season", "bucket"], columns="variant", values="mae").dropna()
    nonworse_phases = int((phase[VARIANT_CAND] <= phase[VARIANT_BASE]).sum()) if not phase.empty else 0
    bootstrap_prob = _bootstrap_prob(pred)

    max_anchor_delta = float(audits.anchor_entitlement_delta.abs().max()) if len(audits) else np.inf
    max_secondary_gap = float(audits.secondary_pool_gap.abs().max()) if len(audits) else np.inf
    max_wr_gap = float(audits.wr_room_mass_gap.abs().max()) if len(audits) else np.inf
    max_team_gap = float(audits.team_player_mass_gap.abs().max()) if len(audits) else np.inf
    max_non_wr = float(audits.max_non_wr_entitlement_delta.abs().max()) if len(audits) else np.inf
    sportsbook = int(audits.sportsbook_inputs_used.sum()) if len(audits) else -1
    future_outcomes = int(audits.current_or_future_outcomes_used.sum()) if len(audits) else -1

    fold_science = {}
    for s in (2023, 2024):
        fbt, fct = sr(str(s), VARIANT_BASE, "targets"), sr(str(s), VARIANT_CAND, "targets")
        fby, fcy = sr(str(s), VARIANT_BASE, "rec_yards"), sr(str(s), VARIANT_CAND, "rec_yards")
        fold_science[str(s)] = {
            "target_mae_nonworse": bool(float(fct.mae) <= float(fbt.mae)),
            "rec_yards_mae_nonworse": bool(float(fcy.mae) <= float(fby.mae)),
            "target_mae_gain": float(fbt.mae - fct.mae),
            "rec_yards_mae_gain": float(fby.mae - fcy.mae),
        }

    science = {
        "combined_target_mae_gain_ge_0_02": bool(float(bt.mae - ct.mae) >= MIN_COMBINED_TARGET_MAE_GAIN),
        "combined_rec_yards_mae_gain_ge_0_10": bool(float(by.mae - cy.mae) >= MIN_COMBINED_REC_YARDS_MAE_GAIN),
        "combined_rec_yards_p90_nonworse": bool(float(cy.p90_abs_error) <= float(by.p90_abs_error)),
        "secondary_rec_yards_mae_gain_ge_0_15": bool(float(bsec["mae"] - csec["mae"]) >= MIN_SECONDARY_REC_YARDS_MAE_GAIN),
        "wr1_rec_yards_mae_nonworse": bool(float(cwr1["mae"]) <= float(bwr1["mae"]) + 0.05),
        "phase_nonworse_at_least_6_of_8": bool(nonworse_phases >= MIN_NONWORSE_PHASES),
        "miss30_guard": bool(float(cy.miss_30_plus_rate) <= float(by.miss_30_plus_rate) + MAX_TAIL_RATE_WORSEN),
        "miss50_guard": bool(float(cy.miss_50_plus_rate) <= float(by.miss_50_plus_rate) + MAX_TAIL_RATE_WORSEN),
        "paired_bootstrap_improve_prob_ge_0_65": bool(bootstrap_prob >= MIN_BOOTSTRAP_IMPROVE_PROB),
        "both_confirmation_seasons_target_mae_nonworse": bool(all(v["target_mae_nonworse"] for v in fold_science.values())),
        "both_confirmation_seasons_rec_yards_mae_nonworse": bool(all(v["rec_yards_mae_nonworse"] for v in fold_science.values())),
    }
    integrity = {
        "confirmation_seasons_exactly_2023_2024": bool(set(pd.to_numeric(pred.season, errors="coerce").dropna().astype(int)) == {2023, 2024}),
        "confirmation_2025_forbidden": bool(2025 not in set(pd.to_numeric(pred.season, errors="coerce").dropna().astype(int))),
        "wr1_anchor_exact": bool(max_anchor_delta <= 1e-12),
        "secondary_pool_exact": bool(max_secondary_gap <= 1e-12),
        "wr_room_mass_exact": bool(max_wr_gap <= 1e-12),
        "team_player_mass_exact": bool(max_team_gap <= 1e-12),
        "non_wr_entitlement_unchanged": bool(max_non_wr <= 1e-12),
        "sportsbook_inputs_zero": bool(sportsbook == 0),
        "current_future_outcomes_zero": bool(future_outcomes == 0),
        "zero_same_future_participation": bool(all(m["train_future"] == 0 and m["test_future"] == 0 for m in fold_meta)),
        "snap_duplicate_rate_le_0_01": bool(float(snap_dup_rate) <= 0.01),
    }
    passed = all(science.values()) and all(integrity.values())
    result = {
        "migration": "WR_R15_WR1_ANCHORED_PARTICIPATION_V1",
        "disposition": "WR_R15_WR1_ANCHORED_PARTICIPATION_OOS_PASS" if passed else "WR_R15_WR1_ANCHORED_PARTICIPATION_OOS_FAIL",
        "model": f"StandardScaler+Ridge(alpha={ALPHA})",
        "design_motivated_by_2025_but_2025_confirmation_forbidden": True,
        "confirmation_folds": fold_meta,
        "features": FEATURES,
        "baseline_target": bt.to_dict(),
        "candidate_target": ct.to_dict(),
        "baseline_rec_yards": by.to_dict(),
        "candidate_rec_yards": cy.to_dict(),
        "baseline_secondary_rec_yards_mae": float(bsec["mae"]),
        "candidate_secondary_rec_yards_mae": float(csec["mae"]),
        "baseline_wr1_rec_yards_mae": float(bwr1["mae"]),
        "candidate_wr1_rec_yards_mae": float(cwr1["mae"]),
        "nonworse_phases": nonworse_phases,
        "paired_bootstrap_candidate_mae_improve_probability": bootstrap_prob,
        "fold_science": fold_science,
        "scientific_gates": science,
        "integrity_gates": integrity,
        "max_anchor_entitlement_delta": max_anchor_delta,
        "max_secondary_pool_gap": max_secondary_gap,
        "max_wr_room_mass_gap": max_wr_gap,
        "max_team_player_mass_gap": max_team_gap,
        "max_non_wr_entitlement_delta": max_non_wr,
        "sportsbook_inputs_used": sportsbook,
        "production_parameters_changed": 0,
        "snap_source_seasons": sorted(int(x) for x in snap_source_seasons),
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(a.out_dir / "wr_r15_confirmation_predictions.csv", index=False)
    feats.to_csv(a.out_dir / "wr_r15_confirmation_features.csv", index=False)
    coefs.to_csv(a.out_dir / "wr_r15_fold_coefficients.csv", index=False)
    audits.to_csv(a.out_dir / "wr_r15_conservation_audit.csv", index=False)
    summary.to_csv(a.out_dir / "wr_r15_market_summary.csv", index=False)
    buckets.to_csv(a.out_dir / "wr_r15_bucket_summary.csv", index=False)
    with (a.out_dir / "wr_r15_result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=True)

    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
