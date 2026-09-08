#!/usr/bin/env python3
"""RB-R10: modern-era stability audit for the frozen RB-R9 receiving-identity mechanism.

Purpose
-------
R9 passed its frozen 2016 historical confirmation. R10 does NOT create a new model,
retune R9, or claim pristine confirmation. It replays the exact R9 mechanism through
modern walk-forward seasons to answer one question: is the R9 receiving-identity
signal stable in the modern NFL rather than an old-era artifact?

Frozen mechanism
----------------
- exact R9 strict-prior receiving-identity features and Ridge model;
- exact R9 rolling-origin training-only reliability slope, bounded to [0, 1];
- exact R9 inference clip and within-RB/FB redistribution;
- preserve each team-game RB/FB target pool exactly;
- preserve total team player target mass exactly;
- preserve all non-RB entitlement exactly;
- zero sportsbook inputs; zero current/future outcomes in features.

Modern walk-forward folds
-------------------------
    train 2021 -> test 2022
    train 2022 -> test 2023
    train 2023 -> test 2024
    train 2024 -> test 2025

These seasons were already visible during the research program, so R10 is a
stability/governance audit, NOT a fresh scientific confirmation. A PASS authorizes
shadow integration and a genuinely prospective 2026 evaluation; it does not promote
R9 by itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.evaluate_wr_room_empirical_bayes_v1 import metric, read
from scripts.backtest import evaluate_rb_r9_receiving_identity_shrinkage_v1 as r9

BASE = r9.BASE
CAND = r9.CAND
FOLDS = ((2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025))
TEST_SEASONS = tuple(te for _, te in FOLDS)

# Frozen before observing any R10 outputs.
MIN_COMBINED_TARGET_MAE_GAIN = 0.02
MIN_COMBINED_REC_YARDS_MAE_GAIN = 0.10
MIN_SEASON_TARGET_NONWORSE = 3
MIN_SEASON_REC_YARDS_NONWORSE = 3
MAX_SINGLE_SEASON_TARGET_MAE_WORSEN = 0.05
MAX_SINGLE_SEASON_REC_YARDS_MAE_WORSEN = 0.25
MIN_COMBINED_TOP20_REC_YARDS_GAIN = 0.10
MAX_COMBINED_REST80_REC_YARDS_WORSEN = 0.10
MIN_TOP20_SEASONS_NONWORSE = 2
MIN_PHASES_NONWORSE = 12  # 12 of 16 modern season-phases.
MIN_COMBINED_BOOTSTRAP_IMPROVE_PROB = 0.65
MAX_COMBINED_TAIL_RATE_WORSEN = 0.0025


def _combined_identity_buckets(x: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, bucket), g in x.groupby(["variant", "identity_bucket"]):
        r = {"variant": variant, "identity_bucket": bucket, **metric(g.actual_rec_yards, g.mc_rec_yards)}
        r["miss_30_plus_rate"] = float(g.abs_err.ge(30).mean())
        r["miss_50_plus_rate"] = float(g.abs_err.ge(50).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def _combined_bootstrap(x: pd.DataFrame, reps: int, seed: int = 91010) -> float:
    keys = ["season", "week", "team", "player_clean_key"]
    b = x.loc[x.variant.eq(BASE), keys + ["actual_rec_yards", "mc_rec_yards"]]
    c = x.loc[x.variant.eq(CAND), keys + ["actual_rec_yards", "mc_rec_yards"]]
    z = b.merge(c, on=keys, suffixes=("_b", "_c"), validate="one_to_one")
    if z.empty:
        raise RuntimeError("R10 combined bootstrap has zero matched rows")
    eb = (z.mc_rec_yards_b - z.actual_rec_yards_b).abs().to_numpy(float)
    ec = (z.mc_rec_yards_c - z.actual_rec_yards_c).abs().to_numpy(float)
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(int(reps)):
        ii = rng.integers(0, len(z), len(z))
        wins += int(ec[ii].mean() < eb[ii].mean())
    return float(wins / reps)


def main() -> int:
    ap = argparse.ArgumentParser()
    for s in range(2021, 2026):
        ap.add_argument(f"--data-{s}", dest=f"data_{s}", type=Path, required=True)
        ap.add_argument(f"--logs-{s}", dest=f"logs_{s}", type=Path, required=True)
    ap.add_argument("--history-start", type=int, default=2013)
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    ap.add_argument("--out-dir", type=Path, default=Path("data/backtests/rb_r10_modern_stability_v1"))
    a = ap.parse_args()

    data = {s: getattr(a, f"data_{s}") for s in range(2021, 2026)}
    logs = {s: read(getattr(a, f"logs_{s}")) for s in range(2021, 2026)}

    pred_parts = []
    audit_parts = []
    coef_parts = []
    oof_parts = []
    fold_meta = []

    for tr, te in FOLDS:
        pred, audit, coef, oof, cal, ntrain = r9._fold(
            train_season=tr,
            test_season=te,
            train_dir=data[tr],
            test_dir=data[te],
            train_logs=logs[tr],
            test_logs=logs[te],
            history_start=a.history_start,
            iterations=a.iterations,
        )
        pred_parts.append(pred)
        audit_parts.append(audit)
        coef_parts.append(coef)
        oof_parts.append(oof)
        fold_meta.append({
            "train_season": tr,
            "test_season": te,
            "role": "modern_stability_not_pristine_confirmation",
            "train_rows": int(ntrain),
            **cal,
        })

    pred = pd.concat(pred_parts, ignore_index=True)
    audit = pd.concat(audit_parts, ignore_index=True)
    coef = pd.concat(coef_parts, ignore_index=True)
    oof = pd.concat(oof_parts, ignore_index=True)
    summary, phases, buckets, x = r9._summaries(pred)
    combined_buckets = _combined_identity_buckets(x)

    def sr(season_bucket, variant, market):
        row = summary.loc[
            summary.season_bucket.eq(str(season_bucket))
            & summary.variant.eq(variant)
            & summary.market.eq(market)
        ]
        if len(row) != 1:
            raise RuntimeError(f"R10 summary lookup failed: {season_bucket=} {variant=} {market=}")
        return row.iloc[0]

    def cbr(variant, bucket):
        row = combined_buckets.loc[
            combined_buckets.variant.eq(variant)
            & combined_buckets.identity_bucket.eq(bucket)
        ]
        if len(row) != 1:
            raise RuntimeError(f"R10 combined bucket lookup failed: {variant=} {bucket=}")
        return row.iloc[0]

    season_rows = []
    target_nonworse = 0
    rec_nonworse = 0
    top20_nonworse = 0
    max_target_worsen = -np.inf
    max_rec_worsen = -np.inf

    for s in TEST_SEASONS:
        bt, ct = sr(s, BASE, "targets"), sr(s, CAND, "targets")
        by, cy = sr(s, BASE, "rec_yards"), sr(s, CAND, "rec_yards")
        btop = buckets.loc[(buckets.season.eq(s)) & buckets.variant.eq(BASE) & buckets.identity_bucket.eq("TOP20")].iloc[0]
        ctop = buckets.loc[(buckets.season.eq(s)) & buckets.variant.eq(CAND) & buckets.identity_bucket.eq("TOP20")].iloc[0]
        target_gain = float(bt.mae - ct.mae)
        rec_gain = float(by.mae - cy.mae)
        top20_gain = float(btop.mae - ctop.mae)
        target_nonworse += int(target_gain >= -1e-12)
        rec_nonworse += int(rec_gain >= -1e-12)
        top20_nonworse += int(top20_gain >= -1e-12)
        max_target_worsen = max(max_target_worsen, -target_gain)
        max_rec_worsen = max(max_rec_worsen, -rec_gain)
        season_rows.append({
            "season": int(s),
            "target_baseline_mae": float(bt.mae),
            "target_candidate_mae": float(ct.mae),
            "target_mae_gain": target_gain,
            "rec_yards_baseline_mae": float(by.mae),
            "rec_yards_candidate_mae": float(cy.mae),
            "rec_yards_mae_gain": rec_gain,
            "rec_yards_baseline_p90": float(by.p90_abs_error),
            "rec_yards_candidate_p90": float(cy.p90_abs_error),
            "baseline_30_plus_rate": float(by.miss_30_plus_rate),
            "candidate_30_plus_rate": float(cy.miss_30_plus_rate),
            "baseline_50_plus_rate": float(by.miss_50_plus_rate),
            "candidate_50_plus_rate": float(cy.miss_50_plus_rate),
            "top20_baseline_mae": float(btop.mae),
            "top20_candidate_mae": float(ctop.mae),
            "top20_mae_gain": top20_gain,
        })

    phase_nonworse = 0
    phase_total = 0
    for s in TEST_SEASONS:
        for ph in phases.loc[phases.season.eq(s), "phase"].dropna().unique():
            b = phases.loc[(phases.season.eq(s)) & phases.phase.eq(ph) & phases.variant.eq(BASE)]
            c = phases.loc[(phases.season.eq(s)) & phases.phase.eq(ph) & phases.variant.eq(CAND)]
            if len(b) and len(c):
                phase_total += 1
                phase_nonworse += int(float(c.iloc[0].mae) <= float(b.iloc[0].mae) + 1e-12)

    cbt, cct = sr("COMBINED", BASE, "targets"), sr("COMBINED", CAND, "targets")
    cby, ccy = sr("COMBINED", BASE, "rec_yards"), sr("COMBINED", CAND, "rec_yards")
    topb, topc = cbr(BASE, "TOP20"), cbr(CAND, "TOP20")
    restb, restc = cbr(BASE, "REST80"), cbr(CAND, "REST80")
    boot = _combined_bootstrap(x, a.bootstrap_reps)

    gates = {
        "combined_target_gain": float(cbt.mae - cct.mae) >= MIN_COMBINED_TARGET_MAE_GAIN,
        "combined_rec_yards_gain": float(cby.mae - ccy.mae) >= MIN_COMBINED_REC_YARDS_MAE_GAIN,
        "season_target_nonworse": target_nonworse >= MIN_SEASON_TARGET_NONWORSE,
        "season_rec_yards_nonworse": rec_nonworse >= MIN_SEASON_REC_YARDS_NONWORSE,
        "single_season_target_worsen_guard": max_target_worsen <= MAX_SINGLE_SEASON_TARGET_MAE_WORSEN + 1e-12,
        "single_season_rec_yards_worsen_guard": max_rec_worsen <= MAX_SINGLE_SEASON_REC_YARDS_MAE_WORSEN + 1e-12,
        "combined_p90_nonworse": float(ccy.p90_abs_error) <= float(cby.p90_abs_error) + 1e-12,
        "combined_cat30_guard": float(ccy.miss_30_plus_rate) <= float(cby.miss_30_plus_rate) + MAX_COMBINED_TAIL_RATE_WORSEN,
        "combined_cat50_guard": float(ccy.miss_50_plus_rate) <= float(cby.miss_50_plus_rate) + MAX_COMBINED_TAIL_RATE_WORSEN,
        "combined_abs_bias_nonworse": abs(float(ccy.bias)) <= abs(float(cby.bias)) + 1e-12,
        "combined_top20_gain": float(topb.mae - topc.mae) >= MIN_COMBINED_TOP20_REC_YARDS_GAIN,
        "combined_top20_p90_nonworse": float(topc.p90_abs_error) <= float(topb.p90_abs_error) + 1e-12,
        "combined_top20_cat30_guard": float(topc.miss_30_plus_rate) <= float(topb.miss_30_plus_rate) + MAX_COMBINED_TAIL_RATE_WORSEN,
        "combined_top20_cat50_guard": float(topc.miss_50_plus_rate) <= float(topb.miss_50_plus_rate) + MAX_COMBINED_TAIL_RATE_WORSEN,
        "combined_rest80_guard": float(restc.mae) <= float(restb.mae) + MAX_COMBINED_REST80_REC_YARDS_WORSEN,
        "top20_seasons_nonworse": top20_nonworse >= MIN_TOP20_SEASONS_NONWORSE,
        "phase_stability": phase_nonworse >= MIN_PHASES_NONWORSE,
        "combined_bootstrap": boot >= MIN_COMBINED_BOOTSTRAP_IMPROVE_PROB,
        "rb_pool_conservation": float(audit.rb_pool_gap.abs().max()) <= 1e-12,
        "team_mass_conservation": float(audit.team_player_mass_gap.abs().max()) <= 1e-12,
        "non_rb_exact": float(audit.max_non_rb_entitlement_delta.max()) <= 1e-12,
        "sportsbook_zero": int(audit.sportsbook_inputs_used.max()) == 0,
        "future_feature_zero": int(audit.current_future_outcomes_used_in_features.max()) == 0,
        "reliability_bounded": bool(((audit.reliability >= 0) & (audit.reliability <= 1)).all()),
    }
    passed = all(gates.values())

    result = {
        "candidate": "RB_R10_MODERN_STABILITY_V1",
        "evaluated_model": "RB_R9_RECEIVING_IDENTITY_SHRINKAGE_V1",
        "disposition": "RB_R10_MODERN_STABILITY_PASS" if passed else "RB_R10_MODERN_STABILITY_FAIL",
        "modern_stability_pass": bool(passed),
        "r9_science_status": "RB_R9_RECEIVING_IDENTITY_SHRINKAGE_OOS_PASS_UNCHANGED",
        "governance_note": "2022-2025 are modern stability seasons already visible to the research program; not pristine confirmation",
        "next_if_pass": "shadow integration plus prospective 2026 grading; no production promotion from R10 alone",
        "history_start": int(a.history_start),
        "sportsbook_inputs_used": 0,
        "features": list(r9.FEATURES),
        "folds": fold_meta,
        "season_target_nonworse_count": int(target_nonworse),
        "season_rec_yards_nonworse_count": int(rec_nonworse),
        "top20_seasons_nonworse_count": int(top20_nonworse),
        "phase_nonworse_count": int(phase_nonworse),
        "phase_total": int(phase_total),
        "combined_bootstrap_improve_probability": float(boot),
        "combined_target_baseline_mae": float(cbt.mae),
        "combined_target_candidate_mae": float(cct.mae),
        "combined_rec_yards_baseline_mae": float(cby.mae),
        "combined_rec_yards_candidate_mae": float(ccy.mae),
        "combined_top20_baseline_mae": float(topb.mae),
        "combined_top20_candidate_mae": float(topc.mae),
        "combined_rest80_baseline_mae": float(restb.mae),
        "combined_rest80_candidate_mae": float(restc.mae),
        "gates": gates,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(a.out_dir / "rb_r10_predictions.csv", index=False)
    audit.to_csv(a.out_dir / "rb_r10_conservation_audit.csv", index=False)
    coef.to_csv(a.out_dir / "rb_r10_coefficients.csv", index=False)
    oof.to_csv(a.out_dir / "rb_r10_reliability_oof.csv", index=False)
    summary.to_csv(a.out_dir / "rb_r10_market_summary.csv", index=False)
    phases.to_csv(a.out_dir / "rb_r10_phase_summary.csv", index=False)
    buckets.to_csv(a.out_dir / "rb_r10_identity_bucket_by_season.csv", index=False)
    combined_buckets.to_csv(a.out_dir / "rb_r10_identity_bucket_combined.csv", index=False)
    pd.DataFrame(season_rows).to_csv(a.out_dir / "rb_r10_season_summary.csv", index=False)
    (a.out_dir / "rb_r10_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print("\n=== R10 season summary ===\n", pd.DataFrame(season_rows).to_string(index=False))
    print("\n=== R10 combined market summary ===\n", summary.loc[summary.season_bucket.eq("COMBINED")].to_string(index=False))
    print("\n=== R10 combined identity buckets ===\n", combined_buckets.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
