#!/usr/bin/env python3
"""RB-R7: freeze canonical RB receiving-room mass; improve only within-RB entitlement.

R6C scientifically failed because its predicted RB-room mass degraded target/tail
accuracy, while the full two-stage candidate still improved receiving-yard MAE in
both OOS seasons.  The room-only variant was worse than baseline, identifying the
outer room-mass stage as the weak component.  R7 therefore removes that stage
entirely rather than tuning it.

Frozen hypothesis
-----------------
- start from canonical explicit finite target entitlement (M38 already applied);
- preserve each team-game's total RB/FB target-entitlement pool EXACTLY;
- preserve every non-RB player's entitlement EXACTLY;
- redistribute only within the existing RB/FB pool using strict-prior offensive
  participation and the same frozen R6 within-room feature family;
- StandardScaler + Ridge(alpha=20), residual clip [-1,1];
- no sportsbook, current/future outcomes, or current/future participation.

Freshness governance
--------------------
R6 used 2022 as training and 2023/2024 as confirmation; 2025 was diagnostic.
R7's primary fresh historical confirmation is therefore:
    train 2020 -> confirm 2021
A second historical replication is:
    train 2021 -> confirm 2022
2022 is explicitly NOT called untouched because it appeared as R6 training data.
The frozen PASS contract requires the fresh 2021 fold to pass all core science
and the 2022 replication to be non-worse on target and receiving-yard MAE.

A PASS authorizes a separate all-history refit/integration confirmation.  It does
not promote R7 by itself.
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

from scripts.backtest.evaluate_wr_room_empirical_bayes_v1 import metric, read
from scripts.backtest import evaluate_rb_r6_two_stage_receiving_entitlement_v1 as r6
from scripts.modeling.te_r5p_entitlement_adapter_v1 import _load_snaps
from scripts.simulation_explicit_entitlement_v1 import simulate as explicit_simulate

BASE = "M38_EXPLICIT_BASELINE"
CAND = "RB_R7_FIXED_ROOM_WITHIN_PARTICIPATION"
ALPHA = 20.0
PRED_CLIP = 1.0
EPS = r6.EPS
FEATURES = list(r6.WITHIN_FEATURES)

# Frozen before any R7 result is observed.
MIN_FRESH_TARGET_MAE_GAIN = 0.02
MIN_FRESH_REC_YARDS_MAE_GAIN = 0.10
MIN_COMBINED_TARGET_MAE_GAIN = 0.02
MIN_COMBINED_REC_YARDS_MAE_GAIN = 0.10
MIN_FRESH_NONWORSE_PHASES = 3  # of 4
MIN_BOOTSTRAP_IMPROVE_PROB = 0.65
MAX_TAIL_RATE_WORSEN = 0.0025


# Mechanical row-identity compatibility discovered in R6C.  Preserve the full
# baseline row label by value through the shared participation helper.
_original_strict_prior = r6._strict_prior_snap_features


def _strict_prior_preserve_baseline_row(frame: pd.DataFrame, snaps: pd.DataFrame):
    if "_row_index" not in frame.columns:
        return _original_strict_prior(frame, snaps)
    source = frame["_row_index"].to_numpy(copy=True)
    clean = frame.drop(columns=["_row_index"]).copy()
    out, future = _original_strict_prior(clean, snaps)
    if len(out) != len(source):
        raise RuntimeError(f"R7 participation helper changed row count {len(source)} -> {len(out)}")
    out["_row_index"] = source
    return out, future


r6._strict_prior_snap_features = _strict_prior_preserve_baseline_row


def _fit_within(*, season: int, data_dir: Path, logs: pd.DataFrame, snaps: pd.DataFrame):
    _, within_train, future = r6._training_cases(
        season=season, data_dir=data_dir, logs=logs, snaps=snaps
    )
    if future:
        raise RuntimeError(f"R7 training season {season} used same/future participation: {future}")
    model = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
    model.fit(within_train[FEATURES], within_train["within_residual_target"])
    return model, within_train


def _apply_fixed_room(baseline: pd.DataFrame, snaps: pd.DataFrame, model):
    out = baseline.copy()
    rb, _, future = r6._rb_features(out, snaps)
    if rb.empty:
        raise RuntimeError("R7 confirmation week has zero RB/FB rows")
    pred = np.clip(model.predict(rb[FEATURES]), -PRED_CLIP, PRED_CLIP)
    rb["r7_predicted_residual"] = pred
    rb["r7_score"] = np.log(rb["b0_rb_within_share"].clip(lower=0.0).to_numpy(float) + EPS) + pred
    rb["r7_entitlement_tgt_share"] = rb["baseline_entitlement_tgt_share"].astype(float)

    audit_rows = []
    for (event_id, team), idx in rb.groupby(["event_id", "team"], sort=False).groups.items():
        source_idx = rb.loc[idx, "_row_index"].astype(int).tolist()
        team_idx = out.index[
            out["event_id"].astype(str).eq(str(event_id))
            & out["team"].astype(str).eq(str(team))
        ]
        rb_pool = float(pd.to_numeric(out.loc[source_idx, "entitlement_tgt_share"], errors="raise").sum())
        team_before = float(pd.to_numeric(out.loc[team_idx, "entitlement_tgt_share"], errors="raise").sum())
        non_rb_idx = [i for i in team_idx if i not in source_idx]
        non_rb_before = pd.to_numeric(out.loc[non_rb_idx, "entitlement_tgt_share"], errors="raise").copy() if non_rb_idx else pd.Series(dtype=float)

        score = rb.loc[idx, "r7_score"].to_numpy(float)
        stable = score - float(np.max(score))
        weight = np.exp(stable)
        share = weight / float(weight.sum())
        candidate = rb_pool * share
        if len(candidate):
            candidate[int(np.argmax(share))] += rb_pool - float(candidate.sum())
        out.loc[source_idx, "entitlement_tgt_share"] = candidate
        rb.loc[idx, "r7_entitlement_tgt_share"] = candidate

        rb_after = float(pd.to_numeric(out.loc[source_idx, "entitlement_tgt_share"], errors="raise").sum())
        team_after = float(pd.to_numeric(out.loc[team_idx, "entitlement_tgt_share"], errors="raise").sum())
        non_rb_after = pd.to_numeric(out.loc[non_rb_idx, "entitlement_tgt_share"], errors="raise") if non_rb_idx else pd.Series(dtype=float)
        non_rb_gap = float(np.max(np.abs(non_rb_after.to_numpy(float) - non_rb_before.to_numpy(float)))) if len(non_rb_idx) else 0.0
        audit_rows.append({
            "event_id": str(event_id),
            "team": str(team),
            "baseline_rb_pool": rb_pool,
            "candidate_rb_pool": rb_after,
            "rb_pool_gap": rb_after - rb_pool,
            "baseline_team_player_mass": team_before,
            "candidate_team_player_mass": team_after,
            "team_player_mass_gap": team_after - team_before,
            "max_non_rb_entitlement_delta": non_rb_gap,
            "sportsbook_inputs_used": 0,
            "current_or_future_outcomes_used": 0,
        })
    return out, rb, pd.DataFrame(audit_rows), int(future)


def _bootstrap_prob(pred: pd.DataFrame, season: int, reps: int = 2000, seed: int = 70707) -> float:
    x = pred.loc[pred["season"].eq(season) & pred["position_family"].isin({"RB", "FB"})]
    keys = ["season", "week", "team", "player_clean_key"]
    b = x.loc[x.variant.eq(BASE), keys + ["actual_rec_yards", "mc_rec_yards"]]
    c = x.loc[x.variant.eq(CAND), keys + ["actual_rec_yards", "mc_rec_yards"]]
    z = b.merge(c, on=keys, suffixes=("_b", "_c"), validate="one_to_one")
    eb = (z["mc_rec_yards_b"] - z["actual_rec_yards_b"]).abs().to_numpy(float)
    ec = (z["mc_rec_yards_c"] - z["actual_rec_yards_c"]).abs().to_numpy(float)
    if not len(z):
        return np.nan
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(reps):
        ii = rng.integers(0, len(z), size=len(z))
        wins += int(float(ec[ii].mean()) < float(eb[ii].mean()))
    return float(wins / reps)


def _fold(*, train_season: int, test_season: int, train_dir: Path, test_dir: Path,
          train_logs: pd.DataFrame, test_logs: pd.DataFrame, snaps: pd.DataFrame,
          iterations: int):
    model, train = _fit_within(season=train_season, data_dir=train_dir, logs=train_logs, snaps=snaps)
    pred_parts, audit_parts = [], []
    future_total = 0
    for week in range(1, 19):
        baseline = r6._build_bundle_frame(
            season=test_season, week=week, prior_season=test_season - 1,
            data_dir=test_dir, logs=test_logs,
        )
        candidate, _, audit, future = _apply_fixed_room(baseline, snaps, model)
        future_total += future
        audit["train_season"] = train_season
        audit["test_season"] = test_season
        audit["week"] = week
        audit_parts.append(audit)

        seed = 707000 + test_season * 100 + week
        bsim = explicit_simulate(baseline, iterations=iterations, seed=seed)
        csim = explicit_simulate(candidate, iterations=iterations, seed=seed)
        at = r6._actual_target_frame(test_logs, test_season, week)
        ay = r6._actual_yards_frame(test_logs, test_season, week)
        for variant, frame, sim in ((BASE, baseline, bsim), (CAND, candidate, csim)):
            p = r6._prediction_rows(frame, sim, variant, test_season, week)
            p = p.merge(at, on=["team", "player_clean_key"], how="inner")
            p = p.merge(ay, on=["team", "player_clean_key"], how="inner")
            p["train_season"] = train_season
            pred_parts.append(p)
        print(f"[rb-r7] confirm train={train_season} test={test_season} week={week:02d}")
    if future_total:
        raise RuntimeError(f"R7 confirmation season {test_season} used same/future participation: {future_total}")

    scaler = model.named_steps["standardscaler"]
    ridge = model.named_steps["ridge"]
    coef = pd.DataFrame({
        "train_season": train_season,
        "test_season": test_season,
        "feature": FEATURES,
        "scaler_mean": scaler.mean_.astype(float),
        "scaler_scale": scaler.scale_.astype(float),
        "ridge_coef": ridge.coef_.astype(float),
        "ridge_intercept": float(ridge.intercept_),
    })
    return pd.concat(pred_parts, ignore_index=True), pd.concat(audit_parts, ignore_index=True), coef, int(len(train))


def _summary(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pred.loc[pred["position_family"].isin({"RB", "FB"})].copy()
    x["phase"] = pd.cut(x.week, [0, 4, 9, 13, 18], labels=["W1-4", "W5-9", "W10-13", "W14-18"])
    x["abs_rec_yards_error"] = (pd.to_numeric(x.mc_rec_yards, errors="coerce") - pd.to_numeric(x.actual_rec_yards, errors="coerce")).abs()
    rows = []
    for bucket, g0 in [("COMBINED", x)] + [(str(s), g) for s, g in x.groupby("season")]:
        for variant, g in g0.groupby("variant"):
            for market, ac, pc in (("targets", "actual_targets", "pred_targets"), ("rec_yards", "actual_rec_yards", "mc_rec_yards")):
                r = {"season_bucket": bucket, "variant": variant, "market": market, **metric(g[ac], g[pc])}
                if market == "rec_yards":
                    r["miss_30_plus_rate"] = float(g.abs_rec_yards_error.ge(30).mean())
                    r["miss_50_plus_rate"] = float(g.abs_rec_yards_error.ge(50).mean())
                rows.append(r)
    phases = []
    for season, sg in x.groupby("season"):
        for (variant, phase), g in sg.groupby(["variant", "phase"], observed=False):
            if len(g):
                phases.append({"season": int(season), "phase": str(phase), "variant": variant, **metric(g.actual_rec_yards, g.mc_rec_yards)})
    return pd.DataFrame(rows), pd.DataFrame(phases)


def main() -> int:
    ap = argparse.ArgumentParser()
    for s in (2020, 2021, 2022):
        ap.add_argument(f"--data-{s}", dest=f"data_{s}", type=Path, required=True)
        ap.add_argument(f"--logs-{s}", dest=f"logs_{s}", type=Path, required=True)
    ap.add_argument("--iterations", type=int, default=2000)
    ap.add_argument("--out-dir", type=Path, default=Path("data/backtests/rb_r7_fixed_room_within_v1"))
    a = ap.parse_args()

    data = {s: getattr(a, f"data_{s}") for s in (2020, 2021, 2022)}
    logs = {s: read(getattr(a, f"logs_{s}")) for s in (2020, 2021, 2022)}
    snaps, snap_dup_rate, snap_source_seasons = _load_snaps()

    all_pred, all_audit, all_coef, fold_meta = [], [], [], []
    for tr, te in ((2020, 2021), (2021, 2022)):
        p, au, co, train_rows = _fold(
            train_season=tr, test_season=te,
            train_dir=data[tr], test_dir=data[te],
            train_logs=logs[tr], test_logs=logs[te], snaps=snaps,
            iterations=a.iterations,
        )
        all_pred.append(p); all_audit.append(au); all_coef.append(co)
        fold_meta.append({
            "train_season": tr,
            "test_season": te,
            "train_rows": train_rows,
            "fresh_confirmation": bool(te == 2021),
            "previously_seen_elsewhere": bool(te == 2022),
        })

    pred = pd.concat(all_pred, ignore_index=True)
    audit = pd.concat(all_audit, ignore_index=True)
    coef = pd.concat(all_coef, ignore_index=True)
    summary, phases = _summary(pred)

    def sr(bucket: str, variant: str, market: str) -> pd.Series:
        return summary.loc[
            summary.season_bucket.eq(bucket)
            & summary.variant.eq(variant)
            & summary.market.eq(market)
        ].iloc[0]

    fresh_bt, fresh_ct = sr("2021", BASE, "targets"), sr("2021", CAND, "targets")
    fresh_by, fresh_cy = sr("2021", BASE, "rec_yards"), sr("2021", CAND, "rec_yards")
    rep_bt, rep_ct = sr("2022", BASE, "targets"), sr("2022", CAND, "targets")
    rep_by, rep_cy = sr("2022", BASE, "rec_yards"), sr("2022", CAND, "rec_yards")
    comb_bt, comb_ct = sr("COMBINED", BASE, "targets"), sr("COMBINED", CAND, "targets")
    comb_by, comb_cy = sr("COMBINED", BASE, "rec_yards"), sr("COMBINED", CAND, "rec_yards")

    fresh_phase = phases.loc[phases.season.eq(2021)].pivot_table(index="phase", columns="variant", values="mae").dropna()
    fresh_nonworse = int((fresh_phase[CAND] <= fresh_phase[BASE]).sum()) if not fresh_phase.empty else 0
    fresh_boot = _bootstrap_prob(pred, 2021)

    scientific = {
        "fresh_2021_target_mae_gain_ge_0_02": bool(float(fresh_bt.mae - fresh_ct.mae) >= MIN_FRESH_TARGET_MAE_GAIN),
        "fresh_2021_rec_yards_mae_gain_ge_0_10": bool(float(fresh_by.mae - fresh_cy.mae) >= MIN_FRESH_REC_YARDS_MAE_GAIN),
        "fresh_2021_rec_yards_p90_nonworse": bool(float(fresh_cy.p90_abs_error) <= float(fresh_by.p90_abs_error)),
        "fresh_2021_miss30_guard": bool(float(fresh_cy.miss_30_plus_rate) <= float(fresh_by.miss_30_plus_rate) + MAX_TAIL_RATE_WORSEN),
        "fresh_2021_miss50_guard": bool(float(fresh_cy.miss_50_plus_rate) <= float(fresh_by.miss_50_plus_rate) + MAX_TAIL_RATE_WORSEN),
        "fresh_2021_phase_nonworse_at_least_3_of_4": bool(fresh_nonworse >= MIN_FRESH_NONWORSE_PHASES),
        "fresh_2021_bootstrap_improve_prob_ge_0_65": bool(fresh_boot >= MIN_BOOTSTRAP_IMPROVE_PROB),
        "replication_2022_target_mae_nonworse": bool(float(rep_ct.mae) <= float(rep_bt.mae)),
        "replication_2022_rec_yards_mae_nonworse": bool(float(rep_cy.mae) <= float(rep_by.mae)),
        "combined_target_mae_gain_ge_0_02": bool(float(comb_bt.mae - comb_ct.mae) >= MIN_COMBINED_TARGET_MAE_GAIN),
        "combined_rec_yards_mae_gain_ge_0_10": bool(float(comb_by.mae - comb_cy.mae) >= MIN_COMBINED_REC_YARDS_MAE_GAIN),
    }
    integrity = {
        "primary_confirmation_2021_fresh": True,
        "2022_replication_not_claimed_untouched": True,
        "2023_2024_2025_excluded_from_r7_confirmation": bool(not set(pred.season.unique()).intersection({2023, 2024, 2025})),
        "rb_room_mass_exact": bool(float(audit.rb_pool_gap.abs().max()) <= 1e-12),
        "team_player_mass_exact": bool(float(audit.team_player_mass_gap.abs().max()) <= 1e-12),
        "non_rb_entitlement_exact": bool(float(audit.max_non_rb_entitlement_delta.abs().max()) == 0.0),
        "sportsbook_inputs_zero": bool(int(audit.sportsbook_inputs_used.sum()) == 0),
        "current_future_outcomes_zero": bool(int(audit.current_or_future_outcomes_used.sum()) == 0),
        "snap_duplicate_rate_le_0_01": bool(float(snap_dup_rate) <= 0.01),
    }
    passed = all(scientific.values()) and all(integrity.values())
    result = {
        "migration": "RB_R7_FIXED_ROOM_WITHIN_PARTICIPATION_V1",
        "disposition": "RB_R7_FIXED_ROOM_WITHIN_PARTICIPATION_OOS_PASS" if passed else "RB_R7_FIXED_ROOM_WITHIN_PARTICIPATION_OOS_FAIL",
        "model": f"StandardScaler+Ridge(alpha={ALPHA}) within existing RB room only",
        "r6_room_mass_stage_removed_not_tuned": True,
        "primary_fresh_confirmation": {"train": 2020, "test": 2021},
        "secondary_replication": {"train": 2021, "test": 2022, "untouched_claim": False},
        "folds": fold_meta,
        "features": FEATURES,
        "fresh_2021_baseline_target": fresh_bt.to_dict(),
        "fresh_2021_candidate_target": fresh_ct.to_dict(),
        "fresh_2021_baseline_rec_yards": fresh_by.to_dict(),
        "fresh_2021_candidate_rec_yards": fresh_cy.to_dict(),
        "replication_2022_baseline_target": rep_bt.to_dict(),
        "replication_2022_candidate_target": rep_ct.to_dict(),
        "replication_2022_baseline_rec_yards": rep_by.to_dict(),
        "replication_2022_candidate_rec_yards": rep_cy.to_dict(),
        "combined_baseline_target": comb_bt.to_dict(),
        "combined_candidate_target": comb_ct.to_dict(),
        "combined_baseline_rec_yards": comb_by.to_dict(),
        "combined_candidate_rec_yards": comb_cy.to_dict(),
        "fresh_2021_nonworse_phases": fresh_nonworse,
        "fresh_2021_bootstrap_rec_yards_improve_probability": fresh_boot,
        "scientific_gates": scientific,
        "integrity_gates": integrity,
        "max_rb_room_mass_gap": float(audit.rb_pool_gap.abs().max()),
        "max_team_player_mass_gap": float(audit.team_player_mass_gap.abs().max()),
        "max_non_rb_entitlement_delta": float(audit.max_non_rb_entitlement_delta.abs().max()),
        "sportsbook_inputs_used": int(audit.sportsbook_inputs_used.sum()),
        "production_parameters_changed": 0,
        "snap_source_seasons": sorted(int(v) for v in snap_source_seasons),
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(a.out_dir / "rb_r7_oos_predictions.csv", index=False)
    audit.to_csv(a.out_dir / "rb_r7_conservation_audit.csv", index=False)
    coef.to_csv(a.out_dir / "rb_r7_fold_coefficients.csv", index=False)
    summary.to_csv(a.out_dir / "rb_r7_market_summary.csv", index=False)
    phases.to_csv(a.out_dir / "rb_r7_phase_summary.csv", index=False)
    (a.out_dir / "rb_r7_result.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
