#!/usr/bin/env python3
"""Diagnostic-only forensic of RB-R10 modern receiving-identity behavior.

R9 remains a historical scientific PASS and R10 remains a modern-stability FAIL.
This script does not fit, tune, promote, or alter either model. It uses the frozen
R10 prediction artifact to localize *where* the receiving-identity adjustment helps
or hurts, with particular attention to the top-20% strict-prior receiving-identity
backs.

Outcome-conditioned target buckets (0-2, 3-4, 5-6, 7+) are forensic only. They are
never deployable inputs. Pregame separability is evaluated separately using only
columns already produced before the game by R9/R10.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "M38_EXPLICIT_BASELINE"
CAND = "RB_R9_IDENTITY_SHRINKAGE"
KEYS = ["season", "week", "team", "player_clean_key"]


def _metric(actual: pd.Series, pred: pd.Series) -> dict:
    a = pd.to_numeric(actual, errors="coerce").to_numpy(float)
    p = pd.to_numeric(pred, errors="coerce").to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(p)
    a, p = a[ok], p[ok]
    if len(a) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "p90_abs_error": np.nan}
    e = p - a
    ae = np.abs(e)
    return {
        "n": int(len(a)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(e.mean()),
        "p90_abs_error": float(np.quantile(ae, 0.90)),
    }


def _bucket_targets(x: pd.Series) -> pd.Categorical:
    v = pd.to_numeric(x, errors="coerce")
    return pd.cut(v, bins=[-np.inf, 2, 4, 6, np.inf], labels=["0-2", "3-4", "5-6", "7+"], right=True)


def _matched(pred: pd.DataFrame) -> pd.DataFrame:
    required = set(KEYS + [
        "variant", "actual_targets", "pred_targets", "actual_rec_yards", "mc_rec_yards",
        "prior_rb_room_share", "r9_raw_r8_residual", "r9_calibrated_residual", "r9_reliability",
    ])
    missing = sorted(required - set(pred.columns))
    if missing:
        raise RuntimeError(f"R10 predictions missing required columns: {missing}")

    b = pred.loc[pred.variant.eq(BASE)].copy()
    c = pred.loc[pred.variant.eq(CAND)].copy()
    keep = KEYS + [
        "actual_targets", "pred_targets", "actual_rec_yards", "mc_rec_yards",
        "prior_rb_room_share", "r9_raw_r8_residual", "r9_calibrated_residual", "r9_reliability",
    ]
    z = b[keep].merge(c[keep], on=KEYS, suffixes=("_b", "_c"), validate="one_to_one")
    if z.empty:
        raise RuntimeError("R10 forensic produced zero matched baseline/candidate rows")

    # Outcomes and strict-prior identity should be identical across variants.
    for c0 in ("actual_targets", "actual_rec_yards", "prior_rb_room_share", "r9_raw_r8_residual", "r9_calibrated_residual", "r9_reliability"):
        a = pd.to_numeric(z[f"{c0}_b"], errors="coerce")
        d = pd.to_numeric(z[f"{c0}_c"], errors="coerce")
        if np.nanmax(np.abs(a.to_numpy(float) - d.to_numpy(float))) > 1e-12:
            raise RuntimeError(f"variant mismatch for invariant column {c0}")

    z["actual_targets"] = pd.to_numeric(z.actual_targets_b, errors="coerce")
    z["actual_rec_yards"] = pd.to_numeric(z.actual_rec_yards_b, errors="coerce")
    z["baseline_pred_targets"] = pd.to_numeric(z.pred_targets_b, errors="coerce")
    z["candidate_pred_targets"] = pd.to_numeric(z.pred_targets_c, errors="coerce")
    z["baseline_pred_rec_yards"] = pd.to_numeric(z.mc_rec_yards_b, errors="coerce")
    z["candidate_pred_rec_yards"] = pd.to_numeric(z.mc_rec_yards_c, errors="coerce")
    z["prior_rb_room_share"] = pd.to_numeric(z.prior_rb_room_share_b, errors="coerce").fillna(0.0)
    z["r9_raw_r8_residual"] = pd.to_numeric(z.r9_raw_r8_residual_b, errors="coerce").fillna(0.0)
    z["r9_calibrated_residual"] = pd.to_numeric(z.r9_calibrated_residual_b, errors="coerce").fillna(0.0)
    z["r9_reliability"] = pd.to_numeric(z.r9_reliability_b, errors="coerce").fillna(0.0)

    z["target_delta"] = z.candidate_pred_targets - z.baseline_pred_targets
    z["rec_yards_delta"] = z.candidate_pred_rec_yards - z.baseline_pred_rec_yards
    z["baseline_abs_rec_error"] = (z.baseline_pred_rec_yards - z.actual_rec_yards).abs()
    z["candidate_abs_rec_error"] = (z.candidate_pred_rec_yards - z.actual_rec_yards).abs()
    z["rec_mae_gain"] = z.baseline_abs_rec_error - z.candidate_abs_rec_error
    z["baseline_abs_target_error"] = (z.baseline_pred_targets - z.actual_targets).abs()
    z["candidate_abs_target_error"] = (z.candidate_pred_targets - z.actual_targets).abs()
    z["target_mae_gain"] = z.baseline_abs_target_error - z.candidate_abs_target_error
    z["target_state"] = _bucket_targets(z.actual_targets)

    # Top-20 receiving identity is defined pregame within each season-week, exactly
    # from strict-prior RB-room share. No outcome contributes to this ranking.
    z["identity_pct"] = z.groupby(["season", "week"])["prior_rb_room_share"].rank(pct=True, method="average")
    z["identity_bucket"] = np.where(z.identity_pct.gt(0.80), "TOP20", "REST80")

    # Did the candidate cross from one side of the eventual outcome to the other?
    bpred = z.baseline_pred_rec_yards
    cpred = z.candidate_pred_rec_yards
    actual = z.actual_rec_yards
    z["crossed_rec_outcome"] = (((bpred <= actual) & (cpred > actual)) | ((bpred >= actual) & (cpred < actual))).astype(int)
    z["candidate_helped_rec"] = (z.candidate_abs_rec_error < z.baseline_abs_rec_error).astype(int)
    return z


def _state_summary(z: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for population, g0 in (("ALL", z), ("TOP20", z.loc[z.identity_bucket.eq("TOP20")]), ("REST80", z.loc[z.identity_bucket.eq("REST80")])):
        for state, g in g0.groupby("target_state", observed=False):
            if g.empty:
                continue
            bm = _metric(g.actual_rec_yards, g.baseline_pred_rec_yards)
            cm = _metric(g.actual_rec_yards, g.candidate_pred_rec_yards)
            btm = _metric(g.actual_targets, g.baseline_pred_targets)
            ctm = _metric(g.actual_targets, g.candidate_pred_targets)
            rows.append({
                "population": population,
                "target_state": str(state),
                "n": int(len(g)),
                "mean_actual_targets": float(g.actual_targets.mean()),
                "mean_baseline_pred_targets": float(g.baseline_pred_targets.mean()),
                "mean_candidate_pred_targets": float(g.candidate_pred_targets.mean()),
                "mean_target_delta": float(g.target_delta.mean()),
                "target_baseline_mae": float(btm["mae"]),
                "target_candidate_mae": float(ctm["mae"]),
                "target_mae_gain": float(btm["mae"] - ctm["mae"]),
                "rec_baseline_mae": float(bm["mae"]),
                "rec_candidate_mae": float(cm["mae"]),
                "rec_mae_gain": float(bm["mae"] - cm["mae"]),
                "rec_baseline_bias": float(bm["bias"]),
                "rec_candidate_bias": float(cm["bias"]),
                "rec_baseline_p90": float(bm["p90_abs_error"]),
                "rec_candidate_p90": float(cm["p90_abs_error"]),
                "baseline_30_plus_rate": float(g.baseline_abs_rec_error.ge(30).mean()),
                "candidate_30_plus_rate": float(g.candidate_abs_rec_error.ge(30).mean()),
                "baseline_50_plus_rate": float(g.baseline_abs_rec_error.ge(50).mean()),
                "candidate_50_plus_rate": float(g.candidate_abs_rec_error.ge(50).mean()),
                "candidate_help_rate": float(g.candidate_helped_rec.mean()),
                "crossed_outcome_rate": float(g.crossed_rec_outcome.mean()),
                "mean_rec_yards_delta": float(g.rec_yards_delta.mean()),
            })
    return pd.DataFrame(rows)


def _signal_quintiles(z: pd.DataFrame) -> pd.DataFrame:
    # All signals below exist before kickoff. The actual 5+/7+ labels are used only
    # to evaluate separability, never as features.
    signals = [
        "prior_rb_room_share",
        "baseline_pred_targets",
        "candidate_pred_targets",
        "target_delta",
        "r9_raw_r8_residual",
        "r9_calibrated_residual",
    ]
    rows = []
    for population, g0 in (("ALL", z), ("TOP20", z.loc[z.identity_bucket.eq("TOP20")])):
        for sig in signals:
            g = g0.copy()
            # Percentile rank avoids qcut duplicate-edge failures and fixes bins at quintiles.
            g["signal_pct"] = g.groupby(["season", "week"])[sig].rank(pct=True, method="average")
            g["signal_quintile"] = np.minimum(5, np.ceil(g.signal_pct * 5).clip(lower=1)).astype(int)
            for q, h in g.groupby("signal_quintile"):
                rows.append({
                    "population": population,
                    "signal": sig,
                    "quintile": int(q),
                    "n": int(len(h)),
                    "mean_signal": float(h[sig].mean()),
                    "actual_targets_mean": float(h.actual_targets.mean()),
                    "rate_5plus_targets": float(h.actual_targets.ge(5).mean()),
                    "rate_7plus_targets": float(h.actual_targets.ge(7).mean()),
                    "rec_yards_mean": float(h.actual_rec_yards.mean()),
                })
    return pd.DataFrame(rows)


def _signal_capture(z: pd.DataFrame) -> pd.DataFrame:
    signals = [
        "prior_rb_room_share",
        "baseline_pred_targets",
        "candidate_pred_targets",
        "target_delta",
        "r9_raw_r8_residual",
        "r9_calibrated_residual",
    ]
    rows = []
    for population, g0 in (("ALL", z), ("TOP20", z.loc[z.identity_bucket.eq("TOP20")])):
        for sig in signals:
            g = g0.copy()
            g["signal_pct"] = g.groupby(["season", "week"])[sig].rank(pct=True, method="average")
            hi = g.signal_pct.gt(0.80)
            for threshold in (5, 7):
                event = g.actual_targets.ge(threshold)
                base_rate = float(event.mean())
                hi_rate = float(event.loc[hi].mean()) if hi.any() else np.nan
                capture = float((event & hi).sum() / event.sum()) if event.sum() else np.nan
                rows.append({
                    "population": population,
                    "signal": sig,
                    "event": f"{threshold}+_targets",
                    "n": int(len(g)),
                    "event_count": int(event.sum()),
                    "top20_signal_n": int(hi.sum()),
                    "overall_event_rate": base_rate,
                    "top20_signal_event_rate": hi_rate,
                    "top20_signal_lift": float(hi_rate / base_rate) if base_rate > 0 and np.isfinite(hi_rate) else np.nan,
                    "event_capture_by_top20_signal": capture,
                })
    return pd.DataFrame(rows)


def _player_summary(z: pd.DataFrame) -> pd.DataFrame:
    g = z.loc[z.identity_bucket.eq("TOP20")].copy()
    rows = []
    for player, h in g.groupby("player_clean_key"):
        rows.append({
            "player_clean_key": player,
            "games": int(len(h)),
            "seasons": int(h.season.nunique()),
            "mean_prior_rb_room_share": float(h.prior_rb_room_share.mean()),
            "actual_targets_pg": float(h.actual_targets.mean()),
            "baseline_targets_pg": float(h.baseline_pred_targets.mean()),
            "candidate_targets_pg": float(h.candidate_pred_targets.mean()),
            "five_plus_target_games": int(h.actual_targets.ge(5).sum()),
            "seven_plus_target_games": int(h.actual_targets.ge(7).sum()),
            "baseline_rec_yards_mae": float(h.baseline_abs_rec_error.mean()),
            "candidate_rec_yards_mae": float(h.candidate_abs_rec_error.mean()),
            "rec_yards_mae_gain": float(h.rec_mae_gain.mean()),
            "baseline_rec_yards_bias": float((h.baseline_pred_rec_yards - h.actual_rec_yards).mean()),
            "candidate_rec_yards_bias": float((h.candidate_pred_rec_yards - h.actual_rec_yards).mean()),
            "candidate_help_rate": float(h.candidate_helped_rec.mean()),
            "crossed_outcome_rate": float(h.crossed_rec_outcome.mean()),
        })
    return pd.DataFrame(rows).sort_values(["games", "rec_yards_mae_gain"], ascending=[False, False])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/backtests/rb_r10_receiving_state_forensic_v1"))
    a = ap.parse_args()

    pred = pd.read_csv(a.predictions, low_memory=False)
    z = _matched(pred)
    state = _state_summary(z)
    quint = _signal_quintiles(z)
    capture = _signal_capture(z)
    players = _player_summary(z)

    top = z.loc[z.identity_bucket.eq("TOP20")].copy()
    low = top.loc[top.actual_targets.le(4)]
    high5 = top.loc[top.actual_targets.ge(5)]
    high7 = top.loc[top.actual_targets.ge(7)]

    def gain(g: pd.DataFrame) -> float:
        return float(g.baseline_abs_rec_error.mean() - g.candidate_abs_rec_error.mean()) if len(g) else np.nan

    result = {
        "diagnostic": "RB_R10_RECEIVING_STATE_FORENSIC_V1",
        "disposition": "DIAGNOSTIC_ONLY_R10_REMAINS_MODERN_STABILITY_FAIL",
        "r9_status": "RB_R9_RECEIVING_IDENTITY_SHRINKAGE_OOS_PASS_UNCHANGED",
        "r10_status": "RB_R10_MODERN_STABILITY_FAIL_UNCHANGED",
        "rows": int(len(z)),
        "top20_rows": int(len(top)),
        "top20_low_state_0_4_rows": int(len(low)),
        "top20_high_state_5plus_rows": int(len(high5)),
        "top20_high_state_7plus_rows": int(len(high7)),
        "top20_low_state_rec_yards_mae_gain": gain(low),
        "top20_high5_rec_yards_mae_gain": gain(high5),
        "top20_high7_rec_yards_mae_gain": gain(high7),
        "top20_low_state_candidate_help_rate": float(low.candidate_helped_rec.mean()) if len(low) else np.nan,
        "top20_high5_candidate_help_rate": float(high5.candidate_helped_rec.mean()) if len(high5) else np.nan,
        "top20_high7_candidate_help_rate": float(high7.candidate_helped_rec.mean()) if len(high7) else np.nan,
        "top20_low_state_crossed_outcome_rate": float(low.crossed_rec_outcome.mean()) if len(low) else np.nan,
        "top20_high5_crossed_outcome_rate": float(high5.crossed_rec_outcome.mean()) if len(high5) else np.nan,
        "mixture_pattern_supported": bool(np.isfinite(gain(low)) and np.isfinite(gain(high5)) and gain(high5) > 0 and gain(low) < 0),
        "interpretation_contract": "Outcome target states localize failure only. Any future deployable state probability must use strict-prior signals and receive a separately frozen OOS test.",
        "sportsbook_inputs_added": 0,
        "model_parameters_changed": 0,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    z.to_csv(a.out_dir / "rb_r10_state_casebook.csv", index=False)
    state.to_csv(a.out_dir / "rb_r10_state_summary.csv", index=False)
    quint.to_csv(a.out_dir / "rb_r10_pregame_signal_quintiles.csv", index=False)
    capture.to_csv(a.out_dir / "rb_r10_pregame_signal_capture.csv", index=False)
    players.to_csv(a.out_dir / "rb_r10_top20_player_summary.csv", index=False)
    (a.out_dir / "rb_r10_receiving_state_forensic_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print("\n=== top20 outcome-state summary ===")
    print(state.loc[state.population.eq("TOP20")].to_string(index=False))
    print("\n=== top20 pregame signal capture ===")
    print(capture.loc[capture.population.eq("TOP20")].to_string(index=False))
    print("\n=== top20 player summary head ===")
    print(players.head(30).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
