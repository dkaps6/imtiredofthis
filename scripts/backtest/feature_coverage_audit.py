"""Audit historical feature availability and Monte Carlo passing-yard inputs.

Diagnostic only. This module never changes projections. It consumes the completed
walk-forward component table, reports which football-context layers were actually
available to the historical model, and produces a QB passing trace for root-cause
analysis of Monte Carlo bias.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    ("player_target_share", "ctx_tgt_share_available", "historical_player_form"),
    ("player_rush_share", "ctx_rush_share_available", "historical_player_form"),
    ("player_ypa", "ctx_ypa_available", "historical_player_form"),
    ("team_success_rate", "ctx_success_rate_available", "reconstructed_prior_pbp"),
    ("team_pace", "ctx_pace_available", "reconstructed_prior_pbp"),
    ("team_play_volume", "ctx_plays_available", "reconstructed_prior_pbp"),
    ("team_proe", "ctx_proe_available", "reconstructed_prior_pbp"),
    ("pressure_matchup", "ctx_pressure_available", "reconstructed_prior_pbp"),
    ("explosive_allowance", "ctx_explosive_available", "reconstructed_prior_pbp"),
    ("defensive_epa", "ctx_def_epa_available", "reconstructed_prior_pbp"),
    ("coverage_scheme", "ctx_coverage_scheme_available", "optional_historical_enrichment"),
    ("box_rates", "ctx_box_rates_available", "optional_historical_enrichment"),
    ("wr_cb_matchup", "ctx_wr_cb_matchup_available", "optional_historical_enrichment"),
    ("injury_report", "ctx_injury_available", "optional_historical_enrichment"),
    ("weather", "ctx_weather_available", "optional_historical_enrichment"),
]


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    x = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    return x.gt(0).astype(bool)


def feature_coverage(predictions: pd.DataFrame) -> pd.DataFrame:
    x = predictions.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    keys = [c for c in ["season", "week", "team", "player_clean_key"] if c in x.columns]
    base = x.drop_duplicates(keys) if keys else x
    rows = []
    total = int(len(base))
    for feature, column, source in FEATURES:
        available = int(_bool_series(base, column).sum())
        coverage = float(available / total) if total else np.nan
        if available == 0:
            status = "missing_or_neutral"
        elif available == total:
            status = "available_all_rows"
        else:
            status = "partially_available"
        rows.append({
            "feature": feature,
            "trace_column": column,
            "source_class": source,
            "available_rows": available,
            "total_player_weeks": total,
            "coverage": coverage,
            "status": status,
        })
    return pd.DataFrame(rows)


def passing_trace(predictions: pd.DataFrame) -> pd.DataFrame:
    x = predictions.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    p = x.loc[x["market"].astype(str).str.lower().eq("pass_yards")].copy()
    if p.empty:
        return p
    for c in [
        "actual", "mc_proj", "actual_opportunities", "mc_projected_plays",
        "mc_pass_rate", "mc_expected_pass_attempts", "mc_base_ypa", "mc_bayes_ypa",
        "mc_rules_ypa", "mc_pass_eff_mult", "mc_off_pressure_allowed",
        "mc_def_pressure_generated", "mc_pressure_mismatch",
    ]:
        if c not in p.columns:
            p[c] = np.nan
        p[c] = pd.to_numeric(p[c], errors="coerce")
    p["mc_error"] = p["mc_proj"] - p["actual"]
    p["mc_abs_error"] = p["mc_error"].abs()
    p["actual_pass_attempts"] = p["actual_opportunities"]
    p["zero_attempt_actual"] = p["actual_pass_attempts"].fillna(0).eq(0).astype(int)
    p["high_attempt_projection"] = p["mc_expected_pass_attempts"].gt(45).fillna(False).astype(int)
    p["high_ypa_projection"] = p["mc_rules_ypa"].gt(9.5).fillna(False).astype(int)
    keep = [c for c in [
        "season", "week", "player", "player_clean_key", "team", "opponent", "position",
        "actual", "mc_proj", "mc_error", "mc_abs_error", "actual_pass_attempts",
        "zero_attempt_actual", "mc_projected_plays", "mc_pass_rate", "mc_expected_pass_attempts",
        "mc_base_ypa", "mc_bayes_ypa", "mc_rules_ypa", "mc_pass_eff_mult",
        "mc_off_pressure_allowed", "mc_def_pressure_generated", "mc_pressure_mismatch",
        "high_attempt_projection", "high_ypa_projection", "rules_applied", "rules_role",
        "prediction_cutoff",
    ] if c in p.columns]
    return p[keep].sort_values(["week", "team", "player"]).reset_index(drop=True)


def passing_summary(trace: pd.DataFrame) -> pd.DataFrame:
    if trace.empty:
        return pd.DataFrame()
    rows = []
    for label, g in [("all", trace), ("actual_pass_attempts_gt_0", trace.loc[trace["actual_pass_attempts"].fillna(0).gt(0)])]:
        if g.empty:
            continue
        err = pd.to_numeric(g["mc_error"], errors="coerce")
        abs_err = pd.to_numeric(g["mc_abs_error"], errors="coerce")
        rows.append({
            "slice": label,
            "n": int(len(g)),
            "mae": float(abs_err.mean()),
            "bias": float(err.mean()),
            "actual_mean": float(pd.to_numeric(g["actual"], errors="coerce").mean()),
            "mc_mean": float(pd.to_numeric(g["mc_proj"], errors="coerce").mean()),
            "actual_pass_attempts_mean": float(pd.to_numeric(g["actual_pass_attempts"], errors="coerce").mean()),
            "projected_pass_attempts_mean": float(pd.to_numeric(g["mc_expected_pass_attempts"], errors="coerce").mean()),
            "zero_attempt_actual_rows": int(pd.to_numeric(g["zero_attempt_actual"], errors="coerce").fillna(0).sum()),
            "high_attempt_projection_rows": int(pd.to_numeric(g["high_attempt_projection"], errors="coerce").fillna(0).sum()),
            "high_ypa_projection_rows": int(pd.to_numeric(g["high_ypa_projection"], errors="coerce").fillna(0).sum()),
        })
    return pd.DataFrame(rows)


def audit(predictions: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    coverage = feature_coverage(predictions)
    trace = passing_trace(predictions)
    summary = passing_summary(trace)
    worst = trace.sort_values("mc_abs_error", ascending=False).head(100).copy() if not trace.empty else trace.copy()

    paths = {
        "feature_coverage": out_dir / "historical_feature_coverage.csv",
        "passing_trace": out_dir / "mc_passing_trace.csv",
        "passing_summary": out_dir / "mc_passing_trace_summary.csv",
        "passing_worst": out_dir / "mc_passing_worst_misses.csv",
    }
    coverage.to_csv(paths["feature_coverage"], index=False)
    trace.to_csv(paths["passing_trace"], index=False)
    summary.to_csv(paths["passing_summary"], index=False)
    worst.to_csv(paths["passing_worst"], index=False)
    return paths


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, default=Path("data/backtests/component_predictions.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/diagnostics"))
    args = p.parse_args()
    if not args.predictions.exists() or args.predictions.stat().st_size == 0:
        raise RuntimeError(f"missing component predictions: {args.predictions}")
    predictions = pd.read_csv(args.predictions)
    paths = audit(predictions, args.out_dir)
    print("[feature_coverage] historical feature availability")
    print(pd.read_csv(paths["feature_coverage"]).to_string(index=False))
    print("\n[mc_passing_trace] summary")
    s = pd.read_csv(paths["passing_summary"])
    print(s.to_string(index=False) if not s.empty else "no passing rows")
    for name, path in paths.items():
        print(f"[feature_coverage] {name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
