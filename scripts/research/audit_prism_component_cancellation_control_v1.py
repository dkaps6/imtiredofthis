#!/usr/bin/env python3
"""PRISM component-cancellation control audit V1.

Diagnostic only. Measures whether small final player-output error can conceal
large opposing opportunity / efficiency / downstream component errors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION = "PRISM_COMPONENT_CANCELLATION_CONTROL_AUDIT_V1"
TOL = 1e-10


def num(v: Any) -> pd.Series:
    return pd.to_numeric(v, errors="coerce")


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def build(component_rows: pd.DataFrame, matched_controls: pd.DataFrame):
    required = {
        "season", "week", "team", "player", "player_clean_key", "position_family",
        "market", "predicted_opportunities", "actual_opportunities",
        "model_effective_efficiency", "actual_efficiency", "baseline_projection",
        "baseline_reconstructed", "actual_output", "grading_identity_valid",
        "component_decomposition_eligible", "sportsbook_inputs_used_upstream",
        "parameters_fit", "automatic_promotion",
    }
    missing = required - set(component_rows.columns)
    if missing:
        raise RuntimeError(f"component rows missing {sorted(missing)}")

    x = component_rows.copy()
    x["season"] = num(x["season"])
    x["week"] = num(x["week"])
    x = x.loc[
        x["season"].eq(2026)
        & x["week"].isin([1, 2, 3, 4])
        & x["grading_identity_valid"].eq(True)
        & x["component_decomposition_eligible"].eq(True)
    ].copy()
    if x.empty:
        raise RuntimeError("zero exact component rows")

    if x["sportsbook_inputs_used_upstream"].astype(bool).any():
        raise RuntimeError("sportsbook upstream use found")
    if num(x["parameters_fit"]).fillna(0).ne(0).any():
        raise RuntimeError("fitted parameter found")
    if x["automatic_promotion"].astype(bool).any():
        raise RuntimeError("automatic promotion found")

    for c in (
        "predicted_opportunities", "actual_opportunities",
        "model_effective_efficiency", "actual_efficiency",
        "baseline_projection", "baseline_reconstructed", "actual_output",
    ):
        x[c] = num(x[c])
        if x[c].isna().any():
            raise RuntimeError(f"nonfinite required component field {c}")

    pn = x["predicted_opportunities"]
    an = x["actual_opportunities"]
    pe = x["model_effective_efficiency"]
    ae = x["actual_efficiency"]

    x["opportunity_contribution"] = (an - pn) * (pe + ae) / 2.0
    x["efficiency_contribution"] = (ae - pe) * (pn + an) / 2.0
    x["net_actual_minus_model"] = x["actual_output"] - x["baseline_projection"]
    x["downstream_contribution"] = (
        x["net_actual_minus_model"]
        - x["opportunity_contribution"]
        - x["efficiency_contribution"]
    )
    x["identity_gap"] = (
        x["opportunity_contribution"]
        + x["efficiency_contribution"]
        + x["downstream_contribution"]
        - x["net_actual_minus_model"]
    ).abs()

    max_gap = float(x["identity_gap"].max())
    if max_gap > TOL:
        raise RuntimeError(f"component identity gap {max_gap}")

    x["final_abs_error"] = x["net_actual_minus_model"].abs()
    x["gross_component_magnitude"] = (
        x["opportunity_contribution"].abs()
        + x["efficiency_contribution"].abs()
        + x["downstream_contribution"].abs()
    )
    x["cancellation_fraction"] = np.where(
        x["gross_component_magnitude"] > 1e-12,
        1.0 - x["final_abs_error"] / x["gross_component_magnitude"],
        0.0,
    )
    if (
        (x["cancellation_fraction"] < -1e-9)
        | (x["cancellation_fraction"] > 1.0 + 1e-9)
    ).any():
        raise RuntimeError("cancellation fraction outside algebraic bounds")
    x["cancellation_fraction"] = x["cancellation_fraction"].clip(0.0, 1.0)

    x["final_error_percentile"] = x.groupby(
        ["position_family", "market"], dropna=False
    )["final_abs_error"].rank(method="average", pct=True)
    x["final_error_decile"] = np.minimum(
        10,
        np.maximum(1, np.ceil(x["final_error_percentile"] * 10).astype(int)),
    )

    def summarize(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "rows": int(len(g)),
            "players": int(g["player_clean_key"].nunique()),
            "median_final_abs_error": float(g["final_abs_error"].median()),
            "median_gross_component_magnitude": float(g["gross_component_magnitude"].median()),
            "median_cancellation_fraction": float(g["cancellation_fraction"].median()),
            "p75_cancellation_fraction": float(g["cancellation_fraction"].quantile(0.75)),
            "p90_cancellation_fraction": float(g["cancellation_fraction"].quantile(0.90)),
        })

    by_week_market = (
        x.groupby(["week", "position_family", "market"], dropna=False)
        .apply(summarize, include_groups=False)
        .reset_index()
    )
    by_decile = (
        x.groupby(["position_family", "market", "final_error_decile"], dropna=False)
        .apply(summarize, include_groups=False)
        .reset_index()
    )

    bottom_decile = x.loc[x["final_error_decile"].eq(1)].copy()
    top_decile = x.loc[x["final_error_decile"].eq(10)].copy()

    controls = matched_controls.copy()
    needed_controls = {
        "case_week", "case_market", "control_player", "control_player_clean_key",
        "control_abs_error",
    }
    miss = needed_controls - set(controls.columns)
    if miss:
        raise RuntimeError(f"matched controls missing {sorted(miss)}")

    lookup = x[[
        "week", "player_clean_key", "market", "position_family",
        "final_abs_error", "gross_component_magnitude", "cancellation_fraction",
        "opportunity_contribution", "efficiency_contribution",
        "downstream_contribution", "predicted_opportunities",
        "actual_opportunities", "model_effective_efficiency", "actual_efficiency",
    ]].copy()
    exact_controls = controls.merge(
        lookup,
        left_on=["case_week", "control_player_clean_key", "case_market"],
        right_on=["week", "player_clean_key", "market"],
        how="inner",
        validate="many_to_one",
    )
    unique_controls = exact_controls.drop_duplicates(
        ["case_week", "control_player_clean_key", "case_market"]
    ).copy()

    examples = unique_controls.sort_values(
        ["cancellation_fraction", "gross_component_magnitude"],
        ascending=[False, False],
    ).head(25).copy()

    result = {
        "version": VERSION,
        "status": "PRISM_COMPONENT_CANCELLATION_CONTROL_AUDIT_COMPLETE__DIAGNOSTIC_ONLY",
        "exact_component_rows": int(len(x)),
        "unique_players": int(x["player_clean_key"].nunique()),
        "weeks": sorted(int(v) for v in x["week"].unique()),
        "max_identity_gap": max_gap,
        "population_median_cancellation_fraction": float(x["cancellation_fraction"].median()),
        "bottom_final_error_decile_rows": int(len(bottom_decile)),
        "bottom_final_error_decile_median_final_abs_error": float(bottom_decile["final_abs_error"].median()),
        "bottom_final_error_decile_median_gross_component_magnitude": float(bottom_decile["gross_component_magnitude"].median()),
        "bottom_final_error_decile_median_cancellation_fraction": float(bottom_decile["cancellation_fraction"].median()),
        "bottom_final_error_decile_p75_cancellation_fraction": float(bottom_decile["cancellation_fraction"].quantile(0.75)),
        "top_final_error_decile_rows": int(len(top_decile)),
        "top_final_error_decile_median_cancellation_fraction": float(top_decile["cancellation_fraction"].median()),
        "matched_control_rows_with_exact_components": int(len(exact_controls)),
        "unique_matched_controls_with_exact_components": int(len(unique_controls)),
        "unique_matched_controls_median_cancellation_fraction": (
            float(unique_controls["cancellation_fraction"].median())
            if len(unique_controls) else None
        ),
        "week6_outcomes_read": 0,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "models_fit": 0,
        "threshold_searches": 0,
        "projection_means_changed": 0,
        "production_changed": False,
        "boundary": (
            "control-validity diagnostic only; cancellation is not a reliability "
            "selector or projection adjustment"
        ),
    }
    return x, by_week_market, by_decile, exact_controls, examples, result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component-rows", type=Path, required=True)
    ap.add_argument("--matched-controls", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows, summary, deciles, controls, examples, result = build(
        read_csv(args.component_rows, "component rows"),
        read_csv(args.matched_controls, "matched controls"),
    )

    rows.to_csv(args.out_dir / "prism_component_cancellation_rows.csv", index=False, float_format="%.12f")
    summary.to_csv(args.out_dir / "prism_component_cancellation_by_week_market.csv", index=False, float_format="%.12f")
    deciles.to_csv(args.out_dir / "prism_component_cancellation_by_error_decile.csv", index=False, float_format="%.12f")
    controls.to_csv(args.out_dir / "prism_component_cancellation_matched_controls.csv", index=False, float_format="%.12f")
    examples.to_csv(args.out_dir / "prism_component_cancellation_control_examples.csv", index=False, float_format="%.12f")
    (args.out_dir / "prism_component_cancellation_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
