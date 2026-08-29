#!/usr/bin/env python3
"""Execution-safe fixes for the frozen Migration 66 frontier audit.

This wrapper changes no M66 model, feature, candidate, threshold, gate, or
historical boundary. It only:
1. enforces that all nine predeclared candidate projections are present; and
2. avoids the pandas DataFrame.mode attribute/column collision in the final
   interpretation step.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.backtest.audit_qb_research_frontier as m


_ORIG_PAIRED_LIBRARY = m.paired_library


def paired_library_fixed(g: pd.DataFrame):
    missing = {
        name: col
        for name, col in m.CANDIDATE_LIBRARY.items()
        if col not in g.columns
    }
    if missing:
        raise RuntimeError(
            "M66 frozen nine-candidate library is incomplete; missing "
            f"{missing}"
        )
    paired, present = _ORIG_PAIRED_LIBRARY(g)
    if set(present) != set(m.CANDIDATE_LIBRARY):
        raise RuntimeError(
            "M66 paired library changed unexpectedly after full-library "
            f"validation: {sorted(present)}"
        )
    return paired, present


def interpretation_fixed(
    library_metrics_df: pd.DataFrame,
    pairs: pd.DataFrame,
    oracle: pd.DataFrame,
    ensembles: pd.DataFrame,
    residuals: pd.DataFrame,
    meanmedian: pd.DataFrame,
) -> pd.DataFrame:
    raw = library_metrics_df[
        library_metrics_df["season"].eq("2025")
        & library_metrics_df["candidate"].eq("raw_attempts")
    ].iloc[0]
    median_abs_resid_corr = (
        float(pairs["abs_residual_corr"].median()) if len(pairs) else np.nan
    )
    oracle_combined = oracle[oracle["season"].eq("combined")].iloc[0]
    models_materially_diverse = bool(
        np.isfinite(median_abs_resid_corr)
        and median_abs_resid_corr <= 0.85
    )
    oracle_headroom = float(oracle_combined["oracle_mae_headroom_vs_raw"])

    eligible_ensembles = []
    for _, r in ensembles.iterrows():
        if str(r["season"]) not in {"2025", "2025_test"}:
            continue
        if (
            float(raw["mae"] - r["mae"]) >= 1.0
            and float(r["corr"] - raw["corr"]) >= 0.03
            and float(r["rmse"]) <= float(raw["rmse"]) + 1e-12
            and int(r["miss_100plus"])
            <= int(np.floor(float(raw["miss_100plus"]) * 0.95))
        ):
            eligible_ensembles.append(str(r["ensemble"]))

    residual_signal_models = []
    for _, r in residuals.iterrows():
        if (
            float(r["residual_corr"]) >= 0.20
            and float(r["mae_gain_vs_raw"]) >= 1.0
            and float(r["corr_gain_vs_raw"]) >= 0.03
        ):
            residual_signal_models.append(str(r["model"]))

    mean_median_signal = False
    raw_median_gain = np.nan
    if not meanmedian.empty:
        mm = meanmedian[
            meanmedian["season"].eq("2025")
            & meanmedian["mode"].eq("raw")
        ]
        if {"mean", "median"}.issubset(set(mm["point_stat"])):
            a = mm[mm["point_stat"].eq("mean")].iloc[0]
            b = mm[mm["point_stat"].eq("median")].iloc[0]
            raw_median_gain = float(a["mae"] - b["mae"])
            mean_median_signal = raw_median_gain >= 0.50

    existing_combination_signal = bool(
        eligible_ensembles or residual_signal_models or mean_median_signal
    )
    if existing_combination_signal:
        verdict = "existing_information_combination_followup"
    elif (not models_materially_diverse) and not residual_signal_models:
        verdict = "current_library_redundant_seek_new_information"
    else:
        verdict = "mixed_frontier_seek_new_information_and_selective_combination"

    return pd.DataFrame([{
        "paired_candidates": int(
            library_metrics_df[
                library_metrics_df["season"].eq("combined")
            ]["candidate"].nunique()
        ),
        "median_abs_pairwise_residual_corr": median_abs_resid_corr,
        "models_materially_diverse_le_0_85": models_materially_diverse,
        "oracle_mae_headroom_vs_raw": oracle_headroom,
        "oracle_headroom_ge_8yd": bool(oracle_headroom >= 8.0),
        "ensemble_followup_eligible": bool(eligible_ensembles),
        "eligible_ensembles": "|".join(eligible_ensembles),
        "residual_information_signal": bool(residual_signal_models),
        "residual_signal_models": "|".join(residual_signal_models),
        "raw_median_mae_gain_2025": raw_median_gain,
        "mean_median_signal_ge_0_50": mean_median_signal,
        "m66_interpretation": verdict,
    }])


m.paired_library = paired_library_fixed
m.interpretation = interpretation_fixed


if __name__ == "__main__":
    raise SystemExit(m.main())
