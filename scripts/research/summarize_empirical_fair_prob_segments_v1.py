#!/usr/bin/env python3
"""Emit frozen season/market/side and chosen-probability diagnostics for V1."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.grade_full_stack_vegas_benchmark_v1 import grade as legacy_grade
from scripts.operations.grade_market_track_record_v1 import num


def _segment_diagnostics(detail: pd.DataFrame, translator: str) -> pd.DataFrame:
    rows: list[dict] = []
    season_values = sorted(pd.to_numeric(detail["season"], errors="coerce").dropna().astype(int).unique())
    season_scopes = [("ALL_SEASONS", detail)] + [
        (str(season), detail.loc[pd.to_numeric(detail["season"], errors="coerce").eq(season)])
        for season in season_values
    ]
    for season_label, season_df in season_scopes:
        markets = sorted(season_df["market"].dropna().astype(str).unique())
        market_scopes = [("ALL_MARKETS", season_df)] + [
            (market, season_df.loc[season_df["market"].astype(str).eq(market)])
            for market in markets
        ]
        for market_label, market_df in market_scopes:
            for side_label in ["ALL_SIDES", "OVER", "UNDER"]:
                g = market_df if side_label == "ALL_SIDES" else market_df.loc[market_df["side"].eq(side_label)]
                decided = g.loc[
                    g["bet_result"].isin(["WIN", "LOSS"]) & num(g["chosen_odds"]).notna()
                ]
                rows.append(
                    {
                        "translator": translator,
                        "season": season_label,
                        "market": market_label,
                        "side": side_label,
                        "matched_rows": int(len(g)),
                        "strong_rows": int(g["signal"].eq("STRONG_EDGE").sum()),
                        "strong_coverage": float(g["signal"].eq("STRONG_EDGE").mean()) if len(g) else np.nan,
                        "decided_bets": int(len(decided)),
                        "win_rate": float(decided["bet_result"].eq("WIN").mean()) if len(decided) else np.nan,
                        "units": float(decided["unit_result"].sum()) if len(decided) else np.nan,
                        "roi_per_unit": float(decided["unit_result"].mean()) if len(decided) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _bet_probability_bins(detail: pd.DataFrame, translator: str) -> pd.DataFrame:
    x = detail.loc[
        detail["bet_result"].isin(["WIN", "LOSS"])
        & num(detail["chosen_odds"]).notna()
        & num(detail["best_model_p"]).notna()
    ].copy()
    x["predicted_win_prob"] = num(x["best_model_p"]).clip(0.0, 1.0)
    x["prob_bin"] = pd.cut(
        x["predicted_win_prob"],
        bins=np.linspace(0.0, 1.0, 11),
        include_lowest=True,
        duplicates="drop",
    )
    rows: list[dict] = []
    for market in list(x["market"].unique()) + ["ALL_MARKETS"]:
        g = x if market == "ALL_MARKETS" else x.loc[x["market"].eq(market)]
        for bucket, b in g.groupby("prob_bin", observed=True):
            rows.append(
                {
                    "translator": translator,
                    "market": market,
                    "prob_bin": str(bucket),
                    "rows": int(len(b)),
                    "mean_predicted_win_prob": float(b["predicted_win_prob"].mean()),
                    "realized_win_rate": float(b["bet_result"].eq("WIN").mean()),
                    "mean_best_ev": float(num(b["best_ev"]).mean()),
                    "roi_per_unit": float(b["unit_result"].mean()),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projection-file", action="append", required=True)
    ap.add_argument("--proj-col", default="ensemble_proj")
    ap.add_argument("--props", type=Path, required=True)
    ap.add_argument("--empirical-detail", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    proj = pd.concat([pd.read_csv(Path(p)) for p in a.projection_file], ignore_index=True)
    props = pd.read_csv(a.props)
    legacy_detail, _ = legacy_grade(proj, props, proj_col=a.proj_col)
    empirical_detail = pd.read_csv(a.empirical_detail)

    if len(legacy_detail) != len(empirical_detail):
        raise RuntimeError(
            f"same-row A/B contract violated for segment audit: "
            f"legacy={len(legacy_detail)} empirical={len(empirical_detail)}"
        )

    segments = pd.concat(
        [
            _segment_diagnostics(legacy_detail, "LEGACY_COMPONENT_SD_NORMAL"),
            _segment_diagnostics(empirical_detail, "EMPIRICAL_MC_RESCALED_V1"),
        ],
        ignore_index=True,
    )
    bins = pd.concat(
        [
            _bet_probability_bins(legacy_detail, "LEGACY_COMPONENT_SD_NORMAL"),
            _bet_probability_bins(empirical_detail, "EMPIRICAL_MC_RESCALED_V1"),
        ],
        ignore_index=True,
    )
    a.out_dir.mkdir(parents=True, exist_ok=True)
    segments.to_csv(a.out_dir / "season_market_side_diagnostics.csv", index=False)
    bins.to_csv(a.out_dir / "bet_probability_bins.csv", index=False)
    print(segments.to_string(index=False))
    print(bins.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
