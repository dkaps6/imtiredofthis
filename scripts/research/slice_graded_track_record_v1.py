#!/usr/bin/env python3
"""Slice an already-graded board track record to find profitable bet categories.

This answers one question: on bets the model actually made and that have
since been graded against real outcomes, was any *category* of bet
profitable, or were the losses spread evenly?

Every slice is reported with its sample size and an exact binomial p-value
against the real break-even win rate implied by the prices actually paid,
then Benjamini-Hochberg FDR correction is applied across every slice tested.
Slicing 438 bets forty ways will always surface some 60% cells by chance;
the correction is what separates those from a real effect. The number of
slices tested is reported so the multiple-comparisons exposure is explicit.

Read-only analysis. Touches no projection, probability, EV, pricing, or
model-selection path.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--detail", type=Path, required=True,
                   help="graded detail CSV from grade_market_track_record_gsis_v1.py --detail-out")
    p.add_argument("--min-n", type=int, default=25,
                   help="minimum decided bets for a slice to be tested (default 25)")
    p.add_argument("--fdr-q", type=float, default=0.10,
                   help="Benjamini-Hochberg false discovery rate (default 0.10)")
    return p.parse_args()


def american_to_decimal_profit(odds: float) -> float:
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def breakeven_rate(profit_per_win: float) -> float:
    """Win rate needed to break even at this average payout: 1/(1+profit)."""
    return 1.0 / (1.0 + profit_per_win)


def summarize(df: pd.DataFrame, label: str, slice_name: str) -> dict:
    n = len(df)
    wins = int((df["bet_result"] == "WIN").sum())
    units = float(df["unit_result"].sum())
    avg_profit = float(np.mean([american_to_decimal_profit(o) for o in df["vegas_odds"]]))
    be = breakeven_rate(avg_profit)
    win_rate = wins / n if n else float("nan")
    # one-sided exact binomial: is this slice's win rate above its own break-even?
    pval = stats.binomtest(wins, n, be, alternative="greater").pvalue if n else float("nan")
    return {
        "slice": slice_name,
        "value": label,
        "n": n,
        "wins": wins,
        "win_rate": win_rate,
        "breakeven_rate": be,
        "units": units,
        "roi": units / n if n else float("nan"),
        "model_closer_rate": float(df["model_closer_than_vegas"].mean()),
        "p_value": pval,
    }


def main() -> int:
    a = parse_args()
    df = pd.read_csv(a.detail, low_memory=False)
    df = df.loc[df["bet_result"].isin(["WIN", "LOSS"])].copy()
    df["vegas_odds"] = pd.to_numeric(df["vegas_odds"], errors="coerce")
    df = df.loc[df["vegas_odds"].notna()]

    overall = summarize(df, "ALL", "overall")
    print("=== OVERALL ===")
    print(f"  n={overall['n']}  wins={overall['wins']}  win_rate={overall['win_rate']:.4f}  "
          f"breakeven={overall['breakeven_rate']:.4f}  units={overall['units']:.2f}  "
          f"roi={overall['roi']:.4f}  model_closer_than_vegas={overall['model_closer_rate']:.4f}")

    results = []
    dims = {
        "market": df["market"],
        "side": df["side"].astype(str).str.upper(),
        "market_x_side": df["market"].astype(str) + " | " + df["side"].astype(str).str.upper(),
    }
    if "edge_pct" in df.columns:
        df["edge_pct"] = pd.to_numeric(df["edge_pct"], errors="coerce")
        df["edge_bin"] = pd.cut(
            df["edge_pct"], [-np.inf, 2, 5, 10, 20, np.inf],
            labels=["0-2", "2-5", "5-10", "10-20", "20+"],
        ).astype(str)
        dims["edge_bin"] = df["edge_bin"]
        dims["market_x_edge"] = df["market"].astype(str) + " | " + df["edge_bin"]
        dims["side_x_edge"] = df["side"].astype(str).str.upper() + " | " + df["edge_bin"]
    else:
        print("NOTE: edge_pct column absent; edge-bucket slices skipped.")
    if "position" in df.columns:
        dims["position"] = df["position"].astype(str)
        dims["position_x_market"] = df["position"].astype(str) + " | " + df["market"].astype(str)

    for dim_name, keys in dims.items():
        for value, grp in df.groupby(keys, dropna=False):
            if len(grp) < a.min_n:
                continue
            results.append(summarize(grp, str(value), dim_name))

    if not results:
        print("\nNo slice met the minimum sample size.")
        return 0

    res = pd.DataFrame(results).sort_values("roi", ascending=False).reset_index(drop=True)

    # Benjamini-Hochberg across every slice tested: sort p ascending, find the
    # largest rank k with p_k <= (k/m)*q, reject every hypothesis up to it.
    m = len(res)
    res = res.sort_values("p_value").reset_index(drop=True)
    res["bh_threshold"] = (res.index + 1) / m * a.fdr_q
    below = np.where(res["p_value"].to_numpy() <= res["bh_threshold"].to_numpy())[0]
    res["survives_fdr"] = False
    if len(below):
        res.loc[: int(below.max()), "survives_fdr"] = True

    res = res.sort_values("roi", ascending=False).reset_index(drop=True)
    print(f"\n=== SLICES TESTED: {m} (min n={a.min_n}, BH-FDR q={a.fdr_q}) ===")
    print(f"{'slice':<18}{'value':<28}{'n':>5}{'win%':>8}{'be%':>8}{'units':>9}{'roi':>8}{'p':>9}  FDR")
    for r in res.itertuples(index=False):
        print(f"{r.slice:<18}{r.value[:27]:<28}{r.n:>5}{r.win_rate*100:>7.1f}%{r.breakeven_rate*100:>7.1f}%"
              f"{r.units:>9.2f}{r.roi*100:>7.1f}%{r.p_value:>9.4f}  {'YES' if r.survives_fdr else '-'}")

    n_survive = int(res["survives_fdr"].sum())
    print(f"\nslices surviving FDR correction: {n_survive} of {m}")
    if n_survive == 0:
        print("DISPOSITION: NO_SLICE_SURVIVES_MULTIPLE_COMPARISONS_CORRECTION")
        print("Every apparently-profitable category is within chance for this sample size.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
