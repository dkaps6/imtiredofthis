#!/usr/bin/env python3
"""Slice an already-graded board track record to find profitable bet categories.

This answers one question: on bets the model actually made and that have
since been graded against real outcomes, was any *category* of bet
profitable, or were the losses spread evenly?

Every slice is reported with its sample size and a game-cluster-aware
one-sided score-test p-value. Each row is centered by the heterogeneous
per-bet break-even probability implied by its actual captured price, then
the centered residuals are summed within NFL games and inference is performed
across independent game clusters. Benjamini-Hochberg FDR correction is
applied only to slices with a valid cluster-aware p-value.
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


def american_breakeven_probability(odds: float) -> float:
    """Per-bet zero-EV win probability implied by the actual American price."""
    o = float(odds)
    if not np.isfinite(o) or o == 0:
        return float("nan")
    return (-o) / (-o + 100.0) if o < 0 else 100.0 / (o + 100.0)


def poisson_binomial_tail(k: int, probabilities) -> float:
    """Exact P(X >= k) for independent Bernoulli trials with unequal p_i."""
    p = np.asarray(list(probabilities), dtype=float)
    if len(p) == 0:
        return float("nan")
    if np.any(~np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        return float("nan")
    if k <= 0:
        return 1.0
    if k > len(p):
        return 0.0
    pmf = np.array([1.0])
    for pi in p:
        pmf = np.convolve(pmf, np.array([1.0 - pi, pi]))
    return float(np.clip(pmf[k:].sum(), 0.0, 1.0))


MIN_CLUSTERS_FOR_INFERENCE = 8


def cluster_score_pvalue(
    df: pd.DataFrame,
    null_p,
    *,
    min_clusters: int = MIN_CLUSTERS_FOR_INFERENCE,
) -> tuple[float, int]:
    """One-sided cluster-robust score test over independent NFL games.

    For bet i, score_i = observed_win_i - price_implied_break_even_i.
    Scores are summed inside each game, so mechanically related props from the
    same game contribute one independent cluster. A one-sample t statistic is
    then formed over the game-level score sums. This is deliberately not
    called exact: it is a small-sample cluster-aware approximation.
    """
    required = ["season", "week", "event_id"]
    if any(c not in df.columns for c in required):
        return float("nan"), 0

    cluster_frame = df[required].copy()
    for c in required:
        if cluster_frame[c].isna().any():
            return float("nan"), 0
        if cluster_frame[c].astype(str).str.strip().eq("").any():
            return float("nan"), 0

    p = np.asarray(list(null_p), dtype=float)
    if len(p) != len(df) or np.any(~np.isfinite(p)):
        return float("nan"), 0

    wins = df["bet_result"].eq("WIN").astype(float).to_numpy()
    scored = cluster_frame.copy()
    scored["_score"] = wins - p
    cluster_scores = (
        scored.groupby(required, dropna=False)["_score"].sum().to_numpy(dtype=float)
    )
    n_clusters = int(len(cluster_scores))
    if n_clusters < int(min_clusters):
        return float("nan"), n_clusters

    sd = float(np.std(cluster_scores, ddof=1))
    if not np.isfinite(sd) or sd <= 0.0:
        return float("nan"), n_clusters

    mean_score = float(np.mean(cluster_scores))
    t_stat = mean_score / (sd / np.sqrt(n_clusters))
    p_value = float(stats.t.sf(t_stat, df=n_clusters - 1))
    return p_value, n_clusters


def summarize(df: pd.DataFrame, label: str, slice_name: str) -> dict:
    n = len(df)
    wins = int((df["bet_result"] == "WIN").sum())
    units = float(df["unit_result"].sum())
    null_p = np.array(
        [american_breakeven_probability(o) for o in df["vegas_odds"]], dtype=float
    )
    be = float(null_p.mean()) if n else float("nan")
    win_rate = wins / n if n else float("nan")
    # Keep the independent-row Poisson-binomial tail as a descriptive
    # reference only. It is NOT used for discovery/FDR because props within
    # the same game are mechanically dependent.
    independent_p = poisson_binomial_tail(wins, null_p) if n else float("nan")
    pval, n_clusters = cluster_score_pvalue(df, null_p)
    return {
        "slice": slice_name,
        "value": label,
        "n": n,
        "clusters": n_clusters,
        "wins": wins,
        "win_rate": win_rate,
        "breakeven_rate": be,
        "units": units,
        "roi": units / n if n else float("nan"),
        "model_closer_rate": float(df["model_closer_than_vegas"].mean()),
        "independent_p_value": independent_p,
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
    print(f"  n={overall['n']}  clusters={overall['clusters']}  wins={overall['wins']}  "
          f"win_rate={overall['win_rate']:.4f}  breakeven={overall['breakeven_rate']:.4f}  "
          f"units={overall['units']:.2f}  roi={overall['roi']:.4f}  "
          f"cluster_p={overall['p_value']:.4f}  "
          f"model_closer_than_vegas={overall['model_closer_rate']:.4f}")

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

    # Benjamini-Hochberg only across slices with a valid game-cluster-aware
    # p-value. Slices with too few independent games are reported but cannot
    # be declared significant.
    res["bh_threshold"] = np.nan
    res["survives_fdr"] = False
    valid = res["p_value"].notna()
    ranked = res.loc[valid].sort_values("p_value").copy()
    m = len(ranked)
    if m:
        ranked["bh_threshold"] = (np.arange(m) + 1) / m * a.fdr_q
        below = np.where(
            ranked["p_value"].to_numpy() <= ranked["bh_threshold"].to_numpy()
        )[0]
        if len(below):
            ranked.iloc[: int(below.max()) + 1, ranked.columns.get_loc("survives_fdr")] = True
        res.loc[ranked.index, "bh_threshold"] = ranked["bh_threshold"]
        res.loc[ranked.index, "survives_fdr"] = ranked["survives_fdr"]

    res = res.sort_values("roi", ascending=False).reset_index(drop=True)
    print(
        f"\n=== SLICES REPORTED: {len(res)}; CLUSTER-TESTED: {m} "
        f"(min n={a.min_n}, min games={MIN_CLUSTERS_FOR_INFERENCE}, BH-FDR q={a.fdr_q}) ==="
    )
    print(
        f"{'slice':<18}{'value':<28}{'n':>5}{'games':>7}{'win%':>8}{'be%':>8}"
        f"{'units':>9}{'roi':>8}{'cl_p':>9}  FDR"
    )
    for r in res.itertuples(index=False):
        p_txt = f"{r.p_value:.4f}" if np.isfinite(r.p_value) else "NA"
        print(
            f"{r.slice:<18}{r.value[:27]:<28}{r.n:>5}{r.clusters:>7}"
            f"{r.win_rate*100:>7.1f}%{r.breakeven_rate*100:>7.1f}%"
            f"{r.units:>9.2f}{r.roi*100:>7.1f}%{p_txt:>9}  "
            f"{'YES' if r.survives_fdr else '-'}"
        )

    n_survive = int(res["survives_fdr"].sum())
    print(f"\nslices surviving cluster-aware FDR correction: {n_survive} of {m} tested")
    if n_survive == 0:
        print("DISPOSITION: NO_SLICE_SURVIVES_MULTIPLE_COMPARISONS_CORRECTION")
        print("No category clears the game-cluster-aware multiple-comparisons gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
