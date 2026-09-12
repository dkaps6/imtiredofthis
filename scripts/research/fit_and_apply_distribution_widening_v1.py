#!/usr/bin/env python3
"""Test whether widening the reconstructed empirical MC distribution toward
realized forecast dispersion improves calibration/ROI, using a genuine
train/holdout split -- never fitting and grading the same season.

Checkpoints 12/20 (Issue #535) confirmed, per market, that the empirical MC
distribution reconstructed in PR #546 is under-dispersed relative to realized
forecast residuals (45-70% of realized SD depending on market). This script
tests the obvious next question: does correcting that specific gap -- and
nothing else -- move calibration/ROI further?

Method, single variable only:
1. FIT phase (one season, no sportsbook data touched at all): for each row,
   rescale its raw historical MC array to its own frozen projection mean
   (exactly production semantics, reused from grade_empirical_fair_prob_v1.py),
   compute that row's within-row simulated SD, and compare the market-level
   mean of those SDs against the market-level realized residual SD
   (std(actual - proj) across rows). The ratio is the widening factor k,
   frozen per market.
2. APPLY phase (the OTHER season only, k frozen from step 1, blind): widen
   each row's mean-aligned rescaled array by k around its own (unchanged)
   mean -- this cannot alter model_mae, since the point projection is never
   touched, only the spread around it. Recompute p_over empirically on the
   widened array. Grade against real Vegas lines with the exact same frozen
   PLAY/LEAN/STRONG gate used everywhere else in this research thread.
3. Both directions run (fit 2024 -> test 2025, and fit 2025 -> test 2024) so
   every reported number is a genuine out-of-sample result, never fit-on-what-
   you-graded.

No mean/weight/threshold change. Not combined with PR #545's heldout weights
(one variable at a time, per the standing agreement in Issue #535). If
widening doesn't help, that is the result -- no post-hoc factor search.

Research only. No production, model, weight, or threshold change.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.grade_full_stack_vegas_benchmark_v1 import (
    ev_roi,
    implied_prob,
    no_vig,
    signal,
)
from scripts.backtest.grade_historical_market_vegas_benchmark_v1 import select_one_book_row
from scripts.operations.grade_market_track_record_v1 import american_profit, num, outcome_side
from scripts.research.grade_empirical_fair_prob_v1 import (
    KEYS,
    _canon_keys,
    _load_metadata,
    empirical_over_probability,
    rescale_outcomes,
)

MARKETS = ["rush_yards", "rec_yards", "rush_rec_yards", "receptions", "pass_yards"]


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        raise RuntimeError(f"missing {label}: {path}")
    out = pd.read_csv(path, low_memory=False)
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _row_arrays(meta: pd.DataFrame, distribution_dir: Path) -> dict:
    cache: dict[str, object] = {}
    out = {}
    for _, row in meta.iterrows():
        fn = str(row["npz_file"])
        if Path(fn).name != fn:
            raise RuntimeError(f"invalid shard path: {fn}")
        if fn not in cache:
            path = distribution_dir / fn
            if not path.exists():
                raise RuntimeError(f"missing shard: {path}")
            cache[fn] = np.load(path, allow_pickle=False)
        out[(row["season"], row["week"], row["team"], row["opponent"], row["player_clean_key"], row["market"])] = (
            np.asarray(cache[fn][str(row["array_key"])], dtype=float)
        )
    return out


def fit_widening_factors(proj: pd.DataFrame, meta: pd.DataFrame, distribution_dir: Path, fit_season: int) -> dict[str, float]:
    fit_meta = _canon_keys(meta.loc[meta.season.eq(fit_season)].copy())
    fit_proj = proj.loc[proj.season.eq(fit_season)].copy()
    merged = fit_proj.merge(fit_meta[KEYS + ["array_key", "npz_file"]], on=KEYS, how="inner", validate="one_to_one")
    if merged.empty:
        raise RuntimeError(f"no rows to fit widening factors for season {fit_season}")

    arrays = _row_arrays(fit_meta, distribution_dir)
    rows = []
    for _, row in merged.iterrows():
        key = (row["season"], row["week"], row["team"], row["opponent"], row["player_clean_key"], row["market"])
        arr = arrays.get(key)
        if arr is None:
            continue
        rescaled = rescale_outcomes(arr, float(row["proj"]))
        row_sd = float(np.std(rescaled, ddof=1)) if len(rescaled) > 1 else np.nan
        rows.append({"market": row["market"], "row_sd": row_sd, "residual": float(row["actual"]) - float(row["proj"])})
    fit_df = pd.DataFrame(rows)

    factors = {}
    for market, g in fit_df.groupby("market"):
        empirical_mc_sd = float(g["row_sd"].mean())
        realized_residual_sd = float(g["residual"].std(ddof=1))
        k = realized_residual_sd / empirical_mc_sd if empirical_mc_sd > 0 else 1.0
        factors[market] = k
        print(f"[widen_fit] season={fit_season} market={market} n={len(g)} empirical_mc_sd={empirical_mc_sd:.4f} "
              f"realized_residual_sd={realized_residual_sd:.4f} k={k:.4f}")
    return factors


def _grade_side(detail: pd.DataFrame) -> pd.DataFrame:
    detail["over_implied"] = detail.over_odds.map(implied_prob)
    detail["under_implied"] = detail.under_odds.map(implied_prob)
    detail["over_novig"] = [no_vig(a, b) for a, b in zip(detail.over_implied, detail.under_implied)]
    detail["under_novig"] = [no_vig(a, b) for a, b in zip(detail.under_implied, detail.over_implied)]
    detail["ev_over"] = [ev_roi(p, o) for p, o in zip(detail.p_over, detail.over_odds)]
    detail["ev_under"] = [ev_roi(p, o) for p, o in zip(detail.p_under, detail.under_odds)]
    best_over = detail.ev_under.isna() | (detail.ev_over.fillna(-np.inf) >= detail.ev_under.fillna(-np.inf))
    detail["side"] = np.where(best_over, "OVER", "UNDER")
    detail["best_ev"] = np.where(best_over, detail.ev_over, detail.ev_under)
    detail["best_model_p"] = np.where(best_over, detail.p_over, detail.p_under)
    detail["best_market_p"] = np.where(best_over, detail.over_novig, detail.under_novig)
    detail["prob_edge"] = detail.best_model_p - detail.best_market_p
    detail["chosen_odds"] = np.where(best_over, detail.over_odds, detail.under_odds)
    detail["signal"] = [signal(e, q) for e, q in zip(detail.best_ev, detail.prob_edge)]
    detail["actual_side"] = [outcome_side(a, l) for a, l in zip(detail.actual, detail.line)]
    detail["bet_result"] = np.select(
        [detail.actual_side.eq("PUSH"), detail.side.eq(detail.actual_side)], ["PUSH", "WIN"], default="LOSS",
    )
    detail["unit_result"] = np.where(
        detail.bet_result.eq("WIN"), [american_profit(o) for o in detail.chosen_odds],
        np.where(detail.bet_result.eq("LOSS"), -1.0, 0.0),
    )
    detail["model_error"] = num(detail.proj) - num(detail.actual)
    detail["vegas_error"] = num(detail.line) - num(detail.actual)
    return detail


def _summarize(z: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tiers = {
        "ALL_NO_FILTER": z,
        "LEAN_OR_STRONG": z.loc[z.signal.isin(["LEAN_EDGE", "STRONG_EDGE"])],
        "STRONG_ONLY_PLAY_TIER": z.loc[z.signal.eq("STRONG_EDGE")],
    }
    for market in list(z.market.unique()) + ["ALL_MARKETS"]:
        for tier_name, tier_df in tiers.items():
            g = tier_df if market == "ALL_MARKETS" else tier_df.loc[tier_df.market.eq(market)]
            decided = g.loc[g.bet_result.isin(["WIN", "LOSS"])]
            y = (num(g.actual) > num(g.line)).astype(float)
            p = np.clip(num(g.p_over), 1e-6, 1 - 1e-6)
            brier = float(np.mean((p - y) ** 2)) if len(g) else np.nan
            log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(g) else np.nan
            rows.append({
                "market": market, "tier": tier_name, "matched_rows": int(len(g)),
                "decided_bets": int(len(decided)),
                "win_rate": float(decided.bet_result.eq("WIN").mean()) if len(decided) else np.nan,
                "roi_per_unit": float(decided.unit_result.mean()) if len(decided) else np.nan,
                "model_mae": float(g.model_error.abs().mean()) if len(g) else np.nan,
                "vegas_mae": float(g.vegas_error.abs().mean()) if len(g) else np.nan,
                "brier": brier, "log_loss": log_loss,
            })
    return pd.DataFrame(rows)


def apply_and_grade(proj: pd.DataFrame, meta: pd.DataFrame, props: pd.DataFrame, distribution_dir: Path,
                     test_season: int, widening_factors: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_meta = _canon_keys(meta.loc[meta.season.eq(test_season)].copy())
    test_proj = proj.loc[proj.season.eq(test_season)].copy()
    merged = test_proj.merge(test_meta[KEYS + ["array_key", "npz_file", "draws"]], on=KEYS, how="inner", validate="one_to_one")
    if len(merged) != len(test_proj):
        raise RuntimeError(f"distribution join dropped rows: proj={len(test_proj)} joined={len(merged)}")

    selected = select_one_book_row(props.loc[props.season.eq(test_season)] if "season" in props.columns else props)
    join_cols = ["game_id", "player_clean_key", "market"]
    keep = join_cols + ["book", "line", "over_odds", "under_odds", "player"]
    matched = merged.merge(selected[keep], on=join_cols, how="inner")
    if matched.empty:
        raise RuntimeError("no matched rows after Vegas join")
    matched["line"] = num(matched.line)

    arrays = _row_arrays(test_meta, distribution_dir)
    p_over_base, p_over_widened, mae_check = [], [], []
    for _, row in matched.iterrows():
        key = (row["season"], row["week"], row["team"], row["opponent"], row["player_clean_key"], row["market"])
        arr = arrays[key]
        rescaled = rescale_outcomes(arr, float(row["proj"]))
        base_mean = float(np.mean(rescaled))
        k = widening_factors.get(row["market"], 1.0)
        widened = base_mean + (rescaled - base_mean) * k
        p_over_base.append(empirical_over_probability(rescaled, float(row["line"])))
        p_over_widened.append(empirical_over_probability(widened, float(row["line"])))
        mae_check.append(abs(float(np.mean(widened)) - base_mean))

    if max(mae_check) > 1e-8:
        raise RuntimeError(f"widening altered the mean: max_abs_mean_shift={max(mae_check):.3g}")

    out_summaries = []
    for label, p_over_col in [("empirical_unwidened", p_over_base), ("empirical_widened", p_over_widened)]:
        z = matched.copy()
        z["p_over"] = p_over_col
        z["p_under"] = 1.0 - z["p_over"]
        z["actual"] = num(z.actual)
        z = _grade_side(z)
        summary = _summarize(z)
        summary["variant"] = label
        out_summaries.append(summary)

    return pd.concat(out_summaries, ignore_index=True), merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projection-file", type=Path, required=True, help="clean full_stack_projection_trace (both seasons)")
    ap.add_argument("--distribution-dir", type=Path, required=True, help="dir with *_metadata.csv and *.npz shards, both seasons")
    ap.add_argument("--props", type=Path, required=True, help="clean historical market props (both seasons)")
    ap.add_argument("--fit-season", type=int, required=True)
    ap.add_argument("--test-season", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    proj = _read(a.projection_file, "projection trace")
    if "ensemble_proj" in proj.columns and "proj" not in proj.columns:
        proj["proj"] = proj["ensemble_proj"]
    meta = pd.concat(
        [pd.read_csv(p) for p in sorted(a.distribution_dir.glob("*_metadata.csv"))], ignore_index=True,
    )
    meta = _canon_keys(meta)
    props = _read(a.props, "historical props")

    a.out_dir.mkdir(parents=True, exist_ok=True)

    widening_factors = fit_widening_factors(proj, meta, a.distribution_dir, a.fit_season)
    pd.DataFrame(
        [{"market": m, "k": k, "fit_season": a.fit_season} for m, k in widening_factors.items()]
    ).to_csv(a.out_dir / f"widening_factors_fit{a.fit_season}.csv", index=False)

    summary, _ = apply_and_grade(proj, meta, props, a.distribution_dir, a.test_season, widening_factors)
    summary.to_csv(a.out_dir / f"widening_test{a.test_season}_fit{a.fit_season}_summary.csv", index=False)

    print(f"\n=== fit={a.fit_season} test={a.test_season} STRONG tier ===")
    strong = summary.loc[summary.tier.eq("STRONG_ONLY_PLAY_TIER") & summary.market.eq("ALL_MARKETS")]
    print(strong.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
