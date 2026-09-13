#!/usr/bin/env python3
"""Grade M89/M90 QB pass_yards synthesis through the empirical MC probability
translator (PR #546) instead of the legacy Normal(mean, component_sd)
translator used in the only existing M89/M90 Vegas comparison (PR #544).

M89/M90 is a pure post-simulation mean correction (Ridge on base_proj; never
touches mc_proj/ml_proj/state_proj or re-enters simulation -- confirmed by
reading run_m89_pregame_synthesis.py directly, and by production's own order
in run_pricing_with_full_roster_universe_v3_core.py, which places M89/M90
after canonical joint MC and after the QB C2 distribution selector). So this
reuses the same mean-rescale semantics already validated for the base
ensemble, distribution widening, and the WR/TE production-order replay --
no new simulation plumbing.

Both base_proj and football_synthesis are graded through the empirical
translator on the identical rows, isolating exactly the translator variable
on top of the already-known M89 mean effect. market_assisted is reported for
continuity with PR #544's three-variant table only -- it is not eligible as
a football-only promotion candidate per M90_QB_SYNTHESIS_CONFIRMATION_PROMOTION.md's
existing frozen rule (it uses market-derived features).

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
    _canon_keys,
    empirical_over_probability,
    rescale_outcomes,
)

KEYS = ["season", "week", "team", "opponent", "player_clean_key", "market"]
VARIANTS = ["base_proj", "football_synthesis", "market_assisted"]


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
    for tier_name, tier_df in tiers.items():
        g = tier_df
        decided = g.loc[g.bet_result.isin(["WIN", "LOSS"])]
        y = (num(g.actual) > num(g.line)).astype(float)
        p = np.clip(num(g.p_over), 1e-6, 1 - 1e-6)
        # Brier/log-loss on decided (non-PUSH) rows only, matching
        # grade_empirical_fair_prob_v1.py's own diagnostics -- a PUSH is not
        # a "not over" outcome.
        yd = (num(decided.actual) > num(decided.line)).astype(float)
        pd_ = np.clip(num(decided.p_over), 1e-6, 1 - 1e-6)
        brier = float(np.mean((pd_ - yd) ** 2)) if len(decided) else np.nan
        log_loss = float(-np.mean(yd * np.log(pd_) + (1 - yd) * np.log(1 - pd_))) if len(decided) else np.nan
        rows.append({
            "tier": tier_name, "matched_rows": int(len(g)),
            "decided_bets": int(len(decided)),
            "win_rate": float(decided.bet_result.eq("WIN").mean()) if len(decided) else np.nan,
            "roi_per_unit": float(decided.unit_result.mean()) if len(decided) else np.nan,
            "model_mae": float(g.model_error.abs().mean()) if len(g) else np.nan,
            "vegas_mae": float(g.vegas_error.abs().mean()) if len(g) else np.nan,
            "brier": brier, "log_loss": log_loss,
        })
    return pd.DataFrame(rows)


def grade_variant(matched: pd.DataFrame, arrays: dict, proj_col: str) -> pd.DataFrame:
    p_over = []
    for _, row in matched.iterrows():
        key = (row["season"], row["week"], row["team"], row["opponent"], row["player_clean_key"], row["market"])
        arr = arrays[key]
        proj = float(row[proj_col])
        rescaled = rescale_outcomes(arr, proj)
        aligned_delta = abs(float(np.mean(rescaled)) - proj)
        if not np.isfinite(aligned_delta) or aligned_delta > 1e-8:
            raise RuntimeError(
                f"rescale_outcomes failed to align mean to {proj_col} (likely a zero-mean MC array) "
                f"season={row['season']} week={row['week']} team={row['team']} "
                f"player={row['player_clean_key']}: mean(rescaled)={float(np.mean(rescaled)):.4f} proj={proj:.4f}"
            )
        p_over.append(empirical_over_probability(rescaled, float(row["line"])))

    z = matched.copy()
    z["proj"] = num(z[proj_col])
    z["p_over"] = p_over
    z["p_under"] = 1.0 - z["p_over"]
    z["actual"] = num(z.actual)
    z = _grade_side(z)
    summary = _summarize(z)
    summary["variant"] = proj_col
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", type=Path, required=True, help="identity-attached M89/M90 synthesis trace")
    ap.add_argument("--distribution-dir", type=Path, required=True, help="dir with *_metadata.csv and *.npz shards")
    ap.add_argument("--props", type=Path, required=True, help="historical QB pass_yards props archive")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    trace = _read(a.trace, "identity-attached M89/M90 trace")
    if "market" not in trace.columns:
        trace["market"] = "pass_yards"
    meta = pd.concat(
        [pd.read_csv(p) for p in sorted(a.distribution_dir.glob("*_metadata.csv"))], ignore_index=True,
    )
    meta = _canon_keys(meta)
    meta = meta.loc[meta["market"].eq("pass_yards")].copy()
    trace = _canon_keys(trace)

    merged = trace.merge(meta[KEYS + ["array_key", "npz_file", "draws"]], on=KEYS, how="inner", validate="one_to_one")
    if len(merged) != len(trace):
        raise RuntimeError(f"distribution join dropped rows: trace={len(trace)} joined={len(merged)}")

    props = _read(a.props, "historical QB pass_yards props")
    props["market"] = "pass_yards"
    selected = select_one_book_row(props)
    join_cols = ["game_id", "player_clean_key", "market"]
    keep = join_cols + ["book", "line", "over_odds", "under_odds", "player"]
    matched = merged.merge(selected[keep], on=join_cols, how="inner")
    if matched.empty:
        raise RuntimeError("no matched rows after Vegas join")
    matched["line"] = num(matched.line)

    arrays = _row_arrays(meta, a.distribution_dir)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for proj_col in VARIANTS:
        if proj_col not in matched.columns:
            continue
        summary = grade_variant(matched, arrays, proj_col)
        all_summaries.append(summary)

    out = pd.concat(all_summaries, ignore_index=True)
    out.to_csv(a.out_dir / "qb_m89_empirical_fair_prob_summary.csv", index=False)

    print("=== QB M89/M90 EMPIRICAL FAIR-PROBABILITY GRADE, STRONG tier ===")
    strong = out.loc[out.tier.eq("STRONG_ONLY_PLAY_TIER")]
    print(strong.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
