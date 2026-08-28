#!/usr/bin/env python3
"""Migration 60C: grade independent QB projections against pregame Vegas props.

The sportsbook line is a benchmark only. It is never used as a feature or as an
input to the football projection. All candidate comparisons are performed on
exactly the same matched player-game rows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BOOK_ORDER = {"draftkings": 0, "fanduel": 1}
CANDIDATES = ["current", "joint_cap_shrink", "attempts_raw_only", "ypa_raw_only", "both_raw"]


def num(v):
    return pd.to_numeric(v, errors="coerce")


def american_profit(odds) -> float:
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if not np.isfinite(o) or o == 0:
        return np.nan
    return o / 100.0 if o > 0 else 100.0 / abs(o)


def implied_prob(odds) -> float:
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if not np.isfinite(o) or o == 0:
        return np.nan
    return 100.0 / (o + 100.0) if o > 0 else abs(o) / (abs(o) + 100.0)


def corr(a, b):
    z = pd.DataFrame({"a": num(a), "b": num(b)}).dropna()
    return float(z.a.corr(z.b)) if len(z) > 2 and z.a.nunique() > 1 and z.b.nunique() > 1 else np.nan


def select_props(props: pd.DataFrame) -> pd.DataFrame:
    if props.empty:
        return props.copy()
    p = props.copy()
    p["book"] = p.book.astype(str).str.lower().str.strip()
    p["book_rank"] = p.book.map(BOOK_ORDER).fillna(99)
    p["price_count"] = num(p.over_odds).notna().astype(int) + num(p.under_odds).notna().astype(int)
    # At the fixed snapshot, prefer DraftKings, then FanDuel, then any other book.
    # If duplicate rows exist, prefer the row with both side prices available.
    p = p.sort_values(
        ["game_id", "player_clean_key", "book_rank", "price_count"],
        ascending=[True, True, True, False],
    )
    return p.drop_duplicates(["game_id", "player_clean_key"], keep="first")


def outcome_side(actual: float, line: float) -> str:
    if actual > line:
        return "OVER"
    if actual < line:
        return "UNDER"
    return "PUSH"


def model_side(proj: float, line: float) -> str:
    if proj > line:
        return "OVER"
    if proj < line:
        return "UNDER"
    return "NO_BET"


def edge_bucket(v: float) -> str:
    if not np.isfinite(v): return "missing"
    if v < 5: return "0-5"
    if v < 10: return "5-10"
    if v < 15: return "10-15"
    if v < 20: return "15-20"
    if v < 30: return "20-30"
    return "30+"


def grade_candidate(frame: pd.DataFrame, candidate: str, *, season_label: str) -> dict:
    col = f"mc_proj_{candidate}"
    z = frame[["actual", "line", "over_odds", "under_odds", col]].copy().dropna(subset=["actual", "line", col])
    z["proj"] = num(z[col])
    z["actual"] = num(z.actual)
    z["line"] = num(z.line)
    e = z.proj - z.actual
    ve = z.line - z.actual
    z["model_side"] = [model_side(p, l) for p, l in zip(z.proj, z.line)]
    z["actual_side"] = [outcome_side(a, l) for a, l in zip(z.actual, z.line)]
    bets = z.loc[z.model_side.ne("NO_BET") & z.actual_side.ne("PUSH")].copy()
    bets["win"] = bets.model_side.eq(bets.actual_side)
    bets["chosen_odds"] = np.where(bets.model_side.eq("OVER"), num(bets.over_odds), num(bets.under_odds))
    bets["unit_result"] = np.where(
        bets.win,
        [american_profit(o) for o in bets.chosen_odds],
        -1.0,
    )
    priced = bets.loc[num(bets.chosen_odds).notna() & pd.to_numeric(bets.unit_result, errors="coerce").notna()].copy()
    return {
        "season": season_label,
        "candidate": candidate,
        "n_matched": len(z),
        "football_mae": float(e.abs().mean()),
        "football_rmse": float(np.sqrt(np.mean(e * e))),
        "football_bias": float(e.mean()),
        "football_correlation": corr(z.actual, z.proj),
        "vegas_line_mae": float(ve.abs().mean()),
        "vegas_line_rmse": float(np.sqrt(np.mean(ve * ve))),
        "vegas_line_bias": float(ve.mean()),
        "vegas_line_correlation": corr(z.actual, z.line),
        "model_closer_than_vegas_rate": float((e.abs() < ve.abs()).mean()),
        "model_tied_vegas_accuracy_rate": float(np.isclose(e.abs(), ve.abs()).mean()),
        "directional_bets": int(len(bets)),
        "wins": int(bets.win.sum()),
        "losses": int((~bets.win).sum()),
        "directional_win_rate": float(bets.win.mean()) if len(bets) else np.nan,
        "priced_bets": int(len(priced)),
        "units": float(priced.unit_result.sum()) if len(priced) else np.nan,
        "roi_per_unit": float(priced.unit_result.mean()) if len(priced) else np.nan,
        "mean_abs_edge_yards": float((z.proj - z.line).abs().mean()),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--projection-file", action="append", required=True)
    p.add_argument("--props", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--min-coverage", type=float, default=0.70)
    a = p.parse_args()

    frames = [pd.read_csv(Path(x)) for x in a.projection_file]
    proj = pd.concat(frames, ignore_index=True)
    props = pd.read_csv(a.props) if a.props.exists() and a.props.stat().st_size else pd.DataFrame()
    if proj.empty:
        raise RuntimeError("no M60 projection rows")
    if props.empty:
        raise RuntimeError("historical prop file is empty; cannot grade market benchmark")

    selected = select_props(props)
    join_cols = ["game_id", "player_clean_key"]
    keep = join_cols + ["book", "line", "over_odds", "under_odds", "snapshot_utc", "kickoff_utc", "book_last_update", "player"]
    selected = selected[[c for c in keep if c in selected]].copy()
    z = proj.merge(selected, on=join_cols, how="left", validate="many_to_one")
    z["has_prop"] = num(z.line).notna()

    coverage_rows = []
    for season, g in z.groupby("season"):
        coverage_rows.append({
            "season": int(season), "stable_qb_rows": len(g), "matched_prop_rows": int(g.has_prop.sum()),
            "coverage": float(g.has_prop.mean()),
        })
    coverage_rows.append({
        "season": "combined", "stable_qb_rows": len(z), "matched_prop_rows": int(z.has_prop.sum()),
        "coverage": float(z.has_prop.mean()),
    })
    coverage = pd.DataFrame(coverage_rows)
    combined_cov = float(z.has_prop.mean())
    claim_status = "benchmark_claims_supported" if combined_cov >= float(a.min_coverage) else "exploratory_only_low_coverage"
    coverage["min_coverage_required"] = float(a.min_coverage)
    coverage["claim_status"] = claim_status

    matched = z.loc[z.has_prop].copy()
    if matched.empty:
        raise RuntimeError("zero projection rows matched a historical QB passing prop")

    metrics = []
    for season, g in matched.groupby("season"):
        for c in CANDIDATES:
            if f"mc_proj_{c}" in g:
                metrics.append(grade_candidate(g, c, season_label=str(int(season))))
    for c in CANDIDATES:
        if f"mc_proj_{c}" in matched:
            metrics.append(grade_candidate(matched, c, season_label="combined"))
    metrics = pd.DataFrame(metrics)

    # Per-game/side grading and edge buckets.
    detailed = []
    for c in CANDIDATES:
        col = f"mc_proj_{c}"
        if col not in matched:
            continue
        d = matched.copy()
        d["candidate"] = c
        d["model_proj"] = num(d[col])
        d["model_error"] = d.model_proj - num(d.actual)
        d["vegas_error"] = num(d.line) - num(d.actual)
        d["edge_yards"] = d.model_proj - num(d.line)
        d["abs_edge_yards"] = d.edge_yards.abs()
        d["edge_bucket"] = d.abs_edge_yards.map(edge_bucket)
        d["model_side"] = [model_side(p, l) for p, l in zip(d.model_proj, num(d.line))]
        d["actual_side"] = [outcome_side(x, l) for x, l in zip(num(d.actual), num(d.line))]
        d["market_result"] = np.select(
            [d.actual_side.eq("PUSH"), d.model_side.eq("NO_BET"), d.model_side.eq(d.actual_side)],
            ["PUSH", "NO_BET", "WIN"],
            default="LOSS",
        )
        d["chosen_odds"] = np.where(d.model_side.eq("OVER"), num(d.over_odds), num(d.under_odds))
        d["unit_result"] = np.nan
        win = d.market_result.eq("WIN") & num(d.chosen_odds).notna()
        loss = d.market_result.eq("LOSS") & num(d.chosen_odds).notna()
        d.loc[win, "unit_result"] = [american_profit(o) for o in d.loc[win, "chosen_odds"]]
        d.loc[loss, "unit_result"] = -1.0
        detailed.append(d)
    detail = pd.concat(detailed, ignore_index=True)

    bucket_rows = []
    threshold_rows = []
    for (candidate, bucket), g in detail.loc[detail.market_result.isin(["WIN", "LOSS"])].groupby(["candidate", "edge_bucket"]):
        priced = g.loc[num(g.chosen_odds).notna() & num(g.unit_result).notna()]
        bucket_rows.append({
            "candidate": candidate, "edge_bucket": bucket, "bets": len(g),
            "wins": int(g.market_result.eq("WIN").sum()),
            "win_rate": float(g.market_result.eq("WIN").mean()),
            "mean_abs_edge_yards": float(num(g.abs_edge_yards).mean()),
            "priced_bets": len(priced),
            "units": float(num(priced.unit_result).sum()) if len(priced) else np.nan,
            "roi_per_unit": float(num(priced.unit_result).mean()) if len(priced) else np.nan,
        })
    for candidate, cg in detail.groupby("candidate"):
        for threshold in (0, 5, 10, 15, 20, 30):
            g = cg.loc[num(cg.abs_edge_yards).ge(threshold) & cg.market_result.isin(["WIN", "LOSS"])].copy()
            priced = g.loc[num(g.chosen_odds).notna() & num(g.unit_result).notna()]
            threshold_rows.append({
                "candidate": candidate, "min_abs_edge_yards": threshold, "bets": len(g),
                "wins": int(g.market_result.eq("WIN").sum()),
                "win_rate": float(g.market_result.eq("WIN").mean()) if len(g) else np.nan,
                "priced_bets": len(priced),
                "units": float(num(priced.unit_result).sum()) if len(priced) else np.nan,
                "roi_per_unit": float(num(priced.unit_result).mean()) if len(priced) else np.nan,
            })
    buckets = pd.DataFrame(bucket_rows)
    thresholds = pd.DataFrame(threshold_rows)

    # Specifically answer the question that motivated M60: when a projection is
    # badly wrong in absolute yards, did it still choose the correct Vegas side?
    tail_rows = []
    for candidate, g in detail.groupby("candidate"):
        for tail, mask in {
            "all_100plus_abs_error": num(g.model_error).abs().ge(100),
            "100plus_overprojection": num(g.model_error).ge(100),
            "100plus_underprojection": num(g.model_error).le(-100),
        }.items():
            t = g.loc[mask & g.market_result.isin(["WIN", "LOSS"])].copy()
            tail_rows.append({
                "candidate": candidate, "tail": tail, "bets": len(t),
                "wins": int(t.market_result.eq("WIN").sum()),
                "win_rate": float(t.market_result.eq("WIN").mean()) if len(t) else np.nan,
                "mean_model_error": float(num(t.model_error).mean()) if len(t) else np.nan,
                "mean_abs_model_error": float(num(t.model_error).abs().mean()) if len(t) else np.nan,
                "mean_abs_edge_yards": float(num(t.abs_edge_yards).mean()) if len(t) else np.nan,
            })
    tails = pd.DataFrame(tail_rows)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(a.out_dir / "m60_market_coverage.csv", index=False)
    metrics.to_csv(a.out_dir / "m60_football_and_market_metrics.csv", index=False)
    buckets.to_csv(a.out_dir / "m60_market_edge_buckets.csv", index=False)
    thresholds.to_csv(a.out_dir / "m60_market_edge_thresholds.csv", index=False)
    tails.to_csv(a.out_dir / "m60_catastrophic_market_diagnostics.csv", index=False)
    detail.to_csv(a.out_dir / "m60_market_game_detail.csv", index=False)

    print("=== M60 MARKET COVERAGE ===")
    print(coverage.to_string(index=False))
    print("\n=== M60 FOOTBALL + VEGAS BENCHMARK ===")
    print(metrics.to_string(index=False))
    print("\n=== M60 EDGE THRESHOLDS ===")
    print(thresholds.to_string(index=False))
    print("\n=== M60 CATASTROPHIC ERRORS VS VEGAS ===")
    print(tails.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
