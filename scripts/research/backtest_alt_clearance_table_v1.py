#!/usr/bin/env python3
"""Out-of-sample calibration test of the alternate-line clearance table.

The table in data/research/alt_line_clearance_v2.csv (fraction-of-line, by
market and line quartile) and alt_line_receptions_absolute_v2.csv (absolute
integer thresholds) were fit on 2024+2025. 2026 is a genuine holdout.

Why this exists: a single graded game -- 2026 Week 9 TNF, BAL/JAX -- showed the
table over-predicting badly (realized 20.2% against 34.1% predicted) and that
was briefly written up as evidence the mid probability band was unusable. That
was wrong. Every alternate rung is an OVER, so a low-volume game depresses all
of them together; TNF produced 27 total receptions with a maximum of 4 by any
player, and only 14 of 72 players caught a pass. One game is one draw of the
game environment, not N independent cells.

This replaces that single game with every archived 2026 board: the paid main
lines are the anchors, nflverse weekly stats are the outcomes.
"""
from __future__ import annotations

import argparse
import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.price_alt_line_offers_v1 import (
    MAIN_TO_CANON, clearance, clearance_integer, load_table, _quartile,
)

STAT = {
    "pass_yards": "passing_yards", "rush_yards": "rushing_yards",
    "rec_yards": "receiving_yards", "rush_rec_yards": "rush_rec",
    "receptions": "receptions",
}
BANDS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


def _board(week: int, ref: str) -> pd.DataFrame:
    path = f"{ref}:data/market_track_record/boards/2026_wk{week:02d}.csv"
    out = subprocess.run(["git", "show", path], capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(out.stdout), low_memory=False)


def run(weeks: list[int], season: int, ref: str) -> pd.DataFrame:
    import nflreadpy as nfl

    table, recp, bounds = load_table()
    w = nfl.load_player_stats(seasons=[season]).to_pandas()
    for col in ("passing_yards", "rushing_yards", "receiving_yards", "receptions"):
        w[col] = pd.to_numeric(w[col], errors="coerce").fillna(0.0)
    w["rush_rec"] = w["rushing_yards"] + w["receiving_yards"]
    fracs = sorted(table["frac_of_LINE"].dropna().unique())

    rows: list[dict] = []
    for week in weeks:
        board = _board(week, ref)
        if board.empty:
            continue
        board = board[board["market"].isin(STAT)]
        anchors = board.groupby(["player", "team", "market"], as_index=False)["vegas_line"].median()
        actual = w[w["week"] == week]
        for r in anchors.itertuples(index=False):
            a = actual[actual["player_display_name"].eq(str(r.player))]
            if a.empty:
                continue
            stat = float(a[STAT[r.market]].iloc[0])
            line = float(r.vegas_line)
            if not np.isfinite(line) or line <= 0:
                continue
            if r.market == "receptions":
                for k in range(1, 6):
                    pred, _ = clearance_integer(recp, line, k - 0.5)
                    if np.isfinite(pred):
                        rows.append({"season": season, "week": week, "market": r.market,
                                     "player": r.player, "team": r.team, "main_line": line,
                                     "threshold": float(k), "predicted": pred,
                                     "actual_stat": stat, "hit": int(stat >= k)})
            else:
                qt = _quartile(line, bounds[r.market]) if r.market in bounds else -1
                for f in fracs:
                    pred, _ = clearance(table, r.market, qt, f)
                    if np.isfinite(pred):
                        rows.append({"season": season, "week": week, "market": r.market,
                                     "player": r.player, "team": r.team, "main_line": line,
                                     "threshold": f * line, "predicted": pred,
                                     "actual_stat": stat, "hit": int(stat > f * line)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weeks", default="1,2,6,7,8")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--ref", default="origin/main", help="git ref the boards are read from")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    g = run([int(x) for x in a.weeks.split(",")], a.season, a.ref)
    if g.empty:
        print("[alt_backtest] no gradable cells")
        return 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(a.out, index=False)
    g["band"] = pd.cut(g["predicted"], BANDS)
    byband = g.groupby("band", observed=True).agg(
        n=("hit", "size"), predicted=("predicted", "mean"), actual=("hit", "mean"))
    byband["deviation"] = byband["actual"] - byband["predicted"]
    print(f"[alt_backtest] cells={len(g)} weeks={sorted(g['week'].unique())} -> {a.out}")
    print(byband.to_string())
    print("\nby market:")
    print(g.groupby("market").agg(n=("hit", "size"), predicted=("predicted", "mean"),
                                  actual=("hit", "mean")).to_string())
    print("\n>=70% band, week by week:")
    print(g[g["predicted"] >= 0.70].groupby("week").agg(
        n=("hit", "size"), predicted=("predicted", "mean"), actual=("hit", "mean")).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
