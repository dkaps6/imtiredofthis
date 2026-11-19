#!/usr/bin/env python3
"""RB rushing-yards UNDER: count of pregame "bad situation" flags.

Research only (2026-11-19, REVERSE_ENGINEER_2026_W1_10_V1.md part 2). Not a
production model and not a ticket rule unless the operator adopts it.

Three flags, all known before kickoff, all strictly prior:
  dog   the RB's team is an underdog by 3+ (nflverse spread_line).
  cold  his yards-per-carry so far is under 95% of the league RB average.
  cut   a plain volume projection is more than 1.2x the book line, i.e. the book
        has already marked him down from his own volume, and the question is
        whether it marked him down far enough.
        volume projection = carries per game x (79% league YPC + 21% own YPC);
        21% because individual YPC repeats week to week at only r = 0.21.

Why these three: volume is stable (RB carry share odd/even-week r = 0.86) while
efficiency is noise (YPC r = 0.21); game script sets volume (losers by 15+ run
20 times, winners by 15+ 31 times); and the book's line sits near the mean of a
right-skewed outcome, so a back in a bad spot lands under it more often than not.

Modes
  backtest  --graded <graded board csv ...>   (grade_mainline_board_v1.py output)
  slate     --board <archived board csv> --week N
Priors: the season's weeks before the target week; for weeks 1-2 the prior
season is appended so there is something to measure.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import nflreadpy as nfl

DOG_SPREAD = 2.5
COLD_RATIO = 0.95
CUT_RATIO = 1.2
OWN_YPC_WEIGHT = 0.21
MIN_GAMES = 3


def spreads(season: int) -> pd.DataFrame:
    s = nfl.load_schedules(seasons=[season]).to_pandas()[["week", "home_team", "away_team", "spread_line"]]
    h = s.rename(columns={"home_team": "tm"}).assign(home=1)
    a = s.rename(columns={"away_team": "tm"}).assign(home=0)
    x = pd.concat([d[["week", "tm", "home", "spread_line"]] for d in (h, a)])
    x["fav_by"] = np.where(x.home == 1, x.spread_line, -x.spread_line)
    return x[["week", "tm", "fav_by"]]


def priors(season: int, weeks) -> pd.DataFrame:
    st = nfl.load_player_stats(seasons=[season - 1, season]).to_pandas()
    st = st[st.season_type == "REG"].copy()
    st["t"] = (st.season - (season - 1)) * 100 + st.week
    out = []
    for w in weeks:
        pr = st[st.t < 100 + w]
        if w > 2:
            pr = pr[pr.season == season]
        rb = pr[pr.position == "RB"]
        lg = rb.rushing_yards.sum() / rb.carries.sum()
        g = pr.groupby("player_display_name").tail(17).groupby("player_display_name")
        f = pd.DataFrame({"pos": g.position.last(), "gp": g.size(), "cpg": g.carries.mean(),
                          "ypc": g.rushing_yards.sum() / g.carries.sum().replace(0, np.nan)})
        f["lg_ypc"], f["week"] = lg, w
        out.append(f.reset_index().rename(columns={"player_display_name": "player"}))
    return pd.concat(out, ignore_index=True)


def flag(board: pd.DataFrame, season: int) -> pd.DataFrame:
    b = board[(board.market == "rush_yards") & (board.book == "fanduel")].copy()
    b["tm"] = b.team.replace({"LAR": "LA"})
    b = b.merge(spreads(season), on=["week", "tm"], how="left")
    b = b.merge(priors(season, sorted(b.week.unique())), on=["player", "week"], how="inner")
    b = b[(b.pos == "RB") & (b.gp >= MIN_GAMES)].copy()
    b["vol_proj"] = b.cpg * ((1 - OWN_YPC_WEIGHT) * b.lg_ypc + OWN_YPC_WEIGHT * b.ypc.fillna(b.lg_ypc))
    b["dog"] = (b.fav_by < -DOG_SPREAD).astype(int)
    b["cold"] = (b.ypc / b.lg_ypc < COLD_RATIO).astype(int)
    b["cut"] = (b.vol_proj / b.vegas_line > CUT_RATIO).astype(int)
    b["n_flags"] = b.dog + b.cold + b.cut
    return b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--graded", nargs="+")
    ap.add_argument("--board"); ap.add_argument("--week", type=int)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    if a.graded:
        g = flag(pd.concat([pd.read_csv(f) for f in a.graded], ignore_index=True), a.season)
        g["period"] = np.where(g.week <= 2, "weeks 1-2", "weeks 6-10")
        t = (g.groupby(["period", "side", "n_flags"]).agg(n=("hit", "size"), hit=("hit", "mean"), roi=("pnl", "mean"))
             .reset_index())
        wk = g.groupby(["period", "side", "n_flags", "week"]).hit.mean().unstack().round(2)
        t["hit_by_week"] = [" ".join(f"w{c}:{v}" for c, v in wk.loc[(p, s, f)].dropna().items())
                            for p, s, f in zip(t.period, t.side, t.n_flags)]
        print(t.round(3).to_string(index=False))
        u = g[(g.side == "UNDER") & (g.n_flags >= 2)]
        print(f"\nUNDER with 2+ flags, all weeks: n={len(u)} hit={u.hit.mean():.1%} roi={u.pnl.mean():+.1%}")
        g.to_csv(a.out, index=False)
    else:
        b = pd.read_csv(a.board); b = b[(b.side == "UNDER") & (b.week == a.week)] if "side" in b else b
        f = flag(b, a.season).sort_values(["n_flags", "vegas_line"], ascending=[False, True])
        cols = ["player", "team", "opponent", "fav_by", "vegas_line", "vegas_under_odds", "model_proj", "cpg", "ypc",
                "lg_ypc", "vol_proj", "dog", "cold", "cut", "n_flags"]
        print(f[cols].round(2).to_string(index=False))
        f[cols].to_csv(a.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
