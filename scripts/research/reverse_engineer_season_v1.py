#!/usr/bin/env python3
"""Reverse-engineer every 2026 game played so far, and test what was knowable pregame.

Asked 2026-11-19 (before week 11): where is PRISM consistent, are there team /
matchup patterns, do game plans show up before kickoff, and what does that say
about player production.

Four parts, all reproducible from nflverse plus the archived boards:

  1. TEAM GAMES. One row per team-game from play-by-play (kneels and spikes
     removed): plays, dropbacks, rushes, neutral-situation pass rate and pass
     rate over expected (win prob 0.2-0.8, downs 1-3, not the last two minutes
     of a half), pass rate when leading/trailing by 8+, pregame spread and
     total, final margin.
  2. WHAT IS STABLE. Odd-week vs even-week correlation across the 32 teams, for
     each offense and for each defense (what it allows). A tendency that does
     not repeat between halves of the season cannot be used pregame.
  3. WALK-FORWARD TEAM VOLUME. Week w is predicted only from weeks < w. Compared:
     production's constant pass share 0.57, the team's own season average,
     team average + spread + total, and + opponent allowed.
  4. THE MARKET BY GAME SCRIPT. Every graded FanDuel main-line offer (both
     sides, model ignored) split by the player's team spread, plus PRISM's own
     picked side (edge_pct > 0) split the same way. Segments were found on weeks
     6-10; weeks 1-2 are the out-of-sample check.

Inputs for part 4 are graded boards from grade_mainline_board_v1.py:
  grade_mainline_board_v1.py --boards data/market_track_record/boards/2026_wk0{1,2}.csv --out W12
  grade_mainline_board_v1.py --boards data/market_track_record/boards/2026_wk{06..10}.csv --out W610
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nflreadpy as nfl

PROD_PASS_SHARE = 0.57   # rules_v2.project_game_script, same for every team


def team_games(season: int) -> pd.DataFrame:
    p = nfl.load_pbp(seasons=[season]).to_pandas()
    x = p[((p["pass"] == 1) | (p["rush"] == 1)) & (p.qb_kneel != 1) & (p.qb_spike != 1) & p.posteam.notna()].copy()
    x["neutral"] = x.wp.between(.2, .8) & (x.down <= 3) & (x.half_seconds_remaining > 120)
    x["script"] = np.select([x.score_differential >= 8, x.score_differential <= -8], ["lead8", "trail8"], "close")
    T = (x.groupby(["game_id", "week", "posteam", "defteam"])
          .agg(plays=("pass", "size"), dropbacks=("pass", "sum"), rushes=("rush", "sum"), epa=("epa", "mean"))
          .reset_index())
    nt = x[x.neutral].groupby(["game_id", "posteam"]).agg(neutral_pass=("pass", "mean"), neutral_proe=("pass_oe", "mean"))
    T = T.merge(nt.reset_index(), on=["game_id", "posteam"], how="left")
    for s in ["lead8", "close", "trail8"]:
        q = x[x.script == s].groupby(["game_id", "posteam"]).agg(**{f"n_{s}": ("pass", "size"), f"pr_{s}": ("pass", "mean")})
        T = T.merge(q.reset_index(), on=["game_id", "posteam"], how="left")
    sch = nfl.load_schedules(seasons=[season]).to_pandas()[
        ["game_id", "home_team", "away_team", "home_score", "away_score", "spread_line", "total_line"]]
    T = T.merge(sch, on="game_id", how="left")
    T["home"] = (T.posteam == T.home_team).astype(int)
    T["fav_by"] = np.where(T.home == 1, T.spread_line, -T.spread_line)   # nflverse: + = home favored
    T["pts"] = np.where(T.home == 1, T.home_score, T.away_score)
    T["opp_pts"] = np.where(T.home == 1, T.away_score, T.home_score)
    T["margin"] = T.pts - T.opp_pts
    T["pass_rate"] = T.dropbacks / T.plays
    return T


def reliability(T: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in ["neutral_proe", "neutral_pass", "pass_rate", "plays", "rushes", "dropbacks"]:
        r = {}
        for side in ["posteam", "defteam"]:
            o = T[T.week % 2 == 1].groupby(side)[c].mean(); e = T[T.week % 2 == 0].groupby(side)[c].mean()
            r[side] = round(o.corr(e), 2)
        rows.append(dict(measure=c, offense_odd_vs_even=r["posteam"], defense_allowed_odd_vs_even=r["defteam"]))
    return pd.DataFrame(rows)


def _feats(T, w):
    tr, te = T[T.week < w], T[T.week == w]
    o = tr.groupby("posteam")[["rushes", "dropbacks", "plays"]].mean().reindex(te.posteam)
    a = tr.groupby("defteam")[["rushes", "dropbacks"]].mean().reindex(te.defteam)
    lg = tr[["rushes", "dropbacks", "plays"]].mean()
    return pd.DataFrame({"c": 1.0, "off_rushes": o.rushes.values, "off_dropbacks": o.dropbacks.values,
                         "opp_rushes": a.rushes.values - lg.rushes, "opp_dropbacks": a.dropbacks.values - lg.dropbacks,
                         "fav": te.fav_by.values, "tot": te.total_line.values - 44.0,
                         "y_rushes": te.rushes.values, "y_dropbacks": te.dropbacks.values,
                         "lg_plays": lg.plays, "week": w})


def walk_forward(T: pd.DataFrame, first: int = 4) -> pd.DataFrame:
    rows = []
    for w in range(first, int(T.week.max()) + 1):
        tr = pd.concat([_feats(T, v) for v in range(3, w)]).dropna()
        te = _feats(T, w)
        for y in ["rushes", "dropbacks"]:
            spread = ["c", f"off_{y}", "fav", "tot"]; full = spread[:2] + [f"opp_{y}"] + spread[2:]
            b1 = np.linalg.lstsq(tr[spread].values, tr[f"y_{y}"].values, rcond=None)[0]
            b2 = np.linalg.lstsq(tr[full].values, tr[f"y_{y}"].values, rcond=None)[0]
            share = PROD_PASS_SHARE if y == "dropbacks" else 1 - PROD_PASS_SHARE
            preds = {"production constant 0.57": te.lg_plays.values * share, "team season average": te[f"off_{y}"].values,
                     "team average + spread + total": te[spread].values @ b1, "+ opponent allowed": te[full].values @ b2}
            for k, pr in preds.items():
                rows.append(dict(week=w, target=y, model=k, mae=float(np.nanmean(np.abs(te[f"y_{y}"].values - pr)))))
    return pd.DataFrame(rows)


def board_by_script(graded: list[str], season: int) -> pd.DataFrame:
    m = pd.concat([pd.read_csv(f) for f in graded], ignore_index=True)
    m = m[m.market != "anytime_td"].copy()
    sch = nfl.load_schedules(seasons=[season]).to_pandas()[["week", "home_team", "away_team", "spread_line", "total_line"]]
    h = sch.rename(columns={"home_team": "tm"}).assign(home=1); a = sch.rename(columns={"away_team": "tm"}).assign(home=0)
    ha = pd.concat([d[["week", "tm", "home", "spread_line", "total_line"]] for d in (h, a)])
    m["tm"] = m.team.replace({"LAR": "LA"})
    m = m.merge(ha, on=["week", "tm"], how="left")
    m["fav_by"] = np.where(m.home == 1, m.spread_line, -m.spread_line)
    st = nfl.load_player_stats(seasons=[season]).to_pandas()
    m["pos"] = m.player.map(st.groupby("player_display_name").position.first())
    m["script"] = np.where(m.fav_by < -2.5, "underdog 3+", np.where(m.fav_by > 2.5, "favored 3+", "pick"))
    sn = nfl.load_snap_counts(seasons=[season]).to_pandas()[["player", "week", "offense_pct"]].sort_values("week")
    sn["prior_snap"] = sn.groupby("player").offense_pct.transform(lambda s: s.shift().expanding().mean())
    m = m.merge(sn, on=["player", "week"], how="left")
    m["snap_state"] = np.where(m.offense_pct < .6 * m.prior_snap, "PARTIAL_OR_ROLE_CUT", "NORMAL")
    return m


def seg(d: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = d.groupby(by, observed=True)
    out = g.agg(n=("pnl", "size"), hit=("hit", "mean"), roi=("pnl", "mean"), line=("vegas_line", "mean"),
                actual_median=("actual", "median")).reset_index()
    wk = d.groupby(by + ["week"], observed=True).pnl.mean().unstack()
    pos = ((wk > 0).sum(axis=1).astype(str) + "/" + wk.notna().sum(axis=1).astype(str)).rename("weeks_positive")
    return out.merge(pos.reset_index(), on=by)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--graded", nargs="+", required=True, help="graded boards from grade_mainline_board_v1.py")
    ap.add_argument("--out-dir", default="data/research/reverse_engineer")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    T = team_games(a.season); T.to_csv(out / f"team_games_{a.season}_v1.csv", index=False)
    R = reliability(T); R.to_csv(out / "team_tendency_reliability_v1.csv", index=False)
    W = walk_forward(T); W.to_csv(out / "team_volume_walkforward_v1.csv", index=False)
    M = board_by_script(a.graded, a.season)
    fd = M[M.book == "fanduel"]
    B = seg(fd, ["market", "pos", "script", "side"]); B.to_csv(out / "fanduel_board_by_script_v1.csv", index=False)
    P = M[M.edge_pct > 0].assign(period=lambda d: np.where(d.week <= 2, "weeks 1-2 (out of sample)", "weeks 6-10"))
    S = seg(P, ["period", "market", "pos", "side"]); S.to_csv(out / "prism_side_by_market_pos_v1.csv", index=False)

    pd.set_option("display.width", 240)
    print(f"{len(T)} team-games, {T.game_id.nunique()} games")
    print("\nBy actual margin:")
    T["result"] = pd.cut(T.margin, [-99, -14.5, -7.5, -.5, 7.5, 14.5, 99],
                         labels=["lost 15+", "lost 8-14", "lost 1-7", "won 0-7", "won 8-14", "won 15+"])
    print(T.groupby("result", observed=True)[["rushes", "dropbacks", "pass_rate"]].mean().round(2).to_string())
    T["pregame"] = pd.cut(T.fav_by, [-30, -6.5, -2.5, 2.5, 6.5, 30], labels=["dog 7+", "dog 3-6", "pick", "fav 3-6", "fav 7+"])
    print("\nBy pregame spread:")
    print(T.groupby("pregame", observed=True)[["rushes", "dropbacks", "pass_rate", "margin"]].mean().round(2).to_string())
    print("\nReliability:"); print(R.to_string(index=False))
    print("\nWalk-forward MAE:"); print(W.groupby(["target", "model"]).mae.mean().round(2).to_string())
    rb = fd[(fd.market == "rush_yards") & (fd.pos == "RB") & (fd.side == "UNDER")]
    print("\nRB rushing UNDER by script (FanDuel):"); print(seg(rb, ["script", "snap_state"]).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
