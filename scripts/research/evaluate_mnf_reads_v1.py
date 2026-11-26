#!/usr/bin/env python3
"""MNF_READS_SINGLE_GAME_TEST_PREREGISTRATION_V1.md, as frozen at 0c8727e.

T1 shootout totals, T2 missing top receiver -> TE/RB, T1+T2, T3 near-projection rungs
(descriptive). Every bet is the main-line OVER at its posted over price; void if the player
took no offensive snap.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nfl

ROWS = "data/research/main_line_screens/main_line_screens_rows_v1.csv.gz"
OUT = Path("data/research/mnf_reads")
SHOOTOUT = 49.5
SEED = 20261126


def key(s) -> str:
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", str(s).lower())
    return re.sub(r"[^a-z]", "", s)


def dec(a: float) -> float:
    return 1 + a / 100 if a > 0 else 1 + 100 / -a


def ci(df: pd.DataFrame, rng) -> tuple[float, float]:
    g = df.groupby("gid").pnl.agg(["sum", "size"])
    s, n = g["sum"].to_numpy(), g["size"].to_numpy()
    if len(g) < 2:
        return (np.nan, np.nan)
    b = [s[i].sum() / n[i].sum() for i in (rng.integers(0, len(g), len(g)) for _ in range(2000))]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def summ(df: pd.DataFrame, label: str, rng) -> dict:
    lo, hi = ci(df, rng) if len(df) else (np.nan, np.nan)
    be = (1 / df.over_odds.map(dec)).mean() if len(df) else np.nan
    return dict(test=label, n=len(df), games=df.gid.nunique() if len(df) else 0,
                hit=df.win.mean() if len(df) else np.nan, breakeven=be,
                roi=df.pnl.mean() if len(df) else np.nan, ci_lo=lo, ci_hi=hi)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = pd.read_csv(ROWS, low_memory=False)
    d = d[d.market.isin(["rec_yards", "receptions"]) & d.pos.isin(["WR", "TE", "RB"])].copy()
    seasons = sorted(d.season.unique().tolist())

    sch = nfl.load_schedules(seasons=seasons).to_pandas()
    sch = sch[sch.game_type == "REG"]
    games = pd.concat([sch.assign(team=sch.home_team, opponent=sch.away_team),
                       sch.assign(team=sch.away_team, opponent=sch.home_team)])[
        ["season", "week", "team", "game_id", "total_line", "gameday", "gametime", "weekday"]]
    d = d.merge(games, on=["season", "week", "team"], how="left")
    d = d.dropna(subset=["game_id"])
    d["gid"] = d.game_id

    sn = nfl.load_snap_counts(seasons=seasons).to_pandas()
    sn = sn[sn.game_type == "REG"]
    sn["pk"] = sn.player.map(key)
    played = set(zip(sn.loc[sn.offense_snaps.fillna(0) > 0, "season"], sn.loc[sn.offense_snaps.fillna(0) > 0, "week"],
                     sn.loc[sn.offense_snaps.fillna(0) > 0, "team"], sn.loc[sn.offense_snaps.fillna(0) > 0, "pk"]))
    d["pk2"] = d.pk.map(key)
    d["played"] = [(s, w, t, k) in played for s, w, t, k in zip(d.season, d.week, d.team, d.pk2)]
    d = d[d.played].copy()

    d["win"] = (d.actual > d.line).astype(float)
    d.loc[d.actual == d.line, "win"] = np.nan
    d = d.dropna(subset=["win", "over_odds"])
    d["pnl"] = np.where(d.win == 1, d.over_odds.map(dec) - 1, -1.0)

    # T2: strictly-prior top target earner (WR/TE, >= 3 games), and whether he played
    st = nfl.load_player_stats(seasons=seasons, summary_level="week").to_pandas()
    st = st[(st.season_type == "REG") & st.position.isin(["WR", "TE"])].copy()
    st["pk"] = st.player_display_name.map(key)
    st = st.sort_values(["season", "team", "week"])
    top = {}
    for (s, t), g in st.groupby(["season", "team"]):
        for w in sorted(sch[(sch.season == s) & ((sch.home_team == t) | (sch.away_team == t))].week.unique()):
            h = g[g.week < w]
            if h.empty:
                continue
            agg = h.groupby("pk").agg(tg=("targets", "sum"), gp=("week", "nunique"))
            agg = agg[agg.gp >= 3]
            if agg.empty:
                continue
            top[(s, w, t)] = agg.tg.idxmax()
    d["top_receiver"] = [top.get((s, w, t)) for s, w, t in zip(d.season, d.week, d.team)]
    d["top_missing"] = [bool(tr) and (s, w, t, tr) not in played for s, w, t, tr in zip(d.season, d.week, d.team, d.top_receiver)]
    d.loc[d.top_receiver.isna(), "top_missing"] = np.nan
    d["shootout"] = d.total_line >= SHOOTOUT

    res = []
    for samp, x in d.groupby(np.where(d.season == 2026, "2026", "2024-25")):
        res.append(summ(x[x.shootout], f"T1 shootout (total>={SHOOTOUT}) all receiving OVERS | {samp}", rng))
        res.append(summ(x[~x.shootout], f"T1 not shootout | {samp}", rng))
        for pos in ("WR", "TE", "RB"):
            res.append(summ(x[x.shootout & x.pos.eq(pos)], f"T1 shootout {pos} | {samp}", rng))
        tr = x[x.pos.isin(["TE", "RB"]) & x.top_missing.notna()]
        res.append(summ(tr[tr.top_missing == True], f"T2 top receiver MISSING, TE/RB OVERS | {samp}", rng))
        res.append(summ(tr[tr.top_missing == False], f"T2 top receiver plays, TE/RB OVERS | {samp}", rng))
        wr = x[x.pos.eq("WR") & x.top_missing.eq(True) & x.pk2.ne(x.top_receiver)]
        res.append(summ(wr, f"T2 top receiver MISSING, other WR OVERS (report) | {samp}", rng))
        both = tr[(tr.top_missing == True) & tr.shootout]
        res.append(summ(both, f"T1+T2 shootout AND top receiver missing, TE/RB OVERS | {samp}", rng))
    R = pd.DataFrame(res)

    # T3: near-projection alt rungs, 2026 weeks 10-11
    alts = []
    for f, wk in [("data/market_track_record/graded/w10_2026_alt_board_graded_complete_v1.csv", 10),
                  ("data/market_track_record/graded/w11_2026_alt_board_graded_v1.csv", 11)]:
        a = pd.read_csv(f, low_memory=False)
        a["week"] = wk
        price = "price_american"
        a = a[(a.side.astype(str).str.upper() == "OVER") & a[price].between(-700, -150)]
        a["cushion"] = a.model_proj / a.line
        a = a[a.cushion.between(1.0, 1.25)].dropna(subset=["hit"])
        ct = pd.to_datetime(a.commence_time, utc=True).dt.tz_convert("America/New_York")
        a["primetime"] = (ct.dt.hour >= 19) & ct.dt.dayofweek.isin([0, 3, 6])
        a["pnl_"] = np.where(a.hit.astype(float) == 1, a[price].map(dec) - 1, -1.0)
        alts.append(a[["week", "event_id", "player", "market", "line", price, "model_proj", "cushion", "primetime", "hit", "pnl_"]])
    A = pd.concat(alts, ignore_index=True)
    t3 = A.groupby("primetime").agg(n=("hit", "size"), games=("event_id", "nunique"), hit=("hit", "mean"),
                                     roi=("pnl_", "mean")).reset_index()
    t3["breakeven"] = A.groupby("primetime")["price_american"].apply(lambda s: (1 / s.map(dec)).mean()).to_numpy()

    d.to_csv(OUT / "mnf_reads_rows_v1.csv.gz", index=False)
    R.to_csv(OUT / "mnf_reads_results_v1.csv", index=False)
    A.to_csv(OUT / "mnf_reads_t3_alt_rows_v1.csv", index=False)
    t3.to_csv(OUT / "mnf_reads_t3_summary_v1.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))
    print(t3.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
