#!/usr/bin/env python3
"""MAIN_LINE_FOOTBALL_SCREENS_PREREGISTRATION_V1.md, exactly as frozen.

Football screens F1-F6 applied to the model's main-line picks, scored within each
market x side cell on 2026 (FanDuel, PRISM model_proj) and 2024-2025 (full-stack
benchmark, mostly DraftKings). All features strictly prior to the game week.

Implementation detail fixed before scoring (the preregistration did not define F3 for
quarterbacks): for pass_yards, F3 = at least 4 prior games and >= 15 pass attempts in
each of his last 3 games (he is the starter).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import nflreadpy as nfl

SLIM26 = "data/research/selective_pick/prism_2026_w1_2_6_10_mainline_graded_slim_v1.csv"
FULL = "data/backtests/full_stack_vegas_benchmark_v1/non_qb_detail.csv"
BASE = "data/backtests/historical_market_vegas_benchmark_v1/historical_market_vegas_benchmark_detail.csv"
MARKETS = ["rush_yards", "rec_yards", "receptions", "pass_yards"]
TEAM_FIX = {"LAR": "LA", "OAK": "LV", "SD": "LAC", "STL": "LA", "WSH": "WAS", "JAC": "JAX"}
nm = lambda s: s.astype(str).str.lower().str.replace(r"[^a-z]", "", regex=True)


def dec(a):
    a = np.asarray(a, float)
    return np.where(a > 0, a / 100, 100 / np.abs(a))


def rows_2026() -> pd.DataFrame:
    m = pd.read_csv(SLIM26)
    m = m[(m.played == 1) & (m.book == "fanduel") & m.market.isin(MARKETS)]
    w = m.pivot_table(index=["season", "week", "player_clean_key", "team", "opponent", "market", "vegas_line", "model_proj", "actual"],
                      columns="side", values="vegas_odds", aggfunc="first").reset_index()
    return w.rename(columns={"OVER": "over_odds", "UNDER": "under_odds", "vegas_line": "line", "model_proj": "proj", "player_clean_key": "pk"})


def rows_hist() -> pd.DataFrame:
    f = pd.read_csv(FULL, low_memory=False)
    f = f[f.market.isin(["rush_yards", "rec_yards", "receptions"])]
    f = f[["season", "week", "team", "opponent", "player_clean_key", "market", "proj", "line", "over_odds", "under_odds", "actual"]]
    b = pd.read_csv(BASE, low_memory=False)
    b = b[b.market == "pass_yards"].rename(columns={"mc_proj": "proj"})
    b = b[["season", "week", "team", "player_clean_key", "market", "proj", "line", "over_odds", "under_odds", "actual"]]
    h = pd.concat([f, b], ignore_index=True)
    return h.rename(columns={"player_clean_key": "pk"})


def pick(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["team"] = d.team.replace(TEAM_FIX)
    d["pk"] = nm(d.pk)
    d = d.dropna(subset=["over_odds", "under_odds", "actual", "line", "proj"])
    d = d[(d.proj != d.line) & (d.actual != d.line)]
    d["side"] = np.where(d.proj > d.line, "OVER", "UNDER")
    d["win"] = np.where(d.side == "OVER", d.actual > d.line, d.actual < d.line).astype(int)
    d["price"] = np.where(d.side == "OVER", d.over_odds, d.under_odds)
    d["pnl"] = np.where(d.win == 1, dec(d.price), -1.0)
    d["gap"] = (d.proj - d.line).abs() / d.line
    return d


def features(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    st = nfl.load_player_stats(seasons=seasons).to_pandas()
    st = st[st.season_type == "REG"].copy()
    for c in ["carries", "targets", "attempts", "rushing_yards", "receiving_yards", "receptions", "passing_yards"]:
        st[c] = st[c].fillna(0)
    st["team"] = st.team.replace(TEAM_FIX)
    st["pk"] = nm(st.player_display_name)
    tm = st.groupby(["season", "week", "team"])[["carries", "targets"]].sum().add_prefix("tm_").reset_index()
    st = st.merge(tm, on=["season", "week", "team"])
    sn = nfl.load_snap_counts(seasons=seasons).to_pandas()
    sn = sn[sn.game_type == "REG"].copy()
    sn["team"] = sn.team.replace(TEAM_FIX)
    sn["pk"] = nm(sn.player)
    sch = nfl.load_schedules(seasons=seasons).to_pandas()
    sch = sch[sch.game_type == "REG"]
    h = sch.assign(team=sch.home_team, opp=sch.away_team, fav_by=sch.spread_line)
    a = sch.assign(team=sch.away_team, opp=sch.home_team, fav_by=-sch.spread_line)
    sp = pd.concat([h, a])[["season", "week", "team", "opp", "fav_by"]]
    sp["team"] = sp.team.replace(TEAM_FIX); sp["opp"] = sp.opp.replace(TEAM_FIX)
    return st, sn, sp, sch


def allowed_pct(st: pd.DataFrame, sp: pd.DataFrame) -> pd.DataFrame:
    """Opponent's strictly-prior yards allowed per game to a position group, as a percentile
    across defenses within (season, week, group). Higher = weaker defense."""
    s = st.merge(sp[["season", "week", "team", "opp"]], on=["season", "week", "team"])
    groups = {"rush_RB": ("RB", "rushing_yards"), "rec_WR": ("WR", "receiving_yards"), "rec_TE": ("TE", "receiving_yards"),
              "rec_RB": ("RB", "receiving_yards"), "rec_cnt_WR": ("WR", "receptions"), "rec_cnt_TE": ("TE", "receptions"),
              "rec_cnt_RB": ("RB", "receptions"), "pass_QB": ("QB", "passing_yards")}
    out = []
    for g, (pos, col) in groups.items():
        gm = s[s.position == pos].groupby(["season", "week", "opp"])[col].sum().reset_index()
        for season, gs in gm.groupby("season"):
            for wk in range(2, 19):
                prior = gs[gs.week < wk].groupby("opp")[col].mean()
                if prior.empty:
                    continue
                pct = prior.rank(pct=True)
                out.append(pd.DataFrame({"season": season, "week": wk, "def": pct.index, "group": g, "def_pct": pct.values}))
    return pd.concat(out, ignore_index=True)


def screen(d: pd.DataFrame, st: pd.DataFrame, sn: pd.DataFrame, sp: pd.DataFrame, ap: pd.DataFrame) -> pd.DataFrame:
    d = d.merge(sp, on=["season", "week", "team"], how="left")
    pos = st.sort_values("week").groupby(["season", "pk"]).position.last().rename("pos").reset_index()
    d = d.merge(pos, on=["season", "pk"], how="left")
    recs = []
    snap_by = {k: g.sort_values("week") for k, g in sn.groupby(["season", "pk"])}
    st_by = {k: g.sort_values("week") for k, g in st.groupby(["season", "pk"])}
    team_weeks = sp.groupby(["season", "team"]).week.apply(lambda x: sorted(x)).to_dict()
    for r in d.itertuples():
        f = dict(F1=False, F2=False, F3=False)
        sh = st_by.get((r.season, r.pk))
        sv = snap_by.get((r.season, r.pk))
        tw = [w for w in team_weeks.get((r.season, r.team), []) if w < r.week]
        if sh is not None:
            h = sh[(sh.week < r.week) & (sh.team == r.team)]
        else:
            h = pd.DataFrame()
        # F1 field
        if r.pos == "QB":
            f["F1"] = True
        elif sv is not None:
            s3 = sv[(sv.week < r.week)].tail(3)
            f["F1"] = len(s3) > 0 and s3.offense_pct.mean() >= 0.50
        # F2 available: played in each of the team's last 2 games
        if sv is not None and len(tw) >= 2:
            played = set(sv[(sv.offense_snaps.fillna(0) > 0)].week)
            f["F2"] = all(w in played for w in tw[-2:])
        elif r.pos == "QB" and len(tw) >= 2 and len(h):
            f["F2"] = all(w in set(h.week) for w in tw[-2:])
        # F3 stable role
        if len(h) >= 4:
            l3 = h.tail(3)
            if r.market == "pass_yards":
                f["F3"] = bool((l3.attempts >= 15).all())
            else:
                num, den = ("carries", "tm_carries") if r.market == "rush_yards" else ("targets", "tm_targets")
                season_share = h[num].sum() / max(h[den].sum(), 1)
                l3_share = l3[num].sum() / max(l3[den].sum(), 1)
                f["F3"] = season_share > 0 and abs(l3_share / season_share - 1) <= 0.30
        recs.append(f)
    d = pd.concat([d.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
    d["F4"] = d.gap.between(0.08, 0.35)
    fb = d.fav_by.fillna(0)
    rush, rec, pas = d.market == "rush_yards", d.market.isin(["rec_yards", "receptions"]), d.market == "pass_yards"
    over, under = d.side == "OVER", d.side == "UNDER"
    d["F5"] = ((rush & under & (fb < 7)) | (rush & over & (fb >= 3)) | (rec & over & (fb < 10)) | (rec & under & (fb > -7))
               | (pas & over & (fb < 10)) | (pas & under & (fb > -7)))
    grp = np.select([rush, (d.market == "rec_yards"), (d.market == "receptions"), pas],
                    ["rush_RB", "rec_" + d.pos.fillna("WR").where(d.pos.isin(["WR", "TE", "RB"]), "WR"),
                     "rec_cnt_" + d.pos.fillna("WR").where(d.pos.isin(["WR", "TE", "RB"]), "WR"), "pass_QB"], "")
    d["group"] = grp
    d = d.merge(ap.rename(columns={"def": "opp"}), on=["season", "week", "opp", "group"], how="left")
    d["F6"] = (over & (d.def_pct >= 0.40)) | (under & (d.def_pct <= 0.60))
    d["SCREENED"] = d[["F1", "F2", "F3", "F4", "F5", "F6"]].all(axis=1)
    return d


def summarize(d: pd.DataFrame) -> dict:
    if len(d) == 0:
        return dict(n=0, hit=np.nan, roi=np.nan, weeks_pos="0/0")
    wk = d.groupby(["season", "week"]).pnl.mean()
    return dict(n=len(d), hit=d.win.mean(), roi=d.pnl.mean(), weeks_pos=f"{int((wk > 0).sum())}/{len(wk)}")


def main() -> int:
    ap_ = argparse.ArgumentParser(); ap_.add_argument("--out-dir", default="data/research/main_line_screens")
    a = ap_.parse_args(); out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    parts = []
    for label, rows, seasons in [("2026", rows_2026(), [2026]), ("2024-25", rows_hist(), [2024, 2025])]:
        d = pick(rows)
        st, sn, sp, _ = features(seasons)
        apct = allowed_pct(st, sp)
        d = screen(d, st, sn, sp, apct).assign(sample=label)
        print(f"{label}: rows {len(d)}, position found {d.pos.notna().mean():.1%}, spread found {d.fav_by.notna().mean():.1%}, "
              f"matchup found {d.def_pct.notna().mean():.1%}, SCREENED {d.SCREENED.sum()}")
        print("   pass rate per screen:", {k: round(float(d[k].mean()), 2) for k in ["F1", "F2", "F3", "F4", "F5", "F6"]})
        parts.append(d)
    D = pd.concat(parts, ignore_index=True)
    rows = []
    for (mk, sd), g in D.groupby(["market", "side"]):
        r = dict(market=mk, side=sd)
        for lab in ["2026", "2024-25"]:
            gg = g[g["sample"] == lab]
            for pre, sub in [("all", gg), ("scr", gg[gg.SCREENED])]:
                s = summarize(sub)
                r.update({f"{lab}_{pre}_{k}": v for k, v in s.items()})
            for yr in (2024, 2025):
                if lab == "2024-25":
                    s = summarize(gg[gg.SCREENED & (gg.season == yr)])
                    r.update({f"{yr}_scr_n": s["n"], f"{yr}_scr_roi": s["roi"]})
        a26, ah = r["2026_scr_roi"], r["2024-25_scr_roi"]
        n_ok = r["2026_scr_n"] >= 40 and r["2024-25_scr_n"] >= 40
        if n_ok and a26 > 0 and ah > 0 and r.get("2024_scr_roi", -1) > -0.02 and r.get("2025_scr_roi", -1) > -0.02:
            v = "HOLDS"
        elif n_ok and a26 > r["2026_all_roi"] and ah > r["2024-25_all_roi"]:
            v = "IMPROVES BUT NOT PROFITABLE"
        else:
            v = "NO EFFECT"
        r["verdict"] = v
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(out / "main_line_screens_v1.csv", index=False)
    D.to_csv(out / "main_line_screens_rows_v1.csv.gz", index=False, compression="gzip")
    pd.set_option("display.width", 260)
    show = ["market", "side", "2026_all_n", "2026_all_hit", "2026_all_roi", "2026_scr_n", "2026_scr_hit", "2026_scr_roi", "2026_scr_weeks_pos",
            "2024-25_all_n", "2024-25_all_roi", "2024-25_scr_n", "2024-25_scr_hit", "2024-25_scr_roi", "2024_scr_roi", "2025_scr_roi", "verdict"]
    print(T[show].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
