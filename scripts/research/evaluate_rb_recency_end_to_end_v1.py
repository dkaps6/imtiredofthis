#!/usr/bin/env python3
"""RB_CARRY_SHARE_RECENCY_END_TO_END_PREREGISTRATION_V1.md, exactly as frozen.

Offline reproduction of production's carry chain for every team-week:
  bayesian_v2 posterior (group mean x3, prior x min(pg,6), current x cg), clip [0,1]
  -> simulation_v2._top_n_shares(5) -> clip 0.95 -> scale to 0.95 if sum > 0.95
  -> x actual team rush attempts.
C0 = production inputs; C1 = rush_share_current replaced by 0.15*std + 0.85*last3 when cg >= 3.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nflreadpy as nfl

from scripts.modeling.bayesian_v2 import GROUP_STRENGTH, PRIOR_PLAYER_CAP
from scripts.simulation_v2 import _top_n_shares

A = 0.85
POS = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}


def player_games(season: int, ids: pd.DataFrame) -> pd.DataFrame:
    st = nfl.load_player_stats(seasons=[season]).to_pandas(); st = st[st.season_type == "REG"]
    sn = nfl.load_snap_counts(seasons=[season]).to_pandas(); sn = sn[sn.offense_snaps.fillna(0) > 0]
    sn = sn.merge(ids, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    g = sn[["gsis_id", "week", "team"]].drop_duplicates().merge(
        st[["player_id", "week", "team", "carries"]].rename(columns={"player_id": "gsis_id"}),
        on=["gsis_id", "week", "team"], how="left")
    g["carries"] = g.carries.fillna(0.0)
    tc = st.groupby(["team", "week"]).carries.sum().rename("team_carries").reset_index()
    g = g.merge(tc, on=["team", "week"], how="inner")
    g["season"] = season
    return g.sort_values("week")


def season_totals(pg: pd.DataFrame) -> pd.DataFrame:
    t = pg.groupby("gsis_id").agg(c=("carries", "sum"), t=("team_carries", "sum"), prior_games=("week", "size"))
    t["rush_share_prior"] = t.c / t.t.replace(0, np.nan)
    return t[["rush_share_prior", "prior_games"]]


def run_season(season: int, ids, weeks) -> pd.DataFrame:
    prev = season_totals(player_games(season - 1, ids))
    cur = player_games(season, ids)
    ro = nfl.load_rosters_weekly(seasons=[season]).to_pandas()
    ro = ro[(ro.status == "ACT") & ro.position.isin(POS)][["gsis_id", "week", "team", "position"]].dropna(subset=["gsis_id"])
    ij = nfl.load_injuries(seasons=[season]).to_pandas()
    out_ij = ij[ij.report_status.astype(str).str.lower().isin(["out", "doubtful"])][["gsis_id", "week"]].assign(out=1)
    team_c = cur.groupby(["team", "week"]).team_carries.first()
    rows = []
    for w in weeks:
        U = ro[ro.week == w].merge(out_ij[out_ij.week == w][["gsis_id", "out"]], on="gsis_id", how="left")
        U = U[U.out.isna()].drop(columns="out").drop_duplicates(["gsis_id", "team"])
        if U.empty:
            continue
        U["grp"] = U.position.map(POS)
        U = U.join(prev, on="gsis_id")
        hist = cur[cur.week < w]
        # current-season inputs with the player's CURRENT team only (PlayerForm current = this team)
        h = hist.merge(U[["gsis_id", "team"]], on=["gsis_id", "team"], how="inner")
        agg = h.groupby("gsis_id").agg(cc=("carries", "sum"), ct=("team_carries", "sum"), current_games=("week", "size"))
        l3 = h.groupby("gsis_id").tail(3).groupby("gsis_id").agg(l3c=("carries", "sum"), l3t=("team_carries", "sum"), l3n=("week", "size"))
        U = U.join(agg, on="gsis_id").join(l3, on="gsis_id")
        U["current_games"] = U.current_games.fillna(0)
        U["std"] = U.cc / U.ct.replace(0, np.nan)
        U["l3"] = np.where(U["l3n"] >= 3, U["l3c"] / U["l3t"].replace(0, np.nan), np.nan)
        U["prior_games"] = U.prior_games.fillna(0)
        gm = {}
        for grp, g in U.groupby("grp"):
            m = g.rush_share_prior.notna() & (g.prior_games > 0)
            gm[grp] = float(np.average(g.rush_share_prior[m], weights=g.prior_games[m])) if m.any() else float(g.rush_share_prior.median())
        U["gm"] = U.grp.map(gm)
        for arm in ("c0", "c1"):
            curv = U["std"] if arm == "c0" else np.where((U.current_games >= 3) & U.l3.notna(), (1 - A) * U["std"] + A * U.l3, U["std"])
            curv = pd.Series(curv, index=U.index)
            wg = GROUP_STRENGTH["rush_share"]
            wp = np.where(U.rush_share_prior.notna() & (U.prior_games > 0), np.minimum(U.prior_games, PRIOR_PLAYER_CAP["rush_share"]), 0.0)
            wc = np.where(curv.notna() & (U.current_games > 0), U.current_games, 0.0)
            num = wg * U.gm.fillna(0) + wp * U.rush_share_prior.fillna(0) + wc * curv.fillna(0)
            U[f"post_{arm}"] = np.clip(num / (wg + wp + wc), 0, 1)
            fin = np.zeros(len(U))
            for team, idx in U.groupby("team").groups.items():
                s = _top_n_shares(U.loc[idx, f"post_{arm}"].to_numpy(), 5)
                s = np.clip(s, 0, 0.95)
                if s.sum() > 0.95:
                    s = s * 0.95 / s.sum()
                fin[U.index.get_indexer(idx)] = s
            U[f"share_{arm}"] = fin
        U["team_carries"] = [team_c.get((t, w), np.nan) for t in U.team]
        played = cur[cur.week == w][["gsis_id", "team", "carries"]]
        S = U[U.grp == "RB"].merge(played, on=["gsis_id", "team"], how="inner")
        S["week"] = w; S["season"] = season
        rows.append(S)
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/research/rb_recency_end_to_end_v1.csv")
    a = ap.parse_args()
    ids = nfl.load_players().to_pandas()[["gsis_id", "pfr_id"]].dropna().drop_duplicates("pfr_id")
    D = pd.concat([run_season(2025, ids, range(4, 19)), run_season(2026, ids, range(4, 11))], ignore_index=True)
    D = D.dropna(subset=["team_carries"])
    for arm in ("c0", "c1"):
        D[f"proj_{arm}"] = D[f"share_{arm}"] * D.team_carries
    D["gid"] = D.season.astype(str) + "_" + D.week.astype(str) + "_" + D.team
    rng = np.random.default_rng(20261120)
    rows = []
    for season, x in D.groupby("season"):
        e0, e1 = (x.carries - x.proj_c0).abs(), (x.carries - x.proj_c1).abs()
        gg = pd.DataFrame({"gid": x.gid, "d": e0 - e1}).groupby("gid").d.agg(["sum", "size"])
        bs = [gg["sum"].to_numpy()[i].sum() / gg["size"].to_numpy()[i].sum()
              for i in (rng.integers(0, len(gg), len(gg)) for _ in range(2000))]
        lead = x.loc[x.groupby("gid").proj_c0.idxmax()]
        l0, l1 = (lead.carries - lead.proj_c0).abs(), (lead.carries - lead.proj_c1).abs()
        lg = pd.DataFrame({"gid": lead.gid, "d": l0 - l1}).groupby("gid").d.sum()
        lbs = [lg.to_numpy()[rng.integers(0, len(lg), len(lg))].mean() for _ in range(2000)]
        # shape: within-team normalized share over the team's RBs
        def shape(col):
            p = x[col] / x.groupby("gid")[col].transform("sum").replace(0, np.nan)
            t = x.carries / x.groupby("gid").carries.transform("sum").replace(0, np.nan)
            return (p - t).abs().mean()
        tm = x.groupby("gid")[["proj_c0", "proj_c1", "carries"]].sum().mean()
        rows.append(dict(season=season, n=len(x), mae_c0=e0.mean(), mae_c1=e1.mean(), gain=e0.mean() - e1.mean(),
                         ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                         lead_n=len(lead), lead_mae_c0=l0.mean(), lead_mae_c1=l1.mean(),
                         lead_gain_ci_lo=np.percentile(lbs, 2.5), lead_gain_ci_hi=np.percentile(lbs, 97.5),
                         lead_bias_c0=(lead.carries - lead.proj_c0).mean(), lead_bias_c1=(lead.carries - lead.proj_c1).mean(),
                         shape_mae_c0=shape("proj_c0"), shape_mae_c1=shape("proj_c1"),
                         team_rb_proj_c0=tm.proj_c0, team_rb_proj_c1=tm.proj_c1, team_rb_actual=tm.carries))
    R = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    D.to_csv(a.out.replace(".csv", "_rows.csv"), index=False); R.to_csv(a.out, index=False)
    pd.set_option("display.width", 250)
    print(R.round(4).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
