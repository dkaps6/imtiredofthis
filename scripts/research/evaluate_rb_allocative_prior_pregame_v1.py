#!/usr/bin/env python3
"""RB_ALLOCATIVE_PRIOR_PREGAME_PREREGISTRATION_V1.md, exactly as frozen.

Pregame universe (rosters ACT minus Out/Doubtful), production carry chain
(bayesian_v2 posterior -> top-5 -> 0.95 cap -> x actual team rushes).
C0 production RB target; C1 allocative target alloc[K][k] * rb_block for RBs;
C2 = C1 + recency (a = 0.85), reported only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nflreadpy as nfl

from scripts.modeling.bayesian_v2 import GROUP_STRENGTH, PRIOR_PLAYER_CAP
from scripts.simulation_v2 import _top_n_shares
from scripts.research.evaluate_rb_recency_end_to_end_v1 import player_games, season_totals

POS = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}
A_RECENCY, K_MAX, MIN_CELL = 0.85, 5, 20


def universe_rows(season: int, ids, weeks) -> pd.DataFrame:
    prev = season_totals(player_games(season - 1, ids))
    cur = player_games(season, ids)
    ro = nfl.load_rosters_weekly(seasons=[season]).to_pandas()
    ro = ro[(ro.status == "ACT") & ro.position.isin(POS)][["gsis_id", "week", "team", "position"]].dropna(subset=["gsis_id"])
    ij = nfl.load_injuries(seasons=[season]).to_pandas()
    out_ij = ij[ij.report_status.astype(str).str.lower().isin(["out", "doubtful"])][["gsis_id", "week"]].drop_duplicates().assign(out=1)
    tc = nfl.load_player_stats(seasons=[season]).to_pandas()
    tc = tc[tc.season_type == "REG"].groupby(["team", "week"]).carries.sum()
    rows = []
    for w in weeks:
        U = ro[ro.week == w].merge(out_ij[out_ij.week == w][["gsis_id", "out"]], on="gsis_id", how="left")
        U = U[U.out.isna()].drop(columns="out").drop_duplicates(["gsis_id", "team"])
        if U.empty:
            continue
        U["grp"] = U.position.map(POS)
        U = U.join(prev, on="gsis_id")
        h = cur[cur.week < w].merge(U[["gsis_id", "team"]], on=["gsis_id", "team"], how="inner")
        agg = h.groupby("gsis_id").agg(cc=("carries", "sum"), ct=("team_carries", "sum"), current_games=("week", "size"))
        l3 = h.groupby("gsis_id").tail(3).groupby("gsis_id").agg(l3c=("carries", "sum"), l3t=("team_carries", "sum"), l3n=("week", "size"))
        U = U.join(agg, on="gsis_id").join(l3, on="gsis_id")
        U["current_games"] = U.current_games.fillna(0); U["prior_games"] = U.prior_games.fillna(0)
        U["std"] = U.cc / U.ct.replace(0, np.nan)
        U["l3"] = np.where(U["l3n"] >= 3, U["l3c"] / U["l3t"].replace(0, np.nan), np.nan)
        gm = {}
        for grp, g in U.groupby("grp"):
            m = g.rush_share_prior.notna() & (g.prior_games > 0)
            gm[grp] = float(np.average(g.rush_share_prior[m], weights=g.prior_games[m])) if m.any() else float(g.rush_share_prior.median())
        U["gm"] = U.grp.map(gm)
        act = cur[cur.week == w][["gsis_id", "team", "carries"]]
        U = U.merge(act, on=["gsis_id", "team"], how="left")
        U["played"] = U.carries.notna(); U["carries"] = U.carries.fillna(0.0)
        U["team_carries"] = [tc.get((t, w), np.nan) for t in U.team]
        U["season"], U["week"] = season, w
        rows.append(U)
    return pd.concat(rows, ignore_index=True)


def add_rank(U: pd.DataFrame) -> pd.DataFrame:
    U = U.copy()
    rb = U.grp == "RB"
    U["proxy"] = np.where(U.current_games > 0, U["std"], U.rush_share_prior).astype(float)
    U["proxy"] = U.proxy.fillna(0.0)
    U["K"] = np.nan; U["k"] = np.nan
    r = U[rb]
    U.loc[rb, "K"] = r.groupby(["season", "week", "team"]).gsis_id.transform("size").clip(upper=K_MAX)
    U.loc[rb, "k"] = r.groupby(["season", "week", "team"]).proxy.rank(ascending=False, method="first")
    return U


def fit_tables(T: pd.DataFrame):
    r = T[(T.grp == "RB")]
    cells = {}
    for _, g in r.groupby(["season", "week", "team"]):
        tot = g.carries.sum()
        if tot <= 0:
            continue
        for K, k, c in zip(g.K, g.k, g.carries):
            cells.setdefault((int(K), int(min(k, K_MAX))), []).append(c / tot)
    alloc = {kk: float(np.mean(v)) for kk, v in cells.items() if len(v) >= MIN_CELL}
    tm = r.groupby(["season", "week", "team"]).agg(rbc=("carries", "sum"), tc=("team_carries", "first"))
    tm = tm[tm.tc > 0]
    return alloc, float((tm.rbc / tm.tc).mean())


def chain(U: pd.DataFrame, target: pd.Series, current: pd.Series) -> np.ndarray:
    wg = GROUP_STRENGTH["rush_share"]
    wp = np.where(U.rush_share_prior.notna() & (U.prior_games > 0), np.minimum(U.prior_games, PRIOR_PLAYER_CAP["rush_share"]), 0.0)
    wc = np.where(current.notna() & (U.current_games > 0), U.current_games, 0.0)
    post = np.clip((wg * target.fillna(0) + wp * U.rush_share_prior.fillna(0) + wc * current.fillna(0)) / (wg + wp + wc), 0, 1)
    fin = np.zeros(len(U))
    for _, idx in U.groupby(["season", "week", "team"]).groups.items():
        loc = U.index.get_indexer(idx)
        s = np.clip(_top_n_shares(post[loc], 5), 0, 0.95)
        if s.sum() > 0.95:
            s = s * 0.95 / s.sum()
        fin[loc] = s
    return fin


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/research/rb_allocative_prior_pregame_v1.csv")
    a = ap.parse_args()
    ids = nfl.load_players().to_pandas()[["gsis_id", "pfr_id"]].dropna().drop_duplicates("pfr_id")
    allw = {s: add_rank(universe_rows(s, ids, range(1, 19 if s < 2026 else 11))) for s in range(2019, 2027)}
    rng = np.random.default_rng(20261121)
    results, tables, scored = [], [], []
    for hold, train in [(2025, range(2019, 2025)), (2026, range(2019, 2026))]:
        alloc, rb_block = fit_tables(pd.concat([allw[s] for s in train], ignore_index=True))
        for (K, k), v in sorted(alloc.items()):
            tables.append(dict(holdout=hold, K=K, k=k, within_rb_share=v, rb_block=rb_block))
        U = allw[hold]; U = U[(U.week >= 4) & U.team_carries.notna()].reset_index(drop=True)
        rb = U.grp == "RB"
        alloc_t = pd.Series([alloc.get((int(K), int(min(k, K_MAX))), np.nan) * rb_block if r else np.nan
                             for K, k, r in zip(U.K.fillna(0), U.k.fillna(0), rb)], index=U.index)
        t1 = U.gm.where(~rb, alloc_t.fillna(U.gm))
        rec = pd.Series(np.where((U.current_games >= 3) & U.l3.notna(), (1 - A_RECENCY) * U["std"] + A_RECENCY * U.l3, U["std"]), index=U.index)
        U["s0"] = chain(U, U.gm, U["std"]); U["s1"] = chain(U, t1, U["std"]); U["s2"] = chain(U, t1, rec)
        for c in ("0", "1", "2"):
            U["p" + c] = U["s" + c] * U.team_carries
        S = U[rb & U.played].copy(); S["gid"] = S.season.astype(str) + "_" + S.week.astype(str) + "_" + S.team
        scored.append(S)

        def gain(x, c):
            e0, e1 = (x.carries - x.p0).abs(), (x.carries - x["p" + c]).abs()
            gg = pd.DataFrame({"gid": x.gid, "d": e0 - e1}).groupby("gid").d.agg(["sum", "size"])
            bs = [gg["sum"].to_numpy()[i].sum() / gg["size"].to_numpy()[i].sum()
                  for i in (rng.integers(0, len(gg), len(gg)) for _ in range(2000))]
            return e0.mean(), e1.mean(), e0.mean() - e1.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        for sub, x in [("all RBs", S), ("lead back (k=1)", S[S.k == 1]), ("committee (K>=3, k>=2)", S[(S.K >= 3) & (S.k >= 2)])]:
            for c in ("1", "2"):
                m0, m1, gn, lo, hi = gain(x, c)
                results.append(dict(holdout=hold, subset=sub, arm="C" + c, n=len(x), mae_c0=m0, mae_arm=m1, gain=gn, ci_lo=lo, ci_hi=hi,
                                    bias_c0=(x.carries - x.p0).mean(), bias_arm=(x.carries - x["p" + c]).mean()))
        tm = U[rb].groupby(["week", "team"]).agg(p0=("p0", "sum"), p1=("p1", "sum"), p2=("p2", "sum"), act=("carries", "sum"))
        results.append(dict(holdout=hold, subset="team RB mass |proj-actual|", arm="C0/C1/C2", n=len(tm),
                            mae_c0=(tm.p0 - tm.act).abs().mean(), mae_arm=(tm.p1 - tm.act).abs().mean(),
                            gain=(tm.p0 - tm.act).abs().mean() - (tm.p1 - tm.act).abs().mean(),
                            bias_c0=(tm.act - tm.p0).mean(), bias_arm=(tm.act - tm.p1).mean(), ci_lo=(tm.p2 - tm.act).abs().mean(), ci_hi=np.nan))
    R = pd.DataFrame(results)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(a.out, index=False); pd.DataFrame(tables).to_csv(a.out.replace(".csv", "_tables.csv"), index=False)
    pd.concat(scored).to_csv(a.out.replace(".csv", "_rows.csv"), index=False)
    pd.set_option("display.width", 250)
    print(R.round(4).to_string(index=False))
    print(pd.DataFrame(tables).query("holdout==2026").round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
