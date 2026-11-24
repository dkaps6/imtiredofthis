#!/usr/bin/env python3
"""TARGET_ALLOCATIVE_PRIOR_BUDGET_PREREGISTRATION_V1.md, exactly as frozen.

Target chain: bayesian_v2 tgt_share posterior -> rules multiplier (TE 1.265, RB/FB 1.20) ->
simulation_v2._sharpen_wr_target_shares (M37) -> per-player clip 0.95 -> team cap 0.95 (C0/T3)
or renormalize to 1.0 (TD1/TN0) -> x actual team targets.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nflreadpy as nfl

from scripts.modeling.bayesian_v2 import GROUP_STRENGTH, PRIOR_PLAYER_CAP
from scripts.simulation_v2 import _sharpen_wr_target_shares

POS = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}
RULES_MULT = {"TE": 1.265, "RB": 1.20, "WR": 1.0, "QB": 1.0}
BLOCKS = ("QB", "RB", "WR", "TE")
K_MAX, MIN_CELL = 5, 20


def player_games(season, ids):
    st = nfl.load_player_stats(seasons=[season]).to_pandas(); st = st[st.season_type == "REG"]
    sn = nfl.load_snap_counts(seasons=[season]).to_pandas(); sn = sn[sn.offense_snaps.fillna(0) > 0]
    sn = sn.merge(ids, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    g = sn[["gsis_id", "week", "team"]].drop_duplicates().merge(
        st[["player_id", "week", "team", "targets"]].rename(columns={"player_id": "gsis_id"}), on=["gsis_id", "week", "team"], how="left")
    g["targets"] = g.targets.fillna(0.0)
    tt = st.groupby(["team", "week"]).targets.sum().rename("team_targets").reset_index()
    return g.merge(tt, on=["team", "week"], how="inner").sort_values("week")


def universe_rows(season, ids, weeks):
    pv = player_games(season - 1, ids)
    prev = pv.groupby("gsis_id").agg(c=("targets", "sum"), t=("team_targets", "sum"), prior_games=("week", "size"))
    prev["tgt_share_prior"] = prev.c / prev.t.replace(0, np.nan); prev = prev[["tgt_share_prior", "prior_games"]]
    cur = player_games(season, ids)
    ro = nfl.load_rosters_weekly(seasons=[season]).to_pandas()
    ro = ro[(ro.status == "ACT") & ro.position.isin(POS)][["gsis_id", "week", "team", "position"]].dropna(subset=["gsis_id"])
    ij = nfl.load_injuries(seasons=[season]).to_pandas()
    out_ij = ij[ij.report_status.astype(str).str.lower().isin(["out", "doubtful"])][["gsis_id", "week"]].drop_duplicates().assign(out=1)
    st = nfl.load_player_stats(seasons=[season]).to_pandas(); tt = st[st.season_type == "REG"].groupby(["team", "week"]).targets.sum()
    rows = []
    for w in weeks:
        U = ro[ro.week == w].merge(out_ij[out_ij.week == w][["gsis_id", "out"]], on="gsis_id", how="left")
        U = U[U.out.isna()].drop(columns="out").drop_duplicates(["gsis_id", "team"])
        if U.empty:
            continue
        U["grp"] = U.position.map(POS); U = U.join(prev, on="gsis_id")
        h = cur[cur.week < w].merge(U[["gsis_id", "team"]], on=["gsis_id", "team"], how="inner")
        agg = h.groupby("gsis_id").agg(cc=("targets", "sum"), ct=("team_targets", "sum"), current_games=("week", "size"))
        U = U.join(agg, on="gsis_id")
        U["current_games"] = U.current_games.fillna(0); U["prior_games"] = U.prior_games.fillna(0)
        U["std"] = U.cc / U.ct.replace(0, np.nan)
        gm = {}
        for grp, g in U.groupby("grp"):
            m = g.tgt_share_prior.notna() & (g.prior_games > 0)
            gm[grp] = float(np.average(g.tgt_share_prior[m], weights=g.prior_games[m])) if m.any() else float(g.tgt_share_prior.median())
        U["gm"] = U.grp.map(gm)
        act = cur[cur.week == w][["gsis_id", "team", "targets"]]
        U = U.merge(act, on=["gsis_id", "team"], how="left")
        U["played"] = U.targets.notna(); U["targets"] = U.targets.fillna(0.0)
        U["team_targets"] = [tt.get((t, w), np.nan) for t in U.team]
        U["season"], U["week"] = season, w
        rows.append(U)
    U = pd.concat(rows, ignore_index=True)
    U["proxy"] = np.where(U.current_games > 0, U["std"], U.tgt_share_prior).astype(float)
    U["proxy"] = U.proxy.fillna(0.0)
    key = ["season", "week", "team", "grp"]
    U["K"] = U.groupby(key).gsis_id.transform("size").clip(upper=K_MAX)
    U["k"] = U.groupby(key).proxy.rank(ascending=False, method="first").clip(upper=K_MAX)
    return U


def fit_tables(T):
    alloc, block = {}, {}
    for P in BLOCKS:
        r = T[T.grp == P]; cells = {}
        for _, g in r.groupby(["season", "week", "team"]):
            tot = g.targets.sum()
            if tot <= 0:
                continue
            for K, k, c in zip(g.K, g.k, g.targets):
                cells.setdefault((int(K), int(k)), []).append(c / tot)
        alloc[P] = {kk: float(np.mean(v)) for kk, v in cells.items() if len(v) >= MIN_CELL}
        tm = r.groupby(["season", "week", "team"]).agg(c=("targets", "sum"), tc=("team_targets", "first"))
        tm = tm[tm.tc > 0]; block[P] = float((tm.c / tm.tc).mean())
    return alloc, block


def target(U, alloc, block):
    t = U.gm.copy()
    for P in BLOCKS:
        m = U.grp == P
        v = [alloc[P].get((int(K), int(k)), np.nan) * block[P] for K, k in zip(U.loc[m, "K"], U.loc[m, "k"])]
        t.loc[m] = pd.Series(v, index=U.index[m]).fillna(U.loc[m, "gm"])
    return t


def chain(U, tg, mode):
    wg = GROUP_STRENGTH["tgt_share"]
    wp = np.where(U.tgt_share_prior.notna() & (U.prior_games > 0), np.minimum(U.prior_games, PRIOR_PLAYER_CAP["tgt_share"]), 0.0)
    wc = np.where(U["std"].notna() & (U.current_games > 0), U.current_games, 0.0)
    post = np.clip((wg * tg.fillna(0) + wp * U.tgt_share_prior.fillna(0) + wc * U["std"].fillna(0)) / (wg + wp + wc), 0, 1)
    post = post * U.grp.map(RULES_MULT).to_numpy()
    fin = np.zeros(len(U))
    for _, idx in U.groupby(["season", "week", "team"]).groups.items():
        loc = U.index.get_indexer(idx)
        s = _sharpen_wr_target_shares(U.loc[idx, ["position"]].reset_index(drop=True), post[loc])
        s = np.clip(s, 0, 0.95); t = s.sum()
        if mode == "cap95" and t > 0.95:
            s = s * 0.95 / t
        elif mode == "norm1" and t > 0:
            s = s / t
        fin[loc] = s
    return fin


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/research/target_allocative_prior_budget_v1.csv")
    a = ap.parse_args()
    ids = nfl.load_players().to_pandas()[["gsis_id", "pfr_id"]].dropna().drop_duplicates("pfr_id")
    allw = {s: universe_rows(s, ids, range(1, 19 if s < 2026 else 11)) for s in range(2019, 2027)}
    rng = np.random.default_rng(20261125)
    results, tables, scored = [], [], []
    for hold, train in [(2025, range(2019, 2025)), (2026, range(2019, 2026))]:
        alloc, block = fit_tables(pd.concat([allw[s] for s in train], ignore_index=True))
        for P in BLOCKS:
            for (K, k), v in sorted(alloc[P].items()):
                tables.append(dict(holdout=hold, block=P, K=K, k=k, within_block_share=v, block_mass=block[P]))
        U = allw[hold]; U = U[(U.week >= 4) & U.team_targets.notna()].reset_index(drop=True)
        t3 = target(U, alloc, block)
        for arm, (tg, mode) in {"C0": (U.gm, "cap95"), "T3": (t3, "cap95"), "TD1": (t3, "norm1"), "TN0": (U.gm, "norm1")}.items():
            U["p_" + arm] = chain(U, tg, mode) * U.team_targets
        U["gid"] = U.season.astype(str) + "_" + U.week.astype(str) + "_" + U.team

        def gain(x, arm):
            e0, e1 = (x.targets - x.p_C0).abs(), (x.targets - x["p_" + arm]).abs()
            gg = pd.DataFrame({"gid": x.gid, "d": e0 - e1}).groupby("gid").d.agg(["sum", "size"])
            bs = [gg["sum"].to_numpy()[i].sum() / gg["size"].to_numpy()[i].sum() for i in (rng.integers(0, len(gg), len(gg)) for _ in range(2000))]
            return dict(n=len(x), mae_c0=e0.mean(), mae_arm=e1.mean(), gain=e0.mean() - e1.mean(), ci_lo=np.percentile(bs, 2.5),
                        ci_hi=np.percentile(bs, 97.5), bias_c0=(x.targets - x.p_C0).mean(), bias_arm=(x.targets - x["p_" + arm]).mean())
        P_ = U[U.played]; scored.append(P_)
        WT = P_[P_.grp.isin(["WR", "TE"])]
        subs = [("WR+TE (primary)", WT), ("WR1 (pregame rank 1 WR)", P_[(P_.grp == "WR") & (P_.k == 1)]),
                ("depth WR/TE (rank>=3)", WT[WT.k >= 3]), ("WR all", P_[P_.grp == "WR"]), ("TE all", P_[P_.grp == "TE"]), ("RB all", P_[P_.grp == "RB"])]
        for sub, x in subs:
            for arm in ("T3", "TD1", "TN0"):
                results.append(dict(holdout=hold, subset=sub, arm=arm, **gain(x, arm)))
        tm = U.groupby("gid")[["p_C0", "p_T3", "p_TD1", "p_TN0", "targets"]].sum()
        for arm in ("T3", "TD1", "TN0"):
            e0 = (tm.p_C0 - tm.targets).abs(); e1 = (tm["p_" + arm] - tm.targets).abs(); d = (e0 - e1).to_numpy()
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            results.append(dict(holdout=hold, subset="team target mass |proj-actual|", arm=arm, n=len(tm), mae_c0=e0.mean(), mae_arm=e1.mean(),
                                gain=e0.mean() - e1.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                                bias_c0=(tm.targets - tm.p_C0).mean(), bias_arm=(tm.targets - tm["p_" + arm]).mean()))
    R = pd.DataFrame(results)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(a.out, index=False); pd.DataFrame(tables).to_csv(a.out.replace(".csv", "_tables.csv"), index=False)
    pd.concat(scored).to_csv(a.out.replace(".csv", "_rows.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(R.round(4).to_string(index=False))
    print(pd.DataFrame(tables).query("holdout==2026 and block=='WR'").round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
