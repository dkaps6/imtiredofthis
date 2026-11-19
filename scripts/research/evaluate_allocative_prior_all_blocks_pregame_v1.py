#!/usr/bin/env python3
"""ALLOCATIVE_PRIOR_ALL_BLOCKS_PREGAME_PREREGISTRATION_V1.md, exactly as frozen.

Same universe and chain as evaluate_rb_allocative_prior_pregame_v1.py. The allocative
target alloc_P[K_P][k] * block_P is applied to every position block P in QB/RB/WR/TE.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nflreadpy as nfl

from scripts.research.evaluate_rb_allocative_prior_pregame_v1 import universe_rows, chain

BLOCKS = ("QB", "RB", "WR", "TE")
A_RECENCY, K_MAX, MIN_CELL = 0.85, 5, 20


def add_ranks(U: pd.DataFrame) -> pd.DataFrame:
    U = U.copy()
    U["proxy"] = np.where(U.current_games > 0, U["std"], U.rush_share_prior).astype(float)
    U["proxy"] = U.proxy.fillna(0.0)
    key = ["season", "week", "team", "grp"]
    U["K"] = U.groupby(key).gsis_id.transform("size").clip(upper=K_MAX)
    U["k"] = U.groupby(key).proxy.rank(ascending=False, method="first").clip(upper=K_MAX)
    return U


def fit_tables(T: pd.DataFrame):
    alloc, block = {}, {}
    for P in BLOCKS:
        r = T[T.grp == P]; cells = {}
        for _, g in r.groupby(["season", "week", "team"]):
            tot = g.carries.sum()
            if tot <= 0:
                continue
            for K, k, c in zip(g.K, g.k, g.carries):
                cells.setdefault((int(K), int(k)), []).append(c / tot)
        alloc[P] = {kk: float(np.mean(v)) for kk, v in cells.items() if len(v) >= MIN_CELL}
        tm = r.groupby(["season", "week", "team"]).agg(c=("carries", "sum"), tc=("team_carries", "first"))
        tm = tm[tm.tc > 0]
        block[P] = float((tm.c / tm.tc).mean())
    return alloc, block


def target(U, alloc, block, blocks):
    t = U.gm.copy()
    for P in blocks:
        m = U.grp == P
        v = [alloc[P].get((int(K), int(k)), np.nan) * block[P] for K, k in zip(U.loc[m, "K"], U.loc[m, "k"])]
        t.loc[m] = pd.Series(v, index=U.index[m]).fillna(U.loc[m, "gm"])
    return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/research/allocative_prior_all_blocks_pregame_v1.csv")
    a = ap.parse_args()
    ids = nfl.load_players().to_pandas()[["gsis_id", "pfr_id"]].dropna().drop_duplicates("pfr_id")
    allw = {s: add_ranks(universe_rows(s, ids, range(1, 19 if s < 2026 else 11))) for s in range(2019, 2027)}
    rng = np.random.default_rng(20261122)
    results, tables, scored = [], [], []
    for hold, train in [(2025, range(2019, 2025)), (2026, range(2019, 2026))]:
        alloc, block = fit_tables(pd.concat([allw[s] for s in train], ignore_index=True))
        for P in BLOCKS:
            for (K, k), v in sorted(alloc[P].items()):
                tables.append(dict(holdout=hold, block=P, K=K, k=k, within_block_share=v, block_mass=block[P]))
        U = allw[hold]; U = U[(U.week >= 4) & U.team_carries.notna()].reset_index(drop=True)
        rec = pd.Series(np.where((U.current_games >= 3) & U.l3.notna(), (1 - A_RECENCY) * U["std"] + A_RECENCY * U.l3, U["std"]), index=U.index)
        t1 = target(U, alloc, block, ("RB",)); t3 = target(U, alloc, block, BLOCKS)
        for c, (tg, cur) in {"0": (U.gm, U["std"]), "1": (t1, U["std"]), "3": (t3, U["std"]), "4": (t3, rec)}.items():
            U["p" + c] = chain(U, tg, cur) * U.team_carries
        U["gid"] = U.season.astype(str) + "_" + U.week.astype(str) + "_" + U.team

        def gain(x, c):
            e0, e1 = (x.carries - x.p0).abs(), (x.carries - x["p" + c]).abs()
            gg = pd.DataFrame({"gid": x.gid, "d": e0 - e1}).groupby("gid").d.agg(["sum", "size"])
            bs = [gg["sum"].to_numpy()[i].sum() / gg["size"].to_numpy()[i].sum()
                  for i in (rng.integers(0, len(gg), len(gg)) for _ in range(2000))]
            return dict(n=len(x), mae_c0=e0.mean(), mae_arm=e1.mean(), gain=e0.mean() - e1.mean(),
                        ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                        bias_c0=(x.carries - x.p0).mean(), bias_arm=(x.carries - x["p" + c]).mean())
        S = U[(U.grp == "RB") & U.played]; Q = U[(U.grp == "QB") & U.played]
        scored.append(U[U.played])
        for sub, x in [("all RBs", S), ("lead back (k=1)", S[S.k == 1]), ("committee (K>=3, k>=2)", S[(S.K >= 3) & (S.k >= 2)]), ("QBs who played", Q)]:
            for c in ("1", "3", "4"):
                results.append(dict(holdout=hold, subset=sub, arm="C" + c, **gain(x, c)))
        for P in ("RB", "QB"):
            tm = U[U.grp == P].groupby("gid")[["p0", "p1", "p3", "p4", "carries"]].sum()
            for c in ("1", "3", "4"):
                results.append(dict(holdout=hold, subset=f"team {P} mass |proj-actual|", arm="C" + c, n=len(tm),
                                    mae_c0=(tm.p0 - tm.carries).abs().mean(), mae_arm=(tm["p" + c] - tm.carries).abs().mean(),
                                    gain=(tm.p0 - tm.carries).abs().mean() - (tm["p" + c] - tm.carries).abs().mean(),
                                    bias_c0=(tm.carries - tm.p0).mean(), bias_arm=(tm.carries - tm["p" + c]).mean()))
    R = pd.DataFrame(results)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(a.out, index=False); pd.DataFrame(tables).to_csv(a.out.replace(".csv", "_tables.csv"), index=False)
    pd.concat(scored).to_csv(a.out.replace(".csv", "_rows.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(R.round(4).to_string(index=False))
    T = pd.DataFrame(tables).query("holdout==2026 and block in ['QB','RB']")
    print(T.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
