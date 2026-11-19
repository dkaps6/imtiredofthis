#!/usr/bin/env python3
"""ALLOCATIVE_PRIOR_PLUS_BUDGET_NORMALIZATION_PREREGISTRATION_V1.md, exactly as frozen.

Arms: C0 production; C3 all-block allocative (cap 0.95); N0 production priors + team stage
renormalized to 1.0; D1 allocative + renormalized to 1.0; D2 allocative + cap raised to 1.0.
Universe, Bayes weights, tables, holdouts unchanged from the all-block study.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nflreadpy as nfl

from scripts.modeling.bayesian_v2 import GROUP_STRENGTH, PRIOR_PLAYER_CAP
from scripts.simulation_v2 import _top_n_shares
from scripts.research.evaluate_rb_allocative_prior_pregame_v1 import universe_rows
from scripts.research.evaluate_allocative_prior_all_blocks_pregame_v1 import add_ranks, fit_tables, target, BLOCKS


def chain(U: pd.DataFrame, tg: pd.Series, cur: pd.Series, mode: str) -> np.ndarray:
    """mode: 'cap95' production; 'norm1' renormalize team sum to 1.0; 'cap1' scale down only above 1.0."""
    wg = GROUP_STRENGTH["rush_share"]
    wp = np.where(U.rush_share_prior.notna() & (U.prior_games > 0), np.minimum(U.prior_games, PRIOR_PLAYER_CAP["rush_share"]), 0.0)
    wc = np.where(cur.notna() & (U.current_games > 0), U.current_games, 0.0)
    post = np.clip((wg * tg.fillna(0) + wp * U.rush_share_prior.fillna(0) + wc * cur.fillna(0)) / (wg + wp + wc), 0, 1)
    fin = np.zeros(len(U))
    for _, idx in U.groupby(["season", "week", "team"]).groups.items():
        loc = U.index.get_indexer(idx)
        s = np.clip(_top_n_shares(post[loc], 5), 0, 0.95)
        t = s.sum()
        if mode == "cap95" and t > 0.95:
            s = s * 0.95 / t
        elif mode == "cap1" and t > 1.0:
            s = s / t
        elif mode == "norm1" and t > 0:
            s = s / t
        fin[loc] = s
    return fin


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/research/allocative_prior_budget_normalization_v1.csv")
    a = ap.parse_args()
    ids = nfl.load_players().to_pandas()[["gsis_id", "pfr_id"]].dropna().drop_duplicates("pfr_id")
    allw = {s: add_ranks(universe_rows(s, ids, range(1, 19 if s < 2026 else 11))) for s in range(2019, 2027)}
    rng = np.random.default_rng(20261123)
    arms = ("C3", "N0", "D1", "D2")
    results, scored = [], []
    for hold, train in [(2025, range(2019, 2025)), (2026, range(2019, 2026))]:
        alloc, block = fit_tables(pd.concat([allw[s] for s in train], ignore_index=True))
        U = allw[hold]; U = U[(U.week >= 4) & U.team_carries.notna()].reset_index(drop=True)
        t3 = target(U, alloc, block, BLOCKS)
        spec = {"C0": (U.gm, "cap95"), "C3": (t3, "cap95"), "N0": (U.gm, "norm1"), "D1": (t3, "norm1"), "D2": (t3, "cap1")}
        for arm, (tg, mode) in spec.items():
            U["p_" + arm] = chain(U, tg, U["std"], mode) * U.team_carries
        U["gid"] = U.season.astype(str) + "_" + U.week.astype(str) + "_" + U.team

        def gain(x, arm):
            e0, e1 = (x.carries - x.p_C0).abs(), (x.carries - x["p_" + arm]).abs()
            gg = pd.DataFrame({"gid": x.gid, "d": e0 - e1}).groupby("gid").d.agg(["sum", "size"])
            bs = [gg["sum"].to_numpy()[i].sum() / gg["size"].to_numpy()[i].sum()
                  for i in (rng.integers(0, len(gg), len(gg)) for _ in range(2000))]
            return dict(n=len(x), mae_c0=e0.mean(), mae_arm=e1.mean(), gain=e0.mean() - e1.mean(),
                        ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                        bias_c0=(x.carries - x.p_C0).mean(), bias_arm=(x.carries - x["p_" + arm]).mean())
        S = U[(U.grp == "RB") & U.played]; Q = U[(U.grp == "QB") & U.played]
        scored.append(U[U.played])
        for sub, x in [("all RBs", S), ("lead back (k=1)", S[S.k == 1]), ("committee (K>=3, k>=2)", S[(S.K >= 3) & (S.k >= 2)]), ("QBs who played", Q)]:
            for arm in arms:
                results.append(dict(holdout=hold, subset=sub, arm=arm, **gain(x, arm)))
        for P in ("RB", "QB"):
            tm = U[U.grp == P].groupby("gid")[[f"p_{x}" for x in ("C0",) + arms] + ["carries"]].sum()
            for arm in arms:
                e0 = (tm.p_C0 - tm.carries).abs(); e1 = (tm["p_" + arm] - tm.carries).abs()
                d = (e0 - e1).to_numpy()
                bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                results.append(dict(holdout=hold, subset=f"team {P} mass |proj-actual|", arm=arm, n=len(tm),
                                    mae_c0=e0.mean(), mae_arm=e1.mean(), gain=e0.mean() - e1.mean(),
                                    ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                                    bias_c0=(tm.carries - tm.p_C0).mean(), bias_arm=(tm.carries - tm["p_" + arm]).mean()))
    R = pd.DataFrame(results)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(a.out, index=False); pd.concat(scored).to_csv(a.out.replace(".csv", "_rows.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(R.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
