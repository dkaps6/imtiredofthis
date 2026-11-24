#!/usr/bin/env python3
"""D1_RUSH_YARDS_VS_MARKET_PREREGISTRATION_V1.md, exactly as frozen.

projection = share (C0 or D1, frozen 2026-holdout tables) x team rolling-8 carries x shrunk YPC.
Graded against FanDuel RB rush_yards main lines, 2026 weeks 6-10.
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
import nflreadpy as nfl

from scripts.research.evaluate_rb_allocative_prior_pregame_v1 import universe_rows
from scripts.research.evaluate_allocative_prior_all_blocks_pregame_v1 import add_ranks, target, BLOCKS
from scripts.research.evaluate_allocative_prior_budget_normalization_v1 import chain

OWN_YPC_W, WINDOW, WEEKS = 0.21, 8, range(6, 11)
SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm(name: str) -> str:
    t = [re.sub(r"[^a-z]", "", w) for w in str(name).lower().replace("-", " ").split()]
    t = [w for w in t if w and w not in SUFFIX]
    return " ".join(t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/research/d1_rush_yards_vs_market_v1.csv")
    a = ap.parse_args()
    ids = nfl.load_players().to_pandas()[["gsis_id", "pfr_id"]].dropna().drop_duplicates("pfr_id")
    T = pd.read_csv("data/research/allocative_prior_all_blocks_pregame_v1_tables.csv").query("holdout == 2026")
    alloc = {P: {(int(r.K), int(r.k)): r.within_block_share for r in T[T.block == P].itertuples()} for P in BLOCKS}
    block = {P: (T[T.block == P].block_mass.iloc[0] if (T.block == P).any() else 0.0) for P in BLOCKS}
    U = add_ranks(universe_rows(2026, ids, WEEKS)).reset_index(drop=True)
    U["sh_R0"] = chain(U, U.gm, U["std"], "cap95"); U["sh_R1"] = chain(U, target(U, alloc, block, BLOCKS), U["std"], "norm1")

    st = nfl.load_player_stats(seasons=[2025, 2026]).to_pandas(); st = st[st.season_type == "REG"].copy()
    st["t"] = (st.season - 2025) * 100 + st.week
    tc = st.groupby(["team", "t"]).carries.sum().reset_index().sort_values("t")
    tc["roll"] = tc.groupby("team").carries.transform(lambda s: s.shift().rolling(WINDOW, min_periods=3).mean())
    rows = []
    for w in WEEKS:
        t = 100 + w
        pr = st[st.t < t]; rb = pr[pr.position == "RB"]
        lg = rb.rushing_yards.sum() / rb.carries.sum()
        own = pr.groupby("player_id").agg(y=("rushing_yards", "sum"), c=("carries", "sum"))
        own["ypc_own"] = own.y / own.c.replace(0, np.nan)
        x = U[U.week == w].join(own[["ypc_own"]], on="gsis_id")
        x["ypc"] = (1 - OWN_YPC_W) * lg + OWN_YPC_W * x.ypc_own.fillna(lg)
        roll = tc[tc.t == t].set_index("team").roll
        x["team_rush"] = x.team.map(roll)
        rows.append(x)
    X = pd.concat(rows)
    for r in ("R0", "R1"):
        X["proj_" + r] = X["sh_" + r] * X.team_rush * X.ypc
    # map board names to gsis via that week's roster
    ro = nfl.load_rosters_weekly(seasons=[2026]).to_pandas()[["gsis_id", "week", "team", "full_name"]].dropna()
    ro["nm"] = ro.full_name.map(norm)
    b = pd.read_csv("data/market_track_record/boards/2026_wk06.csv")
    b = pd.concat([pd.read_csv(f"data/market_track_record/boards/2026_wk{w:02d}.csv") for w in WEEKS])
    b = b[(b.book == "fanduel") & (b.market == "rush_yards") & (b.side == "OVER")].copy()
    b["team"] = b.team.replace({"LAR": "LA"}); b["nm"] = b.player.map(norm)
    b = b.merge(ro, on=["nm", "week", "team"], how="left")
    print(f"board RB/QB rush offers {len(b)}, mapped to gsis {b.gsis_id.notna().sum()}")
    g = b.merge(X[X.grp == "RB"][["gsis_id", "week", "team", "proj_R0", "proj_R1", "sh_R0", "sh_R1", "played"]], on=["gsis_id", "week", "team"], how="inner")
    act = st[st.season == 2026][["player_id", "week", "rushing_yards"]].rename(columns={"player_id": "gsis_id"})
    g = g.merge(act, on=["gsis_id", "week"], how="left")
    g = g[g.played].copy(); g["actual"] = g.rushing_yards.fillna(0.0)
    g = g.dropna(subset=["proj_R0", "proj_R1", "vegas_line", "model_proj"])
    print(f"scored RB offers {len(g)}, games {g.event_id.nunique()}")
    rng = np.random.default_rng(20261124)
    e0, e1 = (g.actual - g.proj_R0).abs(), (g.actual - g.proj_R1).abs()
    gg = pd.DataFrame({"e": g.event_id, "d": e0 - e1}).groupby("e").d.agg(["sum", "size"])
    bs = [gg["sum"].to_numpy()[i].sum() / gg["size"].to_numpy()[i].sum() for i in (rng.integers(0, len(gg), len(gg)) for _ in range(4000))]
    rows = []
    for nm, col in [("R0 production share", "proj_R0"), ("R1 D1 share", "proj_R1"), ("PRISM model_proj", "model_proj"), ("FanDuel line", "vegas_line")]:
        r = dict(projection=nm, n=len(g), mae=(g.actual - g[col]).abs().mean(), bias=(g.actual - g[col]).mean())
        if col != "vegas_line":
            over = g[col] > g.vegas_line; hit = np.where(over, g.actual > g.vegas_line, g.actual < g.vegas_line)
            price = np.where(over, g.vegas_over_odds, g.vegas_under_odds)
            dec = np.where(price > 0, price / 100, 100 / np.abs(price))
            r.update(side_acc=hit.mean(), roi=np.where(hit, dec, -1.0).mean(), share_over=over.mean())
        rows.append(r)
    R = pd.DataFrame(rows)
    summ = dict(primary_gain=e0.mean() - e1.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                mde_side_pp=round(100 * 2.8 * np.sqrt(0.25 / len(g)), 1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(a.out, index=False); g.to_csv(a.out.replace(".csv", "_rows.csv"), index=False)
    pd.set_option("display.width", 220)
    print(R.round(4).to_string(index=False)); print(summ)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
