#!/usr/bin/env python3
"""TD1_REC_YARDS_VS_MARKET_PREREGISTRATION_V1.md, exactly as frozen."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nflreadpy as nfl

from scripts.research.evaluate_target_allocative_prior_budget_v1 import universe_rows, target, chain, BLOCKS
from scripts.research.evaluate_d1_rush_yards_vs_market_v1 import norm

WEEKS, WINDOW = range(6, 11), 8
OWN_YPT_W, OWN_CR_W = 0.27, 0.20


def grade(g, col, actual):
    over = g[col] > g.vegas_line
    hit = np.where(over, g[actual] > g.vegas_line, g[actual] < g.vegas_line)
    price = np.where(over, g.vegas_over_odds, g.vegas_under_odds)
    dec = np.where(price > 0, price / 100, 100 / np.abs(price))
    return hit.mean(), np.where(hit, dec, -1.0).mean(), over.mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/research/td1_rec_yards_vs_market_v1.csv")
    a = ap.parse_args()
    ids = nfl.load_players().to_pandas()[["gsis_id", "pfr_id"]].dropna().drop_duplicates("pfr_id")
    T = pd.read_csv("data/research/target_allocative_prior_budget_v1_tables.csv").query("holdout == 2026")
    alloc = {P: {(int(r.K), int(r.k)): r.within_block_share for r in T[T.block == P].itertuples()} for P in BLOCKS}
    block = {P: (T[T.block == P].block_mass.iloc[0] if (T.block == P).any() else 0.0) for P in BLOCKS}
    U = universe_rows(2026, ids, WEEKS).reset_index(drop=True)
    U["sh_C0"] = chain(U, U.gm, "cap95"); U["sh_TD1"] = chain(U, target(U, alloc, block), "norm1")
    st = nfl.load_player_stats(seasons=[2025, 2026]).to_pandas(); st = st[st.season_type == "REG"].copy()
    st["t"] = (st.season - 2025) * 100 + st.week
    tt = st.groupby(["team", "t"]).targets.sum().reset_index().sort_values("t")
    tt["roll"] = tt.groupby("team").targets.transform(lambda s: s.shift().rolling(WINDOW, min_periods=3).mean())
    rows = []
    for w in WEEKS:
        t = 100 + w; pr = st[st.t < t]
        lg = pr.groupby("position").agg(y=("receiving_yards", "sum"), tg=("targets", "sum"), r=("receptions", "sum"))
        lg["ypt"] = lg.y / lg.tg; lg["cr"] = lg.r / lg.tg
        own = pr.groupby("player_id").agg(y=("receiving_yards", "sum"), tg=("targets", "sum"), r=("receptions", "sum"))
        own["ypt_o"] = own.y / own.tg.replace(0, np.nan); own["cr_o"] = own.r / own.tg.replace(0, np.nan)
        x = U[U.week == w].join(own[["ypt_o", "cr_o"]], on="gsis_id")
        pos = x.grp.replace({"RB": "RB"})
        lypt = pos.map(lg.ypt); lcr = pos.map(lg.cr)
        x["ypt"] = (1 - OWN_YPT_W) * lypt + OWN_YPT_W * x.ypt_o.fillna(lypt)
        x["cr"] = (1 - OWN_CR_W) * lcr + OWN_CR_W * x.cr_o.fillna(lcr)
        x["team_tgt"] = x.team.map(tt[tt.t == t].set_index("team").roll)
        rows.append(x)
    X = pd.concat(rows)
    for arm in ("C0", "TD1"):
        X["tg_" + arm] = X["sh_" + arm] * X.team_tgt
        X["ry_" + arm] = X["tg_" + arm] * X.ypt; X["rec_" + arm] = X["tg_" + arm] * X.cr
    ro = nfl.load_rosters_weekly(seasons=[2026]).to_pandas()[["gsis_id", "week", "team", "full_name"]].dropna()
    ro["nm"] = ro.full_name.map(norm)
    b = pd.concat([pd.read_csv(f"data/market_track_record/boards/2026_wk{w:02d}.csv") for w in WEEKS])
    b = b[(b.book == "fanduel") & b.market.isin(["rec_yards", "receptions"]) & (b.side == "OVER")].copy()
    b["team"] = b.team.replace({"LAR": "LA"}); b["nm"] = b.player.map(norm)
    b = b.merge(ro, on=["nm", "week", "team"], how="left")
    keep = ["gsis_id", "week", "team", "grp", "k", "played"] + [c for c in X.columns if c[:3] in ("ry_", "rec", "tg_")]
    g = b.merge(X[keep], on=["gsis_id", "week", "team"], how="inner")
    act = st[st.season == 2026][["player_id", "week", "receiving_yards", "receptions"]].rename(columns={"player_id": "gsis_id", "receptions": "rec_act", "receiving_yards": "ry_act"})
    g = g.merge(act, on=["gsis_id", "week"], how="left"); g = g[g.played & g.grp.isin(["WR", "TE", "RB"])].copy()
    g[["ry_act", "rec_act"]] = g[["ry_act", "rec_act"]].fillna(0.0)
    rng = np.random.default_rng(20261126)
    out = []
    for mk, a0, a1, act in [("rec_yards", "ry_C0", "ry_TD1", "ry_act"), ("receptions", "rec_C0", "rec_TD1", "rec_act")]:
        m = g[(g.market == mk)].dropna(subset=[a0, a1, "vegas_line", "model_proj"])
        for sub, x in [("all", m), ("WR1 (pregame rank 1 WR)", m[(m.grp == "WR") & (m.k == 1)])]:
            e0, e1 = (x[act] - x[a0]).abs(), (x[act] - x[a1]).abs()
            gg = pd.DataFrame({"e": x.event_id, "d": e0 - e1}).groupby("e").d.agg(["sum", "size"])
            bs = [gg["sum"].to_numpy()[i].sum() / gg["size"].to_numpy()[i].sum() for i in (rng.integers(0, len(gg), len(gg)) for _ in range(4000))]
            row = dict(market=mk, subset=sub, n=len(x), gain=e0.mean() - e1.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                       mde_side_pp=round(100 * 2.8 * np.sqrt(0.25 / max(len(x), 1)), 1))
            for nm_, col in [("C0", a0), ("TD1", a1), ("PRISM", "model_proj"), ("line", "vegas_line")]:
                r = x[act] - x[col]
                row.update({f"mae_{nm_}": r.abs().mean(), f"bias_{nm_}": r.mean(), f"med_{nm_}": r.median()})
                if nm_ != "line":
                    acc, roi, ov = grade(x, col, act); row.update({f"side_{nm_}": acc, f"roi_{nm_}": roi, f"over_{nm_}": ov})
            out.append(row)
    R = pd.DataFrame(out)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(a.out, index=False); g.to_csv(a.out.replace(".csv", "_rows.csv"), index=False)
    pd.set_option("display.width", 250)
    print(R.round(3).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
