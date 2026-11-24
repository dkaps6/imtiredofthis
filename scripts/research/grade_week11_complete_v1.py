#!/usr/bin/env python3
"""Week 11 2026 complete model + slate grading (companion to grade_week11_2026_v1.py).

Every priced main line (FanDuel + DraftKings, TNF through MNF) graded against nflverse:
  1. PRISM's own side (edge_pct > 0), one bet per offer, by book / market / position / side /
     edge size / game window;
  2. projection accuracy vs the market by market x position, for every model component
     (mc, ml, state, ensemble, final), with bias, star compression and the largest misses;
  3. calibration of the model's stated probability;
  4. the FanDuel alt board by market x position.
A row is void if the player took no offensive snap.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import scripts.research.grade_week11_2026_v1 as G

OUT = Path("data/market_track_record/graded")
POSG = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}


def window(event_id: str, kick: dict) -> str:
    t = kick.get(event_id)
    if t is None:
        return "?"
    t = pd.Timestamp(t).tz_convert("America/New_York")
    return {3: "TNF", 6: "SUN", 0: "MNF"}.get(t.dayofweek, "?") if t.hour < 20 or t.dayofweek != 6 else "SNF"


def main() -> int:
    G.ACT, G.SNAP, G.POS = G.actuals()
    pd.set_option("display.width", 260); pd.set_option("display.max_colwidth", 80)
    b = pd.read_csv("data/market_track_record/boards/2026_wk11.csv", low_memory=False)
    b = b[b.market.isin(list(G.STAT)) & b.vegas_odds.notna()].copy()
    g = b.apply(lambda r: G.grade(r.player, r.market, r.side, float(r.vegas_line)), axis=1, result_type="expand")
    b[["actual", "result", "snap"]] = g
    n_all = len(b)
    b = b[b.result.isin(["WIN", "LOSS"])].copy()
    b["win"] = (b.result == "WIN").astype(int)
    b["pnl"] = np.where(b.win == 1, G.dec(b.vegas_odds), -1.0)
    b["pos"] = b.player.map(lambda n: POSG.get(G.POS.get(G.key(n), ""), "?"))
    go = pd.concat([pd.read_csv("data/market_track_record/game_odds/2026_wk11.csv"), pd.read_csv("data/market_track_record/game_odds/2026_wk11_part2.csv")])
    kick = go.drop_duplicates("event_id").set_index("event_id").commence_time.to_dict()
    b["window"] = b.event_id.map(lambda e: window(e, kick))
    b.to_csv(OUT / "w11_2026_mainline_board_graded_complete_v1.csv", index=False)
    print(f"COVERAGE: {n_all} priced rows, {len(b)} graded (voids/pushes dropped {n_all - len(b)}); games {b.event_id.nunique()}; "
          f"players {b.player.nunique()}; books {b.book.value_counts().to_dict()}")

    # 1. PRISM's own side
    side = b[b.edge_pct > 0].copy()
    def t(d, by):
        return d.groupby(by).agg(bets=("win", "size"), hit=("win", "mean"), roi=("pnl", "mean")).round(3)
    print("\n== 1. PRISM-PICKED SIDE (edge_pct > 0), one bet per offer")
    print(f"coin flip (every side): n={len(b)} roi={b.pnl.mean():+.3f} | PRISM side: n={len(side)} hit={side.win.mean():.3f} roi={side.pnl.mean():+.3f}")
    for by in ["book", "market", "pos", "side", "window"]:
        print(t(side, by).to_string()); print()
    side["edge_b"] = pd.cut(side.edge_pct, [0, 0.05, 0.10, 0.20, 1.0])
    print(t(side, "edge_b").to_string())
    print("\nPRISM side by position x market (FanDuel):"); print(t(side[side.book == "fanduel"], ["pos", "market"]).to_string())
    side.to_csv(OUT / "w11_2026_prism_side_graded_v1.csv", index=False)

    # 2. projection accuracy, one row per player-market (FanDuel line)
    f = b[(b.book == "fanduel") & (b.side == "OVER")].drop_duplicates(["player", "market"]).copy()
    for c in ["model_proj", "mc_proj", "ml_proj", "state_proj", "ensemble_proj"]:
        f[f"err_{c}"] = (f[c] - f.actual).abs()
    f["err_vegas"] = (f.vegas_line - f.actual).abs()
    f["closer"] = (f.err_model_proj < f.err_vegas).astype(int)
    f["right_side"] = (np.sign(f.model_proj - f.vegas_line) == np.sign(f.actual - f.vegas_line)).astype(int)
    rows = []
    for (mk, p), d in list(f.groupby(["market", "pos"])) + [((mk, "ALL"), d) for mk, d in f.groupby("market")]:
        if len(d) < 5:
            continue
        rows.append(dict(market=mk, pos=p, n=len(d), prism_mae=d.err_model_proj.mean(), vegas_mae=d.err_vegas.mean(),
                         mc_mae=d.err_mc_proj.mean(), ml_mae=d.err_ml_proj.mean(), state_mae=d.err_state_proj.mean(), ensemble_mae=d.err_ensemble_proj.mean(),
                         prism_bias=(d.model_proj - d.actual).mean(), vegas_bias=(d.vegas_line - d.actual).mean(),
                         closer_than_line=d.closer.mean(), right_side=d.right_side.mean()))
    A = pd.DataFrame(rows).sort_values(["market", "pos"])
    A.to_csv(OUT / "w11_2026_projection_accuracy_v1.csv", index=False)
    print("\n== 2. PROJECTION ACCURACY vs FanDuel line (one row per player-market)"); print(A.round(2).to_string(index=False))
    q = f[f.market.isin(["rec_yards", "rush_yards"])]
    top = q[q.vegas_line >= q.groupby("market").vegas_line.transform(lambda s: s.quantile(0.75))]
    print(f"\nstar compression, top-quartile rec/rush lines (n={len(top)}): PRISM bias {(top.model_proj - top.actual).mean():+.1f} vs Vegas {(top.vegas_line - top.actual).mean():+.1f}")
    f["miss"] = f.actual - f.model_proj
    print("\nlargest misses (actual far ABOVE PRISM):")
    print(f.sort_values("miss", ascending=False).head(12)[["player", "pos", "team", "market", "model_proj", "vegas_line", "actual", "snap"]].round(1).to_string(index=False))
    print("\nlargest misses (actual far BELOW PRISM):")
    print(f.sort_values("miss").head(12)[["player", "pos", "team", "market", "model_proj", "vegas_line", "actual", "snap"]].round(1).to_string(index=False))
    f.to_csv(OUT / "w11_2026_projection_rows_v1.csv", index=False)

    # 3. calibration of the model's stated probability on the side it picked
    side["pb"] = pd.cut(side.fair_prob, [0, 0.5, 0.55, 0.6, 0.7, 0.8, 1.0])
    cal = side.groupby("pb", observed=True).agg(bets=("win", "size"), stated=("fair_prob", "mean"), market=("market_prob", "mean"), actual=("win", "mean"), roi=("pnl", "mean"))
    print("\n== 3. CALIBRATION (PRISM side)"); print(cal.round(3).to_string())

    # 4. alt board by market x position
    a = pd.read_csv(OUT / "w11_2026_alt_board_graded_v1.csv")
    a["pos"] = a.pos.map(lambda p: POSG.get(p, "?"))
    print("\n== 4. ALT BOARD (FanDuel overs, latest capture per game)")
    print(f"all rungs: n={len(a)} hit={a.hit.mean():.3f} needed={a.be.mean():.3f} roi={a.pnl.mean():+.3f}")
    print(a.groupby(["market", "pos"]).agg(rungs=("hit", "size"), hit=("hit", "mean"), needed=("be", "mean"), roi=("pnl", "mean")).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
