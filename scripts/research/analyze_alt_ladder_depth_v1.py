#!/usr/bin/env python3
"""How far up his own alt price ladder did each player actually get?

WHY THE LADDER IS THE UNIT, NOT THE RUNG
----------------------------------------
An alt ladder for one (player, market, book) is monotone by construction: `hit`
is derived from a single `actual`, so if a player cleared the rung priced -200
he necessarily cleared his -400 and -750 rungs too. Counting rungs therefore
double-counts every easy rung inside a winning ladder and triple-counts nothing
inside a losing one. Week 9: 1,321 FanDuel rungs priced >= -200 hit at 33.7%,
while 55.2% of FanDuel LADDERS reached -200 or better. Those two numbers answer
different questions and only the second one is "how often does a player get
there".

So every statistic here is per ladder:

  hwm_price   the price of the DEEPEST rung the player cleared -- his high-water
              mark. NaN means he missed even the cheapest rung offered.
  died_price  the price of the CHEAPEST rung he missed. hwm and died bracket the
              player's actual result in price space.

SURVIVAL, WORKING UP THE LADDER
-------------------------------
For a price level P:
  offered(P)  ladders holding at least one rung priced >= P
  cleared(P)  those where such a rung HIT, i.e. hwm_price >= P
  pick_roi    ROI of betting ONE rung per ladder: the cheapest rung priced >= P,
              which is what "give me about -200 on this guy" actually buys.

pick_roi is the decision-relevant column. `rate` is descriptive: it compares to
breakeven(P) only loosely, because the pick's real price is >= P.

Rows are graded and committed under data/market_track_record/graded/; this script
reads them and writes its tables to data/research/. It decides nothing.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

LEVELS = [-1000, -750, -600, -500, -400, -350, -300, -250, -200, -175, -150,
          -125, -110, 100, 120, 150, 200, 250, 300, 400, 500, 600, 800, 1000]
LADDER_KEY = ["player", "market", "book"]


def dec(american: float) -> float:
    """American odds -> decimal profit per unit staked."""
    return american / 100.0 if american > 0 else 100.0 / (-american)


def load_board(path: str, book: str | None = None) -> pd.DataFrame:
    g = pd.read_csv(path)
    if "pnl" not in g.columns:
        raise SystemExit(f"{path} is not a graded board (no pnl column)")
    g = g[g["pnl"].notna()].copy()
    if book:
        g = g[g["book"].astype(str).str.lower() == book.lower()].copy()
        if g.empty:
            raise SystemExit(f"no rows for book={book}")
    if (g["side"].astype(str).str.upper() != "OVER").any():
        raise SystemExit("ladder logic assumes OVER rungs only; UNDER rows present")
    return g


def check_monotone(g: pd.DataFrame) -> int:
    """Count ladders where a harder rung hit while an easier one missed.

    Should always be 0. A non-zero result means `hit` was not derived from one
    actual per player-market and every statistic below is invalid.
    """
    bad = 0
    for _, d in g.groupby(LADDER_KEY):
        h = d.sort_values("alt_threshold")["hit"].to_numpy()
        if any(h[i] < h[i + 1] for i in range(len(h) - 1)):
            bad += 1
    return bad


def ladders(g: pd.DataFrame) -> pd.DataFrame:
    out = []
    for keys, d in g.groupby(LADDER_KEY):
        d = d.sort_values("alt_threshold")
        hit, miss = d[d.hit == 1], d[d.hit == 0]
        out.append({
            **dict(zip(LADDER_KEY, keys)),
            "actual": d.actual.iloc[0],
            "main_line": d.main_line.iloc[0],
            "n_rungs": len(d), "n_hit": len(hit),
            "lo_thr": d.alt_threshold.min(), "lo_price": d.offered_american.min(),
            "hi_thr": d.alt_threshold.max(), "hi_price": d.offered_american.max(),
            "hwm_thr": hit.alt_threshold.max() if len(hit) else np.nan,
            "hwm_price": hit.offered_american.max() if len(hit) else np.nan,
            "died_thr": miss.alt_threshold.min() if len(miss) else np.nan,
            "died_price": miss.offered_american.min() if len(miss) else np.nan,
        })
    L = pd.DataFrame(out)
    L["cleared_nothing"] = L.hwm_price.isna()
    L["cleared_200_or_better"] = L.hwm_price >= -200
    return L


def survival(g: pd.DataFrame, levels=LEVELS) -> pd.DataFrame:
    grp = list(g.groupby(LADDER_KEY))
    rows = []
    for P in levels:
        offered = cleared = 0
        pnl, prices, hits = [], [], []
        for _, d in grp:
            q = d[d.offered_american >= P]
            if q.empty:
                continue
            offered += 1
            if (q.hit == 1).any():
                cleared += 1
            pick = q.loc[q.offered_american.idxmin()]
            pnl.append(dec(pick.offered_american) if pick.hit == 1 else -1.0)
            prices.append(pick.offered_american)
            hits.append(pick.hit)
        if not offered:
            continue
        be = float(np.mean([1.0 / (1.0 + dec(p)) for p in prices]))
        rows.append({
            "level": P, "offered": offered, "cleared": cleared,
            "clear_rate_pct": round(100 * cleared / offered, 1),
            "pick_n": len(pnl),
            "pick_mean_price": round(float(np.mean(prices)), 1),
            "pick_hit_pct": round(100 * float(np.mean(hits)), 1),
            "pick_breakeven_pct": round(100 * be, 1),
            "pick_edge_pts": round(100 * (float(np.mean(hits)) - be), 1),
            "pick_roi_pct": round(100 * float(np.mean(pnl)), 1),
        })
    return pd.DataFrame(rows)


def bootstrap_level(g: pd.DataFrame, P: int, draws: int = 20000,
                    seed: int = 1111) -> dict:
    """Cluster bootstrap by GAME -- legs inside a game are correlated, so the
    game is the resampling unit. Resampling rungs would understate the CI."""
    picks = []
    for _, d in g.groupby(LADDER_KEY):
        q = d[d.offered_american >= P]
        if q.empty:
            continue
        picks.append(q.loc[q.offered_american.idxmin()])
    if not picks:
        return {}
    pk = pd.DataFrame(picks)
    pk["pnl"] = np.where(pk.hit == 1, pk.offered_american.map(dec), -1.0)
    ev = pk.event_id.unique()
    rng = np.random.default_rng(seed)
    idx = pk.set_index("event_id")
    s = np.array([idx.loc[rng.choice(ev, len(ev), replace=True)].pnl.mean()
                  for _ in range(draws)])
    return {"level": P, "n": len(pk), "games": len(ev),
            "roi_pct": round(100 * pk.pnl.mean(), 1),
            "ci_lo_pct": round(100 * float(np.percentile(s, 2.5)), 1),
            "ci_hi_pct": round(100 * float(np.percentile(s, 97.5)), 1),
            "p_positive_pct": round(100 * float((s > 0).mean()), 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True, help="graded alt board CSV")
    ap.add_argument("--book", default=None, help="restrict to one book")
    ap.add_argument("--out-ladders", default=None)
    ap.add_argument("--out-survival", default=None)
    ap.add_argument("--bootstrap", type=int, nargs="*", default=None,
                    help="price levels to cluster-bootstrap")
    a = ap.parse_args()

    g = load_board(a.board, a.book)
    bad = check_monotone(g)
    L, S = ladders(g), survival(g)

    label = a.book or "all books"
    print(f"=== {label} — {len(L)} ladders, {len(g)} rungs ===")
    print(f"monotonicity violations: {bad}/{len(L)}"
          f"{'  OK' if bad == 0 else '  *** INVALID ***'}")
    print(f"cleared nothing: {int(L.cleared_nothing.sum())} "
          f"({100 * L.cleared_nothing.mean():.1f}%)")
    print(f"cleared -200 or better: {int(L.cleared_200_or_better.sum())} "
          f"({100 * L.cleared_200_or_better.mean():.1f}%)")
    d = L.died_price.dropna()
    if len(d):
        print(f"median price of the first MISSED rung: {d.median():+.0f}")
    print(f"\n{S.to_string(index=False)}")

    for P in (a.bootstrap or []):
        r = bootstrap_level(g, P)
        if r:
            print(f"\nbootstrap @ {P:+d}: ROI {r['roi_pct']:+.1f}% "
                  f"95% CI [{r['ci_lo_pct']:+.1f}%, {r['ci_hi_pct']:+.1f}%] "
                  f"P(>0)={r['p_positive_pct']:.1f}% (n={r['n']}, {r['games']} games)")

    for path, df in ((a.out_ladders, L), (a.out_survival, S)):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            print(f"wrote {path}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
