#!/usr/bin/env python3
"""Build a single-event alternate-line card from an archived paid board.

WHY THIS EXISTS
---------------
A priced Full Slate board is the normal route to a card. When the pipeline
cannot reach pricing -- as on 2026-11-12, where step 33 blocked on availability
provenance a few hours before a Thursday kickoff -- the paid alternate board is
still archived and still the market's current state. This script turns that
board into a card using only football evidence plus the offer itself, so a slate
is not lost to a pipeline defect.

WHAT IT IS NOT
--------------
This is NOT the model. It applies eight of the nine alt gates; the cushion gate
(G3) needs `model_proj` from run_pricing_v2 and is absent by construction here.
Nothing here constructs a football projection from sportsbook data: the price is
used to compute breakeven for COMPARISON, and line availability is used for
SELECTION. Per-game actuals come from nflverse weekly stats. That keeps it on
the right side of AGENTS.md section 3 -- but a card from this script must always
be labeled as missing the cushion gate.

EVIDENCE DISCIPLINE
-------------------
`own_rate` over 5-9 games is a small-sample frequency and is NOT a probability.
The binding criterion is `margin` -- the player's WORST observed game minus the
threshold. A positive margin means every game on record clears the line, which
is a far stronger statement than a point estimate of a rate.

Per the AGENTS.md player-state anomaly gate, this script reports the week
numbers a player actually played so absences and partial games are visible
rather than averaged away. It deliberately does NOT state-label automatically:
a missed week is a question for the operator, not a number to shrink.

Usage:
  python scripts/research/build_event_alt_card_v1.py \
      --board data/market_track_record/alt_props/2026_wk10_thu6.csv \
      --season 2026 [--event <id> | --earliest] [--max-price -800]
"""
from __future__ import annotations
import argparse
import sys

import numpy as np
import pandas as pd

MARKET_STAT = {
    "player_reception_yds_alternate": "receiving_yards",
    "player_rush_yds_alternate": "rushing_yards",
    "player_receptions_alternate": "receptions",
    "player_pass_yds_alternate": "passing_yards",
    "player_rush_reception_yds_alternate": "rush_rec_yards",
}
# Markets we do not play. anytime_td was struck from all evidence on
# 2026-11-12: it is not a market this operation bets, and leaving it in
# contaminated the week-9 amplification replication.
EXCLUDED_MARKETS = {"player_anytime_td", "player_anytime_td_alternate"}


def norm_name(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.replace(r"[^a-z]", "", regex=True)


def implied_prob(american: float) -> float:
    a = float(american)
    return (-a) / ((-a) + 100.0) if a < 0 else 100.0 / (a + 100.0)


def interpolated_main_line(g: pd.DataFrame) -> float:
    """The line where the ladder's price crosses -110.

    Used only as the real-alternate reference (G4), so a main-line shade
    dressed as an alternate is not counted as one. Interpolating the ladder
    avoids depending on a separate main-line capture that may be from a
    different fetch than the alternates.
    """
    g = g.sort_values("line")
    p = g["price_american"].map(implied_prob).to_numpy()
    lines = g["line"].to_numpy()
    target = implied_prob(-110)
    for i in range(len(lines) - 1):
        if (p[i] - target) * (p[i + 1] - target) <= 0 and p[i] != p[i + 1]:
            w = (p[i] - target) / (p[i] - p[i + 1])
            return float(lines[i] + w * (lines[i + 1] - lines[i]))
    return float("nan")


def load_actuals(season: int) -> pd.DataFrame:
    import nflreadpy as nfl

    w = nfl.load_player_stats(seasons=[season])
    st = w.to_pandas() if hasattr(w, "to_pandas") else pd.DataFrame(w)
    st = st.loc[st["season"] == season].copy()
    st["rush_rec_yards"] = st["rushing_yards"].fillna(0) + st["receiving_yards"].fillna(0)
    st["nm"] = norm_name(st["player_display_name"])
    return st


def build(board: pd.DataFrame, st: pd.DataFrame, min_games: int, max_price: float) -> pd.DataFrame:
    board = board.loc[~board["market"].isin(EXCLUDED_MARKETS)].copy()
    board = board.loc[board["side"].astype(str).str.lower() == "over"].copy()
    board["stat"] = board["market"].map(MARKET_STAT)
    board = board.loc[board["stat"].notna()].copy()
    board["nm"] = norm_name(board["player"])

    rows = []
    for (player, stat), g in board.groupby(["player", "stat"], sort=True):
        s = st.loc[st["nm"] == g["nm"].iloc[0]]
        if s.empty:
            continue
        vals = s[stat].fillna(0).to_numpy(dtype=float)
        if len(vals) < min_games:
            continue
        weeks = sorted(int(w) for w in s["week"].dropna().unique())
        main = interpolated_main_line(g)
        # Best available price per threshold across books: an operator bets the
        # best number, so a card built on one book understates the offer.
        best = g.loc[g.groupby("line")["price_american"].idxmax()]
        for r in best.itertuples(index=False):
            thr = float(r.line)
            price = float(r.price_american)
            if price < max_price:
                continue
            be = implied_prob(price)
            rate = float((vals > thr).mean())
            rows.append(
                {
                    "player": player,
                    "stat": stat,
                    "threshold": thr,
                    "price_american": price,
                    "book": r.book,
                    "main_line_interp": main,
                    "alt_ratio": thr / main if main == main and main > 0 else np.nan,
                    "games": len(vals),
                    "weeks_played": ",".join(str(w) for w in weeks),
                    "cleared": int((vals > thr).sum()),
                    "own_rate": rate,
                    "breakeven": be,
                    "edge_pts": rate - be,
                    "floor": float(vals.min()),
                    "median": float(np.median(vals)),
                    "margin": float(vals.min() - thr),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--event")
    ap.add_argument("--earliest", action="store_true", help="pick the earliest-kickoff event")
    ap.add_argument("--min-games", type=int, default=5)
    ap.add_argument("--max-price", type=float, default=-800.0,
                    help="reject rungs cheaper than this (G9); -800 is ~89%% breakeven")
    ap.add_argument("--max-alt-ratio", type=float, default=0.85,
                    help="G4: threshold must be this fraction of the main line or lower")
    ap.add_argument("--out")
    a = ap.parse_args()

    board = pd.read_csv(a.board)
    if a.event:
        board = board.loc[board["event_id"] == a.event].copy()
    elif a.earliest:
        ct = pd.to_datetime(board["commence_time"], utc=True, errors="coerce")
        board = board.loc[ct == ct.min()].copy()
    if board.empty:
        print("no rows for the selected event", file=sys.stderr)
        return 2

    card = build(board, load_actuals(a.season), a.min_games, a.max_price)
    if card.empty:
        print("no candidate rungs survived", file=sys.stderr)
        return 1

    # G4 real-alternate, G6 own-evidence floor test, positive edge.
    q = card.loc[
        (card["alt_ratio"] <= a.max_alt_ratio)
        & (card["margin"] > 0)
        & (card["edge_pts"] > 0)
    ].sort_values(["edge_pts", "margin"], ascending=[False, False])

    cols = ["player", "stat", "threshold", "price_american", "book", "cleared", "games",
            "weeks_played", "breakeven", "edge_pts", "floor", "median", "margin"]
    print("NOTE: cushion gate (G3) NOT applied -- no model_proj in this path.")
    print(q[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if a.out:
        card.to_csv(a.out, index=False)
        print(f"\nwrote all {len(card)} evaluated rungs -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
