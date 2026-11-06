#!/usr/bin/env python3
"""Does an alternate-line edge survive being asked about the NAMED PLAYER?

The market-anchored clearance table expresses a threshold as a fraction of the
sportsbook line and reads the clearance rate off two seasons of outcomes. It is
a POPULATION aggregate: "players whose posted receptions line is 1.5 clear >=1
about 89% of the time." It does not know which player.

That is a direct collision with the AGENTS.md player-projection first
principle, and it matters in the direction that costs money: the table finds
its largest edges on players the book priced LOW, which is exactly the
sub-population a league-wide average over-states.

This audit joins every priced alt row to (a) its table cell's support and
(b) the named player's own realized rate, then reports expected value across
the whole shrinkage range -- player-only, moderate, population-only. A row is
reported ROBUST only when it is positive at every weighting, so a "finding"
that depends entirely on an unestimated shrinkage parameter cannot pass.

THE DENOMINATOR TRAP, which this script exists to avoid
-------------------------------------------------------
nflreadpy.load_player_stats() does not emit a row for a player-game with no
receiving involvement. Counting its rows as games played therefore divides by
"games where he caught something" and inflates every rate -- worst for exactly
the low-volume players the table is being applied to.

Measured on 2026 Week 9 TNF: Matthew Hibner has 5 player_stats rows and 7
games with offensive snaps. The inflated 3/5 = 60% beat the 52.8% break-even
on his >=1 receptions rung at -112 and ranked FIRST on the board at
+0.277 conservative EV. The true 3/7 = 42.9% is below break-even and the row
is negative on his own record.

Games played therefore comes from load_snap_counts() with offense_snaps > 0,
with receiving stats left-joined and missing values filled with zero.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RECP_TABLE = ROOT / "data" / "research" / "alt_line_receptions_absolute_v2.csv"
SHRINK_WEIGHTS = (0, 5, 20, 10**9)


def breakeven(american: float) -> float:
    a = float(american)
    return abs(a) / (abs(a) + 100.0) if a < 0 else 100.0 / (a + 100.0)


def ev_per_dollar(p: float, american: float) -> float:
    a = float(american)
    win = 100.0 / abs(a) if a < 0 else a / 100.0
    return p * win - (1.0 - p)


def player_game_record(
    player: str, threshold_at_least: int, stats: pd.DataFrame, snaps: pd.DataFrame
) -> tuple[int, int]:
    """(hits, games_played) with zero-involvement games INCLUDED."""
    played = snaps.loc[snaps["player"].astype(str).eq(player), ["week"]].drop_duplicates()
    if played.empty:
        return 0, 0
    got = stats.loc[stats["player_display_name"].eq(player), ["week", "receptions"]]
    joined = played.merge(got, on="week", how="left")
    joined["receptions"] = pd.to_numeric(joined["receptions"], errors="coerce").fillna(0.0)
    return int((joined["receptions"] >= threshold_at_least).sum()), int(len(joined))


def audit(edges: pd.DataFrame, stats: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    table = pd.read_csv(RECP_TABLE)
    rows = []
    priceable = edges.loc[edges["unpriceable_reason"].fillna("").eq("")]
    for r in priceable.itertuples(index=False):
        if str(r.market) != "receptions":
            # Only the integer market uses the absolute-threshold table, which
            # is the only table a per-game player rate is directly comparable to.
            continue
        k = math.floor(float(r.alt_threshold)) + 1
        cell = table.loc[
            table["main_line"].eq(float(r.main_line)) & table["threshold_at_least"].eq(k)
        ]
        hits, games = player_game_record(str(r.player), k, stats, snaps)
        if games == 0:
            continue
        p_pop = float(r.clearance)
        be = breakeven(r.offered_american)
        row = {
            "event_id": r.event_id, "player": r.player, "market": r.market,
            "book": r.book, "need_at_least": k, "main_line": r.main_line,
            "alt_threshold": r.alt_threshold, "offered_american": r.offered_american,
            "breakeven": be,
            "table_cleared": p_pop,
            "table_cleared_conservative": float(r.clearance_conservative),
            "table_cell_n": int(cell["n"].iloc[0]) if not cell.empty else -1,
            "table_cell_players": int(cell["players"].iloc[0]) if not cell.empty else -1,
            "post_hoc_line_restriction": bool(cell["post_hoc_line_restriction"].iloc[0]) if not cell.empty else None,
            "player_hits": hits, "player_games": games, "player_rate": hits / games,
        }
        for m in SHRINK_WEIGHTS:
            lab = "ev_player" if m == 0 else ("ev_population" if m == SHRINK_WEIGHTS[-1] else f"ev_shrink_{m}")
            row[lab] = ev_per_dollar((hits + m * p_pop) / (games + m), r.offered_american)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    ev_cols = [c for c in out.columns if c.startswith("ev_")]
    out["robust_all_weights"] = (out[ev_cols] > 0).all(axis=1)
    return out.sort_values(["robust_all_weights", "ev_shrink_5"], ascending=[False, False])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--edges", type=Path, required=True)
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--event-id", default="")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import nflreadpy as nfl

    edges = pd.read_csv(a.edges, low_memory=False)
    if a.event_id:
        edges = edges.loc[edges["event_id"].astype(str).eq(a.event_id)]
    stats = nfl.load_player_stats(seasons=[a.season]).to_pandas()
    snaps = nfl.load_snap_counts(seasons=[a.season]).to_pandas()
    snaps = snaps.loc[pd.to_numeric(snaps["offense_snaps"], errors="coerce").fillna(0) > 0]

    out = audit(edges, stats, snaps)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    if out.empty:
        print("[alt_player_conditional] no comparable rows")
        return 0
    n_rob = int(out["robust_all_weights"].sum())
    print(f"[alt_player_conditional] rows={len(out)} robust_all_weights={n_rob} -> {a.out}")
    cols = ["player", "need_at_least", "offered_american", "breakeven", "table_cleared",
            "player_hits", "player_games", "player_rate", "ev_player", "ev_shrink_5",
            "ev_population", "table_cell_n", "robust_all_weights"]
    print(out[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
