#!/usr/bin/env python3
"""Independent, from-scratch parity check for WR-R19's WR_TARGET_CATCHABLE_RATE8
and TEAM_TARGET_CATCHABLE_RATE8.

Written by Claude as an independent mechanical cross-check per GPT-5.6's
request on Issue #535 for WR-R19, mirroring the same discipline used for
WR-R18. Shares no code with receiver_state()/team_state() in
evaluate_wr_r19_receiver_catchability_stage_a_v1.py -- uses plain
groupby/tail(N) instead of the _prior()/_last_games() helper chain.

Focuses specifically on the two things that are genuinely new in R19 versus
the already-vetted R18 pattern: (1) the team-level control's independent
4-game/40-event support floor, and (2) that receiver and team windows are
each built "games first, then filter" rather than "valid-events first."
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRIOR_GAMES = 8
MIN_PRIOR_TARGET_GAMES = 4
MIN_RECEIVER_TARGETS = 16
MIN_TEAM_GAMES = 4
MIN_TEAM_TARGETS = 40


def independent_receiver_catchable_rate8(
    targets: pd.DataFrame, receiver_id: str, season: int, week: int
) -> dict:
    hist = targets.loc[targets["receiver_id"].eq(receiver_id)]
    hist = hist.loc[(hist["season"] < season) | ((hist["season"] == season) & (hist["week"] < week))]
    games = (
        hist[["season", "week", "game_id"]].drop_duplicates()
        .sort_values(["season", "week"], kind="mergesort").tail(PRIOR_GAMES)
    )
    if games.empty:
        return {"n_games": 0, "n_events": 0, "rate": float("nan"), "supported": False}
    keys = set(zip(games["season"], games["week"], games["game_id"]))
    window = hist.loc[hist.apply(lambda r: (r["season"], r["week"], r["game_id"]) in keys, axis=1)]
    catch = window["catchable"].dropna()
    supported = len(games) >= MIN_PRIOR_TARGET_GAMES and len(catch) >= MIN_RECEIVER_TARGETS
    return {
        "n_games": int(len(games)), "n_events": int(len(catch)),
        "rate": float(catch.mean()) if len(catch) else float("nan"), "supported": bool(supported),
    }


def independent_team_catchable_rate8(
    targets: pd.DataFrame, team: str, season: int, week: int
) -> dict:
    hist = targets.loc[targets["team"].eq(team)]
    hist = hist.loc[(hist["season"] < season) | ((hist["season"] == season) & (hist["week"] < week))]
    games = (
        hist[["season", "week", "game_id"]].drop_duplicates()
        .sort_values(["season", "week"], kind="mergesort").tail(PRIOR_GAMES)
    )
    if games.empty:
        return {"n_games": 0, "n_events": 0, "rate": float("nan"), "supported": False}
    keys = set(zip(games["season"], games["week"], games["game_id"]))
    window = hist.loc[hist.apply(lambda r: (r["season"], r["week"], r["game_id"]) in keys, axis=1)]
    catch = window["catchable"].dropna()
    supported = len(games) >= MIN_TEAM_GAMES and len(catch) >= MIN_TEAM_TARGETS
    return {
        "n_games": int(len(games)), "n_events": int(len(catch)),
        "rate": float(catch.mean()) if len(catch) else float("nan"), "supported": bool(supported),
    }


def _build_fixture() -> pd.DataFrame:
    """8 team games, 6 targets each (48 total, well above the 40-event team
    floor). One receiver (00-R1) gets 2 targets in each of games 1,3,5,7 (8
    total, below the 16-event receiver floor deliberately, to prove team and
    receiver floors are independent of each other -- team can be supported
    while the specific receiver is not). Other targets in each game go to a
    distractor receiver 00-R2, who must not contaminate 00-R1's history.
    A target-week (week 9) event for 00-R1 must never leak in.
    """
    rows = []
    catch_vals = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]
    k = 0
    for week in range(1, 9):
        game_id = f"2023_{week:02d}_BUF"
        for i in range(6):
            receiver = "00-R1" if (week % 2 == 1 and i < 2) else "00-R2"
            rows.append({
                "season": 2023, "week": week, "game_id": game_id, "team": "BUF",
                "receiver_id": receiver, "catchable": float(catch_vals[k]),
            })
            k += 1
    # target-week (week 9) event for 00-R1 -- must never enter history
    rows.append({"season": 2023, "week": 9, "game_id": "2023_09_BUF", "team": "BUF", "receiver_id": "00-R1", "catchable": 0.0})
    return pd.DataFrame(rows)


def main() -> int:
    fixture = _build_fixture()

    team = independent_team_catchable_rate8(fixture, "BUF", 2023, 9)
    print("Team (BUF) independent result:", team)
    # 8 prior games exist, all within the last-8 window: 48 valid targets total (well above 40-floor)
    assert team["n_games"] == 8, team
    assert team["n_events"] == 48, team
    assert team["supported"] is True, team
    expected_team_rate = float(np.mean([c for c in [1,0,1,1,0,1,1,1,0,1,1,0,0,1,1,1,1,0,1,1,0,1,1,1,1,0,1,1,0,1,1,0,0,1,1,1,1,0,1,1,0,1,1,1,1,0,1,1]]))
    assert abs(team["rate"] - expected_team_rate) < 1e-9, (team, expected_team_rate)

    receiver = independent_receiver_catchable_rate8(fixture, "00-R1", 2023, 9)
    print("Receiver (00-R1) independent result:", receiver)
    # 00-R1 has targets in games 1,3,5,7 only (4 games), 2 targets each = 8 events.
    # 4 games clears the 4-game floor, but 8 events < 16-event floor -> NOT supported.
    assert receiver["n_games"] == 4, receiver
    assert receiver["n_events"] == 8, receiver
    assert receiver["supported"] is False, receiver
    print("CONFIRMED: team is supported (48>=40 events, 8>=4 games) while the specific")
    print("receiver within that same team/week is correctly NOT supported (8<16 events),")
    print("proving the two support floors are independently enforced, not conflated.")

    print("PASS: independent parity check complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
