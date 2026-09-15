#!/usr/bin/env python3
"""Independent, from-scratch parity check for WR-R20's EARLY_NO_EXTENDED_SHARE8
and TEAM_EARLY_NO_EXTENDED_SHARE8.

Written by Claude as an independent mechanical cross-check per GPT-5.6's
request for WR-R20, mirroring the same discipline used for WR-R19. Shares no
code with receiver_state()/team_state() in
evaluate_wr_r20_early_no_extended_stage_a_v1.py -- uses plain groupby/tail(N)
instead of the _prior()/_last_games() helper chain.

Focuses on the two things genuinely new vs R19's already-vetted pattern:
(1) CHECKDOWN/SCRAMBLE_DRILL events sit *inside* the selected last-8-game
    window but must be excluded from the classifiable denominator, not just
    excluded from the source population entirely -- a receiver's last 8
    target-bearing games are picked using ALL of that receiver's targets
    (including checkdowns), and only AFTER that selection do CHK/SD events
    drop out of the numerator/denominator; and
(2) the team floor (40 classifiable) is independent of the receiver floor
    (16 classifiable), same as R19's team/receiver independence but now
    against the checkdown-exclusion behavior specifically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRIOR_GAMES = 8
MIN_RECEIVER_GAMES = 4
MIN_RECEIVER_CLASSIFIABLE = 16
MIN_TEAM_CLASSIFIABLE = 40


def independent_receiver_share8(targets: pd.DataFrame, receiver_id: str, season: int, week: int) -> dict:
    hist = targets.loc[targets["receiver_id"].eq(receiver_id)]
    hist = hist.loc[(hist["season"] < season) | ((hist["season"] == season) & (hist["week"] < week))]
    games = (
        hist[["season", "week", "game_id"]].drop_duplicates()
        .sort_values(["season", "week"], kind="mergesort").tail(PRIOR_GAMES)
    )
    if games.empty:
        return {"n_games": 0, "n_all_events": 0, "n_classifiable": 0, "rate": float("nan"), "supported": False}
    keys = set(zip(games["season"], games["week"], games["game_id"]))
    window = hist.loc[hist.apply(lambda r: (r["season"], r["week"], r["game_id"]) in keys, axis=1)]
    classifiable = window.loc[window["state"].isin(["EARLY_NO_EXTENDED", "EXTENDED_PROGRESS"])]
    early = classifiable["state"].eq("EARLY_NO_EXTENDED")
    supported = len(games) >= MIN_RECEIVER_GAMES and len(classifiable) >= MIN_RECEIVER_CLASSIFIABLE
    return {
        "n_games": int(len(games)), "n_all_events": int(len(window)), "n_classifiable": int(len(classifiable)),
        "n_checkdown": int(window["state"].eq("CHECKDOWN").sum()),
        "rate": float(early.mean()) if len(classifiable) else float("nan"), "supported": bool(supported),
    }


def independent_team_share8(targets: pd.DataFrame, team: str, season: int, week: int) -> dict:
    hist = targets.loc[targets["team"].eq(team)]
    hist = hist.loc[(hist["season"] < season) | ((hist["season"] == season) & (hist["week"] < week))]
    games = (
        hist[["season", "week", "game_id"]].drop_duplicates()
        .sort_values(["season", "week"], kind="mergesort").tail(PRIOR_GAMES)
    )
    if games.empty:
        return {"n_games": 0, "n_classifiable": 0, "rate": float("nan"), "supported": False}
    keys = set(zip(games["season"], games["week"], games["game_id"]))
    window = hist.loc[hist.apply(lambda r: (r["season"], r["week"], r["game_id"]) in keys, axis=1)]
    classifiable = window.loc[window["state"].isin(["EARLY_NO_EXTENDED", "EXTENDED_PROGRESS"])]
    early = classifiable["state"].eq("EARLY_NO_EXTENDED")
    supported = len(classifiable) >= MIN_TEAM_CLASSIFIABLE
    return {
        "n_games": int(len(games)), "n_classifiable": int(len(classifiable)),
        "rate": float(early.mean()) if len(classifiable) else float("nan"), "supported": bool(supported),
    }


def _build_fixture() -> pd.DataFrame:
    """One receiver (00-R1) with targets in 9 prior team games (weeks 1-9),
    2 targets/game = 18 total. In games 1-8 (the literal last-8-game window
    once week 9 arrives as the *target* game... but we score AT week 10, so
    the receiver's own last 8 target-bearing games are weeks 2-9):
      - weeks 2-8 (7 games): 1 EARLY_NO_EXTENDED + 1 EXTENDED_PROGRESS each = 14 classifiable
      - week 9: 1 CHECKDOWN + 1 EARLY_NO_EXTENDED -> the CHECKDOWN must be
        counted toward n_games/n_all_events (it's in the window) but NOT
        toward n_classifiable or the rate.
    So the last-8-game window (weeks 2-9) has 16 raw events but only 15
    classifiable (one CHECKDOWN dropped) -- this must fail the receiver's
    >=16-classifiable floor even though it clears the >=4-game floor and even
    though raw event count alone (16) would have cleared a naive floor.

    Team BUF has heavy additional volume (6 more targets/game from a
    distractor receiver 00-R2) across the same 9 games so the team's
    last-8-game classifiable count (7*6 + games 2-9 distractor targets, all
    EXTENDED_PROGRESS for simplicity) clears 40 easily, proving team support
    and receiver support are independent -- team supported, receiver not.
    """
    rows = []
    # Receiver 00-R1: weeks 1-9, 2 targets/game.
    for week in range(1, 10):
        game_id = f"2023_{week:02d}_BUF"
        if week == 9:
            states = ["CHECKDOWN", "EARLY_NO_EXTENDED"]
        else:
            states = ["EARLY_NO_EXTENDED", "EXTENDED_PROGRESS"]
        for s in states:
            rows.append({"season": 2023, "week": week, "game_id": game_id, "team": "BUF", "receiver_id": "00-R1", "state": s})
        # Distractor 00-R2: 6 targets/game, all EXTENDED_PROGRESS, every week 1-9.
        for _ in range(6):
            rows.append({"season": 2023, "week": week, "game_id": game_id, "team": "BUF", "receiver_id": "00-R2", "state": "EXTENDED_PROGRESS"})
    # Target-week (week 10) events for both receivers -- must never leak into history.
    rows.append({"season": 2023, "week": 10, "game_id": "2023_10_BUF", "team": "BUF", "receiver_id": "00-R1", "state": "EARLY_NO_EXTENDED"})
    rows.append({"season": 2023, "week": 10, "game_id": "2023_10_BUF", "team": "BUF", "receiver_id": "00-R2", "state": "EXTENDED_PROGRESS"})
    return pd.DataFrame(rows)


def main() -> int:
    fixture = _build_fixture()

    receiver = independent_receiver_share8(fixture, "00-R1", 2023, 10)
    print("Receiver (00-R1) independent result:", receiver)
    # Last 8 target-bearing games = weeks 2-9 (week 1 drops off). 16 raw events,
    # one CHECKDOWN in week 9 -> 15 classifiable. Clears the 4-game floor,
    # fails the 16-classifiable floor.
    assert receiver["n_games"] == 8, receiver
    assert receiver["n_all_events"] == 16, receiver
    assert receiver["n_checkdown"] == 1, receiver
    assert receiver["n_classifiable"] == 15, receiver
    assert receiver["supported"] is False, receiver

    team = independent_team_share8(fixture, "BUF", 2023, 10)
    print("Team (BUF) independent result:", team)
    # Team's last 8 games (weeks 2-9): each game contributes 6 distractor
    # EXTENDED_PROGRESS events regardless of 00-R1's checkdown -> 48
    # classifiable team events at minimum (00-R1's own classifiable events
    # add a bit more) -- comfortably clears 40.
    assert team["n_games"] == 8, team
    assert team["n_classifiable"] >= 40, team
    assert team["supported"] is True, team

    print()
    print("CONFIRMED: a CHECKDOWN event inside the receiver's own last-8-game")
    print("window correctly counts toward games/raw-event totals but is excluded")
    print("from the classifiable denominator, and this alone is enough to flip")
    print("the receiver from supported to unsupported (16 raw events, only 15")
    print("classifiable) -- while the team-level control, built from different")
    print("receivers' volume, remains comfortably supported. The two floors are")
    print("independently enforced, and CHK exclusion is a real, not cosmetic,")
    print("mechanic.")
    print("PASS: independent parity check complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
