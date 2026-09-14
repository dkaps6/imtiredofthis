#!/usr/bin/env python3
"""Independent, from-scratch parity check for WR-R18's WR_TARGET_CPOE_MEAN8.

Written by Claude as an independent mechanical cross-check per GPT-5.6's
request on Issue #535: a second, structurally different computation of the
same quantity, sharing no code with
evaluate_wr_r18_receiver_target_cpoe_stage_a_v1.py's receiver_state()/
resolve_prior_receiver_history(). Uses a plain groupby + manual last-N-games
selection instead of the row-by-row _prior()/history resolution approach.

This does not touch real nflverse data (no network access in this sandbox);
it validates the ARITHMETIC/WINDOWING logic against a hand-verifiable
synthetic fixture built independently of GPT's own test fixtures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRIOR_GAMES = 8
MIN_PRIOR_TARGET_GAMES = 4
MIN_VALID_CPOE_TARGETS = 16


def independent_receiver_cpoe_mean8(
    targets: pd.DataFrame, receiver_id: str, season: int, week: int
) -> dict:
    """Structurally independent recomputation.

    Strategy: build the full history for this receiver_id, keep only rows
    strictly before (season, week), take the games with >=1 non-null-CPOE
    target, sort them, keep the last 8, then average CPOE over the valid
    events in those games. Implemented with straight pandas groupby/sort,
    no shared helper functions with the evaluator under test.
    """
    hist = targets.loc[targets["receiver_id"].eq(receiver_id)].copy()
    hist = hist.loc[
        (hist["season"] < season) | ((hist["season"] == season) & (hist["week"] < week))
    ]
    valid = hist.loc[hist["cpoe"].notna()]
    valid_games = (
        valid.groupby(["season", "week", "game_id"], as_index=False)
        .size()
        .sort_values(["season", "week"], kind="mergesort")
    )
    last8 = valid_games.tail(PRIOR_GAMES)
    if last8.empty:
        return {"n_games": 0, "n_events": 0, "mean_cpoe": float("nan"), "supported": False}
    keep_keys = set(zip(last8["season"], last8["week"], last8["game_id"]))
    events = valid.loc[
        valid.apply(lambda r: (r["season"], r["week"], r["game_id"]) in keep_keys, axis=1)
    ]
    n_games = len(last8)
    n_events = len(events)
    supported = n_games >= MIN_PRIOR_TARGET_GAMES and n_events >= MIN_VALID_CPOE_TARGETS
    return {
        "n_games": int(n_games),
        "n_events": int(n_events),
        "mean_cpoe": float(events["cpoe"].mean()) if n_events else float("nan"),
        "supported": bool(supported),
    }


def _build_fixture() -> pd.DataFrame:
    """Independent synthetic fixture -- different shape/values than GPT's.

    Receiver 00-PARITY1, team BUF, 5 prior games in 2023 (weeks 1-5), scored
    row is week 6. Game 3 has one null-CPOE event mixed with two valid ones,
    to exercise the null-preserved-but-not-counted path. A different
    receiver (00-OTHER) and a target-week (week 6) event are included as
    distractors that must not leak into the computation.
    """
    rows = []
    game_cpoes = {
        1: [-10.0, 4.0],
        2: [2.0, 6.0, 1.0],
        3: [np.nan, 8.0, -2.0],
        4: [15.0, -5.0, 3.0, 9.0],
        5: [0.0, 12.0],
    }
    for week, cpoes in game_cpoes.items():
        for i, c in enumerate(cpoes):
            rows.append({
                "season": 2023, "week": week, "game_id": f"2023_{week:02d}_BUF",
                "receiver_id": "00-PARITY1", "cpoe": c, "air": 10.0 + i,
            })
    # distractor: different receiver, overlapping weeks -- must not be picked up
    rows.append({"season": 2023, "week": 3, "game_id": "2023_03_BUF", "receiver_id": "00-OTHER", "cpoe": 99.0, "air": 40.0})
    # distractor: target-game (week 6) event for the SAME receiver -- must never enter history
    rows.append({"season": 2023, "week": 6, "game_id": "2023_06_BUF", "receiver_id": "00-PARITY1", "cpoe": -999.0, "air": 1.0})
    return pd.DataFrame(rows)


def main() -> int:
    fixture = _build_fixture()
    result = independent_receiver_cpoe_mean8(fixture, "00-PARITY1", 2023, 6)

    # Hand-computed expectation: games 1-5 all qualify (each has >=1 valid
    # CPOE event), all 5 are within the last-8 window (only 5 exist), so
    # every valid event across games 1-5 counts. Valid events:
    # g1: -10, 4 (2) | g2: 2, 6, 1 (3) | g3: 8, -2 (2, null dropped)
    # g4: 15, -5, 3, 9 (4) | g5: 0, 12 (2)  => 13 valid events total.
    expected_events = [-10.0, 4.0, 2.0, 6.0, 1.0, 8.0, -2.0, 15.0, -5.0, 3.0, 9.0, 0.0, 12.0]
    expected_n_events = len(expected_events)
    expected_mean = float(np.mean(expected_events))
    expected_n_games = 5

    assert result["n_games"] == expected_n_games, (result, expected_n_games)
    assert result["n_events"] == expected_n_events, (result, expected_n_events)
    assert abs(result["mean_cpoe"] - expected_mean) < 1e-9, (result, expected_mean)
    # Only 13 valid events < 16-floor -> must NOT be flagged as supported
    # under the frozen WR-R18 support rule, even though this independent
    # function computes the raw mean regardless of the floor.
    assert result["supported"] is False, result

    print("Independent parity check (hand-verified arithmetic):")
    print(f"  n_games={result['n_games']} (expected {expected_n_games})")
    print(f"  n_events={result['n_events']} (expected {expected_n_events})")
    print(f"  mean_cpoe={result['mean_cpoe']:.6f} (expected {expected_mean:.6f})")
    print(f"  supported={result['supported']} (expected False, 13 < 16-event floor)")
    print("PASS: independent computation matches hand-verified expectation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
