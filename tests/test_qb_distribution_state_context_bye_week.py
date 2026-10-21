#!/usr/bin/env python3
"""Bye-week regression tests for the QB C2 slate guard.

The 2026 Week-7 Full Slate dry run failed with
`expected 32 active team rows, got 28`: the week has 14 games and four teams
idle, and the guard demanded a literal 32. This is the same hard-coded
bye-week guard class that failed the Week-5 run
(`docs/operations/WEEK5_2026_10_08_FULL_SLATE_PRODUCTION_RECOVERY_RESULT.md`
records "failed later on a bye-week source-quality hard-coded 32-team guard").

The replacement asserts the slate is a *closed* set of scheduled teams. These
tests pin both halves: a bye week must pass, and a genuinely broken schedule
map must still fail.
"""
from __future__ import annotations

import pandas as pd
import pytest

CANON = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]


def slate_frame(teams: list[str]) -> pd.DataFrame:
    """Build a closed slate: consecutive pairs face each other."""
    rows = []
    for i in range(0, len(teams), 2):
        a, b = teams[i], teams[i + 1]
        rows.append({"team": a, "opponent": b})
        rows.append({"team": b, "opponent": a})
    return pd.DataFrame(rows)


def guard(slate: pd.DataFrame) -> None:
    """The guard as implemented in run_qb_distribution_state_context.main()."""
    teams = set(slate["team"])
    opps = set(slate["opponent"])
    if (
        (not teams)
        or len(slate) % 2
        or teams != opps
        or len(teams) != len(slate)
        or len(teams) > 32
    ):
        raise RuntimeError(
            f"slate is not a closed set of scheduled teams: rows={len(slate)} "
            f"teams={len(teams)} opponents={len(opps)} "
            f"unmatched={sorted(teams ^ opps)}"
        )


def test_full_slate_of_32_passes():
    guard(slate_frame(CANON))


def test_bye_week_of_28_teams_passes():
    """The exact 2026 week-7 shape: 14 games, four teams idle."""
    playing = [t for t in CANON if t not in ("BAL", "BUF", "KC", "SF")]
    frame = slate_frame(playing)
    assert len(frame) == 28
    guard(frame)


@pytest.mark.parametrize("n_games", [1, 2, 6, 7, 13, 14, 15, 16])
def test_any_even_closed_slate_passes(n_games):
    """Byes vary week to week; every legitimate game count must pass."""
    frame = slate_frame(CANON[: n_games * 2])
    assert len(frame) == n_games * 2
    guard(frame)


def test_empty_slate_still_fails():
    with pytest.raises(RuntimeError, match="closed set"):
        guard(pd.DataFrame({"team": [], "opponent": []}))


def test_odd_participant_count_fails():
    frame = slate_frame(CANON[:4]).iloc[:3]
    with pytest.raises(RuntimeError, match="closed set"):
        guard(frame)


def test_opponent_missing_from_the_slate_fails():
    """A dropped half of a game is exactly what the guard must catch."""
    frame = slate_frame(CANON[:4])
    frame = frame[frame["team"] != CANON[1]]
    with pytest.raises(RuntimeError, match="closed set"):
        guard(frame)


def test_opponent_not_playing_this_week_fails():
    frame = slate_frame(CANON[:4])
    frame.loc[frame.index[0], "opponent"] = "ZZZ"
    with pytest.raises(RuntimeError, match="closed set"):
        guard(frame)


def test_duplicate_team_rows_fail():
    frame = slate_frame(CANON[:4])
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="closed set"):
        guard(frame)


def test_more_than_32_teams_fails():
    frame = slate_frame(CANON + ["XXX", "YYY"])
    with pytest.raises(RuntimeError, match="closed set"):
        guard(frame)


def test_guard_is_stricter_than_a_bare_count():
    """A 32-row slate with a broken pairing passed the old `== 32` check."""
    frame = slate_frame(CANON)
    frame.loc[frame.index[0], "opponent"] = "ZZZ"
    assert len(frame) == 32  # the old guard would have accepted this
    with pytest.raises(RuntimeError, match="closed set"):
        guard(frame)
