#!/usr/bin/env python3
"""Regression tests for participation column resolution in make_team_form.

Full Slate on main failed closed on 2026-10-21 with light_box_rate and
heavy_box_rate missing for all 32 teams. The cause was not the provider:
nflverse participation names the offense `possession_team`, which was absent
from the team-column candidate list, so `team_col` resolved to None and the
box-rate branch was skipped in silence. The strict validator then rejected
every team.

These pin the real column names and, more importantly, pin the *failure mode*:
a resolution miss must not be able to pass as "no box data".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.make_team_form import (
    _BOX_COL_CANDIDATES,
    _PART_TEAM_COL_CANDIDATES,
    compute_personnel_rates,
    resolve_participation_columns,
)


def nflverse_participation(n: int = 6) -> pd.DataFrame:
    """A frame shaped like nflreadpy.load_participation output, lowercased.

    Column names verified against nflreadpy 0.1.5 for seasons 2022-2025.
    """
    return pd.DataFrame(
        {
            "nflverse_game_id": [f"2025_01_ARI_BAL_{i}" for i in range(n)],
            "play_id": list(range(n)),
            "possession_team": ["ARI", "ARI", "ARI", "BAL", "BAL", "BAL"][:n],
            "offense_formation": ["SHOTGUN"] * n,
            "offense_personnel": ["1 RB, 1 TE, 3 WR"] * n,
            "defenders_in_box": [5, 6, 8, 7, 9, 6][:n],
            "defense_personnel": ["4 DL, 2 LB, 5 DB"] * n,
            "defense_coverage_type": ["ZONE"] * n,
        }
    )


def test_possession_team_is_a_recognised_team_column():
    """The exact regression: this name was missing from the candidates."""
    assert "possession_team" in _PART_TEAM_COL_CANDIDATES


def test_defenders_in_box_is_a_recognised_box_column():
    assert "defenders_in_box" in _BOX_COL_CANDIDATES


def test_resolves_the_real_nflverse_schema():
    box_col, team_col = resolve_participation_columns(nflverse_participation())
    assert (box_col, team_col) == ("defenders_in_box", "possession_team")


def test_resolution_prefers_the_real_names_over_later_candidates():
    """A frame carrying both must resolve to the nflverse names."""
    df = nflverse_participation()
    df["team"] = "ZZZ"
    df["box"] = 99
    box_col, team_col = resolve_participation_columns(df)
    assert (box_col, team_col) == ("defenders_in_box", "possession_team")


def test_empty_participation_resolves_to_nothing():
    assert resolve_participation_columns(pd.DataFrame()) == (None, None)


def test_unknown_schema_resolves_to_nothing():
    df = pd.DataFrame({"some_other_col": [1], "defence_in_the_box": [2]})
    assert resolve_participation_columns(df) == (None, None)


def test_box_rates_are_computed_for_every_team_in_participation():
    """The end-to-end behaviour that was broken: real rates, no NaNs."""
    pbp = pd.DataFrame(
        {
            "posteam": ["ARI", "ARI", "BAL", "BAL"],
            "personnel_offense": ["11", "12", "12", "11"],
        }
    )
    out = compute_personnel_rates(pbp, nflverse_participation())
    assert {"light_box_rate", "heavy_box_rate"}.issubset(out.columns)
    got = out.set_index("team")
    for team in ("ARI", "BAL"):
        assert not np.isnan(got.loc[team, "light_box_rate"]), team
        assert not np.isnan(got.loc[team, "heavy_box_rate"]), team
    # ARI boxes 5,6,8 -> light(<=6) 2/3, heavy(>=8) 1/3
    assert got.loc["ARI", "light_box_rate"] == 2 / 3
    assert got.loc["ARI", "heavy_box_rate"] == 1 / 3
    # BAL boxes 7,9,6 -> light 1/3, heavy 1/3
    assert got.loc["BAL", "light_box_rate"] == 1 / 3
    assert got.loc["BAL", "heavy_box_rate"] == 1 / 3


def test_prior_season_participation_still_yields_box_rates():
    """The backfill path: prior-season rows, current-season pbp.

    Box rates are grouped from participation alone, so a prior season's frame
    must still produce per-team rates even though its plays never join to the
    current season's pbp.
    """
    pbp_2026 = pd.DataFrame(
        {
            "posteam": ["ARI", "BAL"],
            "personnel_offense": ["11", "12"],
        }
    )
    out = compute_personnel_rates(pbp_2026, nflverse_participation())
    got = out.set_index("team")
    assert got.loc["ARI", "light_box_rate"] == 2 / 3
    assert got.loc["BAL", "heavy_box_rate"] == 1 / 3


def test_unresolvable_schema_leaves_box_columns_absent_not_wrong():
    """A resolution miss must not fabricate rates.

    It yields no box columns, which the strict validator rejects loudly. That
    is the correct outcome: fail closed, never a plausible-looking number.
    """
    pbp = pd.DataFrame({"posteam": ["ARI"], "personnel_offense": ["11"]})
    unknown = pd.DataFrame({"mystery_team": ["ARI"], "mystery_box": [6]})
    out = compute_personnel_rates(pbp, unknown)
    assert "light_box_rate" not in out.columns
    assert "heavy_box_rate" not in out.columns
