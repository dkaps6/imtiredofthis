"""The depth chart says "Matt Hibner"; the sportsbook says "Matthew Hibner".

Run 42094118645 bought the 2026 Week 9 Thursday odds, captured 738 alternate
offers, and then failed the whole board closed because one correctly-rostered
BAL TE2 was spelled differently by DraftKings than by Ourlads. The same
mismatch is visible in the Week 7 trace, so it recurs.

These tests pin the narrow fallback that fixes it, and -- more importantly --
pin the two properties that keep it from weakening the identity gate.
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.repair_live_prop_identity_v1 import (
    _build_roster_index,
    _infer_team,
    _initial_last_keys,
    _name_keys,
)

EVENTS = {"E1": ("BAL", "JAX")}


def _roles() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"team": "BAL", "player": "Matt Hibner"},
            {"team": "BAL", "player": "Mark Andrews"},
            {"team": "BAL", "player": "Zay Flowers"},
            {"team": "JAX", "player": "Brenton Strange"},
            {"team": "JAX", "player": "Travis Hunter"},
        ]
    )


@pytest.fixture()
def roster() -> dict[str, set[str]]:
    return _build_roster_index(_roles(), required_teams={"BAL", "JAX"})


@pytest.mark.parametrize(
    "posted, expected",
    [
        ("Matthew Hibner", "BAL"),  # the real 2026 W9 failure
        ("Matt Hibner", "BAL"),
        ("Mark Andrews", "BAL"),
        ("Brenton Strange", "JAX"),
        ("Travis Hunter", "JAX"),
    ],
)
def test_resolves_nickname_and_exact_names(roster, posted, expected):
    assert _infer_team(posted, "E1", roster, EVENTS) == expected


def test_unknown_player_still_fails_closed(roster):
    assert _infer_team("Nobody Atall", "E1", roster, EVENTS) == ""


def test_single_token_name_gets_no_fallback_key(roster):
    assert _initial_last_keys("Hibner") == set()
    assert _infer_team("Hibner", "E1", roster, EVENTS) == ""


def test_cross_team_surname_collision_fails_closed():
    """Two same-initial surnames on opposite sides of one game stay ambiguous."""
    roles = pd.DataFrame(
        [
            {"team": "BAL", "player": "Jerry Smith"},
            {"team": "JAX", "player": "Jake Smith"},
        ]
    )
    roster = _build_roster_index(roles, required_teams={"BAL", "JAX"})
    # Neither full name matches, and the fallback matches BOTH teams.
    assert _infer_team("Josh Smith", "E1", roster, EVENTS) == ""
    # An exact name is still resolved, so the collision costs nothing real.
    assert _infer_team("Jake Smith", "E1", roster, EVENTS) == "JAX"


def test_fallback_form_is_namespaced_out_of_name_keys():
    """Quarantine and manual-override matching must stay EXACT.

    Both read _name_keys. If the initial+surname form leaked into it,
    quarantining "Najee Harris" would also quarantine any other N. Harris.
    """
    assert _name_keys("Matt Hibner") == {"matthibner"}
    assert "il:mhibner" not in _name_keys("Matt Hibner")
    assert _initial_last_keys("Matt Hibner") == {"il:mhibner"}
    assert all(k.startswith("il:") for k in _initial_last_keys("Amon-Ra St. Brown"))


def test_suffix_is_stripped_before_the_fallback_key():
    assert _initial_last_keys("Brian Thomas Jr") == {"il:bthomas"}
    assert _initial_last_keys("Brian Thomas") == {"il:bthomas"}
