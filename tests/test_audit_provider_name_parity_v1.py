"""Pin the pre-fetch name-parity guard, including the bug it shipped with.

The first version matched positions against an exact set {"QB","RB","WR",...}.
Ourlads encodes alignment, not position: a receiver is LWR, RWR or SWR and
never "WR", so every receiver name gap passed straight through the strict
gate. That was caught on Joshua/Josh Palmer (BUF) and Mitch/Mitchell Tinsley
(CIN) -- two Week 9 teams -- and is the single most important property here.
"""
from __future__ import annotations

import pandas as pd

from scripts.audit_provider_name_parity_v1 import _initial_last, find_mismatches


def _nflverse() -> pd.DataFrame:
    rows = [
        # Synthetic names for the "a gap IS reported" cases: find_mismatches
        # canonicalizes through the real data/manual_name_overrides.csv, so a
        # real already-fixed name would correctly report nothing and the test
        # would be asserting the opposite of what it means to.
        ("BUF", "Zeb Quillfeather", "WR"),
        ("CIN", "Mortimer Grasswick", "WR"),
        ("BAL", "Nathaniel Dorrington", "TE"),
        ("KC", "Bartholomew Underwood", "OL"),
        # Real names, for the integration case below.
        ("BAL", "Matthew Hibner", "TE"),
        ("BUF", "Josh Palmer", "WR"),
        ("CIN", "Mitchell Tinsley", "WR"),
        ("BAL", "Mark Andrews", "TE"),
    ]
    d = pd.DataFrame(rows, columns=["team", "full_name", "position"])
    from scripts.utils.player_identity_v3 import player_name_key

    d["key"] = d["full_name"].map(lambda v: player_name_key(v, strip_suffix=True))
    d["il"] = d["full_name"].map(_initial_last)
    return d


def test_receiver_alignment_codes_count_as_priced_positions():
    """LWR/RWR/SWR are receivers. An exact-set test silently passed them."""
    roles = pd.DataFrame(
        [
            {"team": "BUF", "player": "Zebediah Quillfeather", "position": "LWR", "role": "WR1"},
            {"team": "CIN", "player": "Mort Grasswick", "position": "SWR", "role": "WR2"},
        ]
    )
    found = find_mismatches(roles, _nflverse())
    assert len(found) == 2
    assert found["priced_position"].all(), "a receiver name gap must be treated as priced"
    assert set(found["nflverse_name"]) == {"Zeb Quillfeather", "Mortimer Grasswick"}


def test_tight_end_gap_is_priced():
    roles = pd.DataFrame([{"team": "BAL", "player": "Nat Dorrington", "position": "TE", "role": "TE2"}])
    found = find_mismatches(roles, _nflverse())
    assert len(found) == 1
    assert bool(found["priced_position"].iloc[0])
    assert found["nflverse_name"].iloc[0] == "Nathaniel Dorrington"


def test_a_name_already_covered_by_an_override_is_not_reported_again():
    """data/manual_name_overrides.csv now carries all three; canonicalize first."""
    roles = pd.DataFrame(
        [
            {"team": "BAL", "player": "Matt Hibner", "position": "TE", "role": "TE2"},
            {"team": "BUF", "player": "Joshua Palmer", "position": "LWR", "role": "WR1"},
            {"team": "CIN", "player": "Mitch Tinsley", "position": "SWR", "role": "WR2"},
        ]
    )
    # The real repository override file is what makes these resolve.
    found = find_mismatches(roles, _nflverse())
    assert found.empty, f"already-overridden names reported again: {found.to_dict('records')}"


def test_matching_names_produce_no_finding():
    roles = pd.DataFrame([{"team": "BAL", "player": "Mark Andrews", "position": "TE", "role": "TE1"}])
    assert find_mismatches(roles, _nflverse()).empty


def test_a_player_absent_from_nflverse_is_not_this_audits_call():
    roles = pd.DataFrame([{"team": "KC", "player": "Totally Unknown", "position": "TE", "role": "TE3"}])
    assert find_mismatches(roles, _nflverse()).empty


def test_non_skill_gap_is_reported_but_not_priced():
    roles = pd.DataFrame([{"team": "KC", "player": "Bart Underwood", "position": "LT", "role": "LT1"}])
    found = find_mismatches(roles, _nflverse())
    assert len(found) == 1
    assert not bool(found["priced_position"].iloc[0])


def test_initial_last_strips_suffixes():
    assert _initial_last("Brian Thomas Jr") == "bthomas"
    assert _initial_last("Amon-Ra St. Brown") == "abrown"
    assert _initial_last("Hibner") == ""
