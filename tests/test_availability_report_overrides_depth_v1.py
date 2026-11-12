"""The club's injury filing outranks a third-party depth scrape.

Week 10 2026: Ourlads flagged Cam Skattebo INACTIVE, which made him
definitively unavailable and removed NYG's RB1 from the PlayerForm universe --
while the official report had him at FULL participation with no designation,
nflverse had him active and RB1 on that day's depth chart, and the sportsbooks
were pricing him. Deshaun Watson, MarShawn Lloyd and Kaleb Johnson were in the
same position with no report entry at all.

That left two production gates in direct contradiction: materialize_live_props
required those priced players to be in PlayerForm, and
validate_playerform_runtime_strict_prior forbade them. No board could be built.

Ourlads lags a return from injury -- Skattebo had not played since week 7 and
the scrape still carried the absence.

PRECEDENCE IS THE POINT. The override must never outrank real evidence of
absence: an official game-day inactive listing, or a hard OUT/IR/PUP
designation, both still win.
"""
from __future__ import annotations

import pytest

from scripts.build.build_current_player_availability_v1 import resolve_state

BASE = dict(official_inactive=False, official_complete=False,
            injury_status="", injury_designation="", ourlads_status="INACTIVE",
            practice_status="", team_report_available=True, roster_status="")


def _state(**kw):
    return resolve_state(**{**BASE, **kw})[0]


# ------------------------------------------------------- the override itself

@pytest.mark.parametrize("practice", [
    "Full Participation in Practice",
    "Limited Participation in Practice",
    "FULL PARTICIPATION IN PRACTICE",
])
def test_practising_overrides_a_stale_depth_inactive(practice):
    """Skattebo. Practising with no designation means he is with the team."""
    assert _state(practice_status=practice) == "AVAILABLE_REPORT_OVERRIDES_DEPTH"


def test_active_roster_membership_overrides_a_stale_depth_inactive():
    """Watson, Lloyd, Kaleb Johnson. OURLADS IS A DEPTH CHART, NOT AN
    AVAILABILITY SOURCE. Its job is who is on the team and in what order; the
    weekly roster decides membership and the injury report decides availability.

    This replaced an earlier "the club filed a report and omitted him" rule,
    which was wrong in a way this test now covers: a CUT or practice-squad
    player never appears on an injury report either, so that heuristic promoted
    exactly the players it should have excluded."""
    assert _state(roster_status="ACT") == "AVAILABLE_ACTIVE_ROSTER"


@pytest.mark.parametrize("status,label", [
    ("DEV", "practice squad"), ("CUT", "released"),
    ("RES", "reserve/IR"), ("RET", "retired"),
])
def test_a_non_member_is_unavailable_whatever_the_depth_chart_says(status, label):
    """The case the report-omission heuristic got wrong. Week 10 had 24 CUT,
    284 RES and 466 DEV players; Josh Johnson reads CUT."""
    assert _state(roster_status=status) == "UNAVAILABLE_NOT_ON_ACTIVE_ROSTER"


def test_membership_outranks_a_did_not_practise(monkeypatch):
    """Consequence worth stating: an ACT player with a DNP and NO designation
    resolves available. That is correct layering -- a depth scrape does not
    decide availability and there is no club designation -- but it means a
    midweek DNP is treated as fully available. That is the open P2
    injury_questionable_untreated item, tracked not silently resolved here."""
    assert _state(roster_status="ACT",
                  practice_status="Did Not Participate In Practice") == "AVAILABLE_ACTIVE_ROSTER"


def test_practice_evidence_still_applies_when_membership_is_unknown():
    """Provider failure must degrade, not break."""
    assert _state(roster_status="",
                  practice_status="Full Participation in Practice") == "AVAILABLE_REPORT_OVERRIDES_DEPTH"


# ------------------------------------------- what must NOT be overridden

def test_a_did_not_practise_player_is_not_promoted_without_membership():
    """With no membership fact and only a DNP, the depth source stands."""
    assert _state(practice_status="Did Not Participate In Practice") == "UNAVAILABLE_DEPTH_SOURCE"


def test_no_filing_at_all_leaves_the_depth_source_standing():
    """Absence from a filed report is evidence. Absence of a filing is not."""
    assert _state(team_report_available=False) == "UNAVAILABLE_DEPTH_SOURCE"


@pytest.mark.parametrize("status", ["Out", "OUT", "IR", "PUP", "Injured Reserve"])
def test_a_hard_designation_still_wins_over_full_practice(status):
    """A player can practise and still be ruled out. The report's designation
    is the decision; practice participation is only evidence."""
    assert _state(injury_status=status, roster_status="ACT",
                  practice_status="Full Participation in Practice") == "UNAVAILABLE_REPORTED"


def test_the_official_inactive_list_still_wins():
    """Game-day inactives are the last word and outrank everything."""
    assert _state(official_inactive=True, official_complete=True, roster_status="ACT",
                  practice_status="Full Participation in Practice") == "UNAVAILABLE_OFFICIAL_INACTIVE"


def test_questionable_still_resolves_uncertain_when_depth_is_active():
    assert _state(ourlads_status="ACTIVE", injury_designation="Questionable") == "UNCERTAIN"


# ------------------------------------------------- unchanged behaviour

def test_an_active_depth_source_is_unaffected_by_the_override():
    assert _state(ourlads_status="ACTIVE") in {
        "AVAILABLE_DEPTH_SOURCE", "AVAILABLE_REPORTED"}


def test_the_override_state_counts_as_available_not_unavailable():
    """definitive_unavailable is computed as state.startswith('UNAVAILABLE_'),
    so the new state must not accidentally read as unavailable."""
    assert not _state(practice_status="Full Participation in Practice").startswith("UNAVAILABLE_")
