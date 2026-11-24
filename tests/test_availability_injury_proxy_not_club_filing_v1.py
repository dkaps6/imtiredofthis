"""An injury-log proxy is not the club's filing.

ESPN's team injury endpoint is a proxy by its own provider's description, and
it carries any "out" entry logged within RECENT_WINDOW_DAYS (6 days). A player
ruled out for Sunday therefore still reads OUT on Thursday. Week 10 2026 had
six players stamped designation=OUT from that proxy -- authority recorded as
`weekly_injury_report` -- while the club's own nfl.com report had three of them
at FULL practice participation with no designation and two of them not on the
report at all. The universe dropped them, six priced sportsbook markets then
named players PlayerForm did not have, and the slate could not price.

These tests pin the authority ladder:

    official game-day inactives
      > the club's own injury-report designation and practice participation
      > ESPN injury-status proxy
      > Ourlads depth (ORDER only)

with nflverse weekly roster membership sitting beside that ladder rather than
inside it, because ACT is a MEMBERSHIP fact and not a game-availability one: a
player can be on the 53 and still be ruled out for Sunday. Membership therefore
refutes Ourlads' INACTIVE (which encodes IR/PUP/practice-squad) but must not
refute an injury-log OUT. Only the club's own report can do that.

Two load-bearing tests:
  * `test_club_out_still_definitive_for_active_roster` -- fixing the proxy must
    NOT promote a player the club actually ruled out.
  * `test_active_roster_alone_does_not_refute_a_game_availability_claim` -- the
    narrower reading above, which is the one the code implements.
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.build.build_current_player_availability_v1 import build, resolve_state

BASE = dict(
    official_inactive=False,
    official_complete=False,
    injury_status="",
    injury_designation="",
    ourlads_status="active",
)


def state(**kw):
    return resolve_state(**{**BASE, **kw})[0]


def authority(**kw):
    return resolve_state(**{**BASE, **kw})[1]


# --------------------------------------------------------------------------
# The proxy loses to roster membership and to the club's own report
# --------------------------------------------------------------------------

def test_active_roster_alone_does_not_refute_a_game_availability_claim():
    """ACT means on the 53, not cleared to play. It must not promote."""
    assert state(
        injury_status="OUT", injury_designation="OUT",
        club_injury_status="", club_designation="",
        espn_proxy_out=True, roster_status="ACT",
    ) == "UNAVAILABLE_REPORTED"


def test_club_practice_refutes_the_proxy():
    """Skattebo: proxy OUT, club report has him at FULL with no designation."""
    assert state(
        injury_status="OUT", injury_designation="OUT",
        club_injury_status="", club_designation="",
        espn_proxy_out=True, roster_status="ACT",
        practice_status="Full Participation in Practice",
    ) == "AVAILABLE_REPORT_OVERRIDES_PROXY"


def test_override_authority_is_the_club_report():
    assert authority(
        injury_status="OUT", injury_designation="OUT",
        club_injury_status="", club_designation="",
        espn_proxy_out=True, practice_status="Full Participation in Practice",
    ) == "weekly_injury_report"


@pytest.mark.parametrize("prac", [
    "Full Participation in Practice",
    "Limited Participation in Practice",
])
def test_proxy_out_loses_to_practice_when_no_roster_fact(prac):
    """Wright/Shaheed: full practice, no designation, roster unavailable."""
    assert state(
        injury_status="OUT", injury_designation="OUT",
        club_injury_status="", club_designation="",
        espn_proxy_out=True, roster_status="", practice_status=prac,
    ) == "AVAILABLE_REPORT_OVERRIDES_PROXY"


def test_proxy_out_stands_with_no_contrary_fact():
    """Fail closed: a proxy OUT with nothing against it still removes a player.

    This is Treylon Burks, MarShawn Lloyd and Deshaun Watson in week 10 2026 --
    DNP or absent from the club report entirely. The books priced all three.
    The football sources say unavailable, and a sportsbook offer is not
    permitted to overrule them (AGENTS.md section 3), so they stay out.
    """
    assert state(
        injury_status="OUT", injury_designation="OUT",
        club_injury_status="", club_designation="",
        espn_proxy_out=True, roster_status="",
        practice_status="Did Not Participate In Practice",
    ) == "UNAVAILABLE_REPORTED"


@pytest.mark.parametrize("unc", ["QUESTIONABLE", "DOUBTFUL"])
def test_club_uncertainty_corroborates_the_proxy(unc):
    """A club QUESTIONABLE alongside an injury-log OUT is agreement, not conflict.

    Full practice participation does not rescue it: the club itself filed
    uncertainty about his game status.
    """
    assert state(
        injury_status="OUT", injury_designation="OUT",
        club_injury_status=unc, club_designation=unc,
        espn_proxy_out=True, roster_status="ACT",
        practice_status="Full Participation in Practice",
    ) == "UNAVAILABLE_REPORTED"


@pytest.mark.parametrize("member", ["CUT", "RES", "DEV", "RET", "EXE"])
def test_proxy_out_with_non_member_roster_is_unavailable(member):
    """A genuine membership fact beats a practice line: Josh Johnson read CUT."""
    assert state(
        injury_status="OUT", injury_designation="OUT",
        club_injury_status="", club_designation="",
        espn_proxy_out=True, roster_status=member,
        practice_status="Full Participation in Practice",
    ) == "UNAVAILABLE_NOT_ON_ACTIVE_ROSTER"


# --------------------------------------------------------------------------
# The gate this fix must not weaken
# --------------------------------------------------------------------------

@pytest.mark.parametrize("desig", ["OUT", "IR", "PUP", "Injured Reserve"])
def test_club_out_still_definitive_for_active_roster(desig):
    """The club ruled him out. Active-roster membership does not undo that."""
    assert state(
        injury_status=desig, injury_designation=desig,
        club_injury_status=desig, club_designation=desig,
        espn_proxy_out=True, roster_status="ACT",
        practice_status="Full Participation in Practice",
    ) == "UNAVAILABLE_REPORTED"


def test_official_inactive_outranks_everything():
    assert state(
        official_inactive=True, official_complete=True,
        club_injury_status="", club_designation="",
        espn_proxy_out=True, roster_status="ACT",
        practice_status="Full Participation in Practice",
    ) == "UNAVAILABLE_OFFICIAL_INACTIVE"


def test_legacy_callers_unchanged():
    """Omitting the new arguments keeps the pre-existing merged-value read."""
    assert state(injury_status="OUT", injury_designation="OUT") == "UNAVAILABLE_REPORTED"
    assert state(injury_status="QUESTIONABLE") == "UNCERTAIN"
    assert state(ourlads_status="active") == "AVAILABLE_DEPTH_SOURCE"


def test_non_proxy_out_is_still_reported():
    """espn_proxy_out False must not route a real designation to the proxy branch."""
    assert state(
        injury_status="OUT", injury_designation="OUT",
        club_injury_status="OUT", club_designation="OUT",
        espn_proxy_out=False, roster_status="ACT",
    ) == "UNAVAILABLE_REPORTED"


# --------------------------------------------------------------------------
# End to end through build()
# --------------------------------------------------------------------------

def _depth(players):
    return pd.DataFrame([
        dict(team=t, player=p, status=s, role=r, position=pos,
             position_group=pg, depth_index=i)
        for i, (t, p, s, r, pos, pg) in enumerate(players, start=1)
    ])


def test_build_keeps_a_returning_player_eligible():
    depth = _depth([
        ("NYG", "Cam Skattebo", "active", "RB1", "RB", "RB"),
        ("NYG", "Devin Singletary", "active", "RB2", "RB", "RB"),
    ])
    injuries = pd.DataFrame([
        dict(team="NYG", player="Cam Skattebo", status="", designation="",
             practice_status="Full Participation in Practice", source="nfl.com"),
    ])
    # ESPN still carries last Sunday's OUT for him.
    official = pd.DataFrame([
        dict(team="NYG", player="Cam Skattebo", section_complete=0,
             source_semantics="INJURY_STATUS_OUT", provider_status="OUT"),
    ])
    out, _meta = build(depth, injuries, official)
    row = out.loc[out.player.eq("Cam Skattebo")].iloc[0]
    assert row.definitive_unavailable == 0
    assert row.final_availability_state == "AVAILABLE_REPORT_OVERRIDES_PROXY"
    assert row.eligible_for_opportunity == 1
    # and he is still RB1 -- the fix must not reshuffle depth order
    assert row.role_after_availability == "RB1"


def test_build_still_removes_a_club_ruled_out_player():
    depth = _depth([
        ("NYG", "Cam Skattebo", "active", "RB1", "RB", "RB"),
        ("NYG", "Devin Singletary", "active", "RB2", "RB", "RB"),
    ])
    injuries = pd.DataFrame([
        dict(team="NYG", player="Cam Skattebo", status="Out", designation="Out",
             practice_status="Did Not Participate In Practice", source="nfl.com"),
    ])
    out, _meta = build(depth, injuries, None)
    row = out.loc[out.player.eq("Cam Skattebo")].iloc[0]
    assert row.definitive_unavailable == 1
    assert row.final_availability_state == "UNAVAILABLE_REPORTED"
    # RB2 is promoted once RB1 is definitively out
    assert out.loc[out.player.eq("Devin Singletary")].iloc[0].role_after_availability == "RB1"


def test_build_reports_no_sportsbook_input():
    depth = _depth([("NYG", "Cam Skattebo", "active", "RB1", "RB", "RB")])
    _out, meta = build(depth, pd.DataFrame(), None)
    assert meta["sportsbook_inputs_used"] == 0
