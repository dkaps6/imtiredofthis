#!/usr/bin/env python3
"""Tests for Player Identity Card V1.

Weighted toward the five correctness defects raised in control-room audit
6420504608, because those are the ones that silently corrupt a named player's
state rather than crashing:

  A  snap counts loaded but never consumed
  B  active zero-opportunity weeks dropped from level/trajectory/stability
  C  target-week team inferred from historical play-by-play mode
  D  universe limited to players with observed prior opportunity
  E  snap/PBP identity joined by name instead of a stable id bridge

The distinction that matters most is B: an active week with zero touches is a
real zero and must count, while a week the player did not play must not be
coerced into a zero. Conflating those biases usage upward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.research.build_player_identity_card_v1 import (
    PARTIAL_PARTICIPATION_RATIO,
    build_cards,
    build_participation,
    player_opportunity_rows,
)

GSIS_A, GSIS_B, GSIS_ROOKIE = "00-0000001", "00-0000002", "00-0000003"
PFR_A, PFR_B, PFR_ROOKIE = "AaaA00", "BbbB00", "CccC00"


def rosters(weeks=(1, 2, 3), season=2026, team="PHI"):
    rows = []
    for w in weeks:
        for gsis, pfr, name, pos in (
            (GSIS_A, PFR_A, "Back A", "RB"),
            (GSIS_B, PFR_B, "Back B", "RB"),
            (GSIS_ROOKIE, PFR_ROOKIE, "Rookie C", "RB"),
        ):
            rows.append({"season": season, "week": w, "gsis_id": gsis,
                         "pfr_id": pfr, "team": team, "player_name": name,
                         "position": pos, "status": "ACT"})
    return pd.DataFrame(rows)


def snaps(spec, season=2026, team="PHI"):
    """spec: {(week, pfr_id): offense_snaps}

    Carries `team`, as real snap counts do: it is the authority for which club
    a player took snaps for in a given week, which the target-team-tenure
    share depends on. Use snaps_no_team() to exercise the degraded path.
    """
    return pd.DataFrame([
        {"season": season, "week": w, "pfr_player_id": p,
         "offense_snaps": n, "team": team}
        for (w, p), n in spec.items()
    ])


def snaps_no_team(spec, season=2026):
    """Snap counts without a team column, for the degraded-source paths."""
    return snaps(spec, season=season).drop(columns=["team"])


def pbp(spec, season=2026, team="PHI"):
    """spec: list of (week, gsis_id, yardline_100, yards)"""
    rows = []
    for i, (w, gsis, yl, yds) in enumerate(spec):
        rows.append({
            "season": season, "week": w, "posteam": team, "defteam": "DAL",
            "yardline_100": yl, "down": 1, "goal_to_go": 0,
            "half_seconds_remaining": 900, "yards_gained": yds,
            "rush_attempt": 1, "pass_attempt": 0,
            "rusher_player_id": gsis, "rusher_player_name": "x",
            "receiver_player_id": None, "play_id": i,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- E: bridge

def test_bridge_joins_on_ids_not_names():
    part, diag = build_participation(
        rosters(), snaps({(1, PFR_A): 30, (1, PFR_B): 10}))
    assert diag["bridge_state"] == "ok"
    assert diag["bridge_matched"] == 2
    assert diag["bridge_unmatched"] == 0
    assert set(part["gsis_id"]) == {GSIS_A, GSIS_B}


def test_bridge_reports_unmatched_rather_than_guessing():
    s = snaps({(1, PFR_A): 30, (1, "ZZZ999"): 25})
    part, diag = build_participation(rosters(), s)
    assert diag["bridge_matched"] == 1
    assert diag["bridge_unmatched"] == 1
    assert GSIS_A in set(part["gsis_id"])


def test_bridge_unavailable_without_snaps():
    part, diag = build_participation(rosters(), pd.DataFrame())
    assert part.empty
    assert diag["bridge_state"] == "unavailable_no_snaps"


def test_bridge_unavailable_when_columns_missing():
    bad = pd.DataFrame([{"season": 2026, "week": 1, "player": "Back A",
                         "offense_pct": 0.8}])
    part, diag = build_participation(rosters(), bad)
    assert part.empty
    assert diag["bridge_state"] == "unavailable_missing_columns"


def test_zero_snap_weeks_are_not_participation():
    part, _ = build_participation(
        rosters(), snaps({(1, PFR_A): 0, (2, PFR_A): 12}))
    assert sorted(part["week"]) == [2]


# ------------------------------------------- A/B: participation denominator

def _cards(part_spec, pbp_spec, week=3):
    ros = rosters()
    part, diag = build_participation(ros, snaps(part_spec))
    assert diag["bridge_state"] == "ok"
    cards, _ = build_cards(pbp(pbp_spec), ros, part, 2026, [week], [])
    return cards


def test_active_zero_opportunity_week_counts_as_a_real_zero():
    """A played weeks 1 and 2; carried only in week 1. Denominator is 2."""
    c = _cards({(1, PFR_A): 30, (2, PFR_A): 30},
               [(1, GSIS_A, 50, 4), (1, GSIS_A, 40, 6)])
    row = c[(c.gsis_id == GSIS_A) & (c.opportunity_kind == "carry")].iloc[0]
    assert row["participation_weeks_prior"] == 2
    assert row["opp_total_current_season"] == 2
    assert row["opp_weeks_observed_current"] == 1
    assert row["active_zero_opportunity_weeks"] == 1
    # 2 carries over 2 participation weeks, NOT over 1 observed week
    assert row["opp_per_participation_week"] == pytest.approx(1.0)


def test_dnp_weeks_are_excluded_not_zeroed():
    """A played only week 1. Week 2 was a DNP and must not become a zero."""
    c = _cards({(1, PFR_A): 30},
               [(1, GSIS_A, 50, 4), (1, GSIS_A, 40, 6)])
    row = c[(c.gsis_id == GSIS_A) & (c.opportunity_kind == "carry")].iloc[0]
    assert row["participation_weeks_prior"] == 1
    assert row["active_zero_opportunity_weeks"] == 0
    # 2 carries over 1 participation week — a DNP must not dilute this to 1.0
    assert row["opp_per_participation_week"] == pytest.approx(2.0)


def test_dropping_active_zeros_would_overstate_usage():
    """Pins the bias direction the audit identified."""
    both = _cards({(1, PFR_A): 30, (2, PFR_A): 30},
                  [(1, GSIS_A, 50, 4), (1, GSIS_A, 40, 6)])
    one = _cards({(1, PFR_A): 30},
                 [(1, GSIS_A, 50, 4), (1, GSIS_A, 40, 6)])
    r_both = both[(both.gsis_id == GSIS_A)
                  & (both.opportunity_kind == "carry")].iloc[0]
    r_one = one[(one.gsis_id == GSIS_A)
                & (one.opportunity_kind == "carry")].iloc[0]
    assert r_both["opp_per_participation_week"] < r_one["opp_per_participation_week"]


def test_stability_includes_the_zero_weeks():
    """sd must be computed over participation weeks, zeros included."""
    c = _cards({(1, PFR_A): 30, (2, PFR_A): 30},
               [(1, GSIS_A, 50, 4), (1, GSIS_A, 40, 6)])
    row = c[(c.gsis_id == GSIS_A) & (c.opportunity_kind == "carry")].iloc[0]
    # counts are [2, 0] -> sd of a two-point sample, not NaN
    assert not np.isnan(row["opp_week_to_week_sd"])
    assert row["opp_week_to_week_sd"] == pytest.approx(pd.Series([2, 0]).std())


def test_rates_fail_closed_without_participation():
    """No snaps means the denominator cannot be trusted: NaN, not a guess."""
    ros = rosters()
    cards, _ = build_cards(pbp([(1, GSIS_A, 50, 4)]), ros,
                           pd.DataFrame(), 2026, [3], [])
    row = cards[(cards.gsis_id == GSIS_A)
                & (cards.opportunity_kind == "carry")].iloc[0]
    assert row["participation_source_state"] == "unavailable"
    assert np.isnan(row["opp_per_participation_week"])
    assert np.isnan(row["opp_week_to_week_sd"])


# ------------------------------------------------- C: target-week team

def test_target_week_team_comes_from_roster_not_pbp_mode():
    """A's history is all with a different club; the roster is authoritative."""
    ros = rosters(team="PHI")
    part, _ = build_participation(ros, snaps({(1, PFR_A): 30, (2, PFR_A): 30}))
    hist = pbp([(1, GSIS_A, 50, 4), (2, GSIS_A, 40, 6)], team="NYG")
    cards, _ = build_cards(hist, ros, part, 2026, [3], [])
    row = cards[(cards.gsis_id == GSIS_A)
                & (cards.opportunity_kind == "carry")].iloc[0]
    assert row["team_target_week"] == "PHI"
    assert "NYG" in row["teams_in_history"]


# --------------------------------------------------------- D: universe

def test_player_with_no_prior_opportunity_still_gets_a_card():
    c = _cards({(1, PFR_A): 30}, [(1, GSIS_A, 50, 4)])
    rookie = c[c.gsis_id == GSIS_ROOKIE]
    assert len(rookie) == 2  # one card per opportunity kind
    assert set(rookie["history_state"]) == {"no_history"}
    assert (rookie["opp_total_all_sources"] == 0).all()


def test_every_rostered_player_is_represented():
    c = _cards({(1, PFR_A): 30}, [(1, GSIS_A, 50, 4)])
    assert set(c["gsis_id"]) == {GSIS_A, GSIS_B, GSIS_ROOKIE}


def test_participation_without_opportunity_is_labelled_distinctly():
    """Played, never touched the ball: not 'no_history'."""
    c = _cards({(1, PFR_B): 22}, [(1, GSIS_A, 50, 4)])
    row = c[(c.gsis_id == GSIS_B) & (c.opportunity_kind == "carry")].iloc[0]
    assert row["history_state"] == "participation_only_zero_opportunity"
    assert row["participation_weeks_prior"] == 1


# ------------------------------------------- strict prior + topology

def test_target_week_never_contributes_to_its_own_card():
    c = _cards({(1, PFR_A): 30, (2, PFR_A): 30, (3, PFR_A): 30},
               [(1, GSIS_A, 50, 4), (3, GSIS_A, 2, 1), (3, GSIS_A, 2, 1)],
               week=3)
    row = c[(c.gsis_id == GSIS_A) & (c.opportunity_kind == "carry")].iloc[0]
    assert row["opp_total_current_season"] == 1  # week 3 excluded
    assert row["inside_5_opp"] == 0              # week-3 goal-line carries excluded
    assert row["participation_weeks_prior"] == 2


def test_opportunity_quality_is_tracked_separately_from_volume():
    """The Dobbins distinction: low volume can own the valuable touches."""
    c = _cards({(1, PFR_A): 30, (1, PFR_B): 30},
               [(1, GSIS_A, 50, 5), (1, GSIS_A, 45, 5), (1, GSIS_A, 40, 5),
                (1, GSIS_B, 3, 1)])
    a = c[(c.gsis_id == GSIS_A) & (c.opportunity_kind == "carry")].iloc[0]
    b = c[(c.gsis_id == GSIS_B) & (c.opportunity_kind == "carry")].iloc[0]
    # A has 3x the volume; B owns the entire inside-5 work
    assert a["team_share_current"] > b["team_share_current"]
    assert b["team_inside_5_share_current"] == pytest.approx(1.0)
    assert a["team_inside_5_share_current"] == pytest.approx(0.0)


def test_situational_flags_are_derived_from_field_position():
    ev = player_opportunity_rows(
        pbp([(1, GSIS_A, 3, 1), (1, GSIS_A, 8, 2), (1, GSIS_A, 18, 3),
             (1, GSIS_A, 55, 4)]))
    assert ev["is_inside_5"].sum() == 1
    assert ev["is_inside_10"].sum() == 2
    assert ev["is_red_zone"].sum() == 3


def test_missing_situational_columns_fail_closed():
    """Absent field position, a quality flag would read as absent work."""
    thin = pbp([(1, GSIS_A, 50, 4)]).drop(columns=["yardline_100"])
    with pytest.raises(RuntimeError, match="situational columns"):
        player_opportunity_rows(thin)


def test_optional_passing_columns_do_not_crash_when_absent():
    """complete_pass/air_yards are not guaranteed; their absence is not fatal."""
    rows = pbp([(1, GSIS_A, 50, 4)])
    rows.loc[:, "pass_attempt"] = 1
    rows.loc[:, "rush_attempt"] = 0
    rows.loc[:, "receiver_player_id"] = GSIS_A
    assert "complete_pass" not in rows.columns
    ev = player_opportunity_rows(rows)
    tgt = ev[ev.opportunity_kind == "target"]
    assert len(tgt) == 1
    # Absent column is unknown, not "incomplete": a 0 here would feed
    # catch_rate a confident "catches nothing".
    assert np.isnan(tgt["complete"].iloc[0])
    assert np.isnan(tgt["air_yards"].iloc[0])


def test_present_but_null_complete_pass_reads_as_incomplete():
    rows = pbp([(1, GSIS_A, 50, 4)])
    rows.loc[:, "pass_attempt"] = 1
    rows.loc[:, "rush_attempt"] = 0
    rows.loc[:, "receiver_player_id"] = GSIS_A
    rows.loc[:, "complete_pass"] = np.nan
    ev = player_opportunity_rows(rows)
    tgt = ev[ev.opportunity_kind == "target"]
    assert tgt["complete"].iloc[0] == 0


def test_cards_never_carry_sportsbook_input():
    c = _cards({(1, PFR_A): 30}, [(1, GSIS_A, 50, 4)])
    assert (c["sportsbook_inputs_used"] == 0).all()
    assert c["strict_prior"].all()


# -------------------------------------- partial bridge denominator trust

def test_player_level_bridge_makes_partial_coverage_impossible():
    """A week whose roster row omits pfr_id must not strand that week.

    pfr_id is a player-level invariant, so one row carrying it is enough to
    bridge every week of that player. This is the regression guard for the
    (season, week, pfr_id) keying that discarded 25% of roster rows.
    """
    ros = rosters(weeks=(1, 2, 3))
    # B's week-2 row has no pfr_id, as a quarter of real roster rows do not.
    ros.loc[(ros.week == 2) & (ros.pfr_id == PFR_B), "pfr_id"] = np.nan
    part, diag = build_participation(
        ros, snaps({(1, PFR_B): 20, (2, PFR_B): 20, (1, PFR_A): 30}))
    # Week 2 still bridges, via the player-level map.
    assert sorted(part.loc[part.gsis_id == GSIS_B, "week"]) == [1, 2]
    assert diag["partial_bridge_player_weeks"] == 0
    assert diag["partial_bridge_players"] == 0
    assert diag["bridge_ambiguous_pfr_ids"] == 0


def test_ambiguous_identity_map_fails_closed():
    """One pfr_id mapping to two gsis ids would silently merge two players."""
    ros = rosters(weeks=(1, 2))
    ros.loc[(ros.week == 2) & (ros.gsis_id == GSIS_B), "pfr_id"] = PFR_A
    part, diag = build_participation(ros, snaps({(1, PFR_A): 30}))
    assert part.empty
    assert diag["bridge_state"] == "unavailable_ambiguous_identity_map"
    assert diag["bridge_ambiguous_pfr_ids"] == 1
    assert PFR_A in diag["bridge_ambiguous_examples"]


def test_never_bridged_player_is_not_counted_as_partial():
    _, diag = build_participation(
        rosters(), snaps({(1, "ZZZ999"): 20, (1, PFR_A): 30}))
    assert diag["partial_bridge_player_weeks"] == 0
    assert diag["unbridged_only_players"] == 1


def test_untrusted_denominator_fails_rates_closed_rather_than_inflating():
    """An understated denominator must yield NaN, never a confident rate."""
    ros = rosters()
    part, _ = build_participation(
        ros, snaps({(1, PFR_A): 30, (2, PFR_A): 30}))
    ev = [(1, GSIS_A, 50, 4), (1, GSIS_A, 40, 6)]

    clean, _ = build_cards(pbp(ev), ros, part, 2026, [3], [])
    r_clean = clean[(clean.gsis_id == GSIS_A)
                    & (clean.opportunity_kind == "carry")].iloc[0]
    assert r_clean["opp_per_participation_week"] == pytest.approx(1.0)
    assert r_clean["participation_source_state"] == "available"

    flagged, _ = build_cards(pbp(ev), ros, part, 2026, [3], [],
                             untrusted_denominator_ids={GSIS_A})
    r_flag = flagged[(flagged.gsis_id == GSIS_A)
                     & (flagged.opportunity_kind == "carry")].iloc[0]
    assert np.isnan(r_flag["opp_per_participation_week"])
    assert np.isnan(r_flag["opp_week_to_week_sd"])
    assert np.isnan(r_flag["active_zero_opportunity_weeks"])
    assert (r_flag["participation_source_state"]
            == "available_untrusted_partial_bridge")
    # Raw counts stay visible — only the rates built on a short denominator go.
    assert r_flag["opp_total_current_season"] == 2


def test_untrusted_flag_does_not_leak_to_other_players():
    ros = rosters()
    part, _ = build_participation(
        ros, snaps({(1, PFR_A): 30, (2, PFR_A): 30, (1, PFR_B): 30}))
    cards, _ = build_cards(
        pbp([(1, GSIS_A, 50, 4), (1, GSIS_B, 30, 3)]), ros, part, 2026, [3], [],
        untrusted_denominator_ids={GSIS_A})
    b = cards[(cards.gsis_id == GSIS_B)
              & (cards.opportunity_kind == "carry")].iloc[0]
    assert b["participation_source_state"] == "available"
    assert not np.isnan(b["opp_per_participation_week"])


def test_roster_coverage_breakdown_distinguishes_the_two_failure_modes():
    """A pfr_id the roster never lists is a different gap from one it lists
    without a gsis_id. Collapsing them hides which source to fix."""
    ros = rosters(weeks=(1,))
    # B is on the roster but carries no gsis_id -> unbridgeable row.
    ros.loc[ros.pfr_id == PFR_B, "gsis_id"] = np.nan
    _, diag = build_participation(
        ros, snaps({(1, PFR_A): 30, (1, PFR_B): 20, (1, "ZZZ999"): 15}))
    assert diag["roster_rows"] == 3
    assert diag["roster_rows_with_pfr"] == 3
    assert diag["roster_rows_bridgeable"] == 2
    assert diag["unbridged_only_players"] == 2          # B and ZZZ999
    assert diag["unbridged_on_roster_without_gsis"] == 1  # B
    assert diag["unbridged_absent_from_roster"] == 1      # ZZZ999


# ----------------------------------- PLAYER-STATE ANOMALY INVESTIGATION GATE

from scripts.research.build_player_identity_card_v1 import (  # noqa: E402
    STATE_HEALTHY, STATE_NO_BASELINE, STATE_UNRESOLVED,
    label_player_week_states,
)

WK = (1, 2, 3, 4, 5, 6)


def test_snap_collapse_is_unresolved_not_a_clean_week():
    part = pd.DataFrame([
        {"season": 2026, "week": w, "gsis_id": GSIS_A, "offense_snaps": n}
        for w, n in [(1, 60), (2, 60), (3, 60), (4, 8), (5, 60)]])
    st = label_player_week_states(part).set_index("week")["player_state"]
    assert st[1] == STATE_NO_BASELINE      # nothing earlier to judge against
    assert st[2] == STATE_NO_BASELINE
    assert st[3] == STATE_HEALTHY
    assert st[4] == STATE_UNRESOLVED       # 8 snaps against a 60 baseline
    assert st[5] == STATE_HEALTHY


def test_return_from_absence_is_unresolved():
    part = pd.DataFrame([
        {"season": 2026, "week": w, "gsis_id": GSIS_A, "offense_snaps": 40}
        for w in (1, 2, 3, 5)])          # week 4 missing entirely
    out = label_player_week_states(part).set_index("week")
    assert bool(out.loc[5, "returned_from_absence"])
    assert out.loc[5, "player_state"] == STATE_UNRESOLVED
    assert out.loc[3, "player_state"] == STATE_HEALTHY


def test_a_week_is_never_judged_against_itself():
    """Baseline uses strictly earlier weeks, so one bad week cannot excuse
    itself by dragging its own baseline down."""
    part = pd.DataFrame([
        {"season": 2026, "week": w, "gsis_id": GSIS_A, "offense_snaps": n}
        for w, n in [(1, 60), (2, 60), (3, 5)]])
    out = label_player_week_states(part).set_index("week")
    assert out.loc[3, "baseline_snaps"] == 60
    assert out.loc[3, "player_state"] == STATE_UNRESOLVED


def _gated_cards(snap_spec, pbp_spec, week=6):
    ros = rosters(weeks=WK)
    part, diag = build_participation(ros, snaps(snap_spec))
    assert diag["bridge_state"] == "ok"
    cards, cdiag = build_cards(pbp(pbp_spec), ros, part, 2026, [week], [])
    return cards, cdiag


def test_gate_withholds_the_anomalous_week_from_trajectory():
    snap = {(w, PFR_A): n for w, n in
            [(1, 60), (2, 60), (3, 60), (4, 8), (5, 60)]}
    ev = ([(w, GSIS_A, 50, 4) for w in (1, 1, 1, 2, 2, 2, 3, 3, 3, 5, 5, 5)])
    cards, _ = _gated_cards(snap, ev)
    r = cards[(cards.gsis_id == GSIS_A)
              & (cards.opportunity_kind == "carry")].iloc[0]
    # 12 carries. Over the 4 explained weeks that is 3.0/wk; over all 5
    # participation weeks (including the 8-snap week) it is 2.4/wk.
    assert r["opp_per_participation_week"] == pytest.approx(3.0)
    assert r["opp_per_participation_week_all_weeks"] == pytest.approx(2.4)
    assert r["gate_weeks_used"] == 4
    assert r["gate_weeks_withheld"] == 1
    assert r["opp_week_to_week_sd"] == pytest.approx(0.0)
    assert r["opp_week_to_week_sd_all_weeks"] > 0
    # Raw counts are per-play facts and the gate must not touch them.
    assert r["opp_total_current_season"] == 12
    assert f"{STATE_UNRESOLVED}=1" in r["gate_state_counts"]


def test_gate_does_not_withhold_a_healthy_zero_opportunity_week():
    """The gate must not undo fix (B): a healthy active week with no touches
    is a real zero and stays in the denominator."""
    snap = {(w, PFR_A): 60 for w in (1, 2, 3, 4, 5)}
    ev = [(w, GSIS_A, 50, 4) for w in (1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4)]
    cards, _ = _gated_cards(snap, ev)   # week 5: played 60 snaps, 0 carries
    r = cards[(cards.gsis_id == GSIS_A)
              & (cards.opportunity_kind == "carry")].iloc[0]
    assert r["gate_weeks_withheld"] == 0
    assert r["gate_weeks_used"] == 5
    # 12 carries over 5 weeks, NOT over the 4 weeks he happened to carry.
    assert r["opp_per_participation_week"] == pytest.approx(2.4)
    assert r["active_zero_opportunity_weeks"] == 1


def test_gate_applies_even_if_participation_arrives_unlabelled():
    """A forgotten labeller call must not silently publish ungated fields."""
    ros = rosters(weeks=WK)
    part, _ = build_participation(ros, snaps(
        {(w, PFR_A): n for w, n in [(1, 60), (2, 60), (3, 60), (4, 8), (5, 60)]}))
    assert "player_state" not in part.columns
    cards, diag = build_cards(
        pbp([(w, GSIS_A, 50, 4) for w in (1, 2, 3, 5)]),
        ros, part, 2026, [6], [])
    r = cards[(cards.gsis_id == GSIS_A)
              & (cards.opportunity_kind == "carry")].iloc[0]
    assert r["gate_weeks_withheld"] == 1
    assert diag["gate_weeks_withheld_total"] >= 1
    assert STATE_UNRESOLVED in diag["player_week_state_counts"]


def test_rate_is_reconstructible_from_its_own_published_inputs():
    snap = {(w, PFR_A): n for w, n in
            [(1, 60), (2, 60), (3, 60), (4, 8), (5, 60)]}
    ev = [(w, GSIS_A, 50, 4) for w in (1, 1, 2, 2, 3, 5)]
    cards, _ = _gated_cards(snap, ev)
    r = cards[(cards.gsis_id == GSIS_A)
              & (cards.opportunity_kind == "carry")].iloc[0]
    assert (r["opp_in_gate_weeks"] / r["gate_weeks_used"]
            == pytest.approx(r["opp_per_participation_week"]))


def _team_part(spec):
    """spec: {(week, gsis_id): (snaps, team)}"""
    return pd.DataFrame([
        {"season": 2026, "week": w, "gsis_id": g,
         "offense_snaps": n, "team": t}
        for (w, g), (n, t) in spec.items()])


def test_bye_week_gap_is_not_an_absence():
    """A team's bye is a gap for every player on it and explains itself.
    Uncorrected, this flags whole rosters and the gate looks like it is
    firing when it is really just reading the schedule."""
    spec = {}
    for g in ("A", "B"):                       # PHI idle in week 5
        for w in (3, 4, 6, 7):
            spec[(w, g)] = (60, "PHI")
    for g in ("E", "F"):                       # DAL plays every week
        for w in (3, 4, 5, 6, 7):
            spec[(w, g)] = (60, "DAL")
    out = label_player_week_states(_team_part(spec))
    phi6 = out[(out.week == 6) & out.team.eq("PHI")]
    assert not phi6["returned_from_absence"].any()
    assert phi6["absence_bye_explained"].all()
    assert (phi6["player_state"] == STATE_HEALTHY).all()


def test_absence_while_the_team_played_is_still_unresolved():
    """The bye correction must not excuse a real missed game."""
    spec = {(w, "E"): (60, "DAL") for w in (3, 4, 5, 6, 7)}
    for w in (3, 4, 6, 7):                     # D missed week 5; DAL played
        spec[(w, "D")] = (60, "DAL")
    out = label_player_week_states(_team_part(spec))
    d6 = out[(out.week == 6) & out.gsis_id.eq("D")].iloc[0]
    assert bool(d6["returned_from_absence"])
    assert not bool(d6["absence_bye_explained"])
    assert d6["player_state"] == STATE_UNRESOLVED


def test_without_a_team_column_the_stricter_reading_is_kept():
    """No team means a bye cannot be told from an absence; fail closed to
    UNRESOLVED rather than silently calling the week clean."""
    part = pd.DataFrame([
        {"season": 2026, "week": w, "gsis_id": GSIS_A, "offense_snaps": 60}
        for w in (3, 4, 6)])
    out = label_player_week_states(part).set_index("week")
    assert bool(out.loc[6, "returned_from_absence"])
    assert out.loc[6, "player_state"] == STATE_UNRESOLVED


def test_participation_reports_whether_bye_detection_is_possible():
    _, diag = build_participation(rosters(), snaps_no_team({(1, PFR_A): 30}))
    assert diag["bye_detection"] == "unavailable_no_team_col"
    _, ok = build_participation(rosters(), snaps({(1, PFR_A): 30}))
    assert ok["bye_detection"] == "available"


def test_efficiency_excludes_opportunities_from_unexplained_weeks():
    """The gate names efficiency, not just trajectory. A carry taken in an
    8-snap week is a real carry, but its yardage is not representative."""
    snap = {(w, PFR_A): n for w, n in
            [(1, 60), (2, 60), (3, 60), (4, 8), (5, 60)]}
    # 5 yards a carry in the healthy weeks, 0 in the collapsed week.
    ev = ([(w, GSIS_A, 50, 5) for w in (1, 1, 2, 2, 3, 3, 5, 5)]
          + [(4, GSIS_A, 50, 0), (4, GSIS_A, 50, 0)])
    cards, _ = _gated_cards(snap, ev)
    r = cards[(cards.gsis_id == GSIS_A)
              & (cards.opportunity_kind == "carry")].iloc[0]
    assert r["yards_per_opp"] == pytest.approx(5.0)        # gated
    assert r["yards_per_opp_all_weeks"] == pytest.approx(4.0)  # 40/10
    assert r["efficiency_opp_used"] == 8
    assert r["efficiency_opp_withheld"] == 2
    # The raw opportunity count is a per-play fact and stays whole.
    assert r["opp_total_current_season"] == 10


def test_efficiency_gate_is_a_no_op_when_no_week_is_withheld():
    snap = {(w, PFR_A): 60 for w in (1, 2, 3, 4, 5)}
    ev = [(w, GSIS_A, 50, 5) for w in (1, 2, 3, 4, 5)]
    cards, _ = _gated_cards(snap, ev)
    r = cards[(cards.gsis_id == GSIS_A)
              & (cards.opportunity_kind == "carry")].iloc[0]
    assert r["efficiency_opp_withheld"] == 0
    assert r["yards_per_opp"] == pytest.approx(r["yards_per_opp_all_weeks"])


# ------------------------------- source-qualified player-state resolution

from scripts.research.build_player_identity_card_v1 import (  # noqa: E402
    DISPOSITION_USABLE, DISPOSITION_WITHHELD, PARTIAL_GAME_SOURCE_STATE,
    STATE_INJURY_LIMITED, STATE_RETURN_FROM_INJURY,
)


def _inj(rows):
    """rows: list of (week, gsis_id, report_status, practice_status)"""
    return pd.DataFrame([
        {"season": 2026, "week": w, "gsis_id": g, "team": "PHI",
         "report_status": r, "practice_status": p}
        for w, g, r, p in rows])


def _snaps_pw(spec, team="PHI"):
    return pd.DataFrame([
        {"season": 2026, "week": w, "gsis_id": g, "offense_snaps": n,
         "team": team}
        for (w, g), n in spec.items()])


# A teammate who plays every week. Without one, a single player's absence
# makes his whole team look idle and the bye correction correctly excuses the
# gap -- real teams field ~25 offensive players, so the fixture needs company.
TEAMMATE = {(w, GSIS_B): 55 for w in range(1, 8)}
STEADY = {**TEAMMATE, **{(w, GSIS_A): 60 for w in (1, 2, 3)}}


def test_abnormal_week_with_a_designation_resolves_to_injury_limited():
    part = _snaps_pw({**STEADY, (4, GSIS_A): 8, (5, GSIS_A): 60})
    out = label_player_week_states(part, _inj([(4, GSIS_A, "Questionable", None)]))
    out = out[out.gsis_id == GSIS_A].set_index("week")
    assert out.loc[4, "player_state"] == STATE_INJURY_LIMITED
    assert out.loc[4, "injury_designation"] == "questionable"
    # Explained is not representative: still withheld.
    assert out.loc[4, "gate_disposition"] == DISPOSITION_WITHHELD


def test_abnormal_week_without_a_designation_stays_unresolved():
    part = _snaps_pw({**STEADY, (4, GSIS_A): 8})
    out = label_player_week_states(
        part, _inj([(4, GSIS_ROOKIE, "Out", None)]))
    out = out[out.gsis_id == GSIS_A].set_index("week")
    assert out.loc[4, "player_state"] == STATE_UNRESOLVED
    assert out.loc[4, "gate_disposition"] == DISPOSITION_WITHHELD


def test_return_after_an_out_designated_gap_resolves_to_return_from_injury():
    part = _snaps_pw({**STEADY, (5, GSIS_A): 60})   # week 4 missing
    out = label_player_week_states(part, _inj([(4, GSIS_A, "Out", None)]))
    out = out[out.gsis_id == GSIS_A].set_index("week")
    assert out.loc[5, "player_state"] == STATE_RETURN_FROM_INJURY
    assert out.loc[5, "injury_designation"] == "out"


def test_return_after_an_unexplained_gap_stays_unresolved():
    part = _snaps_pw({**STEADY, (5, GSIS_A): 60})
    out = label_player_week_states(
        part, _inj([(2, GSIS_A, None, "Full Participation in Practice")]))
    out = out[out.gsis_id == GSIS_A].set_index("week")
    assert out.loc[5, "player_state"] == STATE_UNRESOLVED


def test_a_later_designation_may_not_explain_an_earlier_gap():
    """Chronology: only weeks inside the gap can explain the gap."""
    part = _snaps_pw({**STEADY, (5, GSIS_A): 60})   # gap is week 4 only
    out = label_player_week_states(part, _inj([(6, GSIS_A, "Out", None)]))
    out = out[out.gsis_id == GSIS_A].set_index("week")
    assert out.loc[5, "player_state"] == STATE_UNRESOLVED


def test_resolution_never_changes_which_weeks_are_withheld():
    """The whole point: explanation changes the reason, not the fact. If this
    ever fails, the resolver has started promoting contaminated weeks."""
    part = _snaps_pw({**STEADY, (4, GSIS_A): 8, (6, GSIS_A): 60})
    inj = _inj([(4, GSIS_A, "Questionable", None), (5, GSIS_A, "Out", None)])
    bare = label_player_week_states(part)
    res = label_player_week_states(part, inj)
    key = ["season", "week", "gsis_id"]
    assert (set(map(tuple, bare.loc[bare.gate_disposition.eq(
                DISPOSITION_WITHHELD), key].to_numpy().tolist()))
            == set(map(tuple, res.loc[res.gate_disposition.eq(
                DISPOSITION_WITHHELD), key].to_numpy().tolist())))
    # ...and it did actually resolve something, or the test proves nothing.
    assert set(res["player_state"]) & {STATE_INJURY_LIMITED,
                                       STATE_RETURN_FROM_INJURY}


def test_without_injuries_the_gate_stays_fail_closed():
    part = _snaps_pw({**STEADY, (4, GSIS_A): 8})
    out = label_player_week_states(part)
    out = out[out.gsis_id == GSIS_A].set_index("week")
    assert out.loc[4, "player_state"] == STATE_UNRESOLVED
    assert out.loc[4, "injury_source_state"] == "unavailable"


def test_partial_game_is_declared_source_blocked_not_invented():
    """load_participation covers 2016-2025, so 2026 partial games cannot be
    established. Say so rather than guess from a snap count."""
    out = label_player_week_states(_snaps_pw(STEADY))
    assert (out["partial_game_source_state"] == PARTIAL_GAME_SOURCE_STATE).all()
    assert "SOURCE_BLOCKED" in PARTIAL_GAME_SOURCE_STATE
    assert "PARTIAL_GAME" not in set(out["player_state"])


def test_healthy_weeks_are_usable_and_unaffected_by_the_resolver():
    part = _snaps_pw(STEADY)
    out = label_player_week_states(part, _inj([(3, GSIS_A, "Questionable", None)]))
    out = out[out.gsis_id == GSIS_A].set_index("week")
    # Listed questionable but played a normal snap count: not an anomaly.
    assert out.loc[3, "player_state"] == STATE_HEALTHY
    assert out.loc[3, "gate_disposition"] == DISPOSITION_USABLE


# ------------------------------------- week-matched team-share denominators

def test_team_share_denominator_moves_with_the_numerator():
    """Gating only the player numerator while leaving the team denominator on
    all weeks mechanically depresses every share. Pin the arithmetic.

    A and B share a team. Weeks 1-3 are clean for both; in week 4 A's snaps
    collapse (withheld) and B absorbs the work.

      A carries: 5,5,5 then 1   = 16      B carries: 5,5,5 then 9 = 24
      team all weeks            = 40      team weeks 1-3          = 30

      ungated          16/40 = 0.400
      naive wrong      15/40 = 0.375   <- numerator gated, denominator not
      week-matched     15/30 = 0.500
    """
    snap = {(w, PFR_A): n for w, n in [(1, 60), (2, 60), (3, 60), (4, 8)]}
    snap.update({(w, PFR_B): 60 for w in (1, 2, 3, 4)})
    ev = []
    for w in (1, 2, 3):
        ev += [(w, GSIS_A, 50, 4)] * 5 + [(w, GSIS_B, 50, 4)] * 5
    ev += [(4, GSIS_A, 50, 4)] + [(4, GSIS_B, 50, 4)] * 9

    cards, _ = _gated_cards(snap, ev)
    a = cards[(cards.gsis_id == GSIS_A)
              & (cards.opportunity_kind == "carry")].iloc[0]

    assert a["opp_total_current_season"] == 16          # raw fact, ungated
    assert a["team_share_current_all_weeks"] == pytest.approx(0.400)
    assert a["team_share_current"] == pytest.approx(0.500)
    assert a["team_share_current"] != pytest.approx(0.375)   # the trap
    # Reconstructible from its own published inputs.
    assert a["player_opp_in_gate_weeks_current"] == 15
    assert a["team_opp_in_gate_weeks"] == 30
    assert (a["player_opp_in_gate_weeks_current"] / a["team_opp_in_gate_weeks"]
            == pytest.approx(a["team_share_current"]))


def test_high_value_shares_are_also_week_matched():
    snap = {(w, PFR_A): n for w, n in [(1, 60), (2, 60), (3, 60), (4, 8)]}
    snap.update({(w, PFR_B): 60 for w in (1, 2, 3, 4)})
    ev = []
    for w in (1, 2, 3):
        ev += [(w, GSIS_A, 3, 1), (w, GSIS_B, 3, 1)]       # one inside-5 each
    ev += [(4, GSIS_B, 3, 1)] * 4                           # B owns week 4
    cards, _ = _gated_cards(snap, ev)
    a = cards[(cards.gsis_id == GSIS_A)
              & (cards.opportunity_kind == "carry")].iloc[0]
    # 3 of 6 inside-5 in weeks 1-3; week 4's four go to B and are excluded
    # from BOTH sides of A's share.
    assert a["player_i5_opp_in_gate_weeks"] == 3
    assert a["team_i5_opp_in_gate_weeks"] == 6
    assert a["team_inside_5_share_current"] == pytest.approx(0.5)
    assert a["team_inside_5_share_current_all_weeks"] == pytest.approx(0.3)


def test_share_is_nan_not_zero_when_no_week_is_accepted():
    """An untrusted denominator must not yield a confident 0% share."""
    ros = rosters(weeks=WK)
    part, _ = build_participation(ros, snaps({(1, PFR_A): 60, (2, PFR_A): 60}))
    cards, _ = build_cards(
        pbp([(1, GSIS_A, 50, 4)]), ros, part, 2026, [6], [],
        untrusted_denominator_ids={GSIS_A})
    a = cards[(cards.gsis_id == GSIS_A)
              & (cards.opportunity_kind == "carry")].iloc[0]
    assert np.isnan(a["team_share_current"])
    assert a["team_opp_in_gate_weeks"] == 0


# -------------------------------------------- threshold sensitivity sweep

from scripts.research.sweep_player_state_threshold_v1 import (  # noqa: E402
    SWEEP, _assert_no_outcome_columns, _market_shaped_columns,
)


def test_sweep_covers_exactly_the_authorized_ratios():
    """GPT audit 6425282770 named these five. Not a tunable list."""
    assert SWEEP == (0.30, 0.40, 0.50, 0.60, 0.70)


def test_sweep_refuses_market_columns():
    """The sweep is descriptive. If it is ever pointed at priced data it must
    fail closed rather than quietly produce a fitted result."""
    for col in ("model_proj", "qb_synthesis_proj", "closing_line", "line",
                "over_odds", "implied_prob", "edge", "clv", "bookmaker"):
        with pytest.raises(RuntimeError, match="descriptive"):
            _assert_no_outcome_columns(
                pd.DataFrame({"season": [2026], col: [1]}), "probe")


def test_guard_does_not_false_positive_on_real_football_columns():
    """A substring test flags yardline_100 for "line" and qb_hit for "hit".
    A guard that fires on real data gets loosened until it is useless, so
    matching is token-wise."""
    legit = pd.DataFrame(columns=[
        "yardline_100", "qb_hit", "air_yards", "goal_to_go", "pass_attempt",
        "half_seconds_remaining", "offense_snaps", "pfr_player_id",
        "report_status", "practice_status", "defenders_in_box",
    ])
    assert _market_shaped_columns(legit) == []
    _assert_no_outcome_columns(legit, "legit")


def test_sweep_accepts_the_frames_it_actually_reads():
    _assert_no_outcome_columns(rosters(), "rosters")
    _assert_no_outcome_columns(snaps({(1, PFR_A): 30}), "snaps")


def test_looser_threshold_never_flags_fewer_weeks():
    """Monotonicity is the sweep's own correctness check: raising the ratio
    must flag at least as many weeks abnormal, never fewer."""
    part = _snaps_pw({**TEAMMATE,
                      **{(1, GSIS_A): 60, (2, GSIS_A): 60, (3, GSIS_A): 60,
                         (4, GSIS_A): 36, (5, GSIS_A): 18}})
    withheld = []
    for r in SWEEP:
        out = label_player_week_states(part, partial_ratio=r)
        withheld.append(int((out["gate_disposition"] != DISPOSITION_USABLE).sum()))
    assert withheld == sorted(withheld), withheld
    # ...and the fixture actually exercises the range rather than being flat.
    assert withheld[-1] > withheld[0]


def test_threshold_is_recorded_on_the_labelled_frame():
    out = label_player_week_states(_snaps_pw(STEADY), partial_ratio=0.3)
    assert (out["partial_participation_ratio"] == 0.3).all()


def test_default_threshold_is_unchanged_by_parameterisation():
    a = label_player_week_states(_snaps_pw(STEADY))
    b = label_player_week_states(_snaps_pw(STEADY),
                                 partial_ratio=PARTIAL_PARTICIPATION_RATIO)
    assert a["player_state"].tolist() == b["player_state"].tolist()


# ------------------------- evidence authority + model-facing universe

from scripts.research.build_player_identity_card_v1 import (  # noqa: E402
    EVIDENCE_POSITION_PRIOR_ONLY, UNIVERSE_COMPLETE, UNIVERSE_MODEL_FACING,
    model_facing_universe,
)


def test_zero_history_player_is_position_prior_only_not_merely_no_history():
    """Production does not leave a zero-history player blank: it hands him the
    position prior and lets him compete in the finite room. The card must name
    the authority, not just report absence."""
    c = _cards({(1, PFR_A): 30}, [(1, GSIS_A, 50, 4)])
    r = c[(c.gsis_id == GSIS_ROOKIE) & (c.opportunity_kind == "carry")].iloc[0]
    assert r["history_state"] == "no_history"
    assert r["evidence_authority"] == EVIDENCE_POSITION_PRIOR_ONLY
    assert bool(r["opportunity_from_population_fallback"])
    assert r["participation_evidence"] == "none"


def test_observed_zero_opportunity_is_still_population_fallback_but_labelled():
    """He played and never touched the ball. The opportunity posterior is
    still the prior (n=0), but an observed zero is evidence and the card must
    not conflate it with never having played."""
    c = _cards({(1, PFR_B): 22}, [(1, GSIS_A, 50, 4)])
    r = c[(c.gsis_id == GSIS_B) & (c.opportunity_kind == "carry")].iloc[0]
    assert r["history_state"] == "participation_only_zero_opportunity"
    assert r["evidence_authority"] == EVIDENCE_POSITION_PRIOR_ONLY
    assert r["participation_evidence"] == "observed_zero_opportunity"


def test_player_evidence_is_not_flagged_as_fallback():
    c = _cards({(1, PFR_A): 30, (2, PFR_A): 30},
               [(1, GSIS_A, 50, 4), (2, GSIS_A, 40, 6)])
    r = c[(c.gsis_id == GSIS_A) & (c.opportunity_kind == "carry")].iloc[0]
    assert r["evidence_authority"] == "current_only"
    assert not bool(r["opportunity_from_population_fallback"])
    assert r["participation_evidence"] == "present"


def test_durable_card_is_the_complete_roster_universe():
    c = _cards({(1, PFR_A): 30}, [(1, GSIS_A, 50, 4)])
    assert (c["universe"] == UNIVERSE_COMPLETE).all()


def test_model_facing_universe_is_eligible_only_and_restamped():
    c = _cards({(1, PFR_A): 30}, [(1, GSIS_A, 50, 4)])
    mf = model_facing_universe(c)
    assert len(mf) <= len(c)
    assert mf["can_receive_opportunity"].all()
    assert (mf["universe"] == UNIVERSE_MODEL_FACING).all()
    # The durable card is untouched by taking the subset.
    assert (c["universe"] == UNIVERSE_COMPLETE).all()


def test_model_facing_universe_excludes_non_opportunity_positions():
    ros = rosters()
    ros.loc[ros.gsis_id == GSIS_ROOKIE, "position"] = "OL"
    part, _ = build_participation(ros, snaps({(1, PFR_A): 30}))
    cards, _ = build_cards(pbp([(1, GSIS_A, 50, 4)]), ros, part, 2026, [3], [])
    mf = model_facing_universe(cards)
    assert GSIS_ROOKIE in set(cards["gsis_id"])      # durable card keeps him
    assert GSIS_ROOKIE not in set(mf["gsis_id"])     # model-facing does not


def test_model_facing_universe_refuses_to_guess():
    with pytest.raises(RuntimeError, match="refusing to guess"):
        model_facing_universe(pd.DataFrame({"gsis_id": [GSIS_A]}))


def test_guard_does_not_flag_negative_attestations():
    """A column whose job is to record that a boundary was NOT crossed must
    contain the word it attests about. sportsbook_inputs_used = 0 is the proof
    the boundary held; flagging it inverts its meaning."""
    att = pd.DataFrame(columns=["sportsbook_inputs_used",
                                "injury_source_state",
                                "partial_game_source_state"])
    assert _market_shaped_columns(att) == []


def test_guard_still_catches_a_real_market_column_beside_an_attestation():
    mixed = pd.DataFrame(columns=["sportsbook_inputs_used", "over_odds"])
    assert _market_shaped_columns(mixed) == ["over_odds"]


# ------------------- audit 6425282770 BLOCKER A: NO_BASELINE contamination

def test_first_weeks_with_a_limiting_designation_are_withheld():
    """Blocker A. A player in his first/second participation week with a KNOWN
    limiting designation must not enter trajectory as usable just because
    there is no baseline to judge a snap collapse against."""
    part = _snaps_pw({**TEAMMATE, (1, GSIS_A): 20, (2, GSIS_A): 18})
    out = label_player_week_states(
        part, _inj([(1, GSIS_A, "Questionable", None)]))
    out = out[out.gsis_id == GSIS_A].set_index("week")
    assert out.loc[1, "player_state"] == STATE_INJURY_LIMITED
    assert out.loc[1, "gate_disposition"] == DISPOSITION_WITHHELD
    # The uncontaminated early week is still NO_BASELINE and still usable.
    assert out.loc[2, "player_state"] == STATE_NO_BASELINE
    assert out.loc[2, "gate_disposition"] == DISPOSITION_USABLE


def test_return_from_injury_with_insufficient_baseline_is_withheld():
    """Blocker A. The return state needs no baseline to be meaningful."""
    part = _snaps_pw({**TEAMMATE, (1, GSIS_A): 30, (3, GSIS_A): 30})
    out = label_player_week_states(part, _inj([(2, GSIS_A, "Out", None)]))
    out = out[out.gsis_id == GSIS_A].set_index("week")
    assert out.loc[3, "player_state"] == STATE_RETURN_FROM_INJURY
    assert out.loc[3, "gate_disposition"] == DISPOSITION_WITHHELD


def test_sparse_player_without_contamination_stays_no_baseline():
    """Blocker A, the other direction: absence of a baseline must not be
    read as evidence of being unhealthy."""
    part = _snaps_pw({**TEAMMATE, (1, GSIS_A): 25, (2, GSIS_A): 25})
    out = label_player_week_states(part, _inj([(1, GSIS_B, "Out", None)]))
    out = out[out.gsis_id == GSIS_A].set_index("week")
    assert (out["player_state"] == STATE_NO_BASELINE).all()
    assert (out["gate_disposition"] == DISPOSITION_USABLE).all()
    assert not out["has_limiting_designation"].any()


def test_designation_with_a_normal_snap_count_is_not_contamination():
    """A questionable tag that did not move his workload is not contamination,
    and the fix for blocker A must not start withholding healthy weeks."""
    part = _snaps_pw({**TEAMMATE, **{(w, GSIS_A): 60 for w in (1, 2, 3, 4)}})
    out = label_player_week_states(
        part, _inj([(4, GSIS_A, "Questionable", None)]))
    out = out[out.gsis_id == GSIS_A].set_index("week")
    assert out.loc[4, "player_state"] == STATE_HEALTHY
    assert out.loc[4, "gate_disposition"] == DISPOSITION_USABLE
    assert bool(out.loc[4, "has_limiting_designation"])  # seen, not acted on


def test_no_baseline_is_not_a_bypass_for_the_disposition_rule():
    """Belt and braces: no USABLE card may carry a limiting designation in a
    week the builder could not assess."""
    part = _snaps_pw({**TEAMMATE, (1, GSIS_A): 20, (2, GSIS_A): 20,
                      (3, GSIS_A): 20})
    out = label_player_week_states(
        part, _inj([(1, GSIS_A, "Out", None), (2, GSIS_A, "Doubtful", None)]))
    a = out[out.gsis_id == GSIS_A]
    bypassed = a[(a.gate_disposition == DISPOSITION_USABLE)
                 & a.has_limiting_designation
                 & a.player_state.eq(STATE_NO_BASELINE)]
    assert bypassed.empty, bypassed[["week", "player_state"]].to_dict("records")


# -------------------- audit 6425282770 BLOCKER B: trade share semantics

def test_in_season_trade_cannot_mix_old_team_numerator_with_new_team_denominator():
    """Blocker B, the test the audit asked for: large old-team workload, sparse
    new-team tenure. No old-team opportunity may enter the target-team share.

    A plays weeks 1-3 for NYG with heavy volume, is traded, and plays week 4
    for PHI sparsely. His week-6 card has team_target_week = PHI.

      old team  3 weeks x 10 carries = 30   (must NOT count)
      new team  1 week  x  2 carries =  2
      PHI team total in week 4       =  2 + 8 (teammate) = 10

      target-team-tenure share = 2/10 = 0.20
      the old broken form would have been 32/10 = 3.2, or 2/10 vs a
      season-wide PHI denominator -- either way not a current-role share.
    """
    ros = rosters(weeks=WK, team="PHI")          # target-week roster says PHI
    s_old = snaps({(w, PFR_A): 60 for w in (1, 2, 3)}, team="NYG")
    s_new = snaps({(4, PFR_A): 60, (4, PFR_B): 60}, team="PHI")
    part, diag = build_participation(ros, pd.concat([s_old, s_new],
                                                   ignore_index=True))
    assert diag["bridge_state"] == "ok"

    ev = ([(w, GSIS_A, 50, 4) for w in (1, 1, 2, 2, 3, 3)] * 5)[:30]
    old_rows = pbp(ev, team="NYG")
    new_rows = pbp([(4, GSIS_A, 50, 4)] * 2 + [(4, GSIS_B, 50, 4)] * 8,
                   team="PHI")
    cards, _ = build_cards(pd.concat([old_rows, new_rows], ignore_index=True),
                           ros, part, 2026, [6], [])
    a = cards[(cards.gsis_id == GSIS_A)
              & (cards.opportunity_kind == "carry")].iloc[0]

    assert a["team_target_week"] == "PHI"
    assert "NYG" in a["teams_in_history"]
    assert a["team_share_semantic"] == "TARGET_TEAM_TENURE_SHARE_V1"
    # Only the PHI week counts as tenure.
    assert a["target_team_tenure_weeks"] == 1
    assert a["opp_total_current_season_target_team"] == 2
    # ...while the raw all-teams count is untouched, being a fact not a claim.
    assert a["opp_total_current_season"] == 32
    # The share is PHI-only on both sides.
    assert a["player_opp_in_gate_weeks_current"] == 2
    assert a["team_opp_in_gate_weeks"] == 10
    assert a["team_share_current"] == pytest.approx(0.2)
    assert a["team_share_current_all_weeks"] == pytest.approx(0.2)
    assert not bool(a["no_target_team_tenure_evidence"])


def test_no_target_team_tenure_yields_nan_not_the_previous_team_role():
    """A player with history but no tenure on his target-week team must not
    import his old role. NaN, explicitly flagged."""
    ros = rosters(weeks=WK, team="PHI")
    part, _ = build_participation(
        ros, snaps({(w, PFR_A): 60 for w in (1, 2, 3)}, team="NYG"))
    cards, _ = build_cards(pbp([(w, GSIS_A, 50, 4) for w in (1, 2, 3)],
                               team="NYG"),
                           ros, part, 2026, [6], [])
    a = cards[(cards.gsis_id == GSIS_A)
              & (cards.opportunity_kind == "carry")].iloc[0]
    assert bool(a["no_target_team_tenure_evidence"])
    assert a["target_team_tenure_weeks"] == 0
    assert np.isnan(a["team_share_current"])
    assert np.isnan(a["team_share_current_all_weeks"])
    # His own history is still visible; only the role CLAIM is withheld.
    assert a["opp_total_current_season"] == 3
    assert a["opp_total_current_season_target_team"] == 0


def test_tenure_source_is_reported():
    c = _cards({(1, PFR_A): 30}, [(1, GSIS_A, 50, 4)])
    assert (c["target_team_tenure_source"] == "participation_team").all()


def test_target_week_one_has_no_tenure_authority_and_claims_no_share():
    """Strictly-prior means week 1 has no earlier weeks at all, so there is
    nothing to derive tenure from and nothing to fall back to. It must say so
    rather than implying an inference, and it must publish no share."""
    ros = rosters(weeks=WK)
    part, _ = build_participation(ros, snaps({(1, PFR_A): 60}))
    cards, _ = build_cards(pbp([(1, GSIS_A, 50, 4)]), ros, part, 2026, [1], [])
    assert (cards["target_team_tenure_source"]
            == "no_prior_participation").all()
    assert cards["no_target_team_tenure_evidence"].astype(bool).all()
    assert pd.to_numeric(cards["team_share_current"],
                         errors="coerce").isna().all()
    assert pd.to_numeric(cards["team_share_current_all_weeks"],
                         errors="coerce").isna().all()


def test_a_weak_tenure_source_may_never_publish_a_share():
    """The safety property the CI gate asserts, pinned in code: if the
    participation authority was not used, no tenure and no share."""
    ros = rosters(weeks=WK)
    part, _ = build_participation(ros, snaps_no_team({(1, PFR_A): 60,
                                                      (2, PFR_A): 60}))
    cards, _ = build_cards(pbp([(1, GSIS_A, 50, 4)]), ros, part, 2026, [3], [])
    weak = cards["target_team_tenure_source"].ne("participation_team")
    assert weak.any()
    leaked = weak & ~cards["no_target_team_tenure_evidence"].astype(bool)
    assert not leaked.any()


# ------------- audit 6425282770 MATERIALITY: fallback decomposition

from scripts.research.decompose_position_prior_fallback_materiality_v1 import (  # noqa: E402
    DEFAULTS as PROD_DEFAULTS,
    _position_group as prod_position_group,
    _prior_value,
    annotate,
)


def test_priors_come_from_productions_own_table():
    """Retyping the audit's illustrative values would let them drift from what
    production applies. These are read out of bayesian_v2."""
    assert PROD_DEFAULTS["tgt_share"]["QB"] == 0.0
    assert PROD_DEFAULTS["rush_share"]["TE"] == 0.0
    assert PROD_DEFAULTS["rush_share"]["WR"] == 0.02
    assert PROD_DEFAULTS["rush_share"]["RB"] == 0.32
    assert PROD_DEFAULTS["tgt_share"]["RB"] == 0.08


def test_fullback_and_halfback_map_to_rb_like_production():
    for p in ("FB", "HB", "RB"):
        assert prod_position_group(p) == "RB"
    assert _prior_value("carry", "FB") == 0.32


def test_zero_default_prior_is_not_a_nonzero_candidate():
    """The correction the audit asked for: a QB with no target history and a TE
    with no rush history are position_prior_only, but their production priors
    are 0.0, so counting them as architecture severity overstates it."""
    assert _prior_value("target", "QB") == 0.0
    assert _prior_value("carry", "TE") == 0.0
    cards = pd.DataFrame([
        {"gsis_id": "a", "roster_position": "QB", "opportunity_kind": "target",
         "evidence_authority": "position_prior_only",
         "can_receive_opportunity": True},
        {"gsis_id": "b", "roster_position": "TE", "opportunity_kind": "carry",
         "evidence_authority": "position_prior_only",
         "can_receive_opportunity": True},
        {"gsis_id": "c", "roster_position": "RB", "opportunity_kind": "carry",
         "evidence_authority": "position_prior_only",
         "can_receive_opportunity": True},
    ])
    out = annotate(cards)
    assert out["is_position_prior_only"].all()
    assert out.loc[0, "nonzero_default_prior_candidate"] == False   # noqa: E712
    assert out.loc[1, "nonzero_default_prior_candidate"] == False   # noqa: E712
    assert out.loc[2, "nonzero_default_prior_candidate"] == True    # noqa: E712
    assert out["default_prior_mass_proxy"].tolist() == [0.0, 0.0, 0.32]


def test_player_evidence_rows_contribute_no_proxy_mass():
    cards = pd.DataFrame([
        {"gsis_id": "a", "roster_position": "RB", "opportunity_kind": "carry",
         "evidence_authority": "prior+current", "can_receive_opportunity": True},
    ])
    out = annotate(cards)
    assert not out["is_position_prior_only"].any()
    assert out["default_prior_mass_proxy"].iloc[0] == 0.0
    assert not out["nonzero_default_prior_candidate"].any()


# ------------- re-audit 6427742545: production fallback admission

from scripts.research.reconcile_production_fallback_admission_v1 import (  # noqa: E402
    _BOARD_COLS_ALLOWED, _UNCERTIFIED_MARKETS, reconcile,
)


def _board(rows):
    """rows: (player, team, market, bayes_evidence_state)"""
    return pd.DataFrame([
        {"player": p, "team": t, "market": m, "bayes_evidence_state": s,
         "season": 2026, "week": 7}
        for p, t, m, s in rows])


_CARD7_COLS = ["gsis_id", "player", "evidence_authority", "target_week",
               "can_receive_opportunity", "opportunity_kind"]


def _card7(rows):
    """rows: (gsis_id, player, evidence_authority)"""
    if not rows:
        return pd.DataFrame(columns=_CARD7_COLS)
    return pd.DataFrame([
        {"gsis_id": g, "player": p, "evidence_authority": e,
         "target_week": 7, "can_receive_opportunity": True,
         "opportunity_kind": "carry"}
        for g, p, e in rows])


def test_production_admission_comes_from_the_boards_own_label():
    """The board carries production's bayes_evidence_state, so admission needs
    no inference from DEFAULTS -- the error the re-audit caught."""
    board = _board([
        ("Deep Reserve", "SEA", "anytime_td", "position_prior_only"),
        ("Deep Reserve", "SEA", "rec_yards", "position_prior_only"),
        ("Star Back", "SEA", "rush_yards", "prior+current"),
    ])
    card = _card7([("1", "Deep Reserve", "position_prior_only"),
                   ("2", "Star Back", "prior+current"),
                   ("3", "Never Priced", "position_prior_only")])
    st, detail = reconcile(card, board, 7)
    assert st["production_fallback_rows"] == 2
    assert st["production_fallback_players"] == 1
    # The card says two fallback players; production admitted one.
    assert st["card_fallback_players"] == 2
    assert st["production_fallback_also_fallback_in_card"] == 1
    assert len(detail) == 2


def test_uncertified_market_rows_are_counted_separately():
    """A fallback row in anytime_td is a different problem from one in a
    graded yardage market, and the audit has ruled ATD uncertified."""
    assert "anytime_td" in _UNCERTIFIED_MARKETS
    board = _board([
        ("A", "SEA", "anytime_td", "position_prior_only"),
        ("A", "SEA", "anytime_td", "position_prior_only"),
        ("B", "SEA", "rec_yards", "position_prior_only"),
    ])
    st, detail = reconcile(_card7([]), board, 7)
    assert st["production_fallback_rows_uncertified_market"] == 2
    assert st["production_fallback_rows_certified_market"] == 1
    assert detail.loc[detail.market.eq("anytime_td"),
                      "market_uncertified"].all()


def test_fallback_posterior_is_declared_not_established():
    """The quantity the re-audit actually wanted cannot be computed from this
    repo, and the script must say so rather than substituting DEFAULTS."""
    st, _ = reconcile(_card7([]), _board([("A", "SEA", "rec_yards",
                                           "prior+current")]), 7)
    assert st["fallback_posterior_established"] is False
    assert "player_form_consensus" in st["fallback_posterior_blocker"]
    assert "DEFAULTS" in st["fallback_posterior_blocker"]


def test_reconciliation_reads_no_decision_columns():
    """Outcome-blind: the allowlist must contain no line/odds/edge column."""
    banned = ("line", "odds", "price", "edge", "prob", "proj")
    assert not [c for c in _BOARD_COLS_ALLOWED
                if any(k in c.lower() for k in banned)]


def test_production_fallback_player_missing_from_card_is_surfaced():
    """Disagreement in that direction is a finding, not something to hide."""
    board = _board([("Ghost", "SEA", "rec_yards", "position_prior_only")])
    card = _card7([("1", "Ghost", "prior_only")])
    st, _ = reconcile(card, board, 7)
    assert st["production_fallback_not_fallback_in_card"] == ["Ghost"]


# ---------- audit response 6428042413: week-7 opportunity trace preservation

from scripts.research.preserve_week7_opportunity_trace_v1 import (  # noqa: E402
    _DENY_EXEMPT, _WANTED, _denied, extract,
)


def test_sportsbook_and_decision_columns_are_denied():
    """The artifact contains the priced board, so exclusion must be active."""
    for c in ("vegas_line", "vegas_over_odds", "edge_pct", "fair_prob",
              "market_prob", "model_proj", "mc_proj", "model_sd", "book",
              "implied_prob", "edge_abs"):
        assert _denied(c), c


def test_football_explanatory_columns_are_not_denied():
    for c in ("prior_games", "current_games", "bayes_rush_share",
              "bayes_tgt_share", "rules_rush_share", "rush_share_prior",
              "target_entitlement", "analytic_expected_carries", "rules_role"):
        assert not _denied(c), c


def test_provenance_attestations_are_exempt_from_the_denylist():
    """`sportsbook_inputs_used` proves the boundary held; denying it would
    strip the evidence. `bayes_evidence_state` contains no market data."""
    for c in sorted(_DENY_EXEMPT):
        assert not _denied(c), c
    assert "sportsbook_inputs_used" in _WANTED


def test_extraction_refuses_a_file_with_too_few_explanatory_columns(tmp_path):
    (tmp_path / "noise.csv").write_text("a,b,c\n1,2,3\n")
    found, diag = extract(tmp_path, 7)
    assert found == {}
    assert diag["scanned"][0]["explanatory_columns_matched"] == 0


def test_extraction_keeps_football_state_and_drops_market_columns(tmp_path):
    (tmp_path / "metrics.csv").write_text(
        "player,team,week,prior_games,current_games,bayes_rush_share,"
        "vegas_line,edge_pct\n"
        "Back A,SEA,7,0,0,0.359568,72.5,4.1\n"
        "Back B,SEA,6,3,2,0.210000,60.5,1.0\n")
    found, diag = extract(tmp_path, 7)
    assert list(found) == ["metrics.csv"]
    df = found["metrics.csv"]
    assert "vegas_line" not in df.columns and "edge_pct" not in df.columns
    assert "bayes_rush_share" in df.columns
    # Week filter applied: only the week-7 row survives.
    assert len(df) == 1
    assert df["prior_games"].iloc[0] == 0
    assert df["bayes_rush_share"].iloc[0] == pytest.approx(0.359568)


def test_extraction_reports_columns_absent_from_the_artifact(tmp_path):
    (tmp_path / "m.csv").write_text(
        "player,team,prior_games,current_games\nA,SEA,0,0\n")
    _, diag = extract(tmp_path, 7)
    missing = diag["required_columns_missing_from_artifact"]
    assert "bayes_rush_share" in missing
    assert "player" not in missing
