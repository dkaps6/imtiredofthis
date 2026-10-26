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


def snaps(spec, season=2026):
    """spec: {(week, pfr_id): offense_snaps}"""
    return pd.DataFrame([
        {"season": season, "week": w, "pfr_player_id": p, "offense_snaps": n}
        for (w, p), n in spec.items()
    ])


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
    _, diag = build_participation(rosters(), snaps({(1, PFR_A): 30}))
    assert diag["bye_detection"] == "unavailable_no_team_col"


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
