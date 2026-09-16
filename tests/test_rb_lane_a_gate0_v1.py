import pandas as pd

from scripts.backtest.rb_lane_a_gate0_v1 import (
    RB_POS,
    _name_key,
    gate01_report,
    gate02_report,
    gate03_event_report,
    gate03_report,
)


def test_name_key_strips_punctuation_and_lowercases():
    s = pd.Series(["D'Andre Swift", "A.J. Dillon"])
    out = _name_key(s)
    assert out.tolist() == ["dandreswift", "ajdillon"]


def test_rb_pos_matches_repo_convention():
    assert RB_POS == {"RB", "FB", "HB"}


def test_gate01_report_never_gates_overall_disposition():
    depth_state = pd.DataFrame(
        {
            "season": [2024, 2025],
            "week": [1, 1],
            "team": ["KC", "SF"],
            "name_key": ["a", "b"],
            "depth_rank": [1.0, 1.0],
            "depth_slot": ["RB", "RB"],
            "pregame_source": ["nflverse_native_week_tagged", "nflverse_live_asof_dt_lt_kickoff"],
        }
    )
    report = gate01_report(depth_state, [2024, 2025])
    assert report["semantic_parity_2023_2024_disposition"] == "NOT_CONSTRUCTIBLE_NO_OVERLAP"
    assert report["gates_scored_v1_science"] is False
    assert report["gates_gate0_overall_disposition"] is False


def test_gate02_report_pass_disposition_and_coverage():
    injury_state = pd.DataFrame(
        {
            "season": [2025, 2025],
            "week": [1, 1],
            "status": [None, "Out"],
        }
    )
    report = gate02_report(injury_state, [2025])
    assert report["disposition"] == "PASS"
    assert report["coverage_by_season"]["2025"]["rows"] == 2
    assert report["coverage_by_season"]["2025"]["null_status_rows"] == 1


def test_gate03_report_fails_closed_on_duplicate_identity():
    # Season 2023 deliberately: requirement 4's schedule-coverage check only
    # runs for 2024/2025 (the OOS test seasons), so this stays network-free.
    roster_state = pd.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 1],
            "team": ["KC", "KC"],
            "player_key": ["p1", "p1"],
            "position": ["RB", "RB"],
            "status": ["ACT", "ACT"],
        }
    )
    report = gate03_report(roster_state, [2023])
    assert report["disposition"] == "GATE0_BLOCKED"
    assert any("requirement_3_duplicate_identities" in f for f in report["failures"])


def test_gate03_report_passes_clean_with_no_duplicates():
    roster_state = pd.DataFrame(
        {
            "season": [2023],
            "week": [1],
            "team": ["KC"],
            "player_key": ["p1"],
            "position": ["RB"],
            "status": ["ACT"],
        }
    )
    report = gate03_report(roster_state, [2023])
    assert report["failures"] == []
    # Structural pass only -- per GPT-5.6's adjudication (5702132088), requirements
    # 5/6 (event-level checks) are separate and not folded into this disposition.
    assert report["disposition"] == "PASS_STRUCTURAL_EVENT_CHECKS_PENDING"


def test_gate03_event_report_passes_when_current_and_prior_state_resolvable():
    roster_state = pd.DataFrame(
        {
            "season": [2024, 2024],
            "week": [3, 2],
            "team": ["KC", "KC"],
        }
    )
    event_population = pd.DataFrame(
        {
            "season": [2024],
            "week": [3],
            "team": ["KC"],
            "prior_season": [2024],
            "prior_week": [2],
        }
    )
    report = gate03_event_report(event_population, roster_state)
    assert report["disposition"] == "PASS"
    assert report["failures"] == []
    assert report["events_checked"] == 1


def test_gate03_event_report_fails_closed_on_missing_prior_state():
    roster_state = pd.DataFrame({"season": [2024], "week": [3], "team": ["KC"]})
    event_population = pd.DataFrame(
        {
            "season": [2024],
            "week": [3],
            "team": ["KC"],
            "prior_season": [2024],
            "prior_week": [2],
        }
    )
    report = gate03_event_report(event_population, roster_state)
    assert report["disposition"] == "GATE0_BLOCKED"
    assert any("requirement_5_missing_prior_state" in f for f in report["failures"])


def test_gate03_event_report_fails_closed_on_outcome_columns():
    roster_state = pd.DataFrame(
        {"season": [2024, 2024], "week": [3, 2], "team": ["KC", "KC"]}
    )
    event_population = pd.DataFrame(
        {
            "season": [2024],
            "week": [3],
            "team": ["KC"],
            "prior_season": [2024],
            "prior_week": [2],
            "actual_rush_yards": [88.0],
        }
    )
    report = gate03_event_report(event_population, roster_state)
    assert report["disposition"] == "GATE0_BLOCKED"
    assert any("requirement_6_outcome_columns_present" in f for f in report["failures"])
