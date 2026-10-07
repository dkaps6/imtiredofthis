from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.research.run_player_opportunity_volume_share_decomposition_v1 as d


def _parent() -> pd.DataFrame:
    rows = []
    for week in (1, 2, 3, 4):
        event = f"G{week}"
        rows.extend([
            {
                "season": 2026, "week": week, "event_id": event, "team": "IND", "opponent": "HOU",
                "player": "QB One", "player_clean_key": "qbone", "position_family": "QB",
                "opportunity_type": "pass_attempts",
                "predicted_team_opportunity_mean": 30.0, "final_player_probability": 0.9,
                "expected_opportunities_from_probability": 27.0,
                "actual_opportunities": 32.0, "actual_opportunity_bin": "31_40",
                "sportsbook_inputs_used_upstream": False,
                "rb_week5_room_allocation_shadow_applied": False,
                "linked_yards_error": -40.0, "linked_count_error": np.nan,
            },
            {
                "season": 2026, "week": week, "event_id": event, "team": "IND", "opponent": "HOU",
                "player": "WR One", "player_clean_key": "wrone", "position_family": "WR",
                "opportunity_type": "targets",
                "predicted_team_opportunity_mean": 30.0, "final_player_probability": 0.2,
                "expected_opportunities_from_probability": 6.0,
                "actual_opportunities": 8.0, "actual_opportunity_bin": "06_08",
                "sportsbook_inputs_used_upstream": False,
                "rb_week5_room_allocation_shadow_applied": False,
                "linked_yards_error": -20.0, "linked_count_error": -2.0,
            },
            {
                "season": 2026, "week": week, "event_id": event, "team": "IND", "opponent": "HOU",
                "player": "RB One", "player_clean_key": "rbone", "position_family": "RB",
                "opportunity_type": "carries",
                "predicted_team_opportunity_mean": 25.0, "final_player_probability": 0.4,
                "expected_opportunities_from_probability": 10.0,
                "actual_opportunities": 12.0, "actual_opportunity_bin": "09_14",
                "sportsbook_inputs_used_upstream": False,
                "rb_week5_room_allocation_shadow_applied": False,
                "linked_yards_error": -12.0, "linked_count_error": np.nan,
            },
        ])
    return pd.DataFrame(rows)


def _actual_team() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "week": week, "team": "IND",
            "actual_team_official_pass_attempts": 40,
            "actual_team_rush_attempts": 30,
        }
        for week in (1, 2, 3, 4)
    ])


def test_decomposition_arithmetic_and_full_oracle_identity():
    x = d.build_rows(_parent(), _actual_team())
    qb = x.loc[x["position_family"].eq("QB")].iloc[0]
    wr = x.loc[x["position_family"].eq("WR")].iloc[0]
    rb = x.loc[x["position_family"].eq("RB")].iloc[0]

    assert qb["baseline_expected_opportunity"] == pytest.approx(27.0)
    assert qb["oracle_team_volume_opportunity"] == pytest.approx(36.0)
    assert qb["actual_player_share"] == pytest.approx(0.8)
    assert qb["oracle_player_share_opportunity"] == pytest.approx(24.0)
    assert qb["full_oracle_opportunity"] == pytest.approx(32.0)

    assert wr["baseline_expected_opportunity"] == pytest.approx(6.0)
    assert wr["oracle_team_volume_opportunity"] == pytest.approx(8.0)
    assert wr["actual_player_share"] == pytest.approx(0.2)
    assert wr["oracle_player_share_opportunity"] == pytest.approx(6.0)
    assert wr["full_oracle_opportunity"] == pytest.approx(8.0)

    assert rb["baseline_expected_opportunity"] == pytest.approx(10.0)
    assert rb["oracle_team_volume_opportunity"] == pytest.approx(12.0)
    assert rb["actual_player_share"] == pytest.approx(0.4)
    assert rb["oracle_player_share_opportunity"] == pytest.approx(10.0)
    assert rb["full_oracle_opportunity"] == pytest.approx(12.0)


def test_baseline_parent_expectation_must_match():
    p = _parent()
    p.loc[p["position_family"].eq("WR"), "expected_opportunities_from_probability"] = 7.0
    with pytest.raises(RuntimeError, match="baseline decomposition"):
        d.build_rows(p, _actual_team())


def test_actual_player_share_must_be_valid():
    p = _parent()
    p.loc[p["position_family"].eq("WR"), "actual_opportunities"] = 50.0
    with pytest.raises(RuntimeError, match="actual player share outside"):
        d.build_rows(p, _actual_team())


def test_summarize_distinguishes_team_volume_from_share():
    x = d.build_rows(_parent(), _actual_team())
    summary, high = d.build_summary(x)
    wr = summary.loc[
        summary["position_family"].eq("WR")
        & summary["opportunity_type"].eq("targets")
        & summary["subset"].eq("ALL")
    ].iloc[0]

    # WR example is pure team-volume error: 30 predicted pass attempts vs 40
    # actual, while predicted and actual player target shares are both 0.20.
    assert wr["baseline_mae"] == pytest.approx(2.0)
    assert wr["oracle_team_mae"] == pytest.approx(0.0)
    assert wr["oracle_share_mae"] == pytest.approx(2.0)
    assert wr["team_oracle_mae_improvement"] == pytest.approx(2.0)
    assert wr["share_oracle_mae_improvement"] == pytest.approx(0.0)
    assert wr["team_oracle_fraction_baseline_mae_removed"] == pytest.approx(1.0)
    assert wr["share_oracle_fraction_baseline_mae_removed"] == pytest.approx(0.0)


def test_parent_sportsbook_or_week5_shadow_fails_closed():
    p = _parent()
    p.loc[0, "sportsbook_inputs_used_upstream"] = True
    with pytest.raises(RuntimeError, match="sportsbook"):
        d.build_rows(p, _actual_team())

    p = _parent()
    p.loc[0, "rb_week5_room_allocation_shadow_applied"] = True
    with pytest.raises(RuntimeError, match="Week-5"):
        d.build_rows(p, _actual_team())
