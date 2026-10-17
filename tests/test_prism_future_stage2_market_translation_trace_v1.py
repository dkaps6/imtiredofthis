from __future__ import annotations

import pandas as pd
import pytest

from scripts.research.build_prism_future_stage2_market_translation_trace_v1 import build


def stage2_fixture() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "season": 2026,
            "week": 7,
            "event_id": "2026_07_A_B",
            "team": "A",
            "cell": "RB_RUSHING",
            "family": "RB",
            "player": "Runner One",
            "player_clean_key": "runnerone",
            "player_id": "rb1",
            "p_active": 0.9,
            "stage1_probability_source": "NAMED_PLAYER",
            "canonical_q": 0.40,
            "candidate_expected_q": 0.48,
            "share_shift": 0.08,
            "team_opportunity_mean": 25.0,
            "baseline_expected_opportunities": 10.0,
            "candidate_expected_opportunities": 12.0,
            "expected_opportunity_shift": 2.0,
            "availability_authority": "OFFICIAL",
            "final_availability_state": "ELIGIBLE",
        },
        {
            "season": 2026,
            "week": 7,
            "event_id": "2026_07_C_D",
            "team": "C",
            "cell": "RB_RECEIVING",
            "family": "RB",
            "player": "Back Two",
            "player_clean_key": "backtwo",
            "player_id": "rb2",
            "p_active": 0.85,
            "stage1_probability_source": "NAMED_PLAYER",
            "canonical_q": 0.10,
            "candidate_expected_q": 0.125,
            "share_shift": 0.025,
            "team_opportunity_mean": 40.0,
            "baseline_expected_opportunities": 4.0,
            "candidate_expected_opportunities": 5.0,
            "expected_opportunity_shift": 1.0,
            "availability_authority": "OFFICIAL",
            "final_availability_state": "ELIGIBLE",
        },
        {
            "season": 2026,
            "week": 7,
            "event_id": "2026_07_E_F",
            "team": "E",
            "cell": "TE_RECEIVING",
            "family": "TE",
            "player": "Tight Three",
            "player_clean_key": "tightthree",
            "player_id": "te3",
            "p_active": 0.95,
            "stage1_probability_source": "NAMED_PLAYER",
            "canonical_q": 0.15,
            "candidate_expected_q": 0.1375,
            "share_shift": -0.0125,
            "team_opportunity_mean": 40.0,
            "baseline_expected_opportunities": 6.0,
            "candidate_expected_opportunities": 5.5,
            "expected_opportunity_shift": -0.5,
            "availability_authority": "OFFICIAL",
            "final_availability_state": "ELIGIBLE",
        },
    ])


def universe_fixture() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "event_id": "2026_07_A_B",
            "team": "A",
            "player_clean_key": "runnerone",
            "player_id": "rb1",
            "rules_ypc": 4.5,
            "rules_ypt": 5.0,
            "rules_catch_rate": 0.7,
        },
        {
            "event_id": "2026_07_C_D",
            "team": "C",
            "player_clean_key": "backtwo",
            "player_id": "rb2",
            "rules_ypc": 4.0,
            "rules_ypt": 8.0,
            "rules_catch_rate": 0.75,
        },
        {
            "event_id": "2026_07_E_F",
            "team": "E",
            "player_clean_key": "tightthree",
            "player_id": "te3",
            "rules_ypc": 0.0,
            "rules_ypt": 7.0,
            "rules_catch_rate": 0.60,
        },
    ])


def test_exact_market_translation_holds_efficiency_fixed() -> None:
    rows, result = build(stage2_fixture(), universe_fixture())

    assert len(rows) == 8
    assert result["translation_rows"] == 8
    assert result["players"] == 3
    assert result["target_season"] == 2026
    assert result["target_week"] == 7
    assert result["scientific_lock"] is False
    assert result["week6_backfill"] is False
    assert result["target_outcomes_read"] == 0
    assert result["sportsbook_inputs_used"] == 0
    assert result["efficiency_fields_changed"] == 0
    assert result["max_shift_identity_gap"] <= 1e-12

    rush_yards = rows.loc[
        (rows["player_id"].eq("rb1")) & rows["market"].eq("rush_yards")
    ].iloc[0]
    assert rush_yards["baseline_football_mean"] == pytest.approx(45.0)
    assert rush_yards["candidate_football_mean"] == pytest.approx(54.0)
    assert rush_yards["candidate_minus_baseline_mean"] == pytest.approx(9.0)
    assert rush_yards["efficiency_field"] == "rules_ypc"
    assert rush_yards["efficiency_value"] == pytest.approx(4.5)

    rb_rec = rows.loc[
        (rows["player_id"].eq("rb2")) & rows["market"].eq("receptions")
    ].iloc[0]
    assert rb_rec["baseline_football_mean"] == pytest.approx(3.0)
    assert rb_rec["candidate_football_mean"] == pytest.approx(3.75)
    assert rb_rec["candidate_minus_baseline_mean"] == pytest.approx(0.75)

    rb_yards = rows.loc[
        (rows["player_id"].eq("rb2")) & rows["market"].eq("rec_yards")
    ].iloc[0]
    assert rb_yards["baseline_football_mean"] == pytest.approx(32.0)
    assert rb_yards["candidate_football_mean"] == pytest.approx(40.0)
    assert rb_yards["candidate_minus_baseline_mean"] == pytest.approx(8.0)

    te_yards = rows.loc[
        (rows["player_id"].eq("te3")) & rows["market"].eq("rec_yards")
    ].iloc[0]
    assert te_yards["baseline_football_mean"] == pytest.approx(42.0)
    assert te_yards["candidate_football_mean"] == pytest.approx(38.5)
    assert te_yards["candidate_minus_baseline_mean"] == pytest.approx(-3.5)

    assert rows["efficiency_changed_by_translation"].eq(False).all()
    assert rows["shift_identity_gap"].max() <= 1e-12


def test_week6_backfill_is_rejected() -> None:
    s = stage2_fixture()
    s["week"] = 6
    with pytest.raises(RuntimeError, match="Week-6-or-earlier"):
        build(s, universe_fixture())


def test_sportsbook_column_is_rejected() -> None:
    s = stage2_fixture()
    s["sportsbook_line"] = 50.5
    with pytest.raises(RuntimeError, match="forbidden sportsbook"):
        build(s, universe_fixture())


def test_target_outcome_column_is_rejected() -> None:
    s = stage2_fixture()
    s["actual_targets"] = 10
    with pytest.raises(RuntimeError, match="forbidden target-outcome"):
        build(s, universe_fixture())


def test_missing_required_efficiency_fails_closed() -> None:
    u = universe_fixture()
    u.loc[u["player_id"].eq("te3"), "rules_ypt"] = float("nan")
    with pytest.raises(RuntimeError, match="missing/nonfinite rules_ypt"):
        build(stage2_fixture(), u)


def test_player_identity_mismatch_fails_closed() -> None:
    u = universe_fixture()
    u.loc[u["player_id"].eq("rb1"), "player_clean_key"] = "wrongplayer"
    with pytest.raises(RuntimeError, match="player identity mismatch"):
        build(stage2_fixture(), u)


def test_stage2_probability_and_opportunity_identities_are_enforced() -> None:
    s = stage2_fixture()
    s.loc[s["player_id"].eq("rb1"), "candidate_expected_opportunities"] = 12.1
    with pytest.raises(RuntimeError, match="candidate opportunity identity drift"):
        build(s, universe_fixture())
