import pandas as pd

from scripts.research.audit_prism_rb_receiving_team_context_history_authority_reset_v1_source import (
    attach_reset_source,
    select_first_market_rows,
)


def test_first_market_row_stays_inside_transition_boundary():
    full = pd.DataFrame([
        {"season": 2024, "week": 1, "team": "AAA", "stable_id": "00-1", "transition_kind": "OFFSEASON_TEAM_CHANGE"},
        {"season": 2024, "week": 9, "team": "BBB", "stable_id": "00-1", "transition_kind": "INSEASON_TEAM_CHANGE"},
    ])
    eligible = full.iloc[[0]].copy()
    eligible["transition_id"] = "t1"
    authority = pd.DataFrame([
        {"season": 2024, "week": 5, "team": "AAA", "player_id": "00-1", "game_key": "g5",
         "stage1_p_active": .8, "p_effective": .8, "player_prior_positive_q_rows": 0,
         "hierarchical_expected_benefit": 0.0, "own_authority_disabled": False},
        {"season": 2024, "week": 10, "team": "BBB", "player_id": "00-1", "game_key": "g10",
         "stage1_p_active": .7, "p_effective": .7, "player_prior_positive_q_rows": 0,
         "hierarchical_expected_benefit": 0.0, "own_authority_disabled": False},
    ])
    out, ex = select_first_market_rows(full, eligible, authority)
    assert ex["NO_CANONICAL_RB_RECEIVING_BASELINE"] == 0
    assert len(out) == 1
    assert int(out.iloc[0]["target_week"]) == 5
    assert int(out.iloc[0]["next_transition_week"]) == 9


def test_joint_history_reset_uses_stage1_role_prior_when_named_history_exists():
    selected = pd.DataFrame([{
        "transition_id": "t1", "season": 2024, "transition_week": 1, "team": "AAA",
        "stable_id": "00-1", "transition_kind": "OFFSEASON_TEAM_CHANGE",
        "next_transition_week": None, "target_week": 5, "game_key": "g5",
        "stage1_p_active": .8, "baseline_p_effective": 1.0,
        "player_prior_positive_q_rows": 4, "hierarchical_expected_benefit": -.01,
        "own_authority_disabled": True,
    }])
    stage1 = pd.DataFrame([{
        "season": 2024, "week": 5, "team": "AAA", "player_id": "00-1",
        "p_baseline": .6, "p_candidate": .8, "n_player_prior": 4,
    }])
    out = attach_reset_source(selected, stage1)
    assert bool(out.iloc[0]["stage1_exact_mechanism_row"])
    assert bool(out.iloc[0]["stage1_candidate_matches_current"])
    assert float(out.iloc[0]["reset_p_active"]) == .6
    assert bool(out.iloc[0]["actionable_joint_history_reset"])


def test_unsupported_stage1_player_preserves_existing_role_fallback():
    selected = pd.DataFrame([{
        "transition_id": "t1", "season": 2025, "transition_week": 4, "team": "AAA",
        "stable_id": "00-2", "transition_kind": "INSEASON_TEAM_CHANGE",
        "next_transition_week": None, "target_week": 5, "game_key": "g5",
        "stage1_p_active": .72, "baseline_p_effective": .72,
        "player_prior_positive_q_rows": 0, "hierarchical_expected_benefit": 0.0,
        "own_authority_disabled": False,
    }])
    stage1 = pd.DataFrame(columns=[
        "season","week","team","player_id","p_baseline","p_candidate","n_player_prior"
    ])
    out = attach_reset_source(selected, stage1)
    assert not bool(out.iloc[0]["stage1_exact_mechanism_row"])
    assert float(out.iloc[0]["reset_p_active"]) == .72
    assert not bool(out.iloc[0]["actionable_joint_history_reset"])
