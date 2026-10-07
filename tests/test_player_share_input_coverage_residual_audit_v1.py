from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.research.run_player_share_input_coverage_residual_audit_v1 as a


def test_target_allocator_equivalent_caps_team_mass_and_preserves_missing_evidence():
    x = pd.DataFrame([
        {"event_id":"G","team":"IND","share":0.60},
        {"event_id":"G","team":"IND","share":0.50},
        {"event_id":"G","team":"IND","share":np.nan},
    ])
    out = a._target_allocator_equivalent(x, "share")
    assert pd.isna(out.iloc[2])
    assert float(out.iloc[:2].sum()) <= a.TARGET_MASS_CAP
    expected_scale = a.ALLOCATOR_SAFE_CAP / 1.10
    assert out.iloc[0] == pytest.approx(0.60 * expected_scale)
    assert out.iloc[1] == pytest.approx(0.50 * expected_scale)


def test_rush_allocator_equivalent_retains_top_five_only():
    x = pd.DataFrame([
        {"event_id":"G","team":"IND","share":v}
        for v in [0.30,0.25,0.20,0.15,0.10,0.05]
    ])
    prob, member = a._rush_allocator_equivalent(x, "share")
    assert member.sum() == 5
    assert not bool(member.iloc[5])
    assert prob.iloc[5] == pytest.approx(0.0)
    assert prob.sum() <= a.TARGET_MASS_CAP + 1e-12


def test_long_stage_rows_uses_position_specific_stage_contract():
    rows = pd.DataFrame([
        {
            "season":2026,"week":1,"event_id":"G","team":"IND","opponent":"HOU",
            "player":"WR One","player_clean_key":"wrone","position_family":"WR",
            "opportunity_type":"targets","actual_player_share":0.25,
            "actual_player_opportunity":8.0,"actual_opportunity_bin":"06_08",
            "raw_target_allocator_share":0.15,"bayes_target_allocator_share":0.17,
            "rules_target_allocator_share":0.18,"post_m38_entitlement_tgt_share":0.20,
            "final_target_allocator_share":0.22,
            "bayes_evidence_state":"prior+current","prior_games":17,"current_games":2,
            "specialist_evidence_consumed":True,
            "specialist_prior1_anyteam_available":True,
            "specialist_prior3_anyteam_available":True,
            "specialist_prior1_same_team_available":True,
            "specialist_prior3_same_team_available":True,
        }
    ])
    out=a._long_stage_rows(rows)
    assert out.stage.tolist()==["raw_history","bayes","rules","post_m38","final_specialist"]
    final=out.loc[out.stage.eq("final_specialist")].iloc[0]
    assert final.share_error==pytest.approx(-0.03)
    assert final.absolute_share_error==pytest.approx(0.03)


def test_stage_summary_high_workload_uses_frozen_bins():
    long=pd.DataFrame([
        {
            "position_family":"WR","opportunity_type":"targets","stage":"final_specialist",
            "predicted_share":0.20,"actual_share":0.30,"share_error":-0.10,
            "absolute_share_error":0.10,"actual_player_opportunity":10,
            "actual_opportunity_bin":"09_PLUS",
        },
        {
            "position_family":"WR","opportunity_type":"targets","stage":"final_specialist",
            "predicted_share":0.10,"actual_share":0.10,"share_error":0.0,
            "absolute_share_error":0.0,"actual_player_opportunity":3,
            "actual_opportunity_bin":"03_05",
        },
    ])
    out=a._stage_summary(long)
    high=out.loc[out.subset.eq("HIGH_WORKLOAD")].iloc[0]
    assert high.rows==1
    assert high.share_mae==pytest.approx(0.10)
    assert high.share_bias==pytest.approx(-0.10)


def test_specialist_effect_reports_direction_without_fitting():
    rows=pd.DataFrame([
        {
            "position_family":"WR","opportunity_type":"targets","actual_player_share":0.30,
            "rules_target_allocator_share":0.15,"post_m38_entitlement_tgt_share":0.20,
            "final_target_allocator_share":0.25,"wr_r15_anchor":False,
        },
        {
            "position_family":"TE","opportunity_type":"targets","actual_player_share":0.20,
            "rules_target_allocator_share":0.10,"post_m38_entitlement_tgt_share":0.10,
            "final_target_allocator_share":0.18,
        },
    ])
    out=a._specialist_effect(rows)
    wr=out.loc[out.position_family.eq("WR") & out.scope.eq("ALL")].iloc[0]
    te=out.loc[out.position_family.eq("TE")].iloc[0]
    assert wr.rules_to_final_mae_improvement==pytest.approx(0.10)
    assert te.rules_to_final_mae_improvement==pytest.approx(0.08)


def test_attach_parent_uses_corrected_dropback_parent_schema(tmp_path):
    stage = pd.DataFrame([{
        "season": 2026, "week": 2, "event_id": "G1", "team": "IND",
        "opponent": "HOU", "player": "WR One", "player_clean_key": "wrone",
        "position": "WR", "position_family": "WR",
        "raw_target_allocator_share": 0.10,
        "bayes_target_allocator_share": 0.11,
        "rules_target_allocator_share": 0.12,
        "post_m38_entitlement_tgt_share": 0.13,
        "final_target_allocator_share": 0.14,
        "final_rush_allocator_share": 0.0,
    }])
    parent = pd.DataFrame([{
        "season": 2026, "week": 2, "event_id": "G1", "team": "IND",
        "player_clean_key": "wrone", "position_family": "WR",
        "opportunity_type": "targets", "actual_opportunities": 7.0,
        "actual_team_volume": 35.0, "actual_player_share": 0.20,
        "actual_opportunity_bin": "06_08", "linked_yards_error": -20.0,
        "linked_count_error": -2.0, "final_player_probability": 0.14,
        "sportsbook_inputs_used_upstream": False,
    }])
    p = tmp_path / "parent.csv"
    parent.to_csv(p, index=False)
    out = a._attach_parent(stage, p)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["actual_player_opportunity"] == pytest.approx(7.0)
    assert row["actual_player_share"] == pytest.approx(0.20)
    assert row["predicted_player_share"] == pytest.approx(0.14)

