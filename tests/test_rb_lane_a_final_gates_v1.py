import pandas as pd

from scripts.backtest.rb_lane_a_final_gates_v1 import (
    BASELINE_FAILURE,
    INSUFFICIENT_EVIDENCE,
    NOT_QUALIFIED,
    QUALIFIED,
    assemble_final_disposition,
    p90_catastrophic_protection_check,
    protected_cohort_gate_report,
    transition_mae_gate,
)


def _rows():
    rows = []
    for i in range(40):
        actual_att = 22 if i < 35 else 18
        actual_yards = 105.0 if i < 32 else 80.0
        rows.append(
            {
                "candidate_rush_yards": actual_yards + 4.0,
                "promotion_rush_yards": actual_yards + 8.0,
                "actual_rush_yards": actual_yards,
                "actual_rush_att": actual_att,
            }
        )
    return pd.DataFrame(rows)


def test_transition_mae_requires_strict_improvement():
    x = _rows()
    report = transition_mae_gate(x)
    assert report["disposition"] == "TRANSITION_MAE_GATE_PASS"
    assert report["candidate_mae"] < report["comparator_mae"]

    x["candidate_rush_yards"] = x["promotion_rush_yards"]
    assert transition_mae_gate(x)["disposition"] == "TRANSITION_MAE_GATE_FAILURE"


def test_protected_required_cohorts_are_non_worse_and_25_plus_is_disclosure_only():
    x = _rows()
    report = protected_cohort_gate_report(x)
    assert report["disposition"] == "PROTECTED_COHORT_GATE_PASS"
    assert report["cohorts"]["actual_carries_20_plus"]["required"] is True
    assert report["cohorts"]["actual_rush_yards_100_plus"]["required"] is True
    assert report["cohorts"]["actual_carries_25_plus_disclosure_only"]["required"] is False


def test_p90_protection_is_non_worse():
    x = _rows()
    assert p90_catastrophic_protection_check(x)["disposition"] == "P90_PROTECTION_PASS"
    x.loc[0:10, "candidate_rush_yards"] = x.loc[0:10, "actual_rush_yards"] + 100.0
    assert p90_catastrophic_protection_check(x)["disposition"] == "P90_PROTECTION_FAILURE"


def _final_kwargs():
    return {
        "gate0_disposition": "GATE0_PASS",
        "authority_dispositions": {1: "SAME_JOB_AUTHORITY_RECONSTRUCTION_PASS", 2: "SAME_JOB_AUTHORITY_RECONSTRUCTION_PASS"},
        "mechanism_rotation2_disposition": "MECHANISM_AUTHORITY_RECONSTRUCTION_PASS",
        "constructibility_dispositions": {1: "RUSH_YARD_TRANSLATION_CONSTRUCTIBLE", 2: "RUSH_YARD_TRANSLATION_CONSTRUCTIBLE"},
        "adequacy_dispositions": {1: "ADEQUATE", 2: "ADEQUATE"},
        "transition_mae_dispositions": {1: "TRANSITION_MAE_GATE_PASS", 2: "TRANSITION_MAE_GATE_PASS"},
        "protected_cohort_dispositions": {1: "PROTECTED_COHORT_GATE_PASS", 2: "PROTECTED_COHORT_GATE_PASS"},
        "bootstrap_dispositions": {1: "BOOTSTRAP_GATE_PASS", 2: "BOOTSTRAP_GATE_PASS"},
        "p90_dispositions": {1: "P90_PROTECTION_PASS", 2: "P90_PROTECTION_PASS"},
        "stable_identity_dispositions": {1: "STABLE_IDENTITY_GATE_PASS", 2: "STABLE_IDENTITY_GATE_PASS"},
        "whole_season_safety_dispositions": {1: "WHOLE_SEASON_SAFETY_PASS", 2: "WHOLE_SEASON_SAFETY_PASS"},
        "conservation_dispositions": {1: "CONSERVATION_GATE_PASS", 2: "CONSERVATION_GATE_PASS"},
        "per_season_nonregression_disposition": "PER_SEASON_NONREGRESSION_PASS",
        "sportsbook_inputs_used": 0,
    }


def test_final_disposition_qualifies_only_when_every_required_gate_passes():
    assert assemble_final_disposition(**_final_kwargs()) == QUALIFIED


def test_final_disposition_is_fail_closed_by_stage():
    kw = _final_kwargs()
    kw["authority_dispositions"] = {1: "SAME_JOB_AUTHORITY_RECONSTRUCTION_FAILURE", 2: "SAME_JOB_AUTHORITY_RECONSTRUCTION_PASS"}
    assert assemble_final_disposition(**kw) == BASELINE_FAILURE

    kw = _final_kwargs()
    kw["adequacy_dispositions"] = {1: "INSUFFICIENT_EVIDENCE", 2: "ADEQUATE"}
    assert assemble_final_disposition(**kw) == INSUFFICIENT_EVIDENCE

    kw = _final_kwargs()
    kw["p90_dispositions"] = {1: "P90_PROTECTION_FAILURE", 2: "P90_PROTECTION_PASS"}
    assert assemble_final_disposition(**kw) == NOT_QUALIFIED
