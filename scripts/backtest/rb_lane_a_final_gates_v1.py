"""RB Lane A -- final frozen gate assembly (transition-gated allocation V1).

Completes the preregistered gate set in
``docs/research/RB_LANE_A_TRANSITION_GATED_ALLOCATION_V1_PLAN.md`` before any
real candidate outcome is exposed.  The earlier gate-scoring helper implements
adequacy, dependence-aware bootstraps, per-season non-regression and whole-season
safety.  This module adds the remaining frozen production-endpoint gates:

* strict transition-subpopulation rushing-yard MAE improvement;
* required protected cohorts (actual carries >=20 and actual rushing yards >=100),
  with actual carries >=25 disclosure-only;
* transition-subpopulation p90/catastrophic-error protection; and
* one fail-closed final disposition assembled from every frozen integrity/evidence
  requirement.

Actual carries/yards are evaluation-only here.  Nothing in this module constructs
a pregame candidate or changes model science.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


QUALIFIED = "RB_LANE_A_TRANSITION_ALLOCATION_QUALIFIED"
GATE0_BLOCKED = "RB_LANE_A_TRANSITION_ALLOCATION_GATE0_BLOCKED"
BASELINE_FAILURE = "RB_LANE_A_TRANSITION_ALLOCATION_BASELINE_RECONSTRUCTION_FAILURE"
CONSTRUCTIBILITY_FAILURE = "RUSH_YARD_TRANSLATION_CONSTRUCTIBILITY_FAILURE"
INSUFFICIENT_EVIDENCE = "RB_LANE_A_TRANSITION_ALLOCATION_INSUFFICIENT_EVIDENCE"
NOT_QUALIFIED = "RB_LANE_A_TRANSITION_ALLOCATION_NOT_QUALIFIED"


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise RuntimeError(f"missing required scoring column {column!r}")
    return pd.to_numeric(frame[column], errors="coerce")


def _mae_pair(
    rows: pd.DataFrame,
    *,
    candidate_col: str,
    comparator_col: str,
    actual_col: str,
) -> dict:
    candidate = _numeric(rows, candidate_col)
    comparator = _numeric(rows, comparator_col)
    actual = _numeric(rows, actual_col)
    ok = candidate.notna() & comparator.notna() & actual.notna()
    if not ok.any():
        return {"n": 0, "candidate_mae": None, "comparator_mae": None, "mae_delta": None}
    c_mae = float((candidate[ok] - actual[ok]).abs().mean())
    b_mae = float((comparator[ok] - actual[ok]).abs().mean())
    return {
        "n": int(ok.sum()),
        "candidate_mae": c_mae,
        "comparator_mae": b_mae,
        "mae_delta": c_mae - b_mae,
    }


def transition_mae_gate(
    scored_rows: pd.DataFrame,
    *,
    candidate_col: str = "candidate_rush_yards",
    comparator_col: str = "promotion_rush_yards",
    actual_col: str = "actual_rush_yards",
) -> dict:
    """Decisive endpoint: candidate MAE must be *strictly* lower on scored
    transition rows.  This is evaluated independently in each OOS rotation by
    the caller.
    """
    stats = _mae_pair(
        scored_rows,
        candidate_col=candidate_col,
        comparator_col=comparator_col,
        actual_col=actual_col,
    )
    passed = (
        stats["n"] > 0
        and stats["candidate_mae"] is not None
        and stats["comparator_mae"] is not None
        and stats["candidate_mae"] < stats["comparator_mae"]
    )
    return {
        "disposition": "TRANSITION_MAE_GATE_PASS" if passed else "TRANSITION_MAE_GATE_FAILURE",
        **stats,
    }


def protected_cohort_gate_report(
    scored_rows: pd.DataFrame,
    *,
    candidate_col: str = "candidate_rush_yards",
    comparator_col: str = "promotion_rush_yards",
    actual_yards_col: str = "actual_rush_yards",
    actual_att_col: str = "actual_rush_att",
) -> dict:
    """Frozen evaluation-only cohorts.

    Required cohorts must be non-worse (candidate MAE <= comparator MAE):
    actual carries >=20 and actual rushing yards >=100.  Actual carries >=25
    remains disclosure-only and never gates qualification.
    """
    yards = _numeric(scored_rows, actual_yards_col)
    att = _numeric(scored_rows, actual_att_col)
    masks = {
        "actual_carries_20_plus": att >= 20,
        "actual_rush_yards_100_plus": yards >= 100,
        "actual_carries_25_plus_disclosure_only": att >= 25,
    }
    cohorts: dict[str, dict] = {}
    required_pass = True
    for name, mask in masks.items():
        stats = _mae_pair(
            scored_rows.loc[mask],
            candidate_col=candidate_col,
            comparator_col=comparator_col,
            actual_col=actual_yards_col,
        )
        non_worse = (
            stats["n"] > 0
            and stats["candidate_mae"] is not None
            and stats["comparator_mae"] is not None
            and stats["candidate_mae"] <= stats["comparator_mae"]
        )
        required = name != "actual_carries_25_plus_disclosure_only"
        cohorts[name] = {**stats, "required": required, "non_worse": non_worse}
        if required:
            required_pass = required_pass and non_worse
    return {
        "disposition": "PROTECTED_COHORT_GATE_PASS" if required_pass else "PROTECTED_COHORT_GATE_FAILURE",
        "cohorts": cohorts,
    }


def p90_catastrophic_protection_check(
    scored_rows: pd.DataFrame,
    *,
    candidate_col: str = "candidate_rush_yards",
    comparator_col: str = "promotion_rush_yards",
    actual_col: str = "actual_rush_yards",
) -> dict:
    """Frozen catastrophic-error protection: transition p90 absolute error
    must be non-worse than the promotion comparator.
    """
    candidate = _numeric(scored_rows, candidate_col)
    comparator = _numeric(scored_rows, comparator_col)
    actual = _numeric(scored_rows, actual_col)
    ok = candidate.notna() & comparator.notna() & actual.notna()
    if not ok.any():
        return {
            "disposition": "P90_PROTECTION_FAILURE",
            "n": 0,
            "candidate_p90_abs_error": None,
            "comparator_p90_abs_error": None,
            "p90_delta": None,
        }
    c_err = (candidate[ok] - actual[ok]).abs().to_numpy(float)
    b_err = (comparator[ok] - actual[ok]).abs().to_numpy(float)
    c90 = float(np.quantile(c_err, 0.90))
    b90 = float(np.quantile(b_err, 0.90))
    passed = c90 <= b90
    return {
        "disposition": "P90_PROTECTION_PASS" if passed else "P90_PROTECTION_FAILURE",
        "n": int(ok.sum()),
        "candidate_p90_abs_error": c90,
        "comparator_p90_abs_error": b90,
        "p90_delta": c90 - b90,
    }


def conservation_gate_from_event_meta(event_meta: list[dict]) -> dict:
    """Aggregate the mechanism's per-team-week conservation assertions.

    ``compute_hhi_dampened_reallocation`` itself hard-asserts the identity for
    every constructed event.  The orchestrator additionally records each event's
    post-construction delta so the final artifact has explicit numeric evidence.
    All-zero-history rooms are reported separately and are not constructed/scored,
    per the frozen plan.
    """
    constructed = [m for m in event_meta if not bool(m.get("all_zero_weights"))]
    deltas = [float(m.get("conservation_delta", np.nan)) for m in constructed]
    finite = [d for d in deltas if np.isfinite(d)]
    max_delta = max(finite) if finite else 0.0
    passed = len(finite) == len(constructed) and max_delta <= 1e-6
    return {
        "disposition": "CONSERVATION_GATE_PASS" if passed else "CONSERVATION_GATE_FAILURE",
        "constructed_events": len(constructed),
        "all_zero_weight_events_excluded": int(sum(bool(m.get("all_zero_weights")) for m in event_meta)),
        "max_abs_delta": float(max_delta),
        "tolerance": 1e-6,
    }


def assemble_final_disposition(
    *,
    gate0_disposition: str,
    authority_dispositions: dict[int, str],
    mechanism_rotation2_disposition: str,
    constructibility_dispositions: dict[int, str],
    adequacy_dispositions: dict[int, str],
    transition_mae_dispositions: dict[int, str],
    protected_cohort_dispositions: dict[int, str],
    bootstrap_dispositions: dict[int, str],
    p90_dispositions: dict[int, str],
    stable_identity_dispositions: dict[int, str],
    whole_season_safety_dispositions: dict[int, str],
    conservation_dispositions: dict[int, str],
    per_season_nonregression_disposition: str,
    sportsbook_inputs_used: int = 0,
) -> str:
    """Fail-closed final disposition using the frozen ordering in the plan."""
    if gate0_disposition != "GATE0_PASS":
        return GATE0_BLOCKED

    if (
        set(authority_dispositions) != {1, 2}
        or any(v != "SAME_JOB_AUTHORITY_RECONSTRUCTION_PASS" for v in authority_dispositions.values())
        or mechanism_rotation2_disposition != "MECHANISM_AUTHORITY_RECONSTRUCTION_PASS"
    ):
        return BASELINE_FAILURE

    if (
        set(constructibility_dispositions) != {1, 2}
        or any(v != "RUSH_YARD_TRANSLATION_CONSTRUCTIBLE" for v in constructibility_dispositions.values())
    ):
        return CONSTRUCTIBILITY_FAILURE

    if (
        set(adequacy_dispositions) != {1, 2}
        or any(v != "ADEQUATE" for v in adequacy_dispositions.values())
    ):
        return INSUFFICIENT_EVIDENCE

    required_maps = (
        (transition_mae_dispositions, "TRANSITION_MAE_GATE_PASS"),
        (protected_cohort_dispositions, "PROTECTED_COHORT_GATE_PASS"),
        (bootstrap_dispositions, "BOOTSTRAP_GATE_PASS"),
        (p90_dispositions, "P90_PROTECTION_PASS"),
        (stable_identity_dispositions, "STABLE_IDENTITY_GATE_PASS"),
        (whole_season_safety_dispositions, "WHOLE_SEASON_SAFETY_PASS"),
        (conservation_dispositions, "CONSERVATION_GATE_PASS"),
    )
    for values, expected in required_maps:
        if set(values) != {1, 2} or any(v != expected for v in values.values()):
            return NOT_QUALIFIED

    if per_season_nonregression_disposition != "PER_SEASON_NONREGRESSION_PASS":
        return NOT_QUALIFIED
    if int(sportsbook_inputs_used) != 0:
        return NOT_QUALIFIED
    return QUALIFIED
