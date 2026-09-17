from scripts.backtest.rb_workhorse_gate_v1_classifier import CONFIRMED, INSUFFICIENT_EVIDENCE, NOT_QUALIFIED as NOT_QUALIFIED_ROTATION
from scripts.backtest.run_rb_workhorse_gate_v1 import (
    BOTH_ROTATIONS_CONFIRMED,
    NOT_QUALIFIED,
    assemble_final_disposition,
)


def _rotation(disposition: str) -> dict:
    return {"confirmation": {"disposition": disposition}}


def test_assemble_final_disposition_requires_both_rotations_confirmed():
    reports = {"A": _rotation(CONFIRMED), "B": _rotation(CONFIRMED)}
    assert assemble_final_disposition(reports) == BOTH_ROTATIONS_CONFIRMED


def test_assemble_final_disposition_not_qualified_when_one_rotation_fails():
    reports = {"A": _rotation(CONFIRMED), "B": _rotation(NOT_QUALIFIED_ROTATION)}
    assert assemble_final_disposition(reports) == NOT_QUALIFIED


def test_assemble_final_disposition_insufficient_evidence_dominates_even_if_other_confirms():
    # No rescue / no selective use of a surviving rotation (plan Section
    # 10/14/16): an adequacy failure in either rotation must not be masked
    # by the other rotation confirming.
    reports = {"A": _rotation(INSUFFICIENT_EVIDENCE), "B": _rotation(CONFIRMED)}
    assert assemble_final_disposition(reports) == INSUFFICIENT_EVIDENCE


def test_assemble_final_disposition_not_qualified_when_both_fail_gates():
    reports = {"A": _rotation(NOT_QUALIFIED_ROTATION), "B": _rotation(NOT_QUALIFIED_ROTATION)}
    assert assemble_final_disposition(reports) == NOT_QUALIFIED
