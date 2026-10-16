from scripts.research.audit_qb_team_pass_intent_independent_nli_semantic_validator_v1j import (
    HYPOTHESIS,
    MIN_ENTAILMENT,
    MIN_MARGIN,
    _label_indexes,
    _qualifies_probs,
)


def test_frozen_hypothesis_and_thresholds():
    assert "upcoming game" in HYPOTHESIS
    assert MIN_ENTAILMENT == 0.70
    assert MIN_MARGIN == 0.20


def test_label_indexes_resolve_named_labels():
    entail, neutral, contra = _label_indexes({
        0: "contradiction",
        1: "entailment",
        2: "neutral",
    })
    assert (entail, neutral, contra) == (1, 2, 0)


def test_probability_gate_requires_both_thresholds():
    ok, margin = _qualifies_probs(0.80, 0.10, 0.10)
    assert ok is True
    assert round(margin, 2) == 0.70

    ok, _ = _qualifies_probs(0.69, 0.15, 0.16)
    assert ok is False

    ok, _ = _qualifies_probs(0.70, 0.28, 0.02)
    assert ok is True

    ok, _ = _qualifies_probs(0.70, 0.31, 0.0)
    assert ok is True
