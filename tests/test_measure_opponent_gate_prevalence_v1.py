"""Invariants for the gate prevalence study (audit 6448026674 finding 5).

These pin the things the audit found unprotected: the time-basis contrast, the
source-faithful null treatment, agreement between the status sidecar and the
result document's headline, and the orientation grouping.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.research.measure_opponent_gate_prevalence_v1 import (
    GATES,
    HEAVY_MIN,
    LIGHT_MAX,
    NEAR_CONSTANT,
    NEAR_DEAD,
    gate_summary,
    weekly_rates,
)

STATUS = Path("data/research/gate_prevalence/"
              "opponent_gate_prevalence_v1_status.json")
DOC = Path("docs/research/PRISM_GATE_PREVALENCE_RESULTS_V1.md")


def _frame():
    """Two teams x two weeks, with a deliberate missing box value."""
    return pd.DataFrame({
        "off": ["A"] * 6 + ["B"] * 6,
        "deff": ["B"] * 6 + ["A"] * 6,
        "week": [1, 1, 1, 2, 2, 2] * 2,
        "box": [5, 6, np.nan, 7, 8, 5, 6, 6, 6, 9, 9, np.nan],
        "scrimmage": [True] * 12,
        "is_zone": [True, False, True, True, True, False] * 2,
        "is_man": [False, True, False, False, False, True] * 2,
        "cov_classified": [True] * 12,
    })


def test_production_definitions_match_make_team_form():
    src = Path("scripts/make_team_form.py").read_text(encoding="utf-8")
    assert f"<= {LIGHT_MAX}" in src, "light-box threshold drifted from production"
    assert f">= {HEAVY_MIN}" in src, "heavy-box threshold drifted from production"


def test_literal_production_nulls_keep_missing_box_in_the_denominator():
    """Finding 3: NaN <= 6 is False, so production counts it as not-light."""
    f = _frame()
    strict = weekly_rates(f, "off", False, False, literal_production_nulls=True)
    dropped = weekly_rates(f, "off", False, False, literal_production_nulls=False)
    a1s = strict.loc[(strict["team"] == "A") & (strict["week"] == 1)].iloc[0]
    a1d = dropped.loc[(dropped["team"] == "A") & (dropped["week"] == 1)].iloc[0]
    # week 1 for A: boxes 5, 6, NaN
    assert a1s["box_plays"] == 3 and a1d["box_plays"] == 2
    assert a1s["light_box_rate"] == pytest.approx(2 / 3)   # NaN in denominator
    assert a1d["light_box_rate"] == pytest.approx(2 / 2)   # NaN excluded
    assert a1s["box_null_rows"] == 1


def test_orientation_grouping_actually_differs():
    f = _frame()
    o = weekly_rates(f, "off", True, True)
    d = weekly_rates(f, "deff", True, True)
    assert set(o["team"]) == set(d["team"]) == {"A", "B"}
    oa = o.loc[o["team"] == "A"].set_index("week")["light_box_rate"]
    da = d.loc[d["team"] == "A"].set_index("week")["light_box_rate"]
    assert not np.allclose(oa.values, da.values), (
        "offense-faced and defense-deployed collapsed to the same value; the "
        "orientation arms are not independent")


def test_labels_use_the_preregistered_cutoffs():
    assert NEAR_CONSTANT == 0.90 and NEAR_DEAD == 0.10
    hi = gate_summary(pd.DataFrame({"light_box_rate": [0.95] * 100}))
    lo = gate_summary(pd.DataFrame({"heavy_box_rate": [0.01] * 100}))
    mid = gate_summary(pd.DataFrame({"coverage_zone_rate": [0.7] * 50
                                     + [0.3] * 50}))
    assert hi["light_box_rate"]["label"] == "NEAR_CONSTANT"
    assert lo["heavy_box_rate"]["label"] == "NEAR_DEAD"
    assert mid["coverage_zone_rate"]["label"] == "DISCRIMINATING"


@pytest.mark.skipif(not STATUS.exists(), reason="study not run")
def test_both_time_bases_are_reported_for_every_arm():
    st = json.loads(STATUS.read_text(encoding="utf-8"))
    assert st["arms"], "no arms reported"
    for arm, bases in st["arms"].items():
        assert set(bases) == {"AS_OF_PREGAME", "SINGLE_WEEK"}, (
            f"{arm} does not report both time bases; the basis is a declared "
            "dimension and cannot be collapsed")


@pytest.mark.skipif(not STATUS.exists(), reason="study not run")
def test_only_cross_basis_unanimous_gates_are_called_unconditional():
    st = json.loads(STATUS.read_text(encoding="utf-8"))
    cross = st["cross_basis_label_agreement"]
    for field in GATES:
        labs = cross[field]["labels_across_arm_x_basis"]
        assert cross[field]["unconditional"] == (len(labs) == 1)
    uncond = set(st["unconditional_gates"])
    assert uncond == {f for f in GATES
                      if len(cross[f]["labels_across_arm_x_basis"]) == 1}
    # the measured fact at this target
    assert uncond == {"heavy_box_rate"}


@pytest.mark.skipif(not (STATUS.exists() and DOC.exists()),
                    reason="study not run")
def test_document_headline_cannot_overclaim_beyond_the_artifact():
    """Finding 1 in essence: the doc must not assert unearned robustness."""
    st = json.loads(STATUS.read_text(encoding="utf-8"))
    doc = DOC.read_text(encoding="utf-8")
    uncond = set(st["unconditional_gates"])
    # The withdrawn V1 headline must not be ASSERTED. A document that records
    # what it withdrew is correct and must still pass, so each occurrence only
    # has to sit in a paragraph that marks it withdrawn -- a blunt substring
    # ban would forbid the retraction itself.
    paras = doc.split("\n\n")
    for banned in ("robust across every methodological choice",
                   "all eight arms agree",
                   "robust to every methodological choice"):
        for para in paras:
            if banned in para:
                assert "ithdrawn" in para or "ithdraw" in para, (
                    f"withdrawn V1 claim asserted without retraction: "
                    f"{banned!r}")
    # Every conditional gate must be named as conditional somewhere.
    for f in GATES:
        if f not in uncond:
            assert f in doc
    assert "unmeasured" in doc.lower() or "not measured" in doc.lower(), (
        "middle-open and pressure gates must be declared unmeasured")


@pytest.mark.skipif(not STATUS.exists(), reason="study not run")
def test_sharp_denominator_inference_is_withdrawn_not_asserted():
    st = json.loads(STATUS.read_text(encoding="utf-8"))
    assert "WITHDRAWN_SHARP_DENOMINATOR_INFERENCE" in st
    assert "UNESTABLISHED" in st["WITHDRAWN_SHARP_DENOMINATOR_INFERENCE"]
    assert st["is_a_2026_claim"] is False
    assert st["outcomes_inspected"] is False
    assert st["sportsbook_inputs_used"] == 0
