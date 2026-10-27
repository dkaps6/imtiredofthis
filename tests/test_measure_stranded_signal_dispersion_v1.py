"""Invariants for the stranded-signal dispersion study."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.research.measure_stranded_signal_dispersion_v1 import (
    MIN_RANK_CORR,
    SIGNALS,
    TARGET_WEEKS,
    TRIVIAL_IQR,
    _aggregate,
    _assert_strictly_prior,
    build_as_of,
    dispersion,
)

STATUS = Path("data/research/stranded_dispersion/"
              "stranded_signal_dispersion_v1_status.json")


def _plays(weeks=(1, 2, 3)):
    rows = []
    for w in weeks:
        for t, base in (("A", -0.3), ("B", 0.2)):
            for i in range(30):
                rows.append({"week": w, "defteam": t, "posteam": "X",
                             "play_type": "run" if i % 2 else "pass",
                             "epa": base + (i % 5) * 0.01,
                             "success": float(i % 2),
                             "yards_gained": float(i % 12)})
    return pd.DataFrame(rows)


def test_chronology_guard_raises_on_a_leak():
    p = _plays((1, 2, 3))
    _assert_strictly_prior(p.loc[p["week"] < 3], 3)        # fine
    with pytest.raises(RuntimeError, match="chronology violation"):
        _assert_strictly_prior(p, 3)                        # week 3 present


def test_as_of_uses_only_strictly_prior_weeks():
    p = _plays(tuple(range(1, 8)))
    a = build_as_of(p)
    assert set(a["target_week"]) <= set(TARGET_WEEKS)
    assert 1 not in set(a["target_week"]), "week 1 must be excluded"
    # the week-2 aggregate may only rest on week-1 plays
    w2 = a.loc[a["target_week"].eq(2), "prior_plays"].sum()
    assert w2 == len(p.loc[p["week"].eq(1)])


def test_aggregate_produces_every_signal():
    agg = _aggregate(_plays((1,)))
    for sig in SIGNALS:
        assert sig in agg.columns
        assert agg[sig].notna().any(), f"{sig} is entirely null"


def test_unstable_is_reported_rather_than_material():
    """Pre-committed: wide but rank-unstable must NOT read as material."""
    rng = np.random.default_rng(0)
    rows = []
    for wk in TARGET_WEEKS:
        for i in range(32):
            r = {"defense": f"T{i}", "target_week": wk}
            for sig in SIGNALS:
                r[sig] = float(rng.normal(0, 1))   # wide, no rank persistence
            rows.append(r)
    out = dispersion(pd.DataFrame(rows))
    for sig in SIGNALS:
        assert out[sig]["label"] in ("UNSTABLE", "MATERIAL_DISPERSION")
        if out[sig]["median_rank_corr"] is not None:
            if out[sig]["median_rank_corr"] <= MIN_RANK_CORR:
                assert out[sig]["label"] == "UNSTABLE"


def test_narrow_is_reported_when_spread_is_trivial():
    rows = []
    for wk in TARGET_WEEKS:
        for i in range(32):
            r = {"defense": f"T{i}", "target_week": wk}
            for sig in SIGNALS:
                r[sig] = 0.5 + i * 1e-6      # essentially constant
            rows.append(r)
    out = dispersion(pd.DataFrame(rows))
    for sig in SIGNALS:
        assert out[sig]["label"] == "NARROW"
        assert out[sig]["median_iqr"] <= TRIVIAL_IQR[sig]


@pytest.mark.skipif(not STATUS.exists(), reason="study not run")
def test_study_is_outcome_and_player_blind():
    st = json.loads(STATUS.read_text(encoding="utf-8"))
    assert st["outcomes_inspected"] is False
    assert st["player_level_columns_read"] is False
    assert st["sportsbook_inputs_used"] == 0
    assert st["week1_excluded_no_prior_evidence"] is True
    assert 1 not in st["target_weeks"]


@pytest.mark.skipif(not STATUS.exists(), reason="study not run")
def test_no_predictive_claim_is_recorded():
    st = json.loads(STATUS.read_text(encoding="utf-8"))
    assert "NOT evidence of predictive value" in st["labels_are_dispersion_only"]
    assert any("improve accuracy" in c for c in st["cannot_establish"])
    doc = Path("docs/research/"
               "PRISM_STRANDED_SIGNAL_DISPERSION_RESULTS_V1.md").read_text(
                   encoding="utf-8")
    assert "necessary but not sufficient" in doc
