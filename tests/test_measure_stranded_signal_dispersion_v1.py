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
def test_outcome_language_is_target_week_scoped_not_absolute():
    """Audit 6449349551 blocker 4: prior-play results ARE outcomes."""
    st = json.loads(STATUS.read_text(encoding="utf-8"))
    assert st["target_week_outcomes_inspected"] is False
    assert st["prior_game_play_results_used"] is True
    assert st["model_scoring_performed"] is False
    assert "withdrawn" in st["outcome_language_note"]
    assert "outcomes_inspected" not in st, (
        "the absolute outcome claim must not reappear")
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
    # V2 position: persistence is not established, and the caution runs both
    # ways -- the test is also underpowered to call the signals noise.
    assert "persistence is NOT ESTABLISHED" in doc
    assert "does not establish that the signals *are* noise" in doc or \
        "not establish that the signals" in doc


@pytest.mark.skipif(not STATUS.exists(), reason="study not run")
def test_disjoint_persistence_replaces_the_overlapping_claim():
    """Blocker 1: the withdrawn claim must not survive anywhere."""
    st = json.loads(STATUS.read_text(encoding="utf-8"))
    assert "RETRACTED_OVERLAPPING_PERSISTENCE" in st
    assert "WITHDRAWN" in st["RETRACTED_OVERLAPPING_PERSISTENCE"]
    dp = st["disjoint_persistence"]
    for sig in SIGNALS:
        assert sig in dp
        assert dp[sig]["label"] in ("PERSISTENT", "WEAKLY_PERSISTENT",
                                    "NOT_PERSISTENT", "INSUFFICIENT_ELIGIBLE")
    doc = Path("docs/research/"
               "PRISM_STRANDED_SIGNAL_DISPERSION_RESULTS_V1.md").read_text(
                   encoding="utf-8")
    for para in doc.split("\n\n"):
        if "persistent properties of a defense" in para.lower():
            assert "withdrawn" in para.lower() or "v1" in para.lower()


def test_disjoint_windows_actually_share_no_week():
    from scripts.research.measure_stranded_signal_dispersion_v1 import (
        EARLY_WEEKS, LATE_WEEKS)
    assert set(EARLY_WEEKS).isdisjoint(LATE_WEEKS)
    assert len(EARLY_WEEKS) >= 2 and len(LATE_WEEKS) >= 2


@pytest.mark.skipif(not STATUS.exists(), reason="study not run")
def test_provenance_does_not_call_all_six_stranded():
    """Blocker 3: success_rate_def IS read by production."""
    st = json.loads(STATUS.read_text(encoding="utf-8"))
    prov = st["signal_provenance"]
    assert prov["success_rate_def"]["class"] == "READ_BY_PRODUCTION"
    assert prov["def_rush_epa"]["class"] == "TRANSPORTED_BUT_STRANDED"
    stranded = [k for k, v in prov.items()
                if v["class"] == "TRANSPORTED_BUT_STRANDED"]
    assert stranded == ["def_rush_epa"], (
        "only def_rush_epa is transported-but-stranded on cleared evidence")
    # and production really does read it
    src = Path("scripts/modeling/rules_v2.py").read_text(encoding="utf-8")
    assert "defense.success_rate_def" in src
    assert "success_diff(offense, defense)" in src


@pytest.mark.skipif(not STATUS.exists(), reason="study not run")
def test_iqr_cutoffs_are_labelled_post_hoc():
    st = json.loads(STATUS.read_text(encoding="utf-8"))
    assert "POST-PREREGISTRATION" in st["IQR_CUTOFFS_NOT_PREREGISTERED"]
