"""Guards for the player-specific matchup study.

The study reports a null. A null is only trustworthy if the machinery could have
detected a real effect, and if the decision rule cannot be quietly relaxed later.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.research import measure_player_specific_matchup_response_v1 as M


def _panel(n=400, seed=0, adot_effect=0.0):
    rng = np.random.default_rng(seed)
    pos = rng.choice(list(M.POSITIONS), n)
    adot = rng.uniform(2.0, 16.0, n)
    zone = rng.uniform(0.3, 0.8, n)
    base = rng.uniform(20.0, 90.0, n)
    y = base + adot_effect * adot * zone + rng.normal(0, 5.0, n)
    return pd.DataFrame({
        "pid": [f"p{i%60}" for i in range(n)], "player": [f"P{i%60}" for i in range(n)],
        "pos": pos, "week": rng.integers(4, 19, n), "opponent": "OPP",
        "prior_adot": adot, "prior_ypt": rng.uniform(6, 12, n),
        "prior_tgt_per_game": rng.uniform(2, 9, n), "prior_catch_rate": rng.uniform(.5, .8, n),
        "prior_targets": rng.uniform(20, 120, n), "prior_games": rng.integers(3, 12, n),
        "def_zone_rate": zone, "def_man_rate": 1 - zone,
        "def_light_box_rate": rng.uniform(.2, .6, n), "def_heavy_box_rate": rng.uniform(.1, .4, n),
        "def_prior_labeled_plays": 500.0,
        "y_receiving_yards": y, "y_targets": rng.uniform(1, 12, n),
        "yhat_base": base, "state": "HEALTHY_FULL",
    })


def test_frozen_thresholds_match_the_preregistration():
    assert M.MIN_PRIOR_TARGETS == 20
    assert M.MIN_PRIOR_GAMES_WITH_TARGET == 3
    assert M.MIN_DEF_PRIOR_LABELED_PLAYS == 100
    assert list(M.FIT_WEEKS) == list(range(4, 12))
    assert list(M.HOLDOUT_WEEKS) == list(range(12, 19))
    assert list(M.PERSIST_A) == list(range(4, 11))
    assert list(M.PERSIST_B) == list(range(11, 19))
    # the persistence windows must be DISJOINT -- the lesson from the dispersion study
    assert not (set(M.PERSIST_A) & set(M.PERSIST_B))


def test_main_refuses_to_score_without_the_preregistration(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: False)
    with pytest.raises(RuntimeError, match="preregistration is missing"):
        M.main()


def test_study_detects_a_real_player_specific_effect_when_one_exists():
    """If the machinery cannot find a planted effect, its null means nothing."""
    out, _ = M.run(_panel(n=600, adot_effect=2.5))
    assert out["M2_holdout_mae"] < out["M1_holdout_mae"], "planted aDOT effect not detected"
    assert out["rule_1_M2_beats_M1_player_level"] is True
    assert out["rule_2_sign_persists_disjoint"] is True


def test_study_returns_null_when_no_effect_is_planted():
    out, _ = M.run(_panel(n=600, adot_effect=0.0))
    assert out["H1_SUPPORTED"] is False


def test_h1_requires_all_three_rules():
    """No two-of-three pass may be reported as support."""
    out, _ = M.run(_panel(n=600, adot_effect=0.0))
    rules = (out["rule_1_M2_beats_M1_player_level"],
             out["rule_2_sign_persists_disjoint"],
             out["rule_3_concentrated_off_mean"])
    assert out["H1_SUPPORTED"] == all(rules)


def test_design_matrix_puts_the_player_interaction_where_the_reader_expects():
    df = _panel(n=50)
    x0, x1, x2 = (M._design(df, m).shape[1] for m in ("M0", "M1", "M2"))
    assert x0 == 2 and x1 == 2 + 2 * len(M.POSITIONS) and x2 == x1 + 2
    # the coefficient read for persistence is the aDOT x zone term
    X = M._design(df, "M2")
    assert np.allclose(X[:, -2], df["prior_adot"] * df["def_zone_rate"])
    assert np.allclose(X[:, -1], df["prior_adot"] * df["def_man_rate"])


def test_anomaly_gate_labels_states_before_evidence():
    assert M.state_label(pd.Series({"targets": 0, "carries": 0, "offense_snaps": 0.0}), 40.0) == "DNP_INACTIVE"
    assert M.state_label(pd.Series({"targets": 3, "carries": 0, "offense_snaps": 10.0}), 40.0) == "PARTIAL_GAME"
    assert M.state_label(pd.Series({"targets": 5, "carries": 0, "offense_snaps": 38.0}), 40.0) == "HEALTHY_FULL"
    assert M.state_label(pd.Series({"targets": 0, "carries": 0, "offense_snaps": np.nan}), 40.0) == "UNRESOLVED_CONTEXT"


def test_defense_prior_is_strictly_prior_and_fails_closed_on_thin_evidence():
    dw = pd.DataFrame({
        "defense": ["AAA"] * 4, "week": [1, 2, 3, 9],
        "labeled_plays": [60, 60, 60, 999], "zone_plays": [30, 45, 45, 0],
        "box_known": [60, 60, 60, 999], "light_box": [20, 20, 20, 0], "heavy_box": [10, 10, 10, 0],
    })
    out = M.defense_prior(dw, "AAA", 4)
    assert out["def_prior_labeled_plays"] == 180.0          # week 9 excluded: strictly prior
    assert out["def_zone_rate"] == pytest.approx(120 / 180)
    assert M.defense_prior(dw, "AAA", 2) is None            # 60 < 100: fails closed


def test_secondary_outcomes_are_reported_but_cannot_decide():
    panel = _panel(n=600, adot_effect=0.0)
    sec = M.secondary_outcomes(panel)
    assert "targets" in sec and "M2_minus_M1" in sec["targets"]
    out, _ = M.run(panel)
    assert "targets" not in str(out.get("H1_SUPPORTED"))    # verdict is yards-only
