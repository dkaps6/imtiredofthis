import numpy as np
import pandas as pd
import pytest

from scripts.modeling.rb_rush_rec_conservation_v2 import build_candidate_map
from scripts.simulation_v2 import SimulationResult


def _metrics(*, week=2, position="RB", include_rush=True, include_rec=True):
    markets = ["rush_rec_yards"]
    if include_rush:
        markets.append("rush_yards")
    if include_rec:
        markets.append("rec_yards")
    return pd.DataFrame([
        {
            "event_id": "g1",
            "player": "Test Runner",
            "player_clean_key": "testrunner",
            "team": "AAA",
            "opponent": "BBB",
            "position": position,
            "season": 2026,
            "week": week,
            "market": m,
            "ml_proj": np.nan,
            "state_proj": np.nan,
            # Deliberately include sportsbook-looking columns. The adapter
            # contracts to an allowlist before any computation.
            "line": 50.5,
            "over_odds": -110,
            "under_odds": -110,
        }
        for m in markets
    ])


def _sims():
    rush = np.array([10.0, 20.0, 30.0])
    rec = np.array([1.0, 2.0, 3.0])
    return SimulationResult(
        values={
            ("g1", "testrunner", "rush_yards"): rush,
            ("g1", "testrunner", "rec_yards"): rec,
            ("g1", "testrunner", "rush_rec_yards"): rush + rec,
        },
        iterations=3,
    )


def test_nonweek1_rb_builds_exact_pathwise_sum_with_no_weights():
    out, payload = build_candidate_map(_metrics(), _sims(), pd.DataFrame())
    assert payload["sportsbook_inputs_used"] == 0
    assert payload["week1_rows_changed"] == 0
    assert payload["players"] == 1
    row = out[("g1", "testrunner")]
    np.testing.assert_array_equal(row["draws"], np.array([11.0, 22.0, 33.0]))
    assert row["rush_target_mean"] == pytest.approx(20.0)
    assert row["rec_target_mean"] == pytest.approx(2.0)
    assert row["target_mean"] == pytest.approx(22.0)


def test_week1_is_strict_noop():
    out, payload = build_candidate_map(_metrics(week=1), _sims(), pd.DataFrame())
    assert out == {}
    assert payload["players"] == 0
    assert payload["week1_rows_changed"] == 0


def test_non_rb_is_strict_noop():
    out, payload = build_candidate_map(_metrics(position="WR"), _sims(), pd.DataFrame())
    assert out == {}
    assert payload["players"] == 0


def test_missing_standalone_component_fails_closed():
    with pytest.raises(RuntimeError, match="missing standalone component"):
        build_candidate_map(_metrics(include_rush=False), _sims(), pd.DataFrame())


def test_fb_is_strict_noop():
    out, payload = build_candidate_map(_metrics(position="FB"), _sims(), pd.DataFrame())
    assert out == {}
    assert payload["players"] == 0
