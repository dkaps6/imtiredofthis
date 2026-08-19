import math

import pandas as pd

from scripts.backtest.evaluate_components import add_market_winner, evaluate, prepare_errors, summarize


def _sample_predictions():
    return pd.DataFrame([
        {"season": 2025, "week": 1, "player": "QB One", "team": "IND", "opponent": "TEN", "position": "QB", "market": "pass_yards", "actual": 200, "mc_proj": 220, "ml_proj": 190, "state_proj": 210},
        {"season": 2025, "week": 2, "player": "QB One", "team": "IND", "opponent": "DEN", "position": "QB", "market": "pass_yards", "actual": 250, "mc_proj": 280, "ml_proj": 245, "state_proj": 240},
        {"season": 2025, "week": 7, "player": "WR One", "team": "IND", "opponent": "TEN", "position": "WR", "market": "rec_yards", "actual": 80, "mc_proj": 75, "ml_proj": 70, "state_proj": None},
        {"season": 2025, "week": 13, "player": "WR One", "team": "IND", "opponent": "HOU", "position": "WR", "market": "rec_yards", "actual": 100, "mc_proj": 90, "ml_proj": 95, "state_proj": 85},
    ])


def test_prepare_errors_is_row_level_and_preserves_missing_state():
    errors = prepare_errors(_sample_predictions())
    assert len(errors) == 12
    rec_state = errors.loc[(errors["market"] == "rec_yards") & (errors["week"] == 7) & (errors["model"] == "state")].iloc[0]
    assert not bool(rec_state["available"])
    qb_mc = errors.loc[(errors["week"] == 1) & (errors["model"] == "mc")].iloc[0]
    assert qb_mc["error"] == 20
    assert qb_mc["abs_error"] == 20
    assert qb_mc["season_phase"] == "early_1_6"


def test_summary_computes_mae_bias_coverage_and_winner():
    errors = prepare_errors(_sample_predictions())
    summary = add_market_winner(summarize(errors, ["market", "model"]))
    qb_ml = summary.loc[(summary["market"] == "pass_yards") & (summary["model"] == "ml")].iloc[0]
    assert qb_ml["n"] == 2
    assert math.isclose(qb_ml["mae"], 7.5)
    assert math.isclose(qb_ml["bias"], -7.5)
    assert bool(qb_ml["market_winner"])
    rec_state = summary.loc[(summary["market"] == "rec_yards") & (summary["model"] == "state")].iloc[0]
    assert math.isclose(rec_state["coverage"], 0.5)


def test_evaluate_writes_all_diagnostic_artifacts(tmp_path):
    paths = evaluate(_sample_predictions(), tmp_path)
    assert set(paths) == {"errors", "summary", "by_week", "by_position", "by_phase", "by_player", "worst"}
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0
    phase = pd.read_csv(paths["by_phase"])
    assert {"early_1_6", "mid_7_12", "late_13_18"}.issubset(set(phase["season_phase"]))
