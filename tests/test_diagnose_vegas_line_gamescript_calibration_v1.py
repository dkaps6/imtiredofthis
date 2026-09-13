import numpy as np
import pandas as pd
import pytest

from scripts.research.diagnose_vegas_line_gamescript_calibration_v1 import (
    MIN_ROWS_PER_FOLD,
    _fit_and_apply,
    binned_game_script_accuracy,
    direct_calibration,
    load_game_outcomes,
    out_of_sample_recalibration,
)


def _synthetic_games(n_per_season=40, seasons=(2023, 2024, 2025), bias=5.0, seed=7):
    """Every season shares the same underlying bias structure: actual_total
    runs `bias` points above predicted_total, and actual_margin_home tracks
    predicted_margin_home 1:1 with noise. A linear recalibration fit on any
    two seasons should transfer cleanly to the third and shrink MAE.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for i in range(n_per_season):
            predicted_total = rng.uniform(38, 52)
            predicted_margin = rng.uniform(-14, 14)
            actual_total = predicted_total + bias + rng.normal(0, 1.0)
            actual_margin = predicted_margin + rng.normal(0, 1.0)
            rows.append({
                "season": season, "week": (i % 18) + 1,
                "home_team": "AAA", "away_team": "BBB",
                "predicted_total": predicted_total,
                "predicted_margin_home": predicted_margin,
                "actual_total": actual_total,
                "actual_margin_home": actual_margin,
            })
    return pd.DataFrame(rows)


def test_direct_calibration_reports_pooled_and_per_season_rows():
    games = _synthetic_games()
    out = direct_calibration(games)
    assert "ALL_SEASONS" in out["season"].tolist()
    for season in (2023, 2024, 2025):
        assert str(season) in out["season"].tolist()
    all_row = out.loc[out.season.eq("ALL_SEASONS")].iloc[0]
    # Bias is baked into the synthetic fixture: raw total_bias should recover it.
    assert all_row["total_bias"] == pytest.approx(5.0, abs=0.5)
    assert all_row["margin_bias"] == pytest.approx(0.0, abs=0.5)
    assert all_row["margin_corr"] > 0.9


def test_out_of_sample_recalibration_detects_and_corrects_transferable_bias():
    games = _synthetic_games()
    out = out_of_sample_recalibration(games)
    assert set(out["status"]) == {"OK"}
    total_rows = out.loc[out.target.eq("total")]
    # The +5 bias is present in every season, so a fold trained on the other
    # two seasons should recover it and cut raw MAE substantially on holdout.
    for _, row in total_rows.iterrows():
        assert row["fitted_slope"] == pytest.approx(1.0, abs=0.3)
        assert row["fitted_intercept"] == pytest.approx(5.0, abs=3.0)
        assert row["recalibrated_mae"] < row["raw_line_mae"]
        assert row["recalibration_mae_improvement"] > 0


def test_median_recalibration_beats_ols_recalibration_under_skewed_residuals():
    """Codex P1 on PR #559: OLS (np.polyfit) targets the conditional mean,
    but MAE is optimized by the conditional median. Under right-skewed
    residuals (mean bias > median bias), OLS recalibration overshoots the
    correction and a median (L1) regression should show a larger -- or at
    least not smaller -- MAE improvement.
    """
    def _frame(n, seed_offset):
        r = np.random.default_rng(21 + seed_offset)
        x = r.uniform(30, 60, n)
        noise = r.exponential(scale=4.0, size=n)  # mean=4, median=4*ln(2)~2.77: right-skewed
        y = x + noise
        return pd.DataFrame({"predicted_total": x, "actual_total": y})

    train = _frame(300, 0)
    test = _frame(200, 1)
    result = _fit_and_apply(train, test, pred_col="predicted_total", actual_col="actual_total")
    assert result["status"] == "OK"
    for col in ["median_fitted_slope", "median_fitted_intercept", "median_recalibrated_mae", "median_recalibration_mae_improvement"]:
        assert col in result
    # OLS's mean-targeting intercept overshoots the true median bias under
    # right-skewed noise, so the median-regression arm should do at least as
    # well, and in this constructed case strictly better.
    assert result["median_recalibrated_mae"] <= result["recalibrated_mae"] + 1e-9
    assert result["median_recalibration_mae_improvement"] >= result["recalibration_mae_improvement"] - 1e-9


def test_out_of_sample_recalibration_fails_closed_on_insufficient_rows():
    games = _synthetic_games(n_per_season=5)
    out = out_of_sample_recalibration(games)
    assert set(out["status"]) == {"INSUFFICIENT_ROWS"}
    assert MIN_ROWS_PER_FOLD > 5


def test_binned_game_script_accuracy_shows_monotonic_competitiveness_and_favorite_win_rate():
    rng = np.random.default_rng(11)
    rows = []
    # Construct games so that larger spread buckets have both larger actual
    # margins and a higher favorite win rate -- the expected "Vegas gets the
    # script right" pattern, to check the bucket table actually reflects it.
    for abs_spread, actual_margin_level, win_rate in [(1.0, 2.0, 0.55), (5.0, 6.0, 0.65), (9.0, 10.0, 0.80), (12.0, 13.0, 0.90), (18.0, 17.0, 0.95)]:
        for i in range(30):
            sign = 1.0
            favorite_wins = rng.uniform() < win_rate
            actual_margin = actual_margin_level if favorite_wins else -abs(rng.normal(2.0, 1.0))
            rows.append({
                "season": 2024, "week": (i % 18) + 1,
                "home_team": "AAA", "away_team": "BBB",
                "predicted_total": 44.0,
                "predicted_margin_home": sign * abs_spread,
                "actual_total": 44.0,
                "actual_margin_home": actual_margin,
            })
    games = pd.DataFrame(rows)
    spread_table, total_table = binned_game_script_accuracy(games)
    ordered = spread_table.set_index("spread_bucket").loc[["0-3", "3-7", "7-10", "10-14", "14+"]]
    margins = ordered["mean_actual_abs_margin"].to_numpy()
    assert np.all(np.diff(margins) > 0)
    win_rates = ordered["favorite_win_rate"].to_numpy()
    assert win_rates[-1] > win_rates[0]
    assert not total_table.empty


def test_load_game_outcomes_favors_home_team_when_spread_positive(monkeypatch):
    """Same nflverse sign convention re-verified here independently of PR
    #558: spread_line positive means home is favored, so predicted_margin_home
    should equal spread_line directly (not its negation).
    """
    fake_schedule = pd.DataFrame([{
        "season": 2023, "week": 10, "game_type": "REG", "game_id": "2023_10_NYG_DAL",
        "home_team": "DAL", "away_team": "NYG",
        "spread_line": 17.5, "total_line": 38.5,
        "home_score": 49, "away_score": 17,
    }])

    import sys
    import types

    fake_module = types.ModuleType("nflreadpy")
    fake_module.load_schedules = lambda season: fake_schedule
    monkeypatch.setitem(sys.modules, "nflreadpy", fake_module)

    games = load_game_outcomes([2023])
    row = games.iloc[0]
    assert row["predicted_margin_home"] == pytest.approx(17.5)
    assert row["actual_margin_home"] == pytest.approx(32.0)
    assert row["actual_total"] == pytest.approx(66.0)
    assert row["predicted_total"] == pytest.approx(38.5)


def test_load_game_outcomes_drops_incomplete_games(monkeypatch):
    fake_schedule = pd.DataFrame([
        {
            "season": 2023, "week": 1, "game_type": "REG", "game_id": "played",
            "home_team": "DAL", "away_team": "NYG",
            "spread_line": 3.0, "total_line": 44.0,
            "home_score": 24, "away_score": 20,
        },
        {
            "season": 2023, "week": 20, "game_type": "REG", "game_id": "not_played",
            "home_team": "SF", "away_team": "SEA",
            "spread_line": 3.0, "total_line": 44.0,
            "home_score": None, "away_score": None,
        },
    ])

    import sys
    import types

    fake_module = types.ModuleType("nflreadpy")
    fake_module.load_schedules = lambda season: fake_schedule
    monkeypatch.setitem(sys.modules, "nflreadpy", fake_module)

    games = load_game_outcomes([2023])
    assert len(games) == 1
    assert games.iloc[0]["game_id"] == "played"
