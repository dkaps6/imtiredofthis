import numpy as np
import pandas as pd
import pytest

from scripts.research.diagnose_vegas_confirmation_likelihood_v1 import (
    MONEYLINE_COVERAGE_MIN,
    PRIMARY_THRESHOLD,
    SENSITIVITY_THRESHOLD,
    add_labels,
    add_pregame_features,
    american_implied_prob,
    disposition,
    downstream_summary,
    fit_label,
)


def test_american_implied_prob_matches_known_values():
    # -110 (standard vig line) -> 110/210
    assert american_implied_prob(-110.0) == pytest.approx(110.0 / 210.0)
    # +150 underdog -> 100/250
    assert american_implied_prob(150.0) == pytest.approx(100.0 / 250.0)
    assert np.isnan(american_implied_prob(float("nan")))
    assert np.isnan(american_implied_prob(0.0))


def _fake_games(n=6):
    """season/week/home_team/away_team/game_id + predicted/actual margin+total,
    the shape load_game_outcomes would produce.
    """
    rows = []
    seasons = [2023] * 2 + [2024] * 2 + [2025] * 2
    for i, season in enumerate(seasons):
        rows.append({
            "season": season, "week": i + 1, "game_id": f"g{i}",
            "home_team": "AAA", "away_team": "BBB",
            "predicted_margin_home": 3.0 + i, "predicted_total": 44.0 + i,
            "actual_margin_home": 4.0 + i, "actual_total": 45.0 + i,
        })
    return pd.DataFrame(rows)


def test_add_pregame_features_computes_key_margin_distance():
    games = _fake_games()
    geometry = pd.DataFrame({
        "season": games.season, "week": games.week,
        "home_team": games.home_team, "away_team": games.away_team,
        "home_moneyline": np.nan, "away_moneyline": np.nan,
    })
    x, meta = add_pregame_features(games, geometry)
    assert meta["moneyline_features_used"] is False
    # predicted_margin_home=3.0 -> abs_spread=3.0 -> distance to nearest key {3,7,10,14} is 0.
    row = x.loc[x["predicted_margin_home"].eq(3.0)].iloc[0]
    assert row["key_margin_distance"] == pytest.approx(0.0)
    # predicted_margin_home=5.0 -> abs_spread=5.0 -> nearest key is 3 or 7, distance 2.
    row2 = x.loc[x["predicted_margin_home"].eq(5.0)].iloc[0]
    assert row2["key_margin_distance"] == pytest.approx(2.0)


def test_add_pregame_features_gates_moneyline_block_on_coverage():
    games = _fake_games()
    # Only half the rows carry both moneylines -> below the 0.80 coverage bar.
    geometry = pd.DataFrame({
        "season": games.season, "week": games.week,
        "home_team": games.home_team, "away_team": games.away_team,
        "home_moneyline": [-150.0, np.nan, -150.0, np.nan, -150.0, np.nan],
        "away_moneyline": [130.0, np.nan, 130.0, np.nan, 130.0, np.nan],
    })
    x, meta = add_pregame_features(games, geometry)
    assert meta["moneyline_features_used"] is False
    assert "favorite_ml_prob" not in x.columns


def _fake_games_with_ample_train_rows(n_train=60, n_test=6):
    """add_pregame_features fails closed if the moneyline block passes the
    coverage gate but has under 50 usable train rows (a real safety check,
    not a bug) -- so the moneyline-eligible test needs a fixture large
    enough to clear that floor.
    """
    rows = []
    gid = 0
    for season, n in [(2023, n_train // 2), (2024, n_train - n_train // 2), (2025, n_test)]:
        for i in range(n):
            rows.append({
                "season": season, "week": i + 1, "game_id": f"g{gid}",
                "home_team": "AAA", "away_team": "BBB",
                "predicted_margin_home": 3.0, "predicted_total": 44.0,
                "actual_margin_home": 4.0, "actual_total": 45.0,
            })
            gid += 1
    return pd.DataFrame(rows)


def test_add_pregame_features_computes_novig_favorite_probability_when_eligible():
    games = _fake_games_with_ample_train_rows()
    geometry = pd.DataFrame({
        "season": games.season, "week": games.week,
        "home_team": games.home_team, "away_team": games.away_team,
        "home_moneyline": [-150.0] * len(games), "away_moneyline": [130.0] * len(games),
    })
    x, meta = add_pregame_features(games, geometry)
    assert meta["moneyline_features_used"] is True
    home_p = american_implied_prob(-150.0)
    away_p = american_implied_prob(130.0)
    expected_favorite = max(home_p, away_p) / (home_p + away_p)
    row = x.iloc[0]
    assert row["favorite_ml_prob"] == pytest.approx(expected_favorite)
    # Train-only mapping: the residual must not be identically zero once fit
    # on a 2-point-only synthetic series, but must exist as a real column.
    assert "spread_ml_consistency_resid" in x.columns


def test_add_labels_requires_same_side_not_just_small_error():
    """Same fix as PR #560: a sign reversal / cutoff crossing must not count
    as confirmed even if the absolute error is small.
    """
    x = pd.DataFrame({
        "predicted_margin_home": [3.0, 3.0], "actual_margin_home": [4.0, -3.0],
        "predicted_total": [44.0, 43.0], "actual_total": [45.0, 44.5],
    })
    out = add_labels(x, total_cutoff=44.0)
    # Row 0: same side (both positive), error 1 -> confirmed at both T.
    assert out.loc[0, "margin_confirmed_7"] == 1
    assert out.loc[0, "margin_confirmed_3"] == 1
    # Row 1: predicted +3 (favors home), actual -3 (home lost) -> sign
    # reversal, must NOT be confirmed even though abs error (6) <= T=7.
    assert out.loc[1, "margin_confirmed_7"] == 0
    # Row 1 total: predicted 43 (low side of 44 cutoff), actual 44.5 (high
    # side) -> cutoff crossing, must not be confirmed despite abs error 1.5.
    assert out.loc[1, "total_confirmed_3"] == 0


def _classification_frame(n_per_class=60, seed=1, signal_strength=2.0):
    """Synthetic train(2023-24)+test(2025) frame where the label is
    genuinely predictable from predicted_margin_home so fit_label's gates
    can be exercised in both the pass and fail direction.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for season in [2023, 2024, 2025]:
        for i in range(n_per_class * 2):
            predicted_margin_home = rng.uniform(-14, 14)
            abs_spread = abs(predicted_margin_home)
            predicted_total = rng.uniform(38, 52)
            key_margin_distance = 1.0
            # Label genuinely driven by abs_spread: bigger spreads are more
            # likely confirmed (plausible, testable signal).
            logit = signal_strength * (abs_spread - 7.0) / 7.0
            p = 1.0 / (1.0 + np.exp(-logit))
            label = int(rng.uniform() < p)
            rows.append({
                "season": season, "week": (i % 18) + 1, "game_id": f"{season}_{i}",
                "home_team": "AAA", "away_team": "BBB",
                "predicted_margin_home": predicted_margin_home,
                "abs_spread": abs_spread, "predicted_total": predicted_total,
                "key_margin_distance": key_margin_distance,
                "margin_confirmed_7": label,
            })
    return pd.DataFrame(rows)


def test_fit_label_passes_gates_when_signal_is_genuinely_predictive():
    frame = _classification_frame(signal_strength=3.0, seed=2)
    row, pred = fit_label(
        frame, label_col="margin_confirmed_7",
        features=["predicted_margin_home", "abs_spread", "predicted_total", "key_margin_distance"],
        hypothesis="margin", threshold=PRIMARY_THRESHOLD,
    )
    assert row["auc"] > 0.55
    assert row["classification_gates_pass"] is True
    assert row["selector_cutoff_train_q75"] > 0
    assert len(pred) == row["test_rows"]


def test_fit_label_fails_gates_when_label_is_pure_noise():
    rng = np.random.default_rng(3)
    rows = []
    for season in [2023, 2024, 2025]:
        for i in range(120):
            rows.append({
                "season": season, "week": (i % 18) + 1, "game_id": f"noise_{season}_{i}",
                "home_team": "AAA", "away_team": "BBB",
                "predicted_margin_home": rng.uniform(-14, 14),
                "abs_spread": rng.uniform(0, 14),
                "predicted_total": rng.uniform(38, 52),
                "key_margin_distance": rng.uniform(0, 3),
                "margin_confirmed_7": int(rng.uniform() < 0.4),
            })
    frame = pd.DataFrame(rows)
    row, _ = fit_label(
        frame, label_col="margin_confirmed_7",
        features=["predicted_margin_home", "abs_spread", "predicted_total", "key_margin_distance"],
        hypothesis="margin", threshold=PRIMARY_THRESHOLD,
    )
    assert row["classification_gates_pass"] is False


def test_fit_label_drops_pickem_rows_for_margin_hypothesis():
    """A predicted_margin_home==0.0 (pick'em) row has no defined favored
    side and must be excluded from the margin hypothesis's train/test cohort,
    same as PR #560's exclusion.
    """
    frame = _classification_frame(signal_strength=3.0, seed=4)
    n_2025_before = int(frame["season"].eq(2025).sum())
    frame = pd.concat([frame, pd.DataFrame([{
        "season": 2025, "week": 1, "game_id": "pickem", "home_team": "CCC", "away_team": "DDD",
        "predicted_margin_home": 0.0, "abs_spread": 0.0,
        "predicted_total": 44.0, "key_margin_distance": 3.0, "margin_confirmed_7": 1,
    }])], ignore_index=True)
    row, pred = fit_label(
        frame, label_col="margin_confirmed_7",
        features=["predicted_margin_home", "abs_spread", "predicted_total", "key_margin_distance"],
        hypothesis="margin", threshold=PRIMARY_THRESHOLD,
    )
    assert row["test_rows"] == n_2025_before
    assert "pickem" not in pred["game_id"].to_numpy()


def test_disposition_requires_both_thresholds_and_downstream_to_qualify():
    class_summary = pd.DataFrame([
        {"hypothesis": "margin", "confirm_threshold": 7.0, "classification_gates_pass": True},
        {"hypothesis": "margin", "confirm_threshold": 3.0, "classification_gates_pass": True},
        {"hypothesis": "total", "confirm_threshold": 7.0, "classification_gates_pass": False},
        {"hypothesis": "total", "confirm_threshold": 3.0, "classification_gates_pass": False},
    ])
    downstream = pd.DataFrame([
        {"hypothesis": "margin", "is_primary_metric": True, "arm": "selected_high_confirmation", "downstream_primary_gate_pass": True},
        {"hypothesis": "total", "is_primary_metric": True, "arm": "selected_high_confirmation", "downstream_primary_gate_pass": False},
    ])
    disp = disposition(class_summary, downstream)
    assert disp["margin"]["status"] == "QUALIFIED_PREGAME_CONFIRMATION_CANDIDATE"
    assert disp["total"]["status"] == "NO_ACTIONABLE_PREGAME_CONFIRMATION_STATE"
    assert disp["overall"]["status"] == "NO_COMBINED_PROMOTION"


def test_disposition_predictable_but_not_useful_when_classification_passes_but_downstream_fails():
    class_summary = pd.DataFrame([
        {"hypothesis": "margin", "confirm_threshold": 7.0, "classification_gates_pass": True},
        {"hypothesis": "margin", "confirm_threshold": 3.0, "classification_gates_pass": True},
        {"hypothesis": "total", "confirm_threshold": 7.0, "classification_gates_pass": True},
        {"hypothesis": "total", "confirm_threshold": 3.0, "classification_gates_pass": True},
    ])
    downstream = pd.DataFrame([
        {"hypothesis": "margin", "is_primary_metric": True, "arm": "selected_high_confirmation", "downstream_primary_gate_pass": False},
        {"hypothesis": "total", "is_primary_metric": True, "arm": "selected_high_confirmation", "downstream_primary_gate_pass": True},
    ])
    disp = disposition(class_summary, downstream)
    assert disp["margin"]["status"] == "PREDICTABLE_BUT_NOT_USEFUL_FOR_PLAYER_VOLUME"
    assert disp["total"]["status"] == "QUALIFIED_PREGAME_CONFIRMATION_CANDIDATE"
    assert disp["overall"]["status"] == "NO_COMBINED_PROMOTION"
