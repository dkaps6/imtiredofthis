from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.evaluate_wr_qb_shared_tail_signal_v1 import (
    EXPECTED_QB_ROWS,
    add_features_and_outcomes,
    anti_retest_scoreboard,
    build_cohort,
    spearman,
)


def _synthetic(season: int, n: int, seed: int, *, plant_signal: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    qb_rows, player_rows, actual_rows = [], [], []
    for i in range(n):
        week = 1 + (i % 18)
        event_id = f"{season}_{week:02d}_G{i}"
        team = f"T{i}"
        c2_mean = 220 + rng.normal(0, 20)
        c2_upper = rng.uniform(20, 90)
        c2_p90 = c2_mean + c2_upper
        c2_p10 = c2_mean - rng.uniform(20, 60)
        b0_mean = c2_mean - rng.normal(0, 5)
        qb_rows.append(dict(
            season=season, week=week, event_id=event_id, team=team,
            primary_qb_key=f"qb{i}", football_synthesis=c2_mean, actual_pass_yards=c2_mean + rng.normal(0, 30),
            c2_mean=c2_mean, c2_p10=c2_p10, c2_p90=c2_p90, c2_p95=c2_p90 + 10,
            b0_mean=b0_mean, b0_p90=b0_mean + 40, b0_p10=b0_mean - 40,
        ))
        b0_rec = 55 + rng.normal(0, 15)
        effect = (0.5 * c2_upper if c2_upper > 55 else 0.0) if plant_signal else 0.0
        actual = max(0.0, b0_rec + effect + rng.normal(0, 25))
        player_rows.append(dict(
            season=season, week=week, event_id=event_id, team=team,
            player=f"WR{i}", player_clean_key=f"wr{i}", position="WR",
            b0_target_probability=0.25 + rng.uniform(0, 0.05), b0_rec_yards=b0_rec, c2_rec_yards=b0_rec + 2,
        ))
        actual_rows.append(dict(season=season, week=week, team=team, player_clean_key=f"wr{i}", rec_yards=actual))
    return pd.DataFrame(qb_rows), pd.DataFrame(player_rows), pd.DataFrame(actual_rows)


def _build(*, plant_signal: bool):
    qb24, p24, a24 = _synthetic(2024, EXPECTED_QB_ROWS[2024], seed=0, plant_signal=plant_signal)
    qb25, p25, a25 = _synthetic(2025, EXPECTED_QB_ROWS[2025], seed=1, plant_signal=plant_signal)
    qb = pd.concat([qb24, qb25], ignore_index=True)
    players = pd.concat([p24, p25], ignore_index=True)
    actual = pd.concat([a24, a25], ignore_index=True)
    return add_features_and_outcomes(build_cohort(qb, players, actual))


def test_cohort_row_counts_match_frozen_authority():
    cohort = _build(plant_signal=True)
    assert int((cohort.season == 2024).sum()) == 444
    assert int((cohort.season == 2025).sum()) == 440


def test_duplicate_qb_identity_fails_closed():
    qb24, p24, a24 = _synthetic(2024, EXPECTED_QB_ROWS[2024], seed=0, plant_signal=True)
    qb24_dup = pd.concat([qb24, qb24.iloc[[0]]], ignore_index=True)
    qb25, p25, a25 = _synthetic(2025, EXPECTED_QB_ROWS[2025], seed=1, plant_signal=True)
    qb = pd.concat([qb24_dup, qb25], ignore_index=True)
    players = pd.concat([p24, p25], ignore_index=True)
    actual = pd.concat([a24, a25], ignore_index=True)
    try:
        build_cohort(qb, players, actual)
    except RuntimeError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError for duplicate QB identity")


def test_missing_actual_outcome_fails_closed():
    qb24, p24, a24 = _synthetic(2024, EXPECTED_QB_ROWS[2024], seed=0, plant_signal=True)
    qb25, p25, a25 = _synthetic(2025, EXPECTED_QB_ROWS[2025], seed=1, plant_signal=True)
    qb = pd.concat([qb24, qb25], ignore_index=True)
    players = pd.concat([p24, p25], ignore_index=True)
    actual = pd.concat([a24, a25], ignore_index=True).iloc[1:]  # drop one row's outcome
    try:
        build_cohort(qb, players, actual)
    except RuntimeError as exc:
        assert "missing actual outcome" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for missing actual outcome")


def test_planted_signal_is_detected_via_spearman():
    cohort = _build(plant_signal=True)
    hold = cohort.loc[cohort.season.eq(2025)]
    rho = spearman(hold.qb_c2_upper90.to_numpy(float), hold.wr_residual.to_numpy(float))
    assert rho > 0.3


def test_no_signal_gives_near_zero_correlation():
    cohort = _build(plant_signal=False)
    hold = cohort.loc[cohort.season.eq(2025)]
    rho = spearman(hold.qb_c2_upper90.to_numpy(float), hold.wr_residual.to_numpy(float))
    assert abs(rho) < 0.15


def test_anti_retest_scoreboard_reports_both_seasons_and_all():
    cohort = _build(plant_signal=True)
    board = anti_retest_scoreboard(cohort)
    assert set(board.season) == {2024, 2025, "ALL"}
    assert (board.n == board.season.map({2024: 444, 2025: 440, "ALL": 884})).all()
