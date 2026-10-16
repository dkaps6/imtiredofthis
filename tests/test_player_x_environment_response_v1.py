from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research.audit_player_x_environment_response_v1 import (
    MIN_PRIOR,
    _history_signal,
    build_panel,
)


def _row(season, week, player, position, market, actual, proj, pass_mult=1.0, rush_mult=1.0):
    return {
        "season": season,
        "week": week,
        "game_id": f"{season}_{week:02d}_A_B",
        "team": "A",
        "opponent": "B",
        "player_clean_key": player,
        "position": position,
        "market": market,
        "actual": actual,
        "ensemble_proj": proj,
        "rules_pass_eff_mult": pass_mult,
        "rules_rush_eff_mult": rush_mult,
    }


def test_history_signal_positive_response():
    h = pd.DataFrame({
        "environment_x": np.log([0.90, 0.94, 0.98, 1.02, 1.06, 1.10]),
        "target_residual": [-12.0, -8.0, -3.0, 2.0, 8.0, 13.0],
    })
    rr, rel, signal = _history_signal(h, np.log(1.12))
    assert rr > 0.99
    assert rel > 0
    assert signal > 0


def test_build_panel_uses_strictly_prior_same_season_history_only():
    rows = []
    # Six 2024 prior WR games with monotone environment/residual response.
    for week, mult, resid in zip(
        range(1, 7),
        [0.90, 0.94, 0.98, 1.02, 1.06, 1.10],
        [-12.0, -8.0, -3.0, 2.0, 8.0, 13.0],
    ):
        rows.append(_row(2024, week, "p1", "WR", "rec_yards", 50.0 + resid, 50.0, pass_mult=mult))
    rows.append(_row(2024, 7, "p1", "WR", "rec_yards", 70.0, 50.0, pass_mult=1.12))

    # A prior-season row must not be mixed into 2025 target history.
    rows.append(_row(2023, 18, "p2", "WR", "rec_yards", 100.0, 1.0, pass_mult=1.20))
    for week, mult, resid in zip(
        range(1, 7),
        [0.92, 0.96, 0.99, 1.01, 1.05, 1.08],
        [-9.0, -6.0, -2.0, 1.0, 5.0, 9.0],
    ):
        rows.append(_row(2025, week, "p2", "WR", "rec_yards", 40.0 + resid, 40.0, pass_mult=mult))
    rows.append(_row(2025, 7, "p2", "WR", "rec_yards", 55.0, 40.0, pass_mult=1.10))

    panel, integrity = build_panel(pd.DataFrame(rows))
    q = panel.loc[panel["cell"].eq("WR_REC_YARDS")].sort_values(["season", "week"])
    assert len(q) == 2
    assert q["prior_games"].tolist() == [MIN_PRIOR, MIN_PRIOR]
    assert q["prior_feature_max_week"].tolist() == [6, 6]
    assert (q["prior_feature_max_week"] < q["week"]).all()
    assert integrity["chronology_violations"] == 0
    assert (q["response_signal"] > 0).all()


def test_invalid_environment_multiplier_is_not_used():
    rows = []
    for week in range(1, 7):
        rows.append(_row(2024, week, "p1", "RB", "rush_yards", 50 + week, 50, rush_mult=0.9 + .03 * week))
    rows.append(_row(2024, 7, "p1", "RB", "rush_yards", 60, 50, rush_mult=0.0))
    panel, integrity = build_panel(pd.DataFrame(rows))
    assert panel.empty
    assert integrity["source_rows"] == 7
    assert integrity["valid_environment_rows"] == 6
