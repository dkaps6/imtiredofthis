from __future__ import annotations

import pandas as pd
import pytest

from scripts.backtest.build_qb_synthesis_confirmation_inputs_v1 import _target_season_team_history
from scripts.backtest.run_m89_pregame_synthesis import load_player_logs, load_team_history, prior_rows


def test_qb_trace_builder_scopes_combined_team_history_to_target_season():
    combined = pd.DataFrame(
        [
            {"season": 2022, "week": 18, "team": "KC", "proe": 0.10},
            {"season": 2023, "week": 1, "team": "KC", "proe": 0.20},
            {"season": 2023, "week": 2, "team": "KC", "proe": 0.30},
            {"season": 2024, "week": 1, "team": "KC", "proe": 0.40},
        ]
    )
    scoped = _target_season_team_history(combined, 2023)
    assert set(scoped["season"].astype(int)) == {2023}
    assert scoped["week"].astype(int).tolist() == [1, 2]


def test_qb_trace_builder_rejects_missing_target_season():
    combined = pd.DataFrame([{"season": 2022, "week": 18, "team": "KC"}])
    with pytest.raises(RuntimeError, match="no rows for target season 2023"):
        _target_season_team_history(combined, 2023)


def test_m89_fit_loaders_fail_closed_to_named_season(tmp_path):
    team_path = tmp_path / "team.csv"
    log_path = tmp_path / "logs.csv"
    pd.DataFrame(
        [
            {"season": 2022, "week": 18, "team": "KC", "proe": 9.0},
            {"season": 2023, "week": 1, "team": "KC", "proe": 1.0},
            {"season": 2023, "week": 2, "team": "KC", "proe": 2.0},
        ]
    ).to_csv(team_path, index=False)
    pd.DataFrame(
        [
            {"season": 2022, "week": 18, "team": "KC", "player_clean_key": "Patrick Mahomes", "pass_att": 40},
            {"season": 2023, "week": 1, "team": "KC", "player_clean_key": "Patrick Mahomes", "pass_att": 30},
        ]
    ).to_csv(log_path, index=False)

    team = load_team_history(team_path, 2023)
    logs = load_player_logs(log_path, 2023)
    assert set(team["season"].astype(int)) == {2023}
    assert set(logs["season"].astype(int)) == {2023}
    assert logs["player_clean_key"].tolist() == ["patrickmahomes"]

    # The 2022 row cannot leak into a 2023 history window after season scoping.
    prior = prior_rows(team, 2023, 2, "KC")
    assert prior["week"].astype(int).tolist() == [1]
    assert prior["proe"].tolist() == [1.0]


def test_m89_fit_loaders_reject_missing_named_season(tmp_path):
    team_path = tmp_path / "team.csv"
    log_path = tmp_path / "logs.csv"
    pd.DataFrame([{"season": 2022, "week": 18, "team": "KC"}]).to_csv(team_path, index=False)
    pd.DataFrame([{"season": 2022, "week": 18, "team": "KC", "player_clean_key": "mahomes"}]).to_csv(log_path, index=False)

    with pytest.raises(RuntimeError, match="required season 2023"):
        load_team_history(team_path, 2023)
    with pytest.raises(RuntimeError, match="required season 2023"):
        load_player_logs(log_path, 2023)
