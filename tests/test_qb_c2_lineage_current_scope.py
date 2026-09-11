from pathlib import Path

import pandas as pd
import pytest

from scripts.stamp_qb_c2_pricing_lineage_v1 import _validate_current_c2_scope


ALL_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]
PLAYED = {"LAR", "NE", "SEA", "SF"}
LIVE = [team for team in ALL_TEAMS if team not in PLAYED]


def _c2(teams):
    return pd.DataFrame(
        [{"team": team, "player": f"{team} QB", "player_clean_key": f"{team.lower()}qb"} for team in teams]
    )


def _active_roles(tmp_path: Path, teams) -> Path:
    path = tmp_path / "active_roles.csv"
    pd.DataFrame(
        [{"team": team, "player": f"{team} QB", "position": "QB"} for team in teams]
    ).to_csv(path, index=False)
    return path


def test_c2_scope_accepts_explicit_28_team_current_output(tmp_path):
    path = _active_roles(tmp_path, LIVE)
    scope = _validate_current_c2_scope(
        {"football_qb_rows": len(LIVE)},
        _c2(LIVE),
        active_roles_path=path,
    )
    assert scope["mode"] == "EXPLICIT_CURRENT_AVAILABILITY"
    assert scope["observed_teams"] == 28


def test_c2_scope_rejects_missing_explicit_eligible_team(tmp_path):
    path = _active_roles(tmp_path, LIVE)
    with pytest.raises(RuntimeError, match="certified eligible teams"):
        _validate_current_c2_scope(
            {"football_qb_rows": len(LIVE) - 1},
            _c2(LIVE[:-1]),
            active_roles_path=path,
        )


def test_c2_scope_keeps_legacy_32_team_contract():
    scope = _validate_current_c2_scope(
        {"football_qb_rows": 32},
        _c2(ALL_TEAMS),
        active_roles_path=None,
    )
    assert scope["mode"] == "LEGACY_32_TEAM"
    assert scope["observed_teams"] == 32
