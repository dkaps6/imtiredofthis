"""QB C2 football coverage must match the certified current football scope.

The pricing lineage stamp is the authoritative current-scope gate. Sportsbook
pass-yard coverage may be a smaller downstream subset, but the upstream C2
football audit must contain exactly one starter for every certified current team.
"""
from pathlib import Path

import pandas as pd
import pytest

from scripts.stamp_qb_c2_pricing_lineage_v1 import _validate_current_c2_scope


def _roles(path: Path, teams: list[str]) -> Path:
    pd.DataFrame({"team": teams, "player": [f"{t} Player" for t in teams]}).to_csv(path, index=False)
    return path


def _c2(teams: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "team": teams,
        "player": [f"{t} QB" for t in teams],
        "player_clean_key": [f"{t.lower()}qb" for t in teams],
    })


def test_accepts_exact_current_football_scope(tmp_path):
    teams = ["DAL", "DEN", "KC", "NYG"]
    roles = _roles(tmp_path / "roles.csv", teams)
    status = {"football_qb_rows": 4}

    scope = _validate_current_c2_scope(status, _c2(teams), active_roles_path=roles)

    assert scope["mode"] == "EXPLICIT_CURRENT_AVAILABILITY"
    assert scope["expected_teams"] == 4
    assert scope["observed_teams"] == 4


def test_rejects_football_qb_superset_of_current_scope(tmp_path):
    roles = _roles(tmp_path / "roles.csv", ["DAL", "DEN"])
    with pytest.raises(RuntimeError, match=r"extra=\['KC', 'NYG'\]"):
        _validate_current_c2_scope(
            {"football_qb_rows": 4},
            _c2(["DAL", "DEN", "KC", "NYG"]),
            active_roles_path=roles,
        )


def test_rejects_missing_current_qb_team(tmp_path):
    roles = _roles(tmp_path / "roles.csv", ["DAL", "DEN", "KC", "NYG"])
    with pytest.raises(RuntimeError, match=r"missing=\['NYG'\]"):
        _validate_current_c2_scope(
            {"football_qb_rows": 3},
            _c2(["DAL", "DEN", "KC"]),
            active_roles_path=roles,
        )


def test_rejects_status_row_count_drift(tmp_path):
    teams = ["DAL", "DEN", "KC", "NYG"]
    roles = _roles(tmp_path / "roles.csv", teams)
    with pytest.raises(RuntimeError, match="production audit row count"):
        _validate_current_c2_scope(
            {"football_qb_rows": 32},
            _c2(teams),
            active_roles_path=roles,
        )
