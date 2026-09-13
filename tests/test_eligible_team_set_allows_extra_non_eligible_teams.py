import pandas as pd
import pytest

from scripts.utils.eligible_team_set_v1 import validate_current_team_set

ELIGIBLE_12 = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB"]


def _roles_csv(tmp_path, teams):
    path = tmp_path / "roles_current_production_eligible_v1.csv"
    pd.DataFrame({"team": teams, "player": [f"Player {t}" for t in teams]}).to_csv(path, index=False)
    return path


def test_exact_match_still_passes(tmp_path):
    path = _roles_csv(tmp_path, ELIGIBLE_12)
    result = validate_current_team_set(ELIGIBLE_12, active_roles_path=path, label="test universe")
    assert result["mode"] == "EXPLICIT_CURRENT_AVAILABILITY"
    assert result["extra_non_eligible_teams"] == []


def test_observed_superset_of_eligible_now_passes(tmp_path):
    """This is the real production incident (2026-09-13): PlayerForm's
    football simulation universe correctly covers all 32 teams (its own
    documented contract is sportsbook/kickoff-timing independence), while
    only 12 teams are currently certified-eligible mid-kickoff-wave. That
    must not be fatal -- pricing narrows to eligible teams later, separately.
    """
    path = _roles_csv(tmp_path, ELIGIBLE_12)
    all_32 = ELIGIBLE_12 + ["HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"]
    result = validate_current_team_set(all_32, active_roles_path=path, label="football simulation universe")
    assert result["mode"] == "EXPLICIT_CURRENT_AVAILABILITY"
    assert set(result["extra_non_eligible_teams"]) == set(all_32) - set(ELIGIBLE_12)


def test_missing_eligible_team_still_fails_closed(tmp_path):
    path = _roles_csv(tmp_path, ELIGIBLE_12)
    observed = [t for t in ELIGIBLE_12 if t != "GB"]
    with pytest.raises(RuntimeError, match=r"missing certified eligible teams: \['GB'\]"):
        validate_current_team_set(observed, active_roles_path=path, label="test universe")


def test_legacy_mode_unaffected_when_no_active_roles_path(monkeypatch):
    monkeypatch.delenv("ACTIVE_ROLES_CSV", raising=False)
    all_32_generic = [f"T{i:02d}" for i in range(32)]
    result = validate_current_team_set(all_32_generic, label="test universe")
    assert result["mode"] == "LEGACY_32_TEAM"


def test_legacy_mode_still_fails_on_wrong_count(monkeypatch):
    monkeypatch.delenv("ACTIVE_ROLES_CSV", raising=False)
    with pytest.raises(RuntimeError, match="legacy coverage expected 32 teams"):
        validate_current_team_set(["ARI", "ATL"], label="test universe")
