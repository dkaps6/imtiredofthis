import pandas as pd
import pytest

from scripts.utils.eligible_team_set_v1 import validate_current_team_set

ELIGIBLE_12 = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB"]


def _roles_csv(tmp_path, teams):
    path = tmp_path / "roles_current_production_eligible_v1.csv"
    pd.DataFrame({"team": teams, "player": [f"Player {t}" for t in teams]}).to_csv(path, index=False)
    return path


def test_exact_match_passes(tmp_path):
    path = _roles_csv(tmp_path, ELIGIBLE_12)
    result = validate_current_team_set(ELIGIBLE_12, active_roles_path=path, label="test universe")
    assert result["mode"] == "EXPLICIT_CURRENT_AVAILABILITY"
    assert result["expected_teams"] == 12
    assert result["observed_teams"] == 12
    assert result["canonical_games"] == 6


def test_observed_superset_of_certified_fails_closed(tmp_path):
    """Do not expand a shrinking current Sunday slate back to the full league.

    The weekly schedule can remain 32-team authoritative, but football artifacts
    guarded by this helper must stay on the certified current team universe.
    Sportsbook posting coverage remains a separate downstream concern.
    """
    path = _roles_csv(tmp_path, ELIGIBLE_12)
    all_32 = ELIGIBLE_12 + ["HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"]
    with pytest.raises(RuntimeError, match=r"!= certified eligible teams; missing=\[\] extra="):
        validate_current_team_set(all_32, active_roles_path=path, label="football simulation universe")


def test_missing_certified_team_fails_closed(tmp_path):
    path = _roles_csv(tmp_path, ELIGIBLE_12)
    observed = [t for t in ELIGIBLE_12 if t != "GB"]
    with pytest.raises(RuntimeError, match=r"missing=\['GB'\] extra=\[\]"):
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
