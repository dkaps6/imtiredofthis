import pandas as pd
import pytest

import scripts.repair_live_prop_identity_v1 as repair_mod
from scripts.repair_live_prop_identity_v1 import _build_roster_index


def _roles_for(teams: set[str]) -> pd.DataFrame:
    ordered = sorted(teams)
    return pd.DataFrame(
        {
            "team": ordered,
            "player": [f"Player {team}" for team in ordered],
        }
    )


VALID_OVERRIDE_ROW = {
    "player": "Najee Harris",
    "team": "NYG",
    "reason": "Signed by the Giants on a one-year deal in August 2026",
    "verified_source": "https://www.giants.com/news/najee-harris-signed",
    "verified_date": "2026-09-13",
}


def test_verified_override_rescues_player_missing_from_scraped_roster(tmp_path, monkeypatch):
    overrides_path = tmp_path / "manual_roster_overrides.csv"
    pd.DataFrame([VALID_OVERRIDE_ROW]).to_csv(overrides_path, index=False)
    monkeypatch.setattr(repair_mod, "MANUAL_ROSTER_OVERRIDES", overrides_path)

    roster = _build_roster_index(
        _roles_for({"NYG", "DAL"}), required_teams={"NYG", "DAL"}
    )
    assert "najeeharris" in roster["NYG"]


def test_override_for_team_outside_required_scope_is_not_leaked_in(tmp_path, monkeypatch):
    overrides_path = tmp_path / "manual_roster_overrides.csv"
    pd.DataFrame([VALID_OVERRIDE_ROW]).to_csv(overrides_path, index=False)
    monkeypatch.setattr(repair_mod, "MANUAL_ROSTER_OVERRIDES", overrides_path)

    roster = _build_roster_index(_roles_for({"ARI", "ATL"}), required_teams={"ARI", "ATL"})
    assert set(roster) == {"ARI", "ATL"}


def test_override_missing_verified_source_fails_closed(tmp_path, monkeypatch):
    overrides_path = tmp_path / "manual_roster_overrides.csv"
    bad_row = dict(VALID_OVERRIDE_ROW, verified_source="")
    pd.DataFrame([bad_row]).to_csv(overrides_path, index=False)
    monkeypatch.setattr(repair_mod, "MANUAL_ROSTER_OVERRIDES", overrides_path)

    with pytest.raises(RuntimeError, match="missing verified_source"):
        _build_roster_index(_roles_for({"NYG", "DAL"}), required_teams={"NYG", "DAL"})


def test_override_invalid_team_fails_closed(tmp_path, monkeypatch):
    overrides_path = tmp_path / "manual_roster_overrides.csv"
    bad_row = dict(VALID_OVERRIDE_ROW, team="ZZZ")
    pd.DataFrame([bad_row]).to_csv(overrides_path, index=False)
    monkeypatch.setattr(repair_mod, "MANUAL_ROSTER_OVERRIDES", overrides_path)

    with pytest.raises(RuntimeError, match="invalid team"):
        _build_roster_index(_roles_for({"NYG", "DAL"}), required_teams={"NYG", "DAL"})


def test_absent_overrides_file_is_a_silent_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(repair_mod, "MANUAL_ROSTER_OVERRIDES", tmp_path / "does_not_exist.csv")
    roster = _build_roster_index(_roles_for({"ARI", "ATL"}), required_teams={"ARI", "ATL"})
    assert set(roster) == {"ARI", "ATL"}


def test_tracked_manual_roster_overrides_file_is_well_formed():
    """The real tracked data/manual_roster_overrides.csv must itself pass
    _load_manual_roster_overrides so a bad edit fails CI, not a live run."""
    overrides = repair_mod._load_manual_roster_overrides()
    assert "NYG" in overrides
    assert "najeeharris" in overrides["NYG"]
