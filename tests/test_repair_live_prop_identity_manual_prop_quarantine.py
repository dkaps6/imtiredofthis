import pandas as pd
import pytest

import scripts.repair_live_prop_identity_v1 as repair_mod
from scripts.repair_live_prop_identity_v1 import _quarantine_frame


VALID_QUARANTINE_ROW = {
    "player": "Najee Harris",
    "team": "NYG",
    "reason": "Signed by the Giants in August 2026; not yet in the football-only roster used to build PlayerForm.",
    "verified_source": "https://www.giants.com/news/najee-harris-signed",
    "verified_date": "2026-09-13",
}


def _props(players: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player": players,
            "team_abbr": [""] * len(players),
            "market": ["player_rush_yds"] * len(players),
        }
    )


def test_quarantine_frame_removes_only_matching_player():
    df = _props(["Najee Harris", "Tre Harris", "Marvin Harrison Jr."])
    out, removed = _quarantine_frame(df, {"najeeharris"})
    assert removed == 1
    assert sorted(out["player"]) == ["Marvin Harrison Jr.", "Tre Harris"]


def test_quarantine_frame_is_a_noop_without_keys():
    df = _props(["Najee Harris"])
    out, removed = _quarantine_frame(df, set())
    assert removed == 0
    assert len(out) == 1


def test_load_prop_quarantine_keys_from_valid_file(tmp_path, monkeypatch):
    path = tmp_path / "manual_prop_quarantine.csv"
    pd.DataFrame([VALID_QUARANTINE_ROW]).to_csv(path, index=False)
    monkeypatch.setattr(repair_mod, "MANUAL_PROP_QUARANTINE", path)

    keys = repair_mod._load_prop_quarantine_keys()
    assert "najeeharris" in keys


def test_load_prop_quarantine_keys_missing_verified_source_fails_closed(tmp_path, monkeypatch):
    path = tmp_path / "manual_prop_quarantine.csv"
    bad_row = dict(VALID_QUARANTINE_ROW, verified_source="")
    pd.DataFrame([bad_row]).to_csv(path, index=False)
    monkeypatch.setattr(repair_mod, "MANUAL_PROP_QUARANTINE", path)

    with pytest.raises(RuntimeError, match="missing verified_source"):
        repair_mod._load_prop_quarantine_keys()


def test_load_prop_quarantine_keys_absent_file_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(repair_mod, "MANUAL_PROP_QUARANTINE", tmp_path / "does_not_exist.csv")
    assert repair_mod._load_prop_quarantine_keys() == set()


def test_tracked_manual_prop_quarantine_file_is_well_formed():
    """The real tracked data/manual_prop_quarantine.csv must itself load
    cleanly so a bad edit fails CI, not a live run."""
    keys = repair_mod._load_prop_quarantine_keys()
    assert "najeeharris" in keys
