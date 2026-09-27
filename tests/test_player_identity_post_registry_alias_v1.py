import pandas as pd
import pytest

import scripts.run_player_form_v2_loader as loader
from scripts.utils.player_identity_v3 import resolve_slate_identities


def _registry():
    return pd.DataFrame([
        {
            "player_identity_key": "gsis:00-0036988",
            "player_id": "00-0036988",
            "player": "Josh Palmer",
            "team": "BUF",
            "position": "WR",
            "identity_full_name_key": "joshpalmer",
            "identity_base_name_key": "joshpalmer",
            "last_season": 2026,
            "last_week": 2,
        },
        {
            "player_identity_key": "gsis:00-0040879",
            "player_id": "00-0040879",
            "player": "Matthew Hibner",
            "team": "BAL",
            "position": "TE",
            "identity_full_name_key": "matthewhibner",
            "identity_base_name_key": "matthewhibner",
            "last_season": 2026,
            "last_week": 1,
        },
    ])


def test_post_registry_alias_overlay_preserves_verified_live_names(tmp_path, monkeypatch):
    persistent = tmp_path / "player_identity_aliases.csv"
    current = tmp_path / "player_identity_current_aliases.csv"
    pd.DataFrame([
        {
            "current_name": "Joshua Palmer",
            "historical_name": "Josh Palmer",
            "player_id": "00-0036988",
            "current_team": "BUF",
            "position": "WR",
            "reason": "verified",
            "verified_source": "https://example.com/josh",
            "verified_date": "2026-09-07",
        }
    ]).to_csv(persistent, index=False)
    pd.DataFrame([
        {
            "current_name": "Matt Hibner",
            "historical_name": "Matthew Hibner",
            "player_id": "00-0040879",
            "current_team": "BAL",
            "position": "TE",
            "reason": "verified",
            "verified_source": "https://example.com/matt",
            "verified_date": "2026-09-27",
        }
    ]).to_csv(current, index=False)

    monkeypatch.setattr(loader, "PERSISTENT_IDENTITY_ALIASES", persistent)
    monkeypatch.setattr(loader, "CURRENT_IDENTITY_ALIASES", current)

    overlaid = loader._apply_post_registry_verified_aliases(_registry())
    slate = pd.DataFrame([
        {"player": "Joshua Palmer", "team": "BUF"},
        {"player": "Matt Hibner", "team": "BAL"},
    ])
    resolved = resolve_slate_identities(slate, overlaid)

    assert resolved["player_identity_key"].tolist() == [
        "gsis:00-0036988",
        "gsis:00-0040879",
    ]
    assert resolved["identity_resolution"].tolist() == [
        "team_exact_name",
        "team_exact_name",
    ]
    assert resolved["player_id"].tolist() == ["00-0036988", "00-0040879"]


def test_post_registry_alias_overlay_fails_closed_without_stable_anchor(tmp_path, monkeypatch):
    current = tmp_path / "player_identity_current_aliases.csv"
    pd.DataFrame([
        {
            "current_name": "Matt Hibner",
            "historical_name": "Matthew Hibner",
            "player_id": "00-0040879",
            "current_team": "BAL",
            "position": "TE",
            "reason": "verified",
            "verified_source": "https://example.com/matt",
            "verified_date": "2026-09-27",
        }
    ]).to_csv(current, index=False)

    monkeypatch.setattr(loader, "PERSISTENT_IDENTITY_ALIASES", tmp_path / "missing.csv")
    monkeypatch.setattr(loader, "CURRENT_IDENTITY_ALIASES", current)

    with pytest.raises(RuntimeError, match="stable identity missing"):
        loader._apply_post_registry_verified_aliases(
            _registry().loc[lambda x: ~x["player_id"].eq("00-0040879")].copy()
        )
