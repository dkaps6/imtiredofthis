from pathlib import Path

import pytest

import scripts.run_player_form_current_roles_v1 as current_roles_entry
import scripts.run_player_form_v2_loader as loader


def test_main_honors_explicit_certified_current_roles(tmp_path, monkeypatch):
    """Full Slate must keep PlayerForm on the same certified current-role
    universe used by opportunity/pricing stages. Sportsbook posting coverage is
    still downstream; this env var is football availability/timing authority."""
    explicit = tmp_path / "roles_current_production_eligible_v1.csv"
    explicit.write_text("team,player,role,position,player_clean_key\nKC,Player One,QB1,QB,playerone\n")
    monkeypatch.setenv("ACTIVE_ROLES_CSV", str(explicit))
    monkeypatch.setattr(loader, "main", lambda: 0)
    monkeypatch.setattr(current_roles_entry, "publish_strict_prior_history", lambda: None)

    rc = current_roles_entry.main()

    assert rc == 0
    assert loader.runner.pf.ROLES == explicit


def test_main_uses_reconciled_active_fallback_when_no_explicit_env(tmp_path, monkeypatch):
    fallback = tmp_path / "roles_ourlads_active_v1.csv"
    fallback.write_text("team,player,role,position,player_clean_key\nKC,Player One,QB1,QB,playerone\n")
    monkeypatch.delenv("ACTIVE_ROLES_CSV", raising=False)
    monkeypatch.setattr(current_roles_entry, "DEFAULT_ACTIVE_ROLES", fallback)
    monkeypatch.setattr(current_roles_entry, "resolve_current_roles_path", lambda require_active=True: fallback)
    monkeypatch.setattr(loader, "main", lambda: 0)
    monkeypatch.setattr(current_roles_entry, "publish_strict_prior_history", lambda: None)

    rc = current_roles_entry.main()

    assert rc == 0
    assert loader.runner.pf.ROLES == fallback


def test_main_fails_closed_when_explicit_roles_missing(tmp_path, monkeypatch):
    missing = tmp_path / "does_not_exist.csv"
    monkeypatch.setenv("ACTIVE_ROLES_CSV", str(missing))
    monkeypatch.setattr(loader, "main", lambda: 0)
    monkeypatch.setattr(current_roles_entry, "publish_strict_prior_history", lambda: None)

    with pytest.raises(RuntimeError, match="active reconciled roles missing/empty"):
        current_roles_entry.main()
