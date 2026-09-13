import pytest

import scripts.run_player_form_current_roles_v1 as current_roles_entry
import scripts.run_player_form_v2_loader as loader


def test_main_wires_playerform_roles_to_reconciled_active_roles_regardless_of_env(tmp_path, monkeypatch):
    """PlayerForm's slate universe is contracted to be independent of
    sportsbook/kickoff-timing gating. full-slate.yml sets ACTIVE_ROLES_CSV
    job-wide to a timing-gated file for OTHER stages; this entry point must
    not let that env var redirect PlayerForm's own roles source."""
    fake_active_roles = tmp_path / "roles_ourlads_active_v1.csv"
    fake_active_roles.write_text("team,player\nKC,Player One\n")
    monkeypatch.setattr(current_roles_entry, "DEFAULT_ACTIVE_ROLES", fake_active_roles)
    monkeypatch.setenv("ACTIVE_ROLES_CSV", "data/roles_current_production_eligible_v1.csv")
    monkeypatch.setattr(loader, "main", lambda: 0)
    monkeypatch.setattr(current_roles_entry, "publish_strict_prior_history", lambda: None)

    rc = current_roles_entry.main()

    assert rc == 0
    assert loader.runner.pf.ROLES == fake_active_roles


def test_main_fails_closed_when_reconciled_active_roles_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        current_roles_entry, "DEFAULT_ACTIVE_ROLES", tmp_path / "does_not_exist.csv"
    )
    monkeypatch.setattr(loader, "main", lambda: 0)
    monkeypatch.setattr(current_roles_entry, "publish_strict_prior_history", lambda: None)

    with pytest.raises(RuntimeError, match="reconciled active roles missing/empty"):
        current_roles_entry.main()
