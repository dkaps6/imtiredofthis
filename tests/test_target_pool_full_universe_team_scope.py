"""Target-pool validation must use the same certified current football universe.

The weekly schedule can remain broader, but the simulation/entitlement artifacts
for a run may not silently contain missing or extra teams relative to the
explicit current-role authority.
"""
from __future__ import annotations
import json
import pandas as pd
import pytest
import scripts.validate_team_target_pool_full_universe_v2 as validate_mod


def _write_universe(path, teams):
    pd.DataFrame([
        {"team": t, "player_clean_key": f"{t.lower()}_p1", "tgt_share": 0.2,
         "bayes_tgt_share": 0.2, "rules_tgt_share": 0.2}
        for t in teams
    ]).to_csv(path, index=False)


def _write_universe_audit(path):
    path.write_text(json.dumps({
        "disposition": "FOOTBALL_SIMULATION_UNIVERSE_CERTIFIED",
        "sportsbook_inputs_used_to_generate_football_distributions": False,
        "explicit_target_entitlement_materialized": False,
    }), encoding="utf-8")


def _write_roles(path, teams):
    pd.DataFrame({"team": list(teams)}).to_csv(path, index=False)


@pytest.fixture
def wired(tmp_path, monkeypatch):
    universe = tmp_path / "football_simulation_universe.csv"
    universe_audit = tmp_path / "football_simulation_universe_audit.json"
    monkeypatch.setattr(validate_mod, "UNIVERSE", universe)
    monkeypatch.setattr(validate_mod, "UNIVERSE_AUDIT", universe_audit)
    monkeypatch.setattr(validate_mod, "ENTITLEMENT_TRACE", tmp_path / "target_entitlement_v1_trace.csv")
    monkeypatch.setattr(validate_mod, "ENTITLEMENT_AUDIT", tmp_path / "target_entitlement_v1_audit.json")
    monkeypatch.setattr(validate_mod, "OUT_CSV", tmp_path / "team_target_pool_audit.csv")
    monkeypatch.setattr(validate_mod, "OUT_JSON", tmp_path / "team_target_pool_audit.json")
    roles = tmp_path / "roles_current_production_eligible_v1.csv"
    monkeypatch.setenv("ACTIVE_ROLES_CSV", str(roles))
    return {"universe": universe, "universe_audit": universe_audit, "roles": roles}


def test_universe_exactly_matching_certified_teams_passes(wired):
    teams = ["ARI", "ATL", "BAL", "BUF"]
    _write_universe(wired["universe"], teams)
    _write_universe_audit(wired["universe_audit"])
    _write_roles(wired["roles"], teams)
    assert validate_mod.main() == 0


def test_universe_superset_of_certified_teams_fails_closed(wired):
    _write_universe(wired["universe"], ["ARI", "ATL", "BAL", "BUF"])
    _write_universe_audit(wired["universe_audit"])
    _write_roles(wired["roles"], ["ARI", "ATL"])
    with pytest.raises(RuntimeError, match=r"extra=\['BAL', 'BUF'\]"):
        validate_mod.main()


def test_universe_missing_certified_teams_fails_closed(wired):
    _write_universe(wired["universe"], ["ARI", "ATL"])
    _write_universe_audit(wired["universe_audit"])
    _write_roles(wired["roles"], ["ARI", "ATL", "BAL", "BUF"])
    with pytest.raises(RuntimeError, match=r"missing=\['BAL', 'BUF'\]"):
        validate_mod.main()
