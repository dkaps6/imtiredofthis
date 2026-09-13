"""The target-pool football universe must exactly match the certified current
football teams when explicit availability/timing authority is present.

The weekly schedule may contain more games, but already-started/withheld teams
must not survive into current PlayerForm/simulation/target artifacts.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

import scripts.validate_team_target_pool_full_universe_v2 as validate_mod


def _write_universe(path, teams):
    rows = []
    for team in teams:
        rows.append({
            "team": team, "player_clean_key": f"{team.lower()}_p1",
            "tgt_share": 0.2, "bayes_tgt_share": 0.2, "rules_tgt_share": 0.2,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


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
    entitlement_trace = tmp_path / "target_entitlement_v1_trace.csv"
    entitlement_audit = tmp_path / "target_entitlement_v1_audit.json"
    out_csv = tmp_path / "team_target_pool_audit.csv"
    out_json = tmp_path / "team_target_pool_audit.json"
    monkeypatch.setattr(validate_mod, "UNIVERSE", universe)
    monkeypatch.setattr(validate_mod, "UNIVERSE_AUDIT", universe_audit)
    monkeypatch.setattr(validate_mod, "ENTITLEMENT_TRACE", entitlement_trace)
    monkeypatch.setattr(validate_mod, "ENTITLEMENT_AUDIT", entitlement_audit)
    monkeypatch.setattr(validate_mod, "OUT_CSV", out_csv)
    monkeypatch.setattr(validate_mod, "OUT_JSON", out_json)
    roles = tmp_path / "roles_current_production_eligible_v1.csv"
    monkeypatch.setenv("ACTIVE_ROLES_CSV", str(roles))
    return {"universe": universe, "universe_audit": universe_audit, "roles": roles}


def test_universe_exactly_matching_certified_current_teams_passes(wired):
    teams = ["ARI", "ATL", "BAL", "BUF"]
    _write_universe(wired["universe"], teams)
    _write_universe_audit(wired["universe_audit"])
    _write_roles(wired["roles"], teams)
    assert validate_mod.main() == 0


def test_universe_superset_of_certified_current_teams_fails(wired):
    _write_universe(wired["universe"], ["ARI", "ATL", "BAL", "BUF"])
    _write_universe_audit(wired["universe_audit"])
    _write_roles(wired["roles"], ["ARI", "ATL"])
    with pytest.raises(RuntimeError, match=r"missing=\[\] extra=\['BAL', 'BUF'\]"):
        validate_mod.main()


def test_universe_missing_certified_current_team_fails(wired):
    _write_universe(wired["universe"], ["ARI", "ATL"])
    _write_universe_audit(wired["universe_audit"])
    _write_roles(wired["roles"], ["ARI", "ATL", "BAL", "BUF"])
    with pytest.raises(RuntimeError, match=r"missing=\['BAL', 'BUF'\] extra=\[\]"):
        validate_mod.main()
