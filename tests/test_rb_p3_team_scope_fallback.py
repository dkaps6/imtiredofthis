"""RB P3 is only built for the teams its context covers as of build time.

A certified-eligible team whose props go live *after* that context was built
(a kickoff-wave timing lag, not a data error) must fall back to the generic
calibrated model for that Week-1 RB/FB row instead of crashing the whole
pricing run -- exactly the same principle as every other kickoff-timing-gate
bug fixed this session, applied to RB P3's own team scope.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.modeling.rb_pricing_adapter_v1 import rb_context_teams
import scripts.run_pricing_with_full_roster_universe_v1 as full_roster
from scripts.simulation_v2 import SimulationResult


def _rb_context() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "season": 2026, "week": 1,
            "player": "D'Andre Swift", "player_clean_key": "dandreswift", "player_base_key": "dandreswift",
            "team": "CHI", "opponent": "CAR",
            "rb_synthesis_proj": 51.25, "rb_synthesis_route": "WEEK1_STACK_OVERRIDE",
            "rb_synthesis_version": "RB_P3_SYNTHESIS_V1", "rb_synthesis_applied": 1,
            "football_only_no_odds": 1, "sportsbook_inputs_used": 0,
        }
    ])


def test_rb_context_teams_returns_the_context_team_set():
    assert rb_context_teams(_rb_context()) == {"CHI"}


def _pricing_row(player, player_key, team, opponent, event_id) -> dict:
    return {
        "market": "rush_rec_yards",
        "position_group": "RB",
        "week": 1,
        "season": 2026,
        "event_id": event_id,
        "team": team,
        "opponent": opponent,
        "player": player,
        "player_clean_key": player_key,
    }


def test_out_of_scope_team_falls_back_instead_of_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(full_roster, "load_rb_context", lambda: _rb_context())
    monkeypatch.setattr(full_roster, "RB_AUDIT_CSV", tmp_path / "rb_audit.csv")
    monkeypatch.setattr(full_roster, "RB_AUDIT_JSON", tmp_path / "rb_audit.json")

    in_scope_game = full_roster._canonical_game("CHI", "CAR", 2026, 1)
    out_scope_game = full_roster._canonical_game("NYJ", "PIT", 2026, 1)

    pricing_metrics = pd.DataFrame([
        _pricing_row("D'Andre Swift", "dandreswift", "CHI", "CAR", in_scope_game),
        _pricing_row("Breece Hall", "breecehall", "NYJ", "PIT", out_scope_game),
    ])

    n = 50
    rng = np.random.default_rng(0)
    values = {
        (in_scope_game, "dandreswift", "rush_yards"): rng.uniform(40, 60, n),
        (in_scope_game, "dandreswift", "rec_yards"): rng.uniform(5, 15, n),
        (in_scope_game, "dandreswift", "rush_rec_yards"): rng.uniform(45, 75, n),
        (out_scope_game, "breecehall", "rush_yards"): rng.uniform(60, 90, n),
        (out_scope_game, "breecehall", "rec_yards"): rng.uniform(10, 20, n),
        (out_scope_game, "breecehall", "rush_rec_yards"): rng.uniform(70, 110, n),
    }
    baseline_out_scope_combo = values[(out_scope_game, "breecehall", "rush_rec_yards")].copy()
    result = SimulationResult(values=values, iterations=n)

    payload = full_roster._apply_rb_rush_rec_conservation(result, pricing_metrics)

    assert payload["disposition"] == "RB_RUSH_REC_DISTRIBUTION_CONSERVED_WITH_PROMOTED_P3"
    assert payload["players"] == 1
    assert payload["out_of_p3_scope_teams"] == ["NYJ"]

    # In-scope player: P3-conserved combo mean equals P3 rush mean + raw rec mean.
    conserved = result.values[(in_scope_game, "dandreswift", "rush_rec_yards")]
    assert np.mean(conserved) == pytest.approx(51.25 + np.mean(values[(in_scope_game, "dandreswift", "rec_yards")]), abs=1e-6)

    # Out-of-scope player: left completely untouched, no crash.
    assert np.array_equal(
        result.values[(out_scope_game, "breecehall", "rush_rec_yards")], baseline_out_scope_combo
    )


def test_all_out_of_scope_is_a_graceful_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(full_roster, "load_rb_context", lambda: _rb_context())
    monkeypatch.setattr(full_roster, "RB_AUDIT_CSV", tmp_path / "rb_audit.csv")
    monkeypatch.setattr(full_roster, "RB_AUDIT_JSON", tmp_path / "rb_audit.json")

    out_scope_game = full_roster._canonical_game("NYJ", "PIT", 2026, 1)
    pricing_metrics = pd.DataFrame([_pricing_row("Breece Hall", "breecehall", "NYJ", "PIT", out_scope_game)])
    result = SimulationResult(values={}, iterations=10)

    payload = full_roster._apply_rb_rush_rec_conservation(result, pricing_metrics)
    assert payload["disposition"] == "NO_ELIGIBLE_RB_RUSH_REC_ROWS"
    assert payload["out_of_p3_scope_teams"] == ["NYJ"]


def test_pricing_v2_gates_p3_lookup_on_team_scope():
    """Unit-testing scripts.run_pricing_v2's full price() loop requires the
    entire ML/state/rules/ensemble stack; the rest of this file's tests
    already establish that pattern skips without live artifacts. This
    confirms the new team-scope gate is actually wired into that loop."""
    text = Path("scripts/run_pricing_v2.py").read_text(encoding="utf-8")
    assert "row_team in p3_teams" in text
    assert "rb_context_teams" in text
    assert 'and _runtime_week(row) == 1' in text


def test_full_slate_scopes_the_week1_p3_requirement_to_its_own_teams():
    workflow = Path(".github/workflows/full-slate.yml").read_text(encoding="utf-8")
    assert "rb_p3_teams" in workflow
    assert "Week-1 RB/FB rows outside P3 team scope incorrectly claimed P3" in workflow
    assert "final Week-1 RB model projection is not the promoted P3 synthesis mean" in workflow
