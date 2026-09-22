from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.research import build_rb_pd2_completed_2026_history_v2 as v2
from scripts.operations.grade_market_track_record_gsis_v1 import _suffix_strip
from scripts.research.rb_pd2_forward_shadow_v1 import (
    RUSH_YARDS_FIT_SCOPE,
    RUSH_YARDS_MC_WEIGHT,
    RUSH_YARDS_ML_WEIGHT,
    RUSH_YARDS_PROMOTION_LINEAGE,
    RUSH_YARDS_STATE_WEIGHT,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_source_provenance_requires_artifact_before_first_kickoff():
    schedule = pd.DataFrame([
        {
            "season": 2026,
            "week": 2,
            "home": "BUF",
            "away": "DET",
            "kickoff_utc": "2026-09-17T20:15:00-04:00",
        },
        {
            "season": 2026,
            "week": 2,
            "home": "LAR",
            "away": "NYG",
            "kickoff_utc": "2026-09-21T20:15:00-04:00",
        },
    ])
    run = {
        "id": 123,
        "head_sha": "a" * 40,
        "run_started_at": "2026-09-17T22:26:21Z",
    }
    artifact = {
        "id": 456,
        "name": "run_123",
        "digest": "sha256:" + "b" * 64,
        "created_at": "2026-09-17T22:32:06Z",
        "expired": False,
        "workflow_run": {"id": 123},
    }
    out = v2.validate_source_provenance(
        source_run_metadata=run,
        source_artifact_metadata=artifact,
        completed_week=2,
        schedule=schedule,
    )
    assert out["source_run_id"] == "123"

    late = dict(artifact)
    late["created_at"] = "2026-09-18T01:00:00Z"
    with pytest.raises(RuntimeError, match="not frozen pregame"):
        v2.validate_source_provenance(
            source_run_metadata=run,
            source_artifact_metadata=late,
            completed_week=2,
            schedule=schedule,
        )


def test_full_roster_projection_reconstruction_uses_football_sources_and_exact_priced_parity(
    tmp_path: Path, monkeypatch
):
    source = tmp_path / "source"
    (source / "data").mkdir(parents=True)
    (source / "outputs").mkdir(parents=True)
    game = "2026_02_A_B"

    universe = pd.DataFrame([
        {
            "season": 2026, "week": 2, "event_id": game,
            "player": "Alpha Back III", "player_clean_key": "alphaback",
            "team": "ARI", "opponent": "SEA", "position": "RB",
        },
        {
            "season": 2026, "week": 2, "event_id": game,
            "player": "Beta Back", "player_clean_key": "betaback",
            "team": "SEA", "opponent": "ARI", "position": "FB",
        },
    ])
    universe.to_csv(source / "data/football_simulation_universe.csv", index=False)
    _write_json(source / "data/football_simulation_universe_audit.json", {
        "football_players": 2,
        "football_teams": 2,
        "canonical_games": 1,
        "sportsbook_rows_used_to_define_player_universe": 0,
        "provider_event_ids_used_during_simulation": False,
        "sportsbook_line_odds_book_fields_present": [],
    })

    trace = universe[["event_id", "team", "player_clean_key"]].copy()
    trace["entitlement_tgt_share"] = [0.1, 0.2]
    trace["entitlement_version"] = v2.FINAL_ENTITLEMENT_VERSION
    trace.to_csv(source / "data/target_entitlement_v1_trace.csv", index=False)
    _write_json(source / "data/target_entitlement_v1_audit.json", {
        "sportsbook_inputs_used": False,
    })

    # Diagnostics retain their own football-only identity. Alpha preserves III,
    # while Beta has no suffix.
    ml = universe[["season", "week", "team", "player", "player_clean_key"]].copy()
    ml.loc[ml["player"].eq("Alpha Back III"), "player_clean_key"] = "alphabackiii"
    ml["ml_rush_yards"] = [30.0, 50.0]
    ml.to_csv(source / "data/model_ml_diagnostics.csv", index=False)
    state = universe[["season", "week", "team", "player", "player_clean_key"]].copy()
    state.loc[state["player"].eq("Alpha Back III"), "player_clean_key"] = "alphabackiii"
    state["state_rush_yards"] = [999.0, 999.0]
    state.to_csv(source / "data/model_state_diagnostics.csv", index=False)

    weights = pd.DataFrame([{
        "market": "rush_yards",
        "mc_weight": RUSH_YARDS_MC_WEIGHT,
        "ml_weight": RUSH_YARDS_ML_WEIGHT,
        "state_weight": RUSH_YARDS_STATE_WEIGHT,
        "calibration_rows": 3207,
        "method": "nonnegative_oos_linear_blend_v2",
        "fit_scope": RUSH_YARDS_FIT_SCOPE,
        "promotion_lineage": RUSH_YARDS_PROMOTION_LINEAGE,
    }])
    weights.to_csv(source / "data/model_ensemble_weights.csv", index=False)

    expected = {
        "alphaback": RUSH_YARDS_MC_WEIGHT * 15.0 + RUSH_YARDS_ML_WEIGHT * 30.0,
        "betaback": RUSH_YARDS_MC_WEIGHT * 40.0 + RUSH_YARDS_ML_WEIGHT * 50.0,
    }

    # Beta mimics the Week-2 downstream suffix mismatch: priced identity adds
    # Jr., so its preserved priced path has no ML/State and falls back to MC.
    priced = pd.DataFrame([
        {
            "source_market": "player_rush_yds", "player": "Alpha Back III",
            "player_clean_key": "alphabackiii", "team": "ARI",
            "simulation_iterations": 2, "mc_proj": 15.0,
            "ml_proj": 30.0, "state_proj": 999.0,
            "ensemble_proj": expected["alphaback"], "model_proj": expected["alphaback"],
            "rb_synthesis_applied": 0,
        },
        {
            "source_market": "player_rush_yds", "player": "Beta Back Jr.",
            "player_clean_key": "betabackjr", "team": "SEA",
            "simulation_iterations": 2, "mc_proj": 40.0,
            "ml_proj": np.nan, "state_proj": np.nan,
            "ensemble_proj": 40.0, "model_proj": 40.0,
            "rb_synthesis_applied": 0,
        },
    ])
    priced.to_csv(source / "outputs/props_priced_clean.csv", index=False)

    class FakeResult:
        values = {
            (game, "alphaback", "rush_yards"): np.array([10.0, 20.0]),
            (game, "betaback", "rush_yards"): np.array([30.0, 50.0]),
        }

    monkeypatch.setattr(v2, "explicit_simulate", lambda *args, **kwargs: FakeResult())

    projection, audit = v2.build_projection_frame(
        source_root=source,
        completed_week=2,
        provenance={
            "source_run_id": "123",
            "source_artifact_id": "456",
            "source_git_sha": "a" * 40,
        },
    )
    got = projection.set_index("player_clean_key")["projection_mean"].to_dict()
    assert got["alphaback"] == pytest.approx(expected["alphaback"])
    # Football-only history keeps Beta's ML contribution even though the priced
    # parity arm had a downstream identity miss and fell back to MC.
    assert got["betaback"] == pytest.approx(expected["betaback"])
    assert audit["sportsbook_inputs_used_for_projection"] == 0
    assert audit["priced_parity_rows"] == 2
    assert audit["priced_component_identity_gap_rows"] == 1
    assert audit["max_abs_priced_mc_parity_gap"] <= 1e-12
    assert audit["max_abs_priced_model_parity_gap"] <= 1e-12


def _week1_row() -> dict:
    return {
        "season": 2026,
        "week": 1,
        "team": "CHI",
        "player_clean_key": "alpha",
        "position": "RB",
        "projection_mean": 50.0,
        "actual_rush_yards": 55.0,
        "pregame_lineage_certified": True,
        "projection_lineage": "2026_W1_P3_STACK1",
        "week1_p3_stack1_parity_pass": True,
        "week1_p3_projection": 50.0,
        "week1_stack1_projection": 50.0,
    }


def test_gsis_suffix_strip_uses_longest_suffix_token_first():
    assert _suffix_strip("kennethwalkeriii") == "kennethwalker"
    assert _suffix_strip("playerii") == "player"
    assert _suffix_strip("playeriv") == "player"


def test_attach_verified_actuals_uses_pbp_only_for_unresolved_identity():
    projections = pd.DataFrame([
        {
            "season": 2026, "week": 2, "team": "KC",
            "player": "Kenneth Walker III", "player_clean_key": "kennethwalker",
            "position": "RB", "projection_mean": 50.0,
        },
        {
            "season": 2026, "week": 2, "team": "SEA",
            "player": "Roster Zero", "player_clean_key": "rosterzero",
            "position": "RB", "projection_mean": 5.0,
        },
    ])
    actual = pd.DataFrame(columns=[
        "season", "week", "team", "gsis_id", "player", "player_clean_key", "rush_yards"
    ])
    roster = pd.DataFrame([
        {
            "season": 2026, "week": 2, "team": "SEA", "gsis_id": "00-zero",
            "player_clean_key": "rosterzero", "status": "ACT",
        }
    ])
    pbp = pd.DataFrame([
        {
            "season": 2026, "week": 2, "team": "KC", "gsis_id": "00-walker",
            "player": "Kenneth Walker III", "player_clean_key": "kennethwalkeriii",
            "rush_yards": 117.0,
        }
    ])

    out, audit = v2.attach_verified_actuals(projections, actual, roster, pbp)
    got = out.set_index("player_clean_key")["actual_rush_yards"].to_dict()
    src = out.set_index("player_clean_key")["actual_source"].to_dict()
    assert got["kennethwalker"] == 117.0
    assert src["kennethwalker"] == "pbp_fallback"
    assert got["rosterzero"] == 0.0
    assert audit["verified_pbp_fallback"] == 1
    assert audit["verified_roster_zero"] == 1
    assert audit["excluded_rows"] == 0


def test_attach_verified_actuals_fails_closed_on_any_remaining_unresolved_identity():
    projections = pd.DataFrame([{
        "season": 2026, "week": 2, "team": "KC",
        "player": "Unknown Back", "player_clean_key": "unknownback",
        "position": "RB", "projection_mean": 1.0,
    }])
    empty_actual = pd.DataFrame(columns=[
        "season", "week", "team", "gsis_id", "player", "player_clean_key", "rush_yards"
    ])
    empty_roster = pd.DataFrame(columns=[
        "season", "week", "team", "gsis_id", "player_clean_key", "status"
    ])
    empty_pbp = empty_actual.copy()
    with pytest.raises(RuntimeError, match="unresolved completed rushing outcome"):
        v2.attach_verified_actuals(projections, empty_actual, empty_roster, empty_pbp)


def test_cumulative_append_requires_exact_contiguous_prior_week():
    prior = pd.DataFrame([_week1_row()])
    new = pd.DataFrame([{
        "season": 2026,
        "week": 2,
        "team": "CHI",
        "player_clean_key": "alpha",
        "position": "RB",
        "projection_mean": 52.0,
        "actual_rush_yards": 48.0,
        "pregame_lineage_certified": True,
        "projection_lineage": "2026_W2_GENERIC_ENSEMBLE_FULL_ROSTER",
    }])
    out = v2.append_to_prior_history(prior=prior, new_week=new, completed_week=2)
    assert sorted(out.week.unique().tolist()) == [1, 2]

    with pytest.raises(RuntimeError, match="contiguous through Week 2"):
        v2.append_to_prior_history(prior=prior, new_week=new.assign(week=3), completed_week=3)


def test_projection_reconstruction_rejects_sportsbook_defined_universe(tmp_path: Path):
    source = tmp_path / "source"
    (source / "data").mkdir(parents=True)
    _write_json(source / "data/football_simulation_universe_audit.json", {
        "sportsbook_rows_used_to_define_player_universe": 1,
        "provider_event_ids_used_during_simulation": False,
        "sportsbook_line_odds_book_fields_present": [],
    })
    _write_json(source / "data/target_entitlement_v1_audit.json", {
        "sportsbook_inputs_used": False,
    })
    paths = v2._source_paths(source)
    with pytest.raises(RuntimeError, match="Sportsbook|sportsbook"):
        v2._validate_football_only_source(paths)
