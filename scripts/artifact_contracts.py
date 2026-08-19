"""Canonical artifact contracts for the Full Slate pipeline.

Every production builder and validator should agree on these paths and minimum
schemas. This module intentionally contains no network or model logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"


@dataclass(frozen=True)
class ArtifactContract:
    name: str
    path: Path
    required_columns: Sequence[str]
    min_rows: int = 1
    required: bool = True


CONTRACTS: dict[str, ArtifactContract] = {
    "roles_ourlads": ArtifactContract(
        "roles_ourlads", DATA / "roles_ourlads.csv",
        ("player", "team", "role", "position", "player_key"), min_rows=32,
    ),
    "team_week_map": ArtifactContract(
        "team_week_map", DATA / "team_week_map.csv",
        ("season", "week", "team", "opponent"), min_rows=32,
    ),
    "opponent_map": ArtifactContract(
        "opponent_map", DATA / "opponent_map_from_props.csv",
        ("player", "team", "opponent", "season", "week"), min_rows=1,
    ),
    "props_raw": ArtifactContract(
        "props_raw", OUTPUTS / "props_raw.csv",
        ("player", "market", "line"), min_rows=1,
    ),
    "odds_game": ArtifactContract(
        "odds_game", OUTPUTS / "odds_game.csv",
        ("event_id",), min_rows=1,
    ),
    "team_form": ArtifactContract(
        "team_form", DATA / "team_form.csv",
        ("team", "season"), min_rows=1,
    ),
    "cb_coverage_team": ArtifactContract(
        "cb_coverage_team", DATA / "cb_coverage_team.csv",
        ("team", "season", "week", "man_rate", "zone_rate", "coverage_available", "coverage_source"),
        min_rows=2, required=False,
    ),
    "cb_coverage_player": ArtifactContract(
        "cb_coverage_player", DATA / "cb_coverage_player.csv",
        ("player", "team", "opponent", "season", "week", "primary_cb", "matchup_available", "alignment_available"),
        min_rows=1, required=False,
    ),
    "wr_cb_exposure": ArtifactContract(
        "wr_cb_exposure", DATA / "wr_cb_exposure.csv",
        ("player", "team", "opponent", "season", "week", "exp_vs_man", "exp_vs_zone", "matchup_available", "team_coverage_available"),
        min_rows=1, required=False,
    ),
    "player_game_logs": ArtifactContract(
        "player_game_logs", DATA / "player_game_logs.csv",
        ("season", "week", "game_id", "player", "team", "opponent"), min_rows=1,
    ),
    "player_form": ArtifactContract(
        "player_form", DATA / "player_form.csv",
        ("player", "team", "opponent", "season", "week", "position", "role"), min_rows=1,
    ),
    "player_form_consensus": ArtifactContract(
        "player_form_consensus", DATA / "player_form_consensus.csv",
        (
            "player", "team", "season", "position", "role", "tgt_share", "rush_share",
            "prior_games", "current_games",
            "tgt_share_prior", "tgt_share_current",
            "rush_share_prior", "rush_share_current",
            "ypt_prior", "ypt_current",
            "ypc_prior", "ypc_current",
            "ypa_prior", "ypa_current",
            "receptions_per_target_prior", "receptions_per_target_current",
        ), min_rows=1,
    ),
    "model_context_bridge": ArtifactContract(
        "model_context_bridge", DATA / "model_context_bridge.csv",
        ("player", "team", "opponent", "season", "week", "position", "role"), min_rows=1,
    ),
    "model_bayesian_diagnostics": ArtifactContract(
        "model_bayesian_diagnostics", DATA / "model_bayesian_diagnostics.csv",
        ("player", "team", "season", "position", "bayes_available", "bayes_evidence_state", "bayes_tgt_share", "bayes_rush_share"),
        min_rows=1,
    ),
    "model_ml_diagnostics": ArtifactContract(
        "model_ml_diagnostics", DATA / "model_ml_diagnostics.csv",
        ("player", "team", "season", "week", "position", "hist_games", "ml_available", "ml_method", "ml_training_cutoff"),
        min_rows=1,
    ),
    "model_rule_diagnostics": ArtifactContract(
        "model_rule_diagnostics", DATA / "model_rule_diagnostics.csv",
        ("player", "team", "opponent", "season", "week", "projected_plays", "pass_eff_mult", "rush_eff_mult"), min_rows=1,
    ),
    "model_rule_simulation_inputs": ArtifactContract(
        "model_rule_simulation_inputs", DATA / "model_rule_simulation_inputs.csv",
        ("player", "team", "opponent", "market", "ml_applied", "bayes_applied", "rules_applied", "rules_plays_est", "rules_pass_rate"),
        min_rows=1, required=False,
    ),
    "qb_run_metrics": ArtifactContract(
        "qb_run_metrics", DATA / "qb_run_metrics.csv",
        ("player", "week", "scramble_rate", "designed_run_rate"), min_rows=1,
        required=False,
    ),
    "weather_week": ArtifactContract(
        "weather_week", DATA / "weather_week.csv", (), min_rows=1, required=False,
    ),
    "metrics_ready": ArtifactContract(
        "metrics_ready", DATA / "metrics_ready.csv",
        ("player", "team", "team_abbr", "opponent", "opponent_abbr", "market", "line", "season", "week"),
        min_rows=1,
    ),
    "props_priced": ArtifactContract(
        "props_priced", OUTPUTS / "props_priced_clean.csv",
        ("player", "market", "side", "vegas_line", "model_proj", "fair_prob", "edge_pct"), min_rows=1,
    ),
}


def get_contract(name: str) -> ArtifactContract:
    try:
        return CONTRACTS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown artifact contract: {name}") from exc
