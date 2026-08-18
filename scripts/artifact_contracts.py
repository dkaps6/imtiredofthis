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
        ("player", "team", "season", "position", "role", "tgt_share", "rush_share"), min_rows=1,
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
