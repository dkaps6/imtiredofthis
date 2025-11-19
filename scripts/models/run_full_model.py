"""Top-level orchestration utilities for pricing + predictive models.

Phase 2 wiring focuses on threading team-level script features through the
per-player model inputs without touching the core model math yet.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from scripts.models.shared_types import TeamScriptFeatures
from scripts.models.team_script_features import load_team_script_features
from scripts.models.types import PlayerModelInput

try:  # Reuse the legacy stack for backwards compatibility if present.
    from . import model_stack_patch as _legacy_stack
except Exception:  # pragma: no cover - optional import
    _legacy_stack = None


logger = logging.getLogger(__name__)


def _read_dataframe(path: Path) -> pd.DataFrame:
    """Best-effort CSV reader with informative logging."""

    if not path.exists():
        logger.warning("Metrics file %s does not exist", path)
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception:
        logger.exception("Failed to read metrics file at %s", path)
        return pd.DataFrame()

    if df.empty:
        logger.warning("Metrics file %s was empty", path)

    return df


def _first_nonnull(row: pd.Series, columns: List[str]) -> Optional[str]:
    for col in columns:
        if col in row.index:
            val = row[col]
            if pd.notna(val):
                return val
    return None


def _to_float(val) -> Optional[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except Exception:
        return None


def _build_player_inputs(
    df: pd.DataFrame, team_features: Dict[str, TeamScriptFeatures]
) -> List[PlayerModelInput]:
    inputs: List[PlayerModelInput] = []

    for _, row in df.iterrows():
        player_name = _first_nonnull(row, ["player", "player_name"]) or ""
        team_abbr = (
            _first_nonnull(row, ["team_abbr", "team", "team_code", "team_name"]) or ""
        ).upper()
        opponent_abbr = (
            _first_nonnull(row, ["opponent_abbr", "opponent", "opp_team"]) or ""
        ).upper()
        market = _first_nonnull(row, ["market", "stat_type", "metric"])
        if not market:
            logger.debug(
                "Skipping row for player=%s because no market/stat_type was present",
                player_name,
            )
            continue

        line_value = _to_float(
            _first_nonnull(row, ["line", "vegas_line", "prop_line", "value"])
        )

        player_id = _first_nonnull(
            row,
            [
                "player_id",
                "sportsdata_id",
                "gsis_id",
                "pbp_id",
            ],
        )
        if not player_id:
            player_id = f"{player_name}|{team_abbr}|{market}|{line_value}"

        bookmaker = _first_nonnull(row, ["bookmaker", "book", "source"])
        event_id = _first_nonnull(row, ["event_id", "game_id"])

        offense_script = team_features.get(team_abbr)
        defense_script = team_features.get(opponent_abbr)

        if offense_script is None:
            logger.debug(
                "No TeamScriptFeatures found for offense team_abbr=%s", team_abbr
            )
        if defense_script is None and opponent_abbr:
            logger.debug(
                "No TeamScriptFeatures found for defense team_abbr=%s",
                opponent_abbr,
            )

        model_input = PlayerModelInput(
            player_id=str(player_id),
            player_name=str(player_name),
            team_abbr=team_abbr,
            opponent_abbr=opponent_abbr,
            market=str(market),
            line=line_value,
            stat_type=_first_nonnull(row, ["stat_type", "stat"]),
            bookmaker=bookmaker,
            event_id=event_id,
            model_features=row.to_dict(),
            offense_script=offense_script,
            defense_script=defense_script,
        )

        inputs.append(model_input)

    logger.info("Built %d player inputs", len(inputs))

    return inputs


def run_full_model(
    season: Optional[int] = None,
    week: Optional[int] = None,
    metrics_path: str = "data/metrics_ready.csv",
    team_form_path: str = "data/team_form.csv",
) -> List[PlayerModelInput]:
    """Entry point used by run_predictors and tests."""

    logger.info(
        "Starting run_full_model orchestrator (season=%s, week=%s)",
        season,
        week,
    )

    team_script_features = load_team_script_features(path=team_form_path)
    if not team_script_features:
        logger.warning(
            "No team script features loaded from %s; offense/defense script fields will be None",
            team_form_path,
        )

    df = _read_dataframe(Path(metrics_path))
    if df.empty:
        logger.warning(
            "Metrics dataframe empty from %s; returning without running models",
            metrics_path,
        )
        return []

    player_inputs = _build_player_inputs(df, team_script_features)

    logger.info(
        "Built %d model inputs; example offense_script teams: %s",
        len(player_inputs),
        sorted(
            {
                inp.team_abbr
                for inp in player_inputs
                if getattr(inp, "offense_script", None) is not None
            }
        )[:8],
    )

    if _legacy_stack is not None:
        try:
            _legacy_stack.run_full_model(season=season, week=week)
        except Exception:
            logger.exception("Legacy model stack failed to execute")

    return player_inputs


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the unified model orchestrator")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--metrics", default="data/metrics_ready.csv")
    parser.add_argument("--team-form", default="data/team_form.csv")
    args = parser.parse_args()

    run_full_model(
        season=args.season,
        week=args.week,
        metrics_path=args.metrics,
        team_form_path=args.team_form,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
