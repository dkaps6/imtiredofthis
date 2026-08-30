"""Canonical Team Context v3 -> player modeling bridge.

Team Context v3 is the authoritative team-level football artifact for 2026
production.  Player-level enrichments remain separate and are attached by the
existing Player Context builder.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.modeling.context_bridge import build_player_contexts, player_context_frame
from scripts.modeling.contracts import TeamContext

DATA = Path("data")
TEAM_CONTEXT = DATA / "team_context_v3.csv"


def _read(path: Path, required: bool = False) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        if required:
            raise RuntimeError(f"Required model-context artifact missing: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if required and df.empty:
        raise RuntimeError(f"Required model-context artifact has zero rows: {path}")
    return df


def _num(row: pd.Series, names: Iterable[str]):
    for name in names:
        if name in row.index:
            value = pd.to_numeric(pd.Series([row.get(name)]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
    return None


def build_team_contexts(team_context: pd.DataFrame) -> Dict[str, TeamContext]:
    if team_context is None or team_context.empty:
        raise RuntimeError("Team Context v3 bridge requires non-empty team_context_v3")
    tf = team_context.copy()
    tf.columns = [str(c).strip().lower() for c in tf.columns]
    required = {"team", "season", "team_context_version"}
    missing = required - set(tf.columns)
    if missing:
        raise RuntimeError(f"team_context_v3 missing columns: {sorted(missing)}")
    tf["team"] = tf["team"].map(canon_team)
    if tf["team"].eq("").any() or tf.duplicated("team").any():
        raise RuntimeError("team_context_v3 has invalid/duplicate team identity")
    if not tf["team_context_version"].astype(str).eq("TEAM_CONTEXT_V3").all():
        raise RuntimeError("team_context_v3 contains an unexpected context version")

    out: Dict[str, TeamContext] = {}
    for _, row in tf.iterrows():
        team = str(row["team"])
        out[team] = TeamContext(
            team=team,
            season=int(pd.to_numeric(row["season"], errors="raise")),
            success_rate_off=_num(row, ["success_rate_off"]),
            success_rate_def=_num(row, ["success_rate_def"]),
            pressure_rate_generated=_num(row, ["hit_sack_pressure_rate_generated", "pressure_rate_generated"]),
            pressure_rate_allowed=_num(row, ["hit_sack_pressure_rate_allowed", "pressure_rate_allowed"]),
            neutral_pace=_num(row, ["neutral_pace_true", "neutral_pace"]),
            neutral_pace_last5=_num(row, ["neutral_pace_last5"]),
            sec_per_play_last5=_num(row, ["sec_per_play_last5", "seconds_per_play_last5"]),
            plays_est=_num(row, ["plays_est"]),
            proe=_num(row, ["true_proe", "proe"]),
            explosive_play_rate_allowed=_num(row, ["explosive_play_rate_allowed"]),
            coverage_man_rate=_num(row, ["coverage_man_rate"]),
            coverage_zone_rate=_num(row, ["coverage_zone_rate"]),
            middle_open_rate=_num(row, ["middle_open_rate"]),
            light_box_rate=_num(row, ["light_box_rate"]),
            heavy_box_rate=_num(row, ["heavy_box_rate"]),
            def_pass_epa=_num(row, ["def_pass_epa_allowed", "def_pass_epa"]),
            def_rush_epa=_num(row, ["def_rush_epa"]),
        )
    return out


def load_model_contexts(data_dir: Path = DATA):
    team_context = _read(data_dir / "team_context_v3.csv", required=True)
    player_form = _read(data_dir / "player_form.csv", required=True)
    consensus = _read(data_dir / "player_form_consensus.csv", required=True)
    team_week_map = _read(data_dir / "team_week_map.csv", required=True)
    exposure = _read(data_dir / "wr_cb_exposure.csv")
    injuries = _read(data_dir / "injuries.csv")
    weather = _read(data_dir / "weather_week.csv")

    teams = build_team_contexts(team_context)
    players = build_player_contexts(
        player_form,
        consensus,
        teams,
        exposure,
        injuries,
        weather,
        team_week_map,
    )
    return teams, players


__all__ = ["build_team_contexts", "build_player_contexts", "load_model_contexts", "player_context_frame"]
