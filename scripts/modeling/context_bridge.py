"""Adapters from production v2 artifacts into canonical modeling contracts.

This module is intentionally projection-neutral: it assembles trustworthy
pregame football context but does not alter simulation_v2 or pricing outputs.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.modeling.contracts import PlayerContext, TeamContext
from scripts.utils.canonical_names import canonicalize_player_name_safe

DATA = Path("data")


def _read(path: Path, required: bool = False) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        if required:
            raise RuntimeError(f"Required model-context artifact missing: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        if required:
            raise RuntimeError(f"Unable to read required model-context artifact {path}: {exc}") from exc
        return pd.DataFrame()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if required and df.empty:
        raise RuntimeError(f"Required model-context artifact has zero rows: {path}")
    return df


def _num(row: pd.Series, names: Iterable[str]):
    for name in names:
        if name in row.index:
            val = pd.to_numeric(pd.Series([row[name]]), errors="coerce").iloc[0]
            if pd.notna(val):
                return float(val)
    return None


def _text(row: pd.Series, names: Iterable[str], default: str = "") -> str:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            value = str(row[name]).strip()
            if value and value.lower() != "nan":
                return value
    return default


def _player_key(value) -> str:
    try:
        _, key = canonicalize_player_name_safe(value)
        if key:
            return str(key)
    except Exception:
        pass
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def build_team_contexts(team_form: pd.DataFrame, team_coverage: pd.DataFrame | None = None) -> Dict[str, TeamContext]:
    if team_form is None or team_form.empty:
        raise RuntimeError("TeamContext bridge requires non-empty team_form")
    tf = team_form.copy()
    tf.columns = [str(c).strip().lower() for c in tf.columns]
    if "team" not in tf.columns or "season" not in tf.columns:
        raise RuntimeError("team_form missing team/season")
    tf["team"] = tf["team"].map(canon_team)
    if tf["team"].eq("").any():
        raise RuntimeError("team_form contains unresolvable team identity")
    if tf["team"].duplicated().any():
        raise RuntimeError("team_form must contain exactly one row per team")

    coverage = pd.DataFrame() if team_coverage is None else team_coverage.copy()
    if not coverage.empty:
        coverage.columns = [str(c).strip().lower() for c in coverage.columns]
        if "team" in coverage.columns:
            coverage["team"] = coverage["team"].map(canon_team)
            coverage = coverage.drop_duplicates("team").set_index("team", drop=False)

    out: Dict[str, TeamContext] = {}
    for _, row in tf.iterrows():
        team = str(row["team"])
        cov = coverage.loc[team] if not coverage.empty and team in coverage.index else pd.Series(dtype="object")
        season = int(pd.to_numeric(row["season"], errors="raise"))
        out[team] = TeamContext(
            team=team,
            season=season,
            success_rate_off=_num(row, ["success_rate_off"]),
            success_rate_def=_num(row, ["success_rate_def"]),
            pressure_rate_generated=_num(row, ["pressure_rate_generated", "pressure_rate", "dl_pressure_rate", "def_pressure_rate"]),
            pressure_rate_allowed=_num(row, ["pressure_rate_allowed", "off_pressure_rate_allowed"]),
            neutral_pace=_num(row, ["neutral_pace", "pace"]),
            neutral_pace_last5=_num(row, ["neutral_pace_last5", "neutralpacelast5"]),
            sec_per_play_last5=_num(row, ["sec_per_play_last5", "secplay_last_5", "seconds_per_play_last5"]),
            plays_est=_num(row, ["plays_est", "plays_per_game"]),
            proe=_num(row, ["proe", "pass_rate_over_expected"]),
            explosive_play_rate_allowed=_num(row, ["explosive_play_rate_allowed"]),
            coverage_man_rate=_num(cov, ["man_rate", "coverage_man_rate"]) if not cov.empty else _num(row, ["coverage_man_rate", "man_rate"]),
            coverage_zone_rate=_num(cov, ["zone_rate", "coverage_zone_rate"]) if not cov.empty else _num(row, ["coverage_zone_rate", "zone_rate"]),
            middle_open_rate=_num(row, ["middle_open_rate"]),
            light_box_rate=_num(row, ["light_box_rate"]),
            heavy_box_rate=_num(row, ["heavy_box_rate"]),
            def_pass_epa=_num(row, ["def_pass_epa"]),
            def_rush_epa=_num(row, ["def_rush_epa"]),
        )
    return out


def build_player_contexts(
    player_form: pd.DataFrame,
    consensus: pd.DataFrame,
    teams: Dict[str, TeamContext],
    exposure: pd.DataFrame | None = None,
    injuries: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
) -> list[PlayerContext]:
    if player_form is None or player_form.empty:
        raise RuntimeError("PlayerContext bridge requires non-empty player_form")
    pf = player_form.copy()
    pf.columns = [str(c).strip().lower() for c in pf.columns]
    required = {"player", "team", "opponent", "season", "week", "position"}
    missing = required - set(pf.columns)
    if missing:
        raise RuntimeError(f"player_form missing columns: {sorted(missing)}")
    pf["team"] = pf["team"].map(canon_team)
    pf["opponent"] = pf["opponent"].map(canon_team)
    pf["player_key_bridge"] = pf["player"].map(_player_key)
    if pf[["team", "opponent"]].eq("").any().any():
        raise RuntimeError("player_form contains unresolved team/opponent identity")
    if pf.duplicated(["team", "player_key_bridge"]).any():
        raise RuntimeError("player_form contains duplicate active player/team rows")

    con = consensus.copy() if consensus is not None else pd.DataFrame()
    if not con.empty:
        con.columns = [str(c).strip().lower() for c in con.columns]
        con["team"] = con["team"].map(canon_team)
        con["player_key_bridge"] = con["player"].map(_player_key)
        keep = [c for c in con.columns if c not in {"player", "team", "season", "week", "position", "role"}]
        con = con[["team", "player_key_bridge", *keep]].drop_duplicates(["team", "player_key_bridge"])
        pf = pf.merge(con, on=["team", "player_key_bridge"], how="left", suffixes=("", "_consensus"))

    exp = pd.DataFrame() if exposure is None else exposure.copy()
    if not exp.empty:
        exp.columns = [str(c).strip().lower() for c in exp.columns]
        exp["team"] = exp["team"].map(canon_team)
        exp["player_key_bridge"] = exp["player"].map(_player_key)
        exp_keep = [c for c in ["team", "player_key_bridge", "primary_cb", "exp_vs_man", "exp_vs_zone", "matchup_available", "team_coverage_available"] if c in exp.columns]
        exp = exp[exp_keep].drop_duplicates(["team", "player_key_bridge"])
        pf = pf.merge(exp, on=["team", "player_key_bridge"], how="left", suffixes=("", "_coverage"))

    inj = pd.DataFrame() if injuries is None else injuries.copy()
    if not inj.empty:
        inj.columns = [str(c).strip().lower() for c in inj.columns]
        inj["team"] = inj["team"].map(canon_team)
        inj["player_key_bridge"] = inj["player"].map(_player_key)
        inj_keep = [c for c in ["team", "player_key_bridge", "status", "practice_status", "body_part", "designation", "report_available"] if c in inj.columns]
        inj = inj[inj_keep].drop_duplicates(["team", "player_key_bridge"])
        pf = pf.merge(inj, on=["team", "player_key_bridge"], how="left", suffixes=("", "_injury"))

    wx_by_game: dict[str, dict] = {}
    wx = pd.DataFrame() if weather is None else weather.copy()
    if not wx.empty:
        wx.columns = [str(c).strip().lower() for c in wx.columns]
        if "game_id" in wx.columns:
            wx_by_game = {str(r.get("game_id", "")): r.to_dict() for _, r in wx.iterrows() if pd.notna(r.get("game_id"))}

    contexts: list[PlayerContext] = []
    for _, row in pf.iterrows():
        team = str(row["team"])
        opp = str(row["opponent"])
        if team not in teams or opp not in teams:
            raise RuntimeError(f"Missing TeamContext for active matchup {team} vs {opp}")
        game_id = _text(row, ["game_id"])
        features = {
            "tgt_share": _num(row, ["tgt_share"]),
            "rush_share": _num(row, ["rush_share"]),
            "route_rate": _num(row, ["route_rate"]),
            "yprr": _num(row, ["yprr"]),
            "ypt": _num(row, ["ypt"]),
            "ypc": _num(row, ["ypc"]),
            "ypa": _num(row, ["ypa"]),
            "catch_rate": _num(row, ["receptions_per_target", "catch_rate"]),
            "primary_cb": _text(row, ["primary_cb"]),
            "exp_vs_man": _num(row, ["exp_vs_man"]),
            "exp_vs_zone": _num(row, ["exp_vs_zone"]),
            "matchup_available": int(_num(row, ["matchup_available"]) or 0),
            "team_coverage_available": int(_num(row, ["team_coverage_available"]) or 0),
            "injury_status": _text(row, ["status"]),
            "practice_status": _text(row, ["practice_status"]),
            "injury_designation": _text(row, ["designation"]),
            "injury_report_available": int(_num(row, ["report_available"]) or 0),
        }
        if game_id and game_id in wx_by_game:
            w = wx_by_game[game_id]
            for src, dst in [("temp_f", "weather_temp_f"), ("wind_mph", "weather_wind_mph"), ("precip_flag", "weather_precip_flag"), ("forecast_ok", "weather_forecast_available")]:
                if src in w:
                    features[dst] = w[src]
        contexts.append(PlayerContext(
            player=_text(row, ["player"]),
            team=team,
            opponent=opp,
            season=int(pd.to_numeric(row["season"], errors="raise")),
            week=int(pd.to_numeric(row["week"], errors="raise")),
            position=_text(row, ["position"]).upper(),
            role=_text(row, ["role"]),
            game_id=game_id,
            features=features,
            offense=teams[team],
            defense=teams[opp],
        ))
    return contexts


def load_model_contexts(data_dir: Path = DATA) -> tuple[Dict[str, TeamContext], list[PlayerContext]]:
    team_form = _read(data_dir / "team_form.csv", required=True)
    player_form = _read(data_dir / "player_form.csv", required=True)
    consensus = _read(data_dir / "player_form_consensus.csv", required=True)
    team_cov = _read(data_dir / "cb_coverage_team.csv")
    exposure = _read(data_dir / "wr_cb_exposure.csv")
    injuries = _read(data_dir / "injuries.csv")
    weather = _read(data_dir / "weather_week.csv")
    teams = build_team_contexts(team_form, team_cov)
    players = build_player_contexts(player_form, consensus, teams, exposure, injuries, weather)
    return teams, players


def player_context_frame(players: list[PlayerContext]) -> pd.DataFrame:
    rows = []
    for p in players:
        row = {
            "player": p.player,
            "team": p.team,
            "opponent": p.opponent,
            "season": p.season,
            "week": p.week,
            "position": p.position,
            "role": p.role,
            "game_id": p.game_id,
            **p.features,
        }
        rows.append(row)
    return pd.DataFrame(rows)
