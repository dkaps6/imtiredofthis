"""Leakage-safe historical context factory for walk-forward backtests.

This module establishes the time boundary for historical prediction. It never
uses target-week outcomes to construct player/team features. The caller must
supply a pregame player universe separately (historical roster/depth/active
players); target-week result rows are not used to decide who gets projected.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest.qb_opportunity import add_qb_opportunity
from scripts.modeling.context_bridge import build_player_contexts, build_team_contexts
from scripts.modeling.contracts import PlayerContext, TeamContext
from scripts.utils.canonical_names import canonicalize_player_name_safe


@dataclass(frozen=True)
class HistoricalContextBundle:
    season: int
    week: int
    prior_season: int
    player_history: pd.DataFrame
    team_history: pd.DataFrame
    player_form: pd.DataFrame
    player_consensus: pd.DataFrame
    team_form: pd.DataFrame
    teams: dict[str, TeamContext]
    players: list[PlayerContext]


def _key(value) -> str:
    try:
        _, key = canonicalize_player_name_safe(value)
        if key:
            return str(key)
    except Exception:
        pass
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def before_cutoff(frame: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=[] if frame is None else frame.columns)
    x = frame.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    if not {"season", "week"}.issubset(x.columns):
        raise RuntimeError("historical frame requires season/week")
    s = pd.to_numeric(x["season"], errors="coerce")
    w = pd.to_numeric(x["week"], errors="coerce")
    mask = s.lt(int(season)) | (s.eq(int(season)) & w.lt(int(week)))
    return x.loc[mask].copy()


def assert_no_future_rows(frame: pd.DataFrame, season: int, week: int, label: str) -> None:
    if frame is None or frame.empty:
        return
    x = frame.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    if not {"season", "week"}.issubset(x.columns):
        raise RuntimeError(f"{label} missing season/week")
    s = pd.to_numeric(x["season"], errors="coerce")
    w = pd.to_numeric(x["week"], errors="coerce")
    bad = s.gt(int(season)) | (s.eq(int(season)) & w.ge(int(week)))
    if bad.any():
        sample = x.loc[bad, ["season", "week"]].head(5).to_dict("records")
        raise RuntimeError(f"LEAKAGE: {label} contains target/future rows for {season} W{week}: {sample}")


def _num_series(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def build_historical_player_inputs(player_logs: pd.DataFrame, pregame_universe: pd.DataFrame, season: int, week: int, prior_season: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build historical player form using only games before the target week."""
    hist = before_cutoff(player_logs, season, week)
    assert_no_future_rows(hist, season, week, "player_history")
    if hist.empty:
        raise RuntimeError("No player history available before target cutoff")

    u = pregame_universe.copy()
    u.columns = [str(c).strip().lower() for c in u.columns]
    required = {"player", "team", "opponent", "position"}
    missing = required - set(u.columns)
    if missing:
        raise RuntimeError(f"pregame_universe missing columns: {sorted(missing)}")
    u["team"] = u["team"].map(canon_team)
    u["opponent"] = u["opponent"].map(canon_team)
    u["player_clean_key"] = u.get("player_clean_key", u["player"]).map(_key)
    u["season"] = int(season)
    u["week"] = int(week)
    if "role" not in u.columns:
        u["role"] = ""
    if "game_id" not in u.columns:
        u["game_id"] = ""
    if u[["team", "opponent", "player_clean_key"]].eq("").any().any():
        raise RuntimeError("pregame_universe contains unresolved identity")
    if u.duplicated(["team", "player_clean_key"]).any():
        raise RuntimeError("pregame_universe contains duplicate player/team rows")

    h = hist.copy()
    h["player_clean_key"] = h.get("player_clean_key", h["player"]).map(_key)
    h["team"] = h["team"].map(canon_team)
    stat_map = {
        "tgt_share": "tgt_share_game", "rush_share": "rush_share_game", "route_rate": "route_rate_game",
        "yprr": "yprr_game", "ypt": "ypt_game", "ypc": "ypc_game", "ypa": "ypa_game",
        "receptions_per_target": "catch_rate_game",
    }

    rows = []
    consensus_rows = []
    for _, p in u.iterrows():
        ph = h.loc[h["player_clean_key"].eq(p["player_clean_key"])].sort_values(["season", "week"])
        prior = ph.loc[pd.to_numeric(ph["season"], errors="coerce").eq(int(prior_season))]
        current = ph.loc[pd.to_numeric(ph["season"], errors="coerce").eq(int(season))]
        base = {
            "player": p["player"], "player_clean_key": p["player_clean_key"], "team": p["team"],
            "opponent": p["opponent"], "season": int(season), "week": int(week),
            "position": p["position"], "role": p.get("role", ""), "game_id": p.get("game_id", ""),
        }
        con = dict(base)
        con["prior_games"] = int(prior["week"].nunique()) if not prior.empty else 0
        con["current_games"] = int(current["week"].nunique()) if not current.empty else 0
        for out_col, src in stat_map.items():
            prior_v = _num_series(prior, src).mean() if not prior.empty else np.nan
            current_v = _num_series(current, src).mean() if not current.empty else np.nan
            blended = current_v if pd.notna(current_v) else prior_v
            base[out_col] = blended
            con[out_col] = blended
            con[f"{out_col}_prior"] = prior_v
            con[f"{out_col}_current"] = current_v
        rows.append(base)
        consensus_rows.append(con)

    player_form = pd.DataFrame(rows)
    player_form = add_qb_opportunity(player_form, u, h, season=int(season), prior_season=int(prior_season))
    consensus = pd.DataFrame(consensus_rows).merge(
        player_form[["team", "player_clean_key", "qb_projection_eligible", "qb_pass_att_share", "qb_role_score", "qb_role_source"]],
        on=["team", "player_clean_key"], how="left", validate="one_to_one",
    )
    return hist, player_form, consensus


def build_historical_team_form(team_weekly: pd.DataFrame, season: int, week: int, prior_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    hist = before_cutoff(team_weekly, season, week)
    assert_no_future_rows(hist, season, week, "team_history")
    if hist.empty:
        raise RuntimeError("No team history available before target cutoff")
    x = hist.copy()
    x["team"] = x["team"].map(canon_team)
    numeric_cols = []
    for c in x.columns:
        if c in {"season", "week", "team"}:
            continue
        vals = pd.to_numeric(x[c], errors="coerce")
        if vals.notna().any():
            x[c] = vals
            numeric_cols.append(c)

    rows = []
    for team in sorted(t for t in x["team"].dropna().astype(str).unique() if t):
        part = x.loc[x["team"].eq(team)]
        cur = part.loc[pd.to_numeric(part["season"], errors="coerce").eq(int(season))]
        prior = part.loc[pd.to_numeric(part["season"], errors="coerce").eq(int(prior_season))]
        source = cur if not cur.empty else prior
        if source.empty:
            continue
        row = {"team": team, "season": int(season), "team_history_games": int(len(source)), "team_history_source": "current" if not cur.empty else "prior"}
        for c in numeric_cols:
            row[c] = pd.to_numeric(source[c], errors="coerce").mean()
        rows.append(row)
    return hist, pd.DataFrame(rows)


def build_historical_context_bundle(*, player_logs: pd.DataFrame, team_weekly: pd.DataFrame, pregame_universe: pd.DataFrame, schedule: pd.DataFrame, season: int, week: int, prior_season: int, team_coverage: pd.DataFrame | None = None, exposure: pd.DataFrame | None = None, injuries: pd.DataFrame | None = None, weather: pd.DataFrame | None = None) -> HistoricalContextBundle:
    ph, player_form, consensus = build_historical_player_inputs(player_logs, pregame_universe, season, week, prior_season)
    th, team_form = build_historical_team_form(team_weekly, season, week, prior_season)
    for label, frame in (("team_coverage", team_coverage), ("exposure", exposure), ("injuries", injuries), ("weather", weather)):
        if frame is not None and not frame.empty and {"season", "week"}.issubset({str(c).lower() for c in frame.columns}):
            assert_no_future_rows(frame, season, week, label)
    teams = build_team_contexts(team_form, team_coverage)
    players = build_player_contexts(player_form, consensus, teams, exposure=exposure, injuries=injuries, weather=weather, team_week_map=schedule)
    return HistoricalContextBundle(
        season=int(season), week=int(week), prior_season=int(prior_season), player_history=ph,
        team_history=th, player_form=player_form, player_consensus=consensus, team_form=team_form,
        teams=teams, players=players,
    )
