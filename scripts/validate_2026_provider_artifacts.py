#!/usr/bin/env python3
"""Runtime provider-readiness gate for the 2026 Full Slate pipeline.

This validates artifacts after provider/build steps have run.  It distinguishes
structural/provider failures from legitimate optional-data unavailability.
Sportsbook artifacts are required only when FETCH_LIVE_ODDS is enabled.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from scripts._opponent_map import CANON_TEAM_CODES, canon_team
from scripts.runtime_context import resolve_prior_season, resolve_season, resolve_week

DATA = Path("data")
OUTPUTS = Path("outputs")
OUT = DATA / "provider_readiness_v3.csv"


def _read_csv(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        if required:
            raise RuntimeError(f"required provider artifact missing/empty: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if required and df.empty:
        raise RuntimeError(f"required provider artifact has zero rows: {path}")
    return df


def _result(provider: str, status: str, detail: str, *, fatal: bool = False) -> dict:
    return {
        "provider": provider,
        "status": status,
        "fatal": int(bool(fatal)),
        "detail": detail,
    }


def validate_schedule(df: pd.DataFrame, season: int, week: int) -> tuple[dict, set[str]]:
    required = {"season", "week", "team", "opponent"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"team_week_map missing columns: {sorted(missing)}")
    x = df.copy()
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x = x.loc[x["season"].eq(int(season)) & x["week"].eq(int(week))].copy()
    if x.empty:
        raise RuntimeError(f"team_week_map has no active rows season={season} week={week}")
    x["team"] = x["team"].map(canon_team)
    x["opponent"] = x["opponent"].map(canon_team)
    if x[["team", "opponent"]].eq("").any().any():
        raise RuntimeError("active schedule contains unresolvable team/opponent")
    if x.duplicated("team").any():
        raise RuntimeError("active schedule contains duplicate team rows")
    if len(x) % 2:
        raise RuntimeError(f"active schedule row count must be even; rows={len(x)}")
    teams = set(x["team"])
    if not teams.issubset(CANON_TEAM_CODES) or len(teams) < 24 or len(teams) > 32:
        raise RuntimeError(f"active schedule team set implausible; teams={len(teams)}")
    opp_map = dict(zip(x["team"], x["opponent"]))
    asymmetric = [team for team, opp in opp_map.items() if opp_map.get(opp) != team]
    if asymmetric:
        raise RuntimeError(f"active schedule opponent mapping is not symmetric: {asymmetric[:10]}")
    return _result("schedule", "ready", f"scheduled_teams={len(teams)} games={len(teams)//2}"), teams


def validate_ourlads(df: pd.DataFrame) -> dict:
    required = {"team", "player", "player_clean_key", "position", "role"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Ourlads roles missing columns: {sorted(missing)}")
    x = df.copy()
    x["team"] = x["team"].map(canon_team)
    if x["team"].eq("").any():
        raise RuntimeError("Ourlads contains unresolvable team identity")
    teams = set(x["team"])
    if teams != set(CANON_TEAM_CODES):
        missing_teams = sorted(set(CANON_TEAM_CODES) - teams)
        raise RuntimeError(f"Ourlads does not cover all 32 teams; missing={missing_teams}")
    if x["player"].astype("string").fillna("").str.strip().eq("").any():
        raise RuntimeError("Ourlads contains blank player names")
    if x["player_clean_key"].astype("string").fillna("").str.strip().eq("").any():
        raise RuntimeError("Ourlads contains blank player keys")
    return _result("ourlads", "ready", f"rows={len(x)} teams=32")


def validate_weather(df: pd.DataFrame, scheduled_teams: set[str], season: int, week: int) -> dict:
    required = {"season", "week", "home", "away", "forecast_ok"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"weather_week missing columns: {sorted(missing)}")
    x = df.copy()
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x = x.loc[x["season"].eq(int(season)) & x["week"].eq(int(week))].copy()
    if len(x) != len(scheduled_teams) // 2:
        raise RuntimeError(f"weather game count mismatch expected={len(scheduled_teams)//2} got={len(x)}")
    x["home"] = x["home"].map(canon_team)
    x["away"] = x["away"].map(canon_team)
    weather_teams = set(x["home"]) | set(x["away"])
    if weather_teams != scheduled_teams:
        raise RuntimeError("weather teams do not exactly match active schedule")
    if x.duplicated(["home", "away"]).any():
        raise RuntimeError("weather artifact contains duplicate games")
    ok = int(pd.to_numeric(x["forecast_ok"], errors="coerce").fillna(0).ne(0).sum())
    status = "ready" if ok else "schedule_ready_forecast_pending"
    return _result("weather", status, f"games={len(x)} forecast_ok={ok}/{len(x)}")


def validate_injuries(df: pd.DataFrame, status: dict, season: int, week: int) -> dict:
    if int(status.get("season", -1)) != int(season) or int(status.get("week", -1)) != int(week):
        raise RuntimeError("injury provider status runtime does not match active season/week")
    state = str(status.get("state", "")).strip()
    if state == "provider_outage":
        raise RuntimeError(f"injury provider outage: {status.get('provider_errors', [])}")
    if state not in {"official_report", "no_official_report"}:
        raise RuntimeError(f"unknown injury provider state: {state}")
    if state == "official_report":
        if df.empty:
            raise RuntimeError("injury status says official_report but injuries.csv is empty")
        required = {"player", "team", "season", "week", "report_available"}
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(f"injuries.csv missing columns: {sorted(missing)}")
        s = pd.to_numeric(df["season"], errors="coerce")
        w = pd.to_numeric(df["week"], errors="coerce")
        if not s.eq(int(season)).all() or not w.eq(int(week)).all():
            raise RuntimeError("injury rows contain stale season/week data")
        if not pd.to_numeric(df["report_available"], errors="coerce").fillna(0).eq(1).all():
            raise RuntimeError("official injury rows missing report_available=1")
    elif not df.empty:
        raise RuntimeError("injury status says no_official_report but injuries.csv contains rows")
    return _result("injuries", state, f"rows={len(df)} source={status.get('source','')} errors={len(status.get('provider_errors', []))}")


def validate_coverage(team_cov: pd.DataFrame, exposure: pd.DataFrame, scheduled_teams: set[str]) -> dict:
    if "team" not in team_cov.columns:
        raise RuntimeError("cb_coverage_team missing team")
    tc = team_cov.copy()
    tc["team"] = tc["team"].map(canon_team)
    if set(tc["team"]) != scheduled_teams:
        raise RuntimeError("Coverage v2 team artifact does not exactly match active scheduled teams")
    if tc.duplicated("team").any():
        raise RuntimeError("Coverage v2 team artifact has duplicate team rows")
    if exposure.empty:
        raise RuntimeError("WR-CB exposure artifact is empty")
    required = {"player", "team", "opponent"}
    missing = required - set(exposure.columns)
    if missing:
        raise RuntimeError(f"wr_cb_exposure missing columns: {sorted(missing)}")
    exp = exposure.copy()
    exp["team"] = exp["team"].map(canon_team)
    exp["opponent"] = exp["opponent"].map(canon_team)
    if exp[["team", "opponent"]].eq("").any().any():
        raise RuntimeError("WR-CB exposure has unresolved team/opponent")
    if not set(exp["team"]).issubset(scheduled_teams):
        raise RuntimeError("WR-CB exposure contains players from non-scheduled teams")
    available = int(pd.to_numeric(tc.get("coverage_available", 0), errors="coerce").fillna(0).ne(0).sum())
    direct = int(pd.to_numeric(exp.get("matchup_available", 0), errors="coerce").fillna(0).ne(0).sum())
    status = "ready" if available else "structurally_ready_scheme_optional_unavailable"
    return _result("coverage_v2", status, f"team_scheme={available}/{len(tc)} direct_matchups={direct}/{len(exp)} wr_rows={len(exp)}")


def validate_player_history(df: pd.DataFrame, season: int, prior_season: int, week: int) -> dict:
    required = {"season", "week", "player", "team", "player_identity_key"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"player_game_logs missing columns: {sorted(missing)}")
    if df.empty:
        raise RuntimeError("player_game_logs is empty")
    s = pd.to_numeric(df["season"], errors="coerce")
    w = pd.to_numeric(df["week"], errors="coerce")
    if not s.eq(int(prior_season)).any():
        raise RuntimeError(f"player history contains no prior-season rows for {prior_season}")
    future = s.gt(int(season)) | (s.eq(int(season)) & w.ge(int(week)))
    if future.any():
        sample = df.loc[future, ["season", "week", "player", "team"]].head(10).to_dict("records")
        raise RuntimeError(f"player_game_logs contains same/future-week rows: {sample}")
    keys = df["player_identity_key"].astype("string").fillna("").str.strip()
    if keys.eq("").any():
        raise RuntimeError("player_game_logs contains unresolved player identities")
    return _result("nflverse_player_history", "ready", f"rows={len(df)} prior_season={prior_season}")


def validate_identity_summary(df: pd.DataFrame) -> dict:
    if not {"metric", "value"}.issubset(df.columns):
        raise RuntimeError("player_identity_validation missing metric/value")
    vals = {str(r.metric): int(r.value) for r in df.itertuples()}
    players = int(vals.get("slate_players", 0))
    stable = int(vals.get("stable_gsis", 0))
    temporary = int(vals.get("temporary_new_or_unmapped", 0))
    if players <= 0:
        raise RuntimeError("Player Identity v3 validation reports zero slate players")
    if stable <= 0:
        raise RuntimeError("Player Identity v3 resolved zero stable GSIS identities")
    temp_rate = temporary / players
    if temp_rate > 0.50:
        raise RuntimeError(f"Player Identity v3 temporary identity rate is implausibly high: {temp_rate:.1%}")
    status = "ready" if temp_rate <= 0.15 else "ready_high_temporary_identity_share"
    return _result("player_identity_v3", status, f"players={players} stable={stable} temporary={temporary} temp_rate={temp_rate:.1%}")


def validate_live_odds(props: pd.DataFrame, odds_game: pd.DataFrame) -> dict:
    if props.empty or odds_game.empty:
        raise RuntimeError("live odds mode enabled but props/game odds artifact is empty")
    if "player" not in props.columns:
        raise RuntimeError("live props artifact missing player")
    if props["player"].astype("string").fillna("").str.strip().eq("").any():
        raise RuntimeError("live props contain blank player names")
    return _result("odds_api", "ready", f"prop_rows={len(props)} game_odds_rows={len(odds_game)}")


def run_provider_readiness(
    season: int,
    prior_season: int,
    week: int,
    *,
    live_odds_enabled: bool,
    data_dir: Path = DATA,
    outputs_dir: Path = OUTPUTS,
) -> pd.DataFrame:
    results: list[dict] = []
    schedule = _read_csv(data_dir / "team_week_map.csv")
    result, scheduled_teams = validate_schedule(schedule, season, week)
    results.append(result)
    results.append(validate_ourlads(_read_csv(data_dir / "roles_ourlads.csv")))
    results.append(validate_weather(_read_csv(data_dir / "weather_week.csv"), scheduled_teams, season, week))

    status_path = data_dir / "injuries_source_status.json"
    if not status_path.exists() or status_path.stat().st_size == 0:
        raise RuntimeError("injuries_source_status.json missing; injury provider health is unknown")
    injury_status = json.loads(status_path.read_text(encoding="utf-8"))
    injuries = _read_csv(data_dir / "injuries.csv", required=False)
    results.append(validate_injuries(injuries, injury_status, season, week))

    results.append(validate_coverage(
        _read_csv(data_dir / "cb_coverage_team.csv"),
        _read_csv(data_dir / "wr_cb_exposure.csv"),
        scheduled_teams,
    ))
    results.append(validate_player_history(
        _read_csv(data_dir / "player_game_logs.csv"), season, prior_season, week
    ))
    results.append(validate_identity_summary(_read_csv(data_dir / "player_identity_validation.csv")))

    if live_odds_enabled:
        props_path = outputs_dir / "props_raw.csv"
        if not props_path.exists() or props_path.stat().st_size == 0:
            props_path = data_dir / "props_raw.csv"
        results.append(validate_live_odds(_read_csv(props_path), _read_csv(outputs_dir / "odds_game.csv")))
    else:
        results.append(_result("odds_api", "skipped_no_credit_mode", "FETCH_LIVE_ODDS=false"))

    return pd.DataFrame(results)


def main() -> int:
    season = int(resolve_season())
    prior = int(resolve_prior_season())
    week = int(resolve_week())
    live = os.getenv("FETCH_LIVE_ODDS", "false").strip().lower() in {"1", "true", "yes", "on"}
    summary = run_provider_readiness(season, prior, week, live_odds_enabled=live)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT, index=False)
    print(f"[provider_readiness_v3] season={season} week={week} live_odds={live}")
    for row in summary.itertuples(index=False):
        print(f"[provider_readiness_v3] {row.provider}: {row.status} | {row.detail}")
    print(f"[provider_readiness_v3] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
