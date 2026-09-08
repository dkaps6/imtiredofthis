#!/usr/bin/env python3
"""Fail-closed semantic audit for Full Slate artifacts before PlayerForm/modeling.

A successful GitHub step is not evidence that a data product is usable. This gate
validates critical pre-model artifacts and distinguishes execution readiness from
fully available optional provider features. Structural failures raise; declared
limitations remain visible and are never disguised as an unconditional PASS.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import CANON_TEAM_CODES, canon_team
from scripts.runtime_context import resolve_season, resolve_week
from scripts.team_context_v3 import GUARDED_PBP_FIELDS, PROMOTED_FIELDS

DATA = Path("data")
OUTPUTS = Path("outputs")
OUT_CSV = DATA / "full_slate_pre_model_semantic_audit.csv"
OUT_JSON = DATA / "full_slate_pre_model_semantic_audit.json"


def _read(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        if required:
            raise RuntimeError(f"required semantic-gate artifact missing/empty: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise RuntimeError(f"semantic-gate CSV parse failed path={path}: {exc}") from exc
    df.columns = [str(c).strip().lower() for c in df.columns]
    if required and df.empty:
        raise RuntimeError(f"required semantic-gate artifact has zero rows: {path}")
    return df


def _text(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.strip()


def _finite(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")
    bad: dict[str, int] = {}
    for col in cols:
        x = pd.to_numeric(df[col], errors="coerce")
        n = int((x.isna() | ~np.isfinite(x)).sum())
        if n:
            bad[col] = n
    if bad:
        raise RuntimeError(f"{label} required numeric fields contain missing/non-finite values: {bad}")


def _row(component: str, status: str, detail: str) -> dict[str, object]:
    return {"component": component, "status": status, "detail": detail}


def _schedule(season: int, week: int) -> tuple[pd.DataFrame, set[str]]:
    df = _read(DATA / "team_week_map.csv")
    need = {"season", "week", "team", "opponent"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"team_week_map missing columns: {sorted(miss)}")
    s = pd.to_numeric(df["season"], errors="coerce")
    w = pd.to_numeric(df["week"], errors="coerce")
    x = df.loc[s.eq(season) & w.eq(week)].copy()
    x["team"] = x["team"].map(canon_team)
    x["opponent"] = x["opponent"].map(canon_team)
    if x.empty or x["team"].eq("").any() or x["opponent"].eq("").any():
        raise RuntimeError("active team-week map empty or has unresolved team identity")
    if x.duplicated("team").any():
        raise RuntimeError("active team-week map contains duplicate team rows")
    teams = set(x["team"])
    if week == 1 and teams != set(CANON_TEAM_CODES):
        raise RuntimeError(f"Week 1 schedule must contain all 32 teams; got {len(teams)}")
    if not teams.issubset(CANON_TEAM_CODES) or len(teams) < 24 or len(teams) % 2:
        raise RuntimeError(f"active scheduled team set implausible: {len(teams)}")
    opp = dict(zip(x["team"], x["opponent"]))
    asymmetric = [t for t, o in opp.items() if opp.get(o) != t]
    if asymmetric:
        raise RuntimeError(f"opponent mapping is not symmetric: {asymmetric[:10]}")
    return x, teams


def audit(season: int, week: int, *, live_odds_enabled: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    schedule, teams = _schedule(season, week)
    rows.append(_row("schedule", "PASS", f"teams={len(teams)} games={len(teams)//2}"))

    roles = _read(DATA / "roles_ourlads.csv")
    need = {"team", "player", "player_clean_key", "position", "role"}
    miss = need - set(roles.columns)
    if miss:
        raise RuntimeError(f"Ourlads missing columns: {sorted(miss)}")
    roles["team"] = roles["team"].map(canon_team)
    if set(roles["team"]) != set(CANON_TEAM_CODES):
        raise RuntimeError(f"Ourlads must cover 32 teams; got {roles['team'].nunique()}")
    if _text(roles["player"]).eq("").any() or _text(roles["player_clean_key"]).eq("").any():
        raise RuntimeError("Ourlads contains blank player identity")
    if roles.duplicated(["team", "player_clean_key"]).any():
        raise RuntimeError("Ourlads contains duplicate team/player identities")
    rows.append(_row("ourlads", "PASS", f"rows={len(roles)} teams=32 players={roles['player_clean_key'].nunique()}"))

    tf = _read(DATA / "team_form.csv")
    tf["team"] = tf["team"].map(canon_team)
    if len(tf) != 32 or set(tf["team"]) != set(CANON_TEAM_CODES) or tf.duplicated("team").any():
        raise RuntimeError(f"TeamForm must be one row for all 32 teams; rows={len(tf)} teams={tf['team'].nunique()}")
    _finite(tf, list(GUARDED_PBP_FIELDS), "TeamForm guarded PBP")
    legacy_all_null = [c for c in tf.columns if tf[c].isna().all()]
    rows.append(_row(
        "team_form",
        "WARN_OPTIONAL_FEATURES_UNAVAILABLE" if legacy_all_null else "PASS",
        f"rows=32 guarded_fields_complete=1 all_null_optional_columns={legacy_all_null}",
    ))

    qb = _read(DATA / "qb_promoted_team_context.csv")
    qb["team"] = qb["team"].map(canon_team)
    if len(qb) != 32 or set(qb["team"]) != set(CANON_TEAM_CODES) or qb.duplicated("team").any():
        raise RuntimeError(f"promoted QB context must cover exactly 32 teams; rows={len(qb)} teams={qb['team'].nunique()}")
    _finite(qb, list(PROMOTED_FIELDS), "M89/M90 promoted QB context")
    conv = pd.to_numeric(qb["pass_attempts_per_dropback"], errors="coerce")
    if not conv.between(0.50, 1.0, inclusive="both").all():
        raise RuntimeError("promoted QB attempt conversion outside [0.50,1.00]")
    rows.append(_row("qb_m89_m90_context", "PASS", "rows=32 promoted_fields_complete=1"))

    weather = _read(DATA / "weather_week.csv")
    need = {"season", "week", "home", "away", "forecast_ok"}
    miss = need - set(weather.columns)
    if miss:
        raise RuntimeError(f"weather missing columns: {sorted(miss)}")
    ws = pd.to_numeric(weather["season"], errors="coerce")
    ww = pd.to_numeric(weather["week"], errors="coerce")
    wx = weather.loc[ws.eq(season) & ww.eq(week)].copy()
    wx["home"] = wx["home"].map(canon_team)
    wx["away"] = wx["away"].map(canon_team)
    if len(wx) != len(teams)//2 or (set(wx["home"]) | set(wx["away"])) != teams:
        raise RuntimeError("weather artifact does not exactly cover the active schedule")
    forecast_ok = int(pd.to_numeric(wx["forecast_ok"], errors="coerce").fillna(0).ne(0).sum())
    rows.append(_row(
        "weather", "PASS" if forecast_ok == len(wx) else "WARN_FORECAST_PARTIAL",
        f"games={len(wx)} forecast_ok={forecast_ok}/{len(wx)}",
    ))

    status_path = DATA / "injuries_source_status.json"
    if not status_path.exists() or status_path.stat().st_size <= 0:
        raise RuntimeError("injury source status missing")
    injury_status = json.loads(status_path.read_text(encoding="utf-8"))
    if int(injury_status.get("season", -1)) != season or int(injury_status.get("week", -1)) != week:
        raise RuntimeError("injury provider status is stale/wrong runtime")
    state = str(injury_status.get("state", ""))
    injuries = _read(DATA / "injuries.csv", required=False)
    if state == "official_report":
        if injuries.empty:
            raise RuntimeError("injury provider says official_report but injuries.csv has zero rows")
        need = {"player", "team", "season", "week", "report_available"}
        miss = need - set(injuries.columns)
        if miss:
            raise RuntimeError(f"injuries.csv missing columns: {sorted(miss)}")
        injury_teams = injuries["team"].map(canon_team)
        if _text(injuries["player"]).eq("").any() or injury_teams.eq("").any():
            raise RuntimeError("official injury report contains blank player/team identity")
        if not set(injury_teams).issubset(teams):
            raise RuntimeError("official injury report contains non-scheduled team identity")
        if not pd.to_numeric(injuries["season"], errors="coerce").eq(season).all() or not pd.to_numeric(injuries["week"], errors="coerce").eq(week).all():
            raise RuntimeError("injury rows contain stale season/week")
        scope_proven = bool(injury_status.get("all_scheduled_teams_checked", False))
        if scope_proven:
            scope = _read(DATA / "injury_team_scope.csv")
            if set(scope["team"].map(canon_team)) != teams or len(scope) != len(teams):
                raise RuntimeError("injury source claims full scope but scope ledger does not match schedule")
            rows.append(_row(
                "injuries", "PASS_REPORT_SCOPE_CERTIFIED",
                f"official_rows={len(injuries)} teams_with_rows={injury_teams.nunique()} teams_checked={len(scope)} "
                f"explicit_no_injuries={int(scope['scope_state'].astype(str).eq('NO_INJURIES_REPORTED_BY_SOURCE').sum())} "
                f"source={injury_status.get('source','')}",
            ))
        else:
            rows.append(_row(
                "injuries", "WARN_PARTIAL_SCOPE_UNPROVEN",
                f"official_rows={len(injuries)} teams_with_rows={injury_teams.nunique()}/{len(teams)} source={injury_status.get('source','')}",
            ))
    elif state == "no_official_report":
        if not injuries.empty:
            raise RuntimeError("injury provider says no_official_report but injuries.csv contains rows")
        rows.append(_row("injuries", "PASS_NO_OFFICIAL_REPORT", "rows=0"))
    else:
        raise RuntimeError(f"injury provider state is not production-usable: {state}")

    cov = _read(DATA / "cb_coverage_team.csv")
    cov["team"] = cov["team"].map(canon_team)
    if len(cov) != len(teams) or set(cov["team"]) != teams or cov.duplicated("team").any():
        raise RuntimeError("team coverage artifact does not exactly cover active teams")
    avail = pd.to_numeric(cov.get("coverage_available", 0), errors="coerce").fillna(0)
    if not avail.eq(1).all():
        raise RuntimeError(f"team coverage unavailable for {int(avail.ne(1).sum())} active teams")
    exposure = _read(DATA / "wr_cb_exposure.csv")
    need = {"player", "team", "opponent", "team_coverage_available", "matchup_available"}
    miss = need - set(exposure.columns)
    if miss:
        raise RuntimeError(f"WR-CB exposure missing columns: {sorted(miss)}")
    exposure["team"] = exposure["team"].map(canon_team)
    exposure["opponent"] = exposure["opponent"].map(canon_team)
    if exposure[["team", "opponent"]].eq("").any().any():
        raise RuntimeError("WR-CB exposure contains unresolved team/opponent identity")
    if not set(exposure["team"]).issubset(teams):
        raise RuntimeError("WR-CB exposure contains off-slate teams")
    team_cov_rows = int(pd.to_numeric(exposure["team_coverage_available"], errors="coerce").fillna(0).eq(1).sum())
    direct_flags = pd.to_numeric(exposure["matchup_available"], errors="coerce").fillna(0)
    direct = int(direct_flags.eq(1).sum())
    if direct == 0 and not direct_flags.eq(0).all():
        raise RuntimeError("WR-CB direct matchup flags contain invalid unavailable states")
    rows.append(_row(
        "coverage_v2",
        "PASS_WITH_DIRECT_MATCHUPS" if direct > 0 else "WARN_DIRECT_MATCHUP_UNAVAILABLE_GATED",
        f"team_rows={len(cov)} wr_rows={len(exposure)} team_coverage_rows={team_cov_rows} direct_matchup_rows={direct}",
    ))

    if live_odds_enabled:
        live_path = DATA / "live_odds_status.json"
        if not live_path.exists() or live_path.stat().st_size <= 0:
            raise RuntimeError("live odds enabled but live_odds_status.json missing")
        live = json.loads(live_path.read_text(encoding="utf-8"))
        if not bool(live.get("available")):
            rows.append(_row("live_odds", "PASS_NO_MARKETS", f"status={live.get('status')}"))
        else:
            if live.get("artifact_hardening_disposition") != "LIVE_ODDS_ARTIFACTS_SEMANTICALLY_HARDENED":
                raise RuntimeError(f"live odds semantic hardening not confirmed: {live.get('artifact_hardening_disposition')}")
            if live.get("core_prop_identity_disposition") != "LIVE_PROP_IDENTITY_READY":
                raise RuntimeError(f"live core prop identity not ready: {live.get('core_prop_identity_disposition')}")
            if int(live.get("core_prop_identity_unresolved_rows", -1)) != 0:
                raise RuntimeError("live core prop identity has unresolved real rows")
            props = _read(OUTPUTS / "props_raw.csv")
            placeholder = pd.to_numeric(props.get("bookmaker_missing", 0), errors="coerce").fillna(0).eq(1)
            actual = props.loc[~placeholder].copy()
            if actual.empty:
                raise RuntimeError("live odds status says available but compact props have zero actual rows")
            player_col = next((c for c in ("canonical_player_name", "player_canonical", "player") if c in actual.columns), None)
            if player_col is None:
                raise RuntimeError("live compact props have no canonical player column")
            if _text(actual[player_col]).eq("").any():
                raise RuntimeError("actual live prop rows contain blank canonical player")
            for col in ("team_abbr", "opponent_abbr", "event_id", "market"):
                if col not in actual.columns or _text(actual[col]).eq("").any():
                    raise RuntimeError(f"actual live prop rows contain missing {col}")
            if actual.duplicated().any():
                raise RuntimeError("compact live prop artifact contains exact duplicate rows after hardening")
            rows.append(_row(
                "live_odds", "PASS",
                f"actual_rows={len(actual)} core_rows={live.get('core_prop_identity_rows',0)} hardening=1 unresolved=0 duplicates_removed={live.get('raw_artifact_exact_duplicates_removed',0)}",
            ))
    else:
        rows.append(_row("live_odds", "SKIP_NO_CREDIT_MODE", "FETCH_LIVE_ODDS=false"))
    return pd.DataFrame(rows)


def main() -> int:
    season = int(resolve_season())
    week = int(resolve_week())
    live = os.getenv("FETCH_LIVE_ODDS", "false").strip().lower() in {"1", "true", "yes", "on"}
    try:
        out = audit(season, week, live_odds_enabled=live)
        warnings = out["status"].astype(str).str.startswith("WARN")
        disposition = (
            "FULL_SLATE_PRE_MODEL_EXECUTION_READY_WITH_DECLARED_LIMITATIONS"
            if warnings.any() else "FULL_SLATE_PRE_MODEL_SEMANTIC_GATE_PASS"
        )
        fatal_error = ""
    except Exception as exc:
        out = pd.DataFrame([_row("semantic_gate", "FAIL", str(exc))])
        disposition = "FULL_SLATE_PRE_MODEL_SEMANTIC_GATE_FAIL"
        fatal_error = str(exc)
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT_CSV, index=False)
        OUT_JSON.write_text(json.dumps({
            "disposition": disposition, "season": season, "week": week,
            "live_odds_enabled": live, "fatal_error": fatal_error,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(out.to_string(index=False))
        raise

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    payload = {
        "disposition": disposition,
        "season": season,
        "week": week,
        "live_odds_enabled": live,
        "fatal_error": fatal_error,
        "components": out.to_dict("records"),
        "warning_components": out.loc[out["status"].astype(str).str.startswith("WARN"), "component"].tolist(),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(out.to_string(index=False))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
