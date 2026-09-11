#!/usr/bin/env python3
"""Classify Full Slate data quality beyond process-level green checks.

Execution usability and production certification are separate. Missing optional
features are acceptable only when their unavailability is explicit and the
production path proves they are gated off rather than silently consumed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.runtime_context import resolve_season, resolve_week

DATA = Path("data")
OUTPUTS = Path("outputs")
OUT_CSV = DATA / "full_slate_data_quality_audit.csv"
OUT_JSON = DATA / "full_slate_data_quality_audit.json"


def _read(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        if required:
            raise RuntimeError(f"data-quality artifact missing/empty: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if required and df.empty:
        raise RuntimeError(f"data-quality artifact has zero rows: {path}")
    return df


def _json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"data-quality JSON missing/empty: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _row(component: str, status: str, detail: str, *, blocker: bool = False) -> dict:
    return {"component": component, "status": status, "certification_blocker": int(bool(blocker)), "detail": detail}


def _nonblank_count(series: pd.Series) -> int:
    return int(series.astype("string").fillna("").str.strip().ne("").sum())


def _validate_current_roster_scope(
    scheduled_teams: set[str],
    roles: pd.DataFrame,
    game_odds: pd.DataFrame,
) -> tuple[set[str], set[str]]:
    """Require roster coverage for the already-gated live event universe.

    The live odds gate has already removed off-week events and later rematches.
    Teams whose Week-1 games have already been played may legitimately be absent
    from the current Ourlads snapshot, so the quality classifier must not restore
    the old all-32 roster invariant. Extra roster teams are allowed only when they
    are part of the authoritative active-week schedule, and any missing live-event
    team remains fatal.
    """
    if "team" not in roles.columns:
        raise RuntimeError("current Ourlads roster missing team column")
    required_game_cols = {"home_team", "away_team"}
    missing_game_cols = required_game_cols - set(game_odds.columns)
    if missing_game_cols:
        raise RuntimeError(
            f"odds_game missing columns required for current-roster scope: {sorted(missing_game_cols)}"
        )

    role_series = roles["team"].map(canon_team).astype("string").fillna("").str.strip()
    if role_series.eq("").any():
        raise RuntimeError("current Ourlads roster contains unresolvable team identity")
    role_teams = set(role_series)

    live_event_teams: set[str] = set()
    for col in ("home_team", "away_team"):
        event_series = game_odds[col].map(canon_team).astype("string").fillna("").str.strip()
        if event_series.eq("").any():
            raise RuntimeError("live odds event scope contains unresolvable team identity")
        live_event_teams.update(event_series)
    if not live_event_teams:
        raise RuntimeError("live odds event scope contains zero teams")

    off_schedule_events = sorted(live_event_teams - scheduled_teams)
    if off_schedule_events:
        raise RuntimeError(
            f"live odds event scope contains teams outside active schedule: {off_schedule_events}"
        )
    off_schedule_roster = sorted(role_teams - scheduled_teams)
    if off_schedule_roster:
        raise RuntimeError(
            f"current Ourlads roster contains teams outside active schedule: {off_schedule_roster}"
        )
    missing_live = sorted(live_event_teams - role_teams)
    if missing_live:
        raise RuntimeError(
            f"current Ourlads roster missing live-event teams: {missing_live}"
        )

    return role_teams, live_event_teams


def _derive_positive_row_injury_scope(
    injuries: pd.DataFrame,
    scheduled_teams: set[str],
    *,
    source: str,
) -> pd.DataFrame | None:
    """Prove complete injury-source scope when every scheduled team has a row.

    Positive report rows are sufficient to prove that a team was represented by
    the source. They are *not* sufficient to infer that a team with no rows had
    no injuries. Therefore this helper certifies only exact 32/32 positive-row
    coverage; partial coverage remains unproven and fail-closed.
    """
    if injuries.empty or "team" not in injuries.columns:
        return None
    teams = injuries["team"].map(canon_team).astype("string").fillna("").str.strip()
    if teams.eq("").any():
        raise RuntimeError("injury artifact contains unresolvable team identity")
    injury_teams = set(teams)
    off_schedule = sorted(injury_teams - scheduled_teams)
    if off_schedule:
        raise RuntimeError(f"injury artifact contains teams outside active schedule: {off_schedule}")
    if injury_teams != scheduled_teams:
        return None

    counts = teams.value_counts().to_dict()
    return pd.DataFrame([
        {
            "team": team,
            "scope_state": "OFFICIAL_REPORT_ROWS",
            "injury_rows": int(counts.get(team, 0)),
            "source": str(source),
        }
        for team in sorted(scheduled_teams)
    ])


def audit() -> dict:
    rows: list[dict] = []
    season = int(resolve_season())
    week = int(resolve_week())

    schedule = _read(DATA / "team_week_map.csv")
    if not {"season", "week", "team"}.issubset(schedule.columns):
        raise RuntimeError("team_week_map missing runtime season/week/team columns")
    active = schedule.loc[
        pd.to_numeric(schedule["season"], errors="coerce").eq(season)
        & pd.to_numeric(schedule["week"], errors="coerce").eq(week)
    ].copy()
    if active.empty:
        raise RuntimeError(f"team_week_map has no active rows season={season} week={week}")
    active["team"] = active["team"].map(canon_team)
    scheduled_teams = set(active["team"].dropna().astype(str))
    if len(scheduled_teams) != 32:
        raise RuntimeError(f"active schedule must contain 32 teams; got {len(scheduled_teams)}")
    rows.append(_row("schedule", "CERTIFIED", f"teams=32 games=16"))

    roles = _read(DATA / "roles_ourlads.csv")
    game_odds = _read(OUTPUTS / "odds_game.csv")
    role_teams, live_event_teams = _validate_current_roster_scope(
        scheduled_teams,
        roles,
        game_odds,
    )
    rows.append(_row(
        "current_roster",
        "LIVE_PROVIDER_ROSTER_PRESENT",
        f"source=ourlads rows={len(roles)} teams={len(role_teams)} "
        f"live_event_teams={len(live_event_teams)} scheduled_teams=32 players={roles['player'].nunique()}",
    ))

    identity = _json(DATA / "player_identity_semantic_audit.json")
    suspicious = int(identity.get("temporary_possible_historical_aliases", -1))
    temp = int(identity.get("temporary_players", -1))
    temp_props = int(identity.get("temporary_sportsbook_players", -1))
    mismatches = int(identity.get("sportsbook_current_roster_mismatches", -1))
    if mismatches != 0:
        raise RuntimeError(f"sportsbook/current-roster identity mismatches remain: {mismatches}")
    rows.append(_row("sportsbook_roster_identity", "CERTIFIED", "current_roster_mismatches=0"))
    if suspicious > 0:
        rows.append(_row(
            "historical_player_identity", "REVIEW_REQUIRED_POSSIBLE_VETERAN_ALIAS",
            f"temporary={temp} temporary_with_props={temp_props} possible_historical_aliases={suspicious}", blocker=True,
        ))
    elif temp > 0:
        rows.append(_row(
            "historical_player_identity", "NEW_OR_UNMAPPED_PLAYERS_EXPLICIT",
            f"temporary={temp} temporary_with_props={temp_props} possible_historical_aliases=0",
        ))
    else:
        rows.append(_row("historical_player_identity", "CERTIFIED", "temporary=0"))

    live = _json(DATA / "live_odds_status.json")
    compact = _read(OUTPUTS / "props_raw_compact.csv")
    if int(live.get("production_compact_rows", -1)) != len(compact):
        raise RuntimeError("compact live-prop row count drift")
    quarantine = _read(DATA / "live_odds_placeholder_rows.csv", required=False)
    if int(live.get("production_quarantined_rows", -1)) != len(quarantine):
        raise RuntimeError("quarantine evidence drift")
    q_reasons = (
        quarantine["quarantine_reason"].astype(str).value_counts().sort_index().astype(int).to_dict()
        if not quarantine.empty and "quarantine_reason" in quarantine.columns else {}
    )
    expected_reasons = {str(k): int(v) for k, v in live.get("production_quarantine_reasons", {}).items()}
    if q_reasons != expected_reasons:
        raise RuntimeError("quarantine reason ledger does not reconcile with live_odds_status")
    rows.append(_row("sportsbook_quarantine", "CERTIFIED_IMMUTABLE", f"compact_rows={len(compact)} quarantined_rows={len(quarantine)} reasons={q_reasons}"))

    tf = _read(DATA / "team_form.csv")
    all_null = [c for c in tf.columns if tf[c].isna().all()]
    rows.append(_row(
        "team_form_optional_features",
        "DECLARED_OPTIONAL_UNAVAILABLE" if all_null else "CERTIFIED",
        f"all_null_columns={all_null}",
    ))

    # Injury certification is about source scope, not forcing every team to have
    # a player row. The NFL page can explicitly state No Injuries Reported. A
    # source with positive current-week rows for all 32 teams also proves scope
    # without making any inference about teams that have no rows.
    injury_status = _json(DATA / "injuries_source_status.json")
    injuries = _read(DATA / "injuries.csv", required=False)
    injury_state = str(injury_status.get("state", ""))
    scope_proven = False
    if injury_state == "official_report":
        if injuries.empty:
            raise RuntimeError("injury status says official_report but injury artifact is empty")
        injury_teams = set(injuries["team"].map(canon_team).dropna().astype(str))
        practice_nonblank = _nonblank_count(injuries.get("practice_status", pd.Series("", index=injuries.index)))
        game_status_nonblank = _nonblank_count(injuries.get("status", pd.Series("", index=injuries.index)))
        body_nonblank = _nonblank_count(injuries.get("body_part", pd.Series("", index=injuries.index)))
        scope_proven = bool(injury_status.get("all_scheduled_teams_checked", False))
        if scope_proven:
            scope = _read(DATA / "injury_team_scope.csv")
        else:
            scope = _derive_positive_row_injury_scope(
                injuries,
                scheduled_teams,
                source=str(injury_status.get("source", "")),
            )
            if scope is not None:
                scope_proven = True
                scope.to_csv(DATA / "injury_team_scope.csv", index=False)
                injury_status = dict(injury_status)
                injury_status.update({
                    "all_scheduled_teams_checked": True,
                    "scheduled_teams_checked": 32,
                    "teams_with_report_rows": 32,
                    "teams_explicit_no_injuries_reported": 0,
                    "scope_basis": "complete_current_week_positive_report_row_coverage",
                    "scope_ledger": str(DATA / "injury_team_scope.csv"),
                })
                (DATA / "injuries_source_status.json").write_text(
                    json.dumps(injury_status, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        if scope_proven:
            if not {"team", "scope_state"}.issubset(scope.columns):
                raise RuntimeError("injury scope ledger missing team/scope_state")
            scope_teams = set(scope["team"].map(canon_team).dropna().astype(str))
            if len(scope) != 32 or scope_teams != scheduled_teams:
                raise RuntimeError("injury scope ledger does not exactly match active scheduled teams")
            allowed = {"OFFICIAL_REPORT_ROWS", "NO_INJURIES_REPORTED_BY_SOURCE"}
            bad_states = sorted(set(scope["scope_state"].astype(str)) - allowed)
            if bad_states:
                raise RuntimeError(f"injury scope ledger contains unresolved states: {bad_states}")
            report_teams = int(scope["scope_state"].eq("OFFICIAL_REPORT_ROWS").sum())
            explicit_none = int(scope["scope_state"].eq("NO_INJURIES_REPORTED_BY_SOURCE").sum())
            rows.append(_row(
                "injuries", "CERTIFIED_REPORT_SCOPE",
                f"rows={len(injuries)} teams_checked=32 teams_with_report_rows={report_teams} "
                f"teams_explicit_no_injuries_reported={explicit_none} practice_status={practice_nonblank}/{len(injuries)} "
                f"game_status={game_status_nonblank}/{len(injuries)} body_part={body_nonblank}/{len(injuries)} "
                f"source={injury_status.get('source','')}",
            ))
        else:
            rows.append(_row(
                "injuries", "PARTIAL_SCOPE_NOT_PROVEN",
                f"rows={len(injuries)} teams_with_rows={len(injury_teams)}/32 practice_status={practice_nonblank}/{len(injuries)} "
                f"game_status={game_status_nonblank}/{len(injuries)} body_part={body_nonblank}/{len(injuries)} source={injury_status.get('source','')}",
                blocker=True,
            ))
    elif injury_state == "no_official_report":
        rows.append(_row("injuries", "NO_OFFICIAL_REPORT_CONFIRMED", "rows=0"))
    else:
        raise RuntimeError(f"injury provider state not usable: {injury_state}")

    # Direct WR/CB assignments are optional and fail-closed. Team scheme coverage
    # is production-available; direct shadow penalties activate only when the
    # row-level matchup_available flag is 1. Zero direct rows therefore means the
    # direct feature is explicitly gated off, not silently imputed.
    team_cov = _read(DATA / "cb_coverage_team.csv")
    exposure = _read(DATA / "wr_cb_exposure.csv")
    team_available = int(pd.to_numeric(team_cov.get("coverage_available", 0), errors="coerce").fillna(0).eq(1).sum())
    direct_flags = pd.to_numeric(exposure.get("matchup_available", 0), errors="coerce").fillna(0)
    direct = int(direct_flags.eq(1).sum())
    if team_available != 32:
        raise RuntimeError(f"team coverage incomplete teams={team_available}/32")
    if direct <= 0:
        if not direct_flags.eq(0).all():
            raise RuntimeError("WR/CB exposure contains nonzero/non-one matchup flags")
        payload_nonblank = {}
        for col in ("primary_cb", "wr_cb_advantage", "matchup_source", "shadow_flag"):
            if col in exposure.columns:
                payload_nonblank[col] = _nonblank_count(exposure[col])
        # Numeric shadow_flag can stringify nan safely; only meaningful payloads
        # are disallowed when matchup_available=0.
        meaningful = {
            k: v for k, v in payload_nonblank.items()
            if v > 0 and not (k == "shadow_flag" and pd.to_numeric(exposure[k], errors="coerce").notna().sum() == 0)
        }
        if meaningful:
            raise RuntimeError(f"direct WR/CB payload present while matchup_available=0: {meaningful}")
        rows.append(_row(
            "coverage_v2", "DIRECT_MATCHUP_UNAVAILABLE_GATED_OFF",
            f"team_coverage={team_available}/32 wr_rows={len(exposure)} direct_matchups=0 "
            "direct_matchup_consumption_eligible_rows=0; team_scheme_coverage_remains_available",
        ))
    else:
        rows.append(_row(
            "coverage_v2", "CERTIFIED_WITH_DIRECT_MATCHUPS",
            f"team_coverage={team_available}/32 wr_rows={len(exposure)} direct_matchups={direct}",
        ))

    out = pd.DataFrame(rows)
    blockers = int(pd.to_numeric(out["certification_blocker"], errors="coerce").fillna(0).sum())
    fully_certified = {
        "CERTIFIED", "CERTIFIED_IMMUTABLE", "CERTIFIED_REPORT_SCOPE",
        "CERTIFIED_WITH_DIRECT_MATCHUPS", "LIVE_PROVIDER_ROSTER_PRESENT",
    }
    limitations = int((~out["status"].isin(fully_certified)).sum())
    disposition = (
        "FULL_SLATE_DATA_QUALITY_NOT_CERTIFIED" if blockers
        else "FULL_SLATE_DATA_QUALITY_PASS_WITH_DECLARED_LIMITATIONS" if limitations
        else "FULL_SLATE_DATA_QUALITY_CERTIFIED"
    )
    result = {"disposition": disposition, "certification_blockers": blockers, "declared_limitations": limitations, "components": rows}
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    readiness_path = DATA / "provider_readiness_v3.csv"
    if readiness_path.exists() and readiness_path.stat().st_size > 0:
        readiness = pd.read_csv(readiness_path)
        if {"provider", "status", "detail"}.issubset(readiness.columns):
            if (readiness["provider"] == "injuries").any():
                mask = readiness["provider"].eq("injuries")
                readiness.loc[mask, "status"] = "report_scope_certified" if scope_proven else "partial_scope_not_proven"
            if (readiness["provider"] == "coverage_v2").any():
                mask = readiness["provider"].eq("coverage_v2")
                readiness.loc[mask, "status"] = "direct_matchup_unavailable_gated_off" if direct <= 0 else "direct_matchups_available"
            if (readiness["provider"] == "player_identity_v3").any():
                mask = readiness["provider"].eq("player_identity_v3")
                readiness.loc[mask, "status"] = "identity_review_required" if suspicious else "new_or_unmapped_players_explicit"
                readiness.loc[mask, "detail"] = readiness.loc[mask, "detail"].astype(str) + f" | possible_historical_aliases={suspicious}"
            readiness.to_csv(readiness_path, index=False)

    print("[full_slate_data_quality] " + json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
