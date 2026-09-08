#!/usr/bin/env python3
"""Classify Full Slate data quality beyond process-level green checks.

This audit intentionally separates "the pipeline can continue" from "the input is
production-certified complete." Optional/degraded providers may remain usable for
an execution audit, but they are durable limitations and can block the final
production-clean disposition until their scope is proven.
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
    return {
        "component": component,
        "status": status,
        "certification_blocker": int(bool(blocker)),
        "detail": detail,
    }


def audit() -> dict:
    rows: list[dict] = []

    schedule = _read(DATA / "team_week_map.csv")
    active = schedule.copy()
    season = int(resolve_season())
    week = int(resolve_week())
    if not {"season", "week"}.issubset(active.columns):
        raise RuntimeError("team_week_map missing runtime season/week columns")
    active = active.loc[
        pd.to_numeric(active["season"], errors="coerce").eq(season)
        & pd.to_numeric(active["week"], errors="coerce").eq(week)
    ].copy()
    if active.empty:
        raise RuntimeError(f"team_week_map has no active rows season={season} week={week}")
    active["team"] = active["team"].map(canon_team)
    scheduled_teams = set(active["team"].dropna().astype(str))
    rows.append(_row("schedule", "CERTIFIED", f"teams={len(scheduled_teams)} games={len(scheduled_teams)//2}"))

    roles = _read(DATA / "roles_ourlads.csv")
    role_teams = set(roles["team"].map(canon_team))
    if role_teams != scheduled_teams:
        raise RuntimeError("current Ourlads roster team set does not match scheduled teams")
    rows.append(
        _row(
            "current_roster",
            "LIVE_PROVIDER_ROSTER_PRESENT",
            f"source=ourlads rows={len(roles)} teams={len(role_teams)} players={roles['player'].nunique()}",
        )
    )

    identity = _json(DATA / "player_identity_semantic_audit.json")
    suspicious = int(identity.get("temporary_possible_historical_aliases", -1))
    temp = int(identity.get("temporary_players", -1))
    temp_props = int(identity.get("temporary_sportsbook_players", -1))
    mismatches = int(identity.get("sportsbook_current_roster_mismatches", -1))
    if mismatches != 0:
        raise RuntimeError(f"sportsbook/current-roster identity mismatches remain: {mismatches}")
    rows.append(_row("sportsbook_roster_identity", "CERTIFIED", "current_roster_mismatches=0"))
    if suspicious > 0:
        rows.append(
            _row(
                "historical_player_identity",
                "REVIEW_REQUIRED_POSSIBLE_VETERAN_ALIAS",
                f"temporary={temp} temporary_with_props={temp_props} possible_historical_aliases={suspicious}",
                blocker=True,
            )
        )
    elif temp > 0:
        rows.append(
            _row(
                "historical_player_identity",
                "NEW_OR_UNMAPPED_PLAYERS_EXPLICIT",
                f"temporary={temp} temporary_with_props={temp_props} possible_historical_aliases=0",
            )
        )
    else:
        rows.append(_row("historical_player_identity", "CERTIFIED", "temporary=0"))

    live = _json(DATA / "live_odds_status.json")
    compact = _read(OUTPUTS / "props_raw_compact.csv")
    expected_compact = int(live.get("production_compact_rows", -1))
    if expected_compact != len(compact):
        raise RuntimeError(
            f"compact live-prop row count drift status={expected_compact} artifact={len(compact)}"
        )
    quarantine = _read(DATA / "live_odds_placeholder_rows.csv", required=False)
    expected_q = int(live.get("production_quarantined_rows", -1))
    if expected_q != len(quarantine):
        raise RuntimeError(
            f"quarantine evidence drift status={expected_q} artifact={len(quarantine)}"
        )
    q_reasons = (
        quarantine["quarantine_reason"].astype(str).value_counts().sort_index().astype(int).to_dict()
        if not quarantine.empty and "quarantine_reason" in quarantine.columns
        else {}
    )
    if q_reasons != {str(k): int(v) for k, v in live.get("production_quarantine_reasons", {}).items()}:
        raise RuntimeError("quarantine reason ledger does not reconcile with live_odds_status")
    rows.append(
        _row(
            "sportsbook_quarantine",
            "CERTIFIED_IMMUTABLE",
            f"compact_rows={len(compact)} quarantined_rows={len(quarantine)} reasons={q_reasons}",
        )
    )

    tf = _read(DATA / "team_form.csv")
    all_null = [c for c in tf.columns if tf[c].isna().all()]
    if all_null:
        rows.append(
            _row(
                "team_form_optional_features",
                "DECLARED_OPTIONAL_UNAVAILABLE",
                f"all_null_columns={all_null}",
            )
        )
    else:
        rows.append(_row("team_form_optional_features", "CERTIFIED", "all_null_columns=[]"))

    injury_status = _json(DATA / "injuries_source_status.json")
    injuries = _read(DATA / "injuries.csv", required=False)
    injury_state = str(injury_status.get("state", ""))
    if injury_state == "official_report":
        if injuries.empty:
            raise RuntimeError("injury status says official_report but injury artifact is empty")
        injury_teams = set(injuries["team"].map(canon_team).dropna().astype(str))
        practice_nonblank = int(
            injuries.get("practice_status", pd.Series("", index=injuries.index))
            .astype("string").fillna("").str.strip().ne("").sum()
        )
        game_status_nonblank = int(
            injuries.get("status", pd.Series("", index=injuries.index))
            .astype("string").fillna("").str.strip().ne("").sum()
        )
        body_nonblank = int(
            injuries.get("body_part", pd.Series("", index=injuries.index))
            .astype("string").fillna("").str.strip().ne("").sum()
        )
        scope_proven = bool(injury_status.get("all_scheduled_teams_checked", False))
        if not scope_proven:
            rows.append(
                _row(
                    "injuries",
                    "PARTIAL_SCOPE_NOT_PROVEN",
                    f"rows={len(injuries)} teams_with_rows={len(injury_teams)}/{len(scheduled_teams)} "
                    f"practice_status={practice_nonblank}/{len(injuries)} game_status={game_status_nonblank}/{len(injuries)} "
                    f"body_part={body_nonblank}/{len(injuries)} source={injury_status.get('source','')}",
                    blocker=True,
                )
            )
        else:
            rows.append(
                _row(
                    "injuries",
                    "CERTIFIED_REPORT_SCOPE",
                    f"rows={len(injuries)} teams_checked={len(scheduled_teams)} practice_status={practice_nonblank}",
                )
            )
    elif injury_state == "no_official_report":
        rows.append(_row("injuries", "NO_OFFICIAL_REPORT_CONFIRMED", "rows=0"))
    else:
        raise RuntimeError(f"injury provider state not usable: {injury_state}")

    team_cov = _read(DATA / "cb_coverage_team.csv")
    exposure = _read(DATA / "wr_cb_exposure.csv")
    team_available = int(
        pd.to_numeric(team_cov.get("coverage_available", 0), errors="coerce").fillna(0).eq(1).sum()
    )
    direct = int(
        pd.to_numeric(exposure.get("matchup_available", 0), errors="coerce").fillna(0).eq(1).sum()
    )
    if team_available != len(scheduled_teams):
        raise RuntimeError(
            f"team coverage incomplete teams={team_available}/{len(scheduled_teams)}"
        )
    if direct <= 0:
        rows.append(
            _row(
                "coverage_v2",
                "TEAM_COVERAGE_ONLY_DIRECT_MATCHUPS_UNAVAILABLE",
                f"team_coverage={team_available}/{len(scheduled_teams)} wr_rows={len(exposure)} direct_matchups=0",
                blocker=True,
            )
        )
    else:
        rows.append(
            _row(
                "coverage_v2",
                "CERTIFIED_WITH_DIRECT_MATCHUPS",
                f"team_coverage={team_available}/{len(scheduled_teams)} wr_rows={len(exposure)} direct_matchups={direct}",
            )
        )

    out = pd.DataFrame(rows)
    blockers = int(pd.to_numeric(out["certification_blocker"], errors="coerce").fillna(0).sum())
    limitations = int(
        (~out["status"].isin({
            "CERTIFIED", "CERTIFIED_IMMUTABLE", "CERTIFIED_REPORT_SCOPE",
            "CERTIFIED_WITH_DIRECT_MATCHUPS", "LIVE_PROVIDER_ROSTER_PRESENT",
        })).sum()
    )
    disposition = (
        "FULL_SLATE_DATA_QUALITY_NOT_CERTIFIED"
        if blockers
        else "FULL_SLATE_DATA_QUALITY_PASS_WITH_DECLARED_LIMITATIONS"
        if limitations
        else "FULL_SLATE_DATA_QUALITY_CERTIFIED"
    )
    result = {
        "disposition": disposition,
        "certification_blockers": blockers,
        "declared_limitations": limitations,
        "components": rows,
    }
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    readiness_path = DATA / "provider_readiness_v3.csv"
    if readiness_path.exists() and readiness_path.stat().st_size > 0:
        readiness = pd.read_csv(readiness_path)
        if {"provider", "status", "detail"}.issubset(readiness.columns):
            if (
                (readiness["provider"] == "injuries").any()
                and injury_state == "official_report"
                and not bool(injury_status.get("all_scheduled_teams_checked", False))
            ):
                mask = readiness["provider"].eq("injuries")
                readiness.loc[mask, "status"] = "partial_scope_not_proven"
                readiness.loc[mask, "detail"] = (
                    readiness.loc[mask, "detail"].astype(str)
                    + " | full scheduled-team report coverage not proven"
                )
            if (readiness["provider"] == "coverage_v2").any() and direct <= 0:
                mask = readiness["provider"].eq("coverage_v2")
                readiness.loc[mask, "status"] = "team_coverage_only_no_direct_matchups"
            if (readiness["provider"] == "player_identity_v3").any():
                mask = readiness["provider"].eq("player_identity_v3")
                readiness.loc[mask, "status"] = (
                    "identity_review_required" if suspicious else "new_or_unmapped_players_explicit"
                )
                readiness.loc[mask, "detail"] = (
                    readiness.loc[mask, "detail"].astype(str)
                    + f" | possible_historical_aliases={suspicious}"
                )
            readiness.to_csv(readiness_path, index=False)

    print("[full_slate_data_quality] " + json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
