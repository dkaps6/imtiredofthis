#!/usr/bin/env python3
"""Restore acquisition-time football state for replayed games that cross T-75.

A preserved paid sportsbook snapshot is tied to the football state that existed
when those prices were acquired.  If a canonical replay starts later and a
still-upcoming game has crossed the T-75 official-inactives boundary, the
current availability gate can correctly fail closed even though the paid
snapshot was captured while that game was still eligible.

For preserved-odds replay only, this operation restores the *whole game's*
availability/eligible-role state from the exact paid source artifact when all of
these are true:
  * source game was production eligible;
  * source snapshot was >= 75 minutes before kickoff;
  * current game is still pre-kickoff;
  * current game is withheld only because official inactive sections are
    required but missing;
  * game identity, matchup and kickoff are unchanged.

It never restores a kicked-off game, never overrides a substantive current
unavailability decision, never changes sportsbook bytes, and never performs a
provider request.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team

DATA = Path("data")
OUTPUTS = Path("outputs")
CUR_CERT = DATA / "current_player_availability_game_certification.csv"
CUR_AVAIL = DATA / "current_player_availability.csv"
CUR_ROLES = DATA / "roles_current_production_eligible_v1.csv"
AUDIT = DATA / "preserved_pregame_game_lock_audit.json"
CERT_META = DATA / "current_player_availability_game_certification.json"
AVAIL_STATUS = DATA / "current_player_availability_status.json"

LOCK_STATE = "PRESERVED_PAID_ACQUISITION_LOCK"
CURRENT_WITHHELD_STATE = "REQUIRED_MISSING_FAIL_CLOSED"
MIN_SOURCE_MINUTES = 75.0


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"required file missing/empty: {path}")
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise RuntimeError(f"required file has zero rows: {path}")
    return df


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _game_key(row: pd.Series) -> tuple[str, str, str, str]:
    return (
        str(row["game_id"]),
        canon_team(row["away_team"]),
        canon_team(row["home_team"]),
        str(row["kickoff_utc"]),
    )


def _missing_sections_only(reason: object, away: str, home: str) -> bool:
    text = str(reason or "").strip()
    if not text or text.lower() == "nan":
        return False
    pieces = {p.strip() for p in text.split("|") if p.strip()}
    expected = {f"missing_complete_section:{away}", f"missing_complete_section:{home}"}
    return pieces == expected


def _validate_schema(cur_cert: pd.DataFrame, src_cert: pd.DataFrame) -> None:
    required = {
        "season", "week", "game_id", "away_team", "home_team", "kickoff_utc",
        "asof_utc", "minutes_to_kickoff", "official_required",
        "away_official_section_complete", "home_official_section_complete",
        "official_snapshot_asof_utc", "certification_state",
        "production_eligible", "failure_reason",
    }
    for label, df in (("current cert", cur_cert), ("source cert", src_cert)):
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(f"{label} missing columns: {sorted(missing)}")


def restore(source_root: Path, source_run_id: int, source_artifact_id: int) -> dict:
    source_data = source_root / "data"
    src_cert = _read(source_data / "current_player_availability_game_certification.csv")
    src_avail = _read(source_data / "current_player_availability.csv")
    src_roles = _read(source_data / "roles_current_production_eligible_v1.csv")
    cur_cert = _read(CUR_CERT)
    cur_avail = _read(CUR_AVAIL)
    cur_roles = _read(CUR_ROLES)

    _validate_schema(cur_cert, src_cert)

    src_by_game = {str(r["game_id"]): r for _, r in src_cert.iterrows()}
    locked_game_ids: list[str] = []
    locked_teams: set[str] = set()
    evidence: list[dict] = []

    for idx, cur in cur_cert.iterrows():
        game_id = str(cur["game_id"])
        src = src_by_game.get(game_id)
        if src is None:
            continue

        # Exact event identity must be stable across source/current snapshots.
        if _game_key(cur) != _game_key(src):
            raise RuntimeError(
                f"preserved replay game identity drift for {game_id}: "
                f"current={_game_key(cur)} source={_game_key(src)}"
            )

        source_eligible = bool(_truthy(pd.Series([src["production_eligible"]])).iloc[0])
        current_eligible = bool(_truthy(pd.Series([cur["production_eligible"]])).iloc[0])
        if current_eligible or not source_eligible:
            continue

        away = canon_team(cur["away_team"])
        home = canon_team(cur["home_team"])
        cur_minutes = pd.to_numeric(pd.Series([cur["minutes_to_kickoff"]]), errors="coerce").iloc[0]
        src_minutes = pd.to_numeric(pd.Series([src["minutes_to_kickoff"]]), errors="coerce").iloc[0]

        # Never revive a game that has kicked off or whose acquisition-time
        # snapshot was already inside the official-inactives window.
        if pd.isna(cur_minutes) or float(cur_minutes) <= 0:
            continue
        if pd.isna(src_minutes) or float(src_minutes) < MIN_SOURCE_MINUTES:
            continue
        if str(src["certification_state"]) != "NOT_YET_REQUIRED":
            continue
        if bool(src["official_required"]):
            continue

        # The only acceptable current blocker is the mechanical T-75
        # missing-official-section state for both teams.
        if str(cur["certification_state"]) != CURRENT_WITHHELD_STATE:
            continue
        if not bool(cur["official_required"]):
            continue
        if not _missing_sections_only(cur["failure_reason"], away, home):
            continue

        source_team_roles = src_roles[src_roles["team"].map(canon_team).isin({away, home})].copy()
        source_team_avail = src_avail[src_avail["team"].map(canon_team).isin({away, home})].copy()
        if source_team_roles.empty or set(source_team_roles["team"].map(canon_team)) != {away, home}:
            raise RuntimeError(f"source eligible roles incomplete for preserved game {game_id}")
        if source_team_avail.empty or set(source_team_avail["team"].map(canon_team)) != {away, home}:
            raise RuntimeError(f"source availability incomplete for preserved game {game_id}")

        # Restore the exact source certification row, but stamp a distinct
        # replay lock state so downstream consumers cannot mistake it for a
        # freshly certified current game.
        replacement = src.copy()
        replacement["certification_state"] = LOCK_STATE
        replacement["production_eligible"] = True
        replacement["failure_reason"] = ""
        cur_cert.loc[idx, :] = replacement[cur_cert.columns].values

        locked_game_ids.append(game_id)
        locked_teams.update({away, home})
        evidence.append({
            "game_id": game_id,
            "away_team": away,
            "home_team": home,
            "kickoff_utc": str(cur["kickoff_utc"]),
            "source_asof_utc": str(src["asof_utc"]),
            "source_minutes_to_kickoff": float(src_minutes),
            "current_asof_utc": str(cur["asof_utc"]),
            "current_minutes_to_kickoff": float(cur_minutes),
            "current_failure_reason": str(cur["failure_reason"]),
        })

    if not locked_game_ids:
        result = {
            "disposition": "NO_PREGAME_ACQUISITION_LOCK_NEEDED",
            "source_run_id": int(source_run_id),
            "source_artifact_id": int(source_artifact_id),
            "locked_game_ids": [],
            "locked_teams": [],
            "provider_requests_used": False,
            "sportsbook_inputs_changed": False,
        }
        AUDIT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("[preserved_pregame_lock] " + json.dumps(result, sort_keys=True))
        return result

    team_mask_avail = cur_avail["team"].map(canon_team).isin(locked_teams)
    cur_avail = cur_avail.loc[~team_mask_avail].copy()
    src_locked_avail = src_avail[src_avail["team"].map(canon_team).isin(locked_teams)].copy()
    cur_avail = pd.concat([cur_avail, src_locked_avail], ignore_index=True, sort=False)

    team_mask_roles = cur_roles["team"].map(canon_team).isin(locked_teams)
    cur_roles = cur_roles.loc[~team_mask_roles].copy()
    src_locked_roles = src_roles[src_roles["team"].map(canon_team).isin(locked_teams)].copy()
    cur_roles = pd.concat([cur_roles, src_locked_roles], ignore_index=True, sort=False)

    # Strong invariants: complete-game restoration, no duplicate player/team
    # identity, and eligible-game/team count parity.
    if cur_roles.duplicated(["team", "player_clean_key"]).any():
        dup = cur_roles.loc[
            cur_roles.duplicated(["team", "player_clean_key"], keep=False),
            ["team", "player", "player_clean_key"],
        ].head(20).to_dict("records")
        raise RuntimeError(f"duplicate roles after preserved game lock: {dup}")

    eligible_games = int(_truthy(cur_cert["production_eligible"]).sum())
    eligible_teams = set(cur_roles["team"].map(canon_team))
    if len(eligible_teams) != eligible_games * 2:
        raise RuntimeError(
            f"preserved game lock eligible universe mismatch: "
            f"games={eligible_games} role_teams={len(eligible_teams)}"
        )
    if not locked_teams.issubset(eligible_teams):
        raise RuntimeError("preserved game lock did not restore all locked teams to eligible roles")

    cur_cert.to_csv(CUR_CERT, index=False)
    cur_avail.to_csv(CUR_AVAIL, index=False)
    cur_roles.to_csv(CUR_ROLES, index=False)

    # Keep the companion status ledgers internally consistent with the
    # acquisition-time lock.  The distinct state makes the provenance visible
    # while production_eligible remains mechanically coherent.
    old_cert_meta = {}
    if CERT_META.exists() and CERT_META.stat().st_size:
        old_cert_meta = json.loads(CERT_META.read_text(encoding="utf-8"))
    eligible_mask = _truthy(cur_cert["production_eligible"])
    withheld_rows = cur_cert.loc[~eligible_mask]
    withheld_meta_teams = sorted({
        canon_team(t)
        for col in ("away_team", "home_team")
        for t in withheld_rows[col]
        if canon_team(t)
    })
    cert_meta = {
        "asof_utc": old_cert_meta.get("asof_utc", ""),
        "eligible_games": int(eligible_mask.sum()),
        "games": int(len(cur_cert)),
        "require_minutes_before_kickoff": old_cert_meta.get(
            "require_minutes_before_kickoff", MIN_SOURCE_MINUTES
        ),
        "sportsbook_inputs_used": 0,
        "state_counts": {
            str(k): int(v)
            for k, v in cur_cert["certification_state"].value_counts().to_dict().items()
        },
        "withheld_games": int((~eligible_mask).sum()),
        "withheld_teams": withheld_meta_teams,
        "preserved_acquisition_lock_games": sorted(locked_game_ids),
        "preserved_acquisition_lock_teams": sorted(locked_teams),
        "preserved_source_run_id": int(source_run_id),
        "preserved_source_artifact_id": int(source_artifact_id),
    }
    CERT_META.write_text(
        json.dumps(cert_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    old_status = {}
    if AVAIL_STATUS.exists() and AVAIL_STATUS.stat().st_size:
        old_status = json.loads(AVAIL_STATUS.read_text(encoding="utf-8"))
    definitive = pd.to_numeric(
        cur_avail["definitive_unavailable"], errors="coerce"
    ).fillna(0)
    complete = pd.to_numeric(
        cur_avail["official_inactive_section_complete"], errors="coerce"
    ).fillna(0)
    status_meta = dict(old_status)
    status_meta.update({
        "rows": int(len(cur_avail)),
        "teams": int(cur_avail["team"].map(canon_team).nunique()),
        "definitive_unavailable": int(definitive.sum()),
        "uncertain": int(cur_avail["final_availability_state"].astype(str).eq("UNCERTAIN").sum()),
        "unknown": int(cur_avail["final_availability_state"].astype(str).eq("UNKNOWN").sum()),
        "official_complete_teams": int(
            cur_avail.loc[complete.eq(1), "team"].map(canon_team).nunique()
        ),
        "sportsbook_inputs_used": 0,
        "preserved_acquisition_lock_games": sorted(locked_game_ids),
        "preserved_acquisition_lock_teams": sorted(locked_teams),
        "preserved_source_run_id": int(source_run_id),
        "preserved_source_artifact_id": int(source_artifact_id),
    })
    AVAIL_STATUS.write_text(
        json.dumps(status_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    cur_avail.to_csv(OUTPUTS / "current_player_availability.csv", index=False)
    cur_cert.to_csv(OUTPUTS / "current_player_availability_game_certification.csv", index=False)
    cur_roles.to_csv(OUTPUTS / "roles_current_production_eligible_v1.csv", index=False)

    result = {
        "disposition": "PRESERVED_PREGAME_ACQUISITION_LOCK_RESTORED",
        "source_run_id": int(source_run_id),
        "source_artifact_id": int(source_artifact_id),
        "locked_game_ids": sorted(locked_game_ids),
        "locked_teams": sorted(locked_teams),
        "locked_games": evidence,
        "eligible_games_after_restore": eligible_games,
        "eligible_teams_after_restore": len(eligible_teams),
        "restored_role_rows": int(len(src_locked_roles)),
        "restored_availability_rows": int(len(src_locked_avail)),
        "provider_requests_used": False,
        "sportsbook_inputs_changed": False,
        "football_state_source": "EXACT_PAID_ARTIFACT_ACQUISITION_TIME",
    }
    AUDIT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[preserved_pregame_lock] " + json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--source-run-id", required=True, type=int)
    ap.add_argument("--source-artifact-id", required=True, type=int)
    args = ap.parse_args()
    restore(Path(args.source_root), args.source_run_id, args.source_artifact_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
