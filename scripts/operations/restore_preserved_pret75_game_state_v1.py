#!/usr/bin/env python3
"""Restore whole-game eligibility from a pinned pre-T75 Full Slate artifact.

Replay-only operational seam. A later canonical run can cross the T-75 official
inactive boundary after the paid snapshot was acquired. If that later run
withholds an otherwise-scheduled game only because required official inactive
sections are missing, this seam may restore *game eligibility* from the pinned
source run provided that source certified the exact same game as
NOT_YET_REQUIRED before T-75.

Important: player-level availability is NOT rolled back. After game eligibility
is restored, production-eligible roles are rebuilt from the current
football-only active-role artifact, so any newer definitive-unavailable facts
remain authoritative.

No sportsbook file is read. No odds are changed. Kicked-off games are never
restored.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.build.build_production_eligible_active_roles_v1 import build as build_eligible_roles

DATA = Path("data")
AUDIT = DATA / "preserved_pret75_game_state_audit.json"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"required artifact missing/empty: {path}")
    return pd.read_csv(path, low_memory=False)


def _bool_value(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _game_key(row) -> tuple[str, str]:
    return tuple(sorted((canon_team(row["away_team"]), canon_team(row["home_team"]))))


def restore(source_root: Path, source_run_id: int) -> dict:
    current_cert_path = DATA / "current_player_availability_game_certification.csv"
    current_meta_path = DATA / "current_player_availability_game_certification.json"
    current_avail_path = DATA / "current_player_availability.csv"
    current_active_roles_path = DATA / "roles_ourlads_active_v1.csv"
    current_roles_path = DATA / "roles_current_production_eligible_v1.csv"
    current_roles_status_path = DATA / "roles_current_production_eligible_v1_status.json"

    source_cert_path = source_root / "data/current_player_availability_game_certification.csv"
    source_meta_path = source_root / "data/current_player_availability_game_certification.json"

    if not source_meta_path.exists() or source_meta_path.stat().st_size <= 0:
        raise RuntimeError("pinned source availability certification metadata missing")
    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
    if int(source_meta.get("sportsbook_inputs_used", -1)) != 0:
        raise RuntimeError(f"pinned source availability is not football-only: {source_meta}")

    cur_cert = _read(current_cert_path)
    src_cert = _read(source_cert_path)
    cur_avail = _read(current_avail_path)
    current_active_roles = _read(current_active_roles_path)

    for df, label in ((cur_cert, "current certification"), (src_cert, "source certification")):
        need = {"season", "week", "away_team", "home_team", "kickoff_utc", "certification_state", "production_eligible"}
        missing = need - set(df.columns)
        if missing:
            raise RuntimeError(f"{label} missing columns: {sorted(missing)}")
        df["away_team"] = df["away_team"].map(canon_team)
        df["home_team"] = df["home_team"].map(canon_team)

    if cur_cert[["season", "week"]].drop_duplicates().shape[0] != 1:
        raise RuntimeError("current certification is not a single season/week")
    season = int(pd.to_numeric(cur_cert["season"], errors="raise").iloc[0])
    week = int(pd.to_numeric(cur_cert["week"], errors="raise").iloc[0])

    src_scope = src_cert[
        pd.to_numeric(src_cert["season"], errors="coerce").eq(season)
        & pd.to_numeric(src_cert["week"], errors="coerce").eq(week)
    ].copy()
    if src_scope.empty:
        raise RuntimeError(f"source certification has no season={season} week={week}")

    src_by_game = {_game_key(r): r for _, r in src_scope.iterrows()}
    if len(src_by_game) != len(src_scope):
        raise RuntimeError("source certification contains duplicate weekly game keys")

    now = pd.Timestamp(datetime.now(timezone.utc))
    restored: list[dict] = []
    restored_teams: set[str] = set()

    # Snapshot the current player-level unavailable set so we can prove this seam
    # never weakens it.
    if not {"team", "player_clean_key", "definitive_unavailable"}.issubset(cur_avail.columns):
        raise RuntimeError("current availability missing player-level unavailability columns")
    current_unavailable = {
        (canon_team(t), str(k))
        for t, k, u in zip(
            cur_avail["team"], cur_avail["player_clean_key"], cur_avail["definitive_unavailable"]
        )
        if _bool_value(u)
    }

    for idx, row in cur_cert.iterrows():
        state = str(row.get("certification_state", ""))
        eligible = _bool_value(row.get("production_eligible"))
        if state != "REQUIRED_MISSING_FAIL_CLOSED" or eligible:
            continue

        kickoff = pd.to_datetime(row["kickoff_utc"], utc=True, errors="coerce")
        if pd.isna(kickoff) or kickoff <= now:
            continue

        key = _game_key(row)
        src = src_by_game.get(key)
        if src is None:
            raise RuntimeError(f"withheld game absent from pinned source certification: {key}")

        src_eligible = _bool_value(src.get("production_eligible"))
        src_state = str(src.get("certification_state", ""))
        src_asof = pd.to_datetime(src.get("asof_utc"), utc=True, errors="coerce")
        src_kickoff = pd.to_datetime(src.get("kickoff_utc"), utc=True, errors="coerce")
        src_minutes = pd.to_numeric(pd.Series([src.get("minutes_to_kickoff")]), errors="coerce").iloc[0]

        if not src_eligible or src_state != "NOT_YET_REQUIRED":
            raise RuntimeError(
                f"pinned source did not certify withheld game pre-T75: game={key} "
                f"state={src_state} eligible={src_eligible}"
            )
        if pd.isna(src_asof) or pd.isna(src_kickoff) or src_asof >= src_kickoff:
            raise RuntimeError(f"pinned source certification is not pre-kickoff for game={key}")
        if pd.isna(src_minutes) or float(src_minutes) <= 75.0:
            raise RuntimeError(
                f"pinned source was not outside T-75 for game={key}: minutes={src_minutes}"
            )
        if abs((src_kickoff - kickoff).total_seconds()) > 1:
            raise RuntimeError(f"kickoff drift between current and source certification for game={key}")

        # Preserve the source's pre-T75 authority, but mark this as a replay
        # restoration rather than pretending current official inactives exist.
        for col in (
            "replay_source_run_id",
            "replay_source_certification_state",
            "replay_source_asof_utc",
            "replay_restored_at_utc",
        ):
            if col not in cur_cert.columns:
                cur_cert[col] = pd.NA
        cur_cert.loc[idx, "certification_state"] = "PRESERVED_PRE_T75_REPLAY"
        cur_cert.loc[idx, "production_eligible"] = True
        cur_cert.loc[idx, "failure_reason"] = ""
        cur_cert.loc[idx, "replay_source_run_id"] = int(source_run_id)
        cur_cert.loc[idx, "replay_source_certification_state"] = src_state
        cur_cert.loc[idx, "replay_source_asof_utc"] = src_asof.isoformat()
        cur_cert.loc[idx, "replay_restored_at_utc"] = now.isoformat()

        teams = {key[0], key[1]}
        restored_teams |= teams
        restored.append({
            "game": list(key),
            "source_state": src_state,
            "source_asof_utc": src_asof.isoformat(),
            "source_minutes_to_kickoff": float(src_minutes),
            "kickoff_utc": kickoff.isoformat(),
        })

    if not restored:
        result = {
            "disposition": "NO_PRE_T75_GAME_RESTORE_REQUIRED",
            "season": season,
            "week": week,
            "source_run_id": int(source_run_id),
            "restored_games": 0,
            "restored_teams": [],
            "sportsbook_inputs_used": 0,
        }
        AUDIT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("[pret75_replay_restore] " + json.dumps(result, sort_keys=True))
        return result

    # Rebuild eligible roles from *current* football-only active roles. This is
    # the key safety rule: pinned source authorizes the game, not stale player
    # participation. Newer definitive-unavailable information remains in force.
    rebuilt_roles, role_status = build_eligible_roles(current_active_roles, cur_cert)

    role_teams = set(rebuilt_roles["team"].map(canon_team).dropna().astype(str))
    if not restored_teams.issubset(role_teams):
        raise RuntimeError(f"restored teams missing from rebuilt eligible roles: {sorted(restored_teams-role_teams)}")

    # Prove no currently definitive-unavailable player was resurrected.
    role_keys = {
        (canon_team(t), str(k))
        for t, k in zip(rebuilt_roles["team"], rebuilt_roles["player_clean_key"])
    }
    resurrected = sorted(current_unavailable & role_keys)
    if resurrected:
        raise RuntimeError(
            f"pre-T75 game restore resurrected current definitive-unavailable players: {resurrected[:20]}"
        )

    elig = cur_cert["production_eligible"].map(_bool_value)
    cert_teams = set(cur_cert.loc[elig, "away_team"]) | set(cur_cert.loc[elig, "home_team"])
    if role_teams != cert_teams:
        raise RuntimeError(
            f"rebuilt eligible role teams differ from patched certification: "
            f"roles_only={sorted(role_teams-cert_teams)} cert_only={sorted(cert_teams-role_teams)}"
        )

    cur_cert.to_csv(current_cert_path, index=False)
    rebuilt_roles.to_csv(current_roles_path, index=False)

    counts = cur_cert["certification_state"].astype(str).value_counts().to_dict()
    withheld = set(cur_cert.loc[~elig, "away_team"]) | set(cur_cert.loc[~elig, "home_team"])

    meta = json.loads(current_meta_path.read_text(encoding="utf-8")) if current_meta_path.exists() else {}
    meta.update({
        "games": int(len(cur_cert)),
        "eligible_games": int(elig.sum()),
        "withheld_games": int((~elig).sum()),
        "state_counts": {str(k): int(v) for k, v in counts.items()},
        "withheld_teams": sorted(withheld),
        "replay_preserved_pre_t75_games": int(len(restored)),
        "replay_preserved_pre_t75_teams": sorted(restored_teams),
        "replay_source_run_id": int(source_run_id),
        "sportsbook_inputs_used": 0,
    })
    current_meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    role_status.update({
        "replay_preserved_pre_t75_games": int(len(restored)),
        "replay_source_run_id": int(source_run_id),
        "newer_definitive_unavailable_preserved": int(len(current_unavailable)),
        "sportsbook_inputs_used": 0,
    })
    current_roles_status_path.write_text(json.dumps(role_status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = {
        "disposition": "PRESERVED_PRE_T75_GAME_ELIGIBILITY_RESTORED_CURRENT_PLAYER_STATE_PRESERVED",
        "season": season,
        "week": week,
        "source_run_id": int(source_run_id),
        "restored_games": int(len(restored)),
        "restored_teams": sorted(restored_teams),
        "games": restored,
        "current_definitive_unavailable_count": int(len(current_unavailable)),
        "current_definitive_unavailable_resurrected": 0,
        "sportsbook_inputs_used": 0,
        "sportsbook_files_read": [],
        "game_eligibility_source": "pinned_pre_t75_full_slate_artifact",
        "player_availability_source": "current_football_only_availability",
        "eligible_roles_source": "current_roles_ourlads_active_v1_plus_restored_game_certification",
    }
    AUDIT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[pret75_replay_restore] " + json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--source-run-id", type=int, required=True)
    args = ap.parse_args()
    restore(args.source_root, args.source_run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
