#!/usr/bin/env python3
"""Restore whole-game football state from a pinned pre-T75 Full Slate artifact.

Replay-only operational seam. A later canonical run can cross the T-75 official
inactive boundary after the paid snapshot was acquired. If that later run
withholds an otherwise-scheduled game only because required official inactive
sections are missing, this seam may restore the exact football-only state from
the pinned source run provided that source certified the whole game as
NOT_YET_REQUIRED before T-75.

This never reads sportsbook files, never changes odds, and never restores games
that have kicked off.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team

DATA = Path("data")
AUDIT = DATA / "preserved_pret75_game_state_audit.json"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"required artifact missing/empty: {path}")
    return pd.read_csv(path, low_memory=False)


def _bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _game_key(row) -> tuple[str, str]:
    return tuple(sorted((canon_team(row["away_team"]), canon_team(row["home_team"]))))


def _replace_team_rows(current: pd.DataFrame, source: pd.DataFrame, teams: set[str], *, label: str) -> pd.DataFrame:
    if "team" not in current.columns or "team" not in source.columns:
        raise RuntimeError(f"{label} missing team column")
    cur = current.copy()
    src = source.copy()
    cur["team"] = cur["team"].map(canon_team)
    src["team"] = src["team"].map(canon_team)
    src_rows = src[src["team"].isin(teams)].copy()
    missing = teams - set(src_rows["team"].dropna().astype(str))
    if missing:
        raise RuntimeError(f"{label} source missing restored teams: {sorted(missing)}")
    cols = list(dict.fromkeys(list(cur.columns) + list(src_rows.columns)))
    cur = cur.reindex(columns=cols)
    src_rows = src_rows.reindex(columns=cols)
    out = pd.concat([cur[~cur["team"].isin(teams)], src_rows], ignore_index=True)
    return out


def restore(source_root: Path, source_run_id: int) -> dict:
    current_cert_path = DATA / "current_player_availability_game_certification.csv"
    current_meta_path = DATA / "current_player_availability_game_certification.json"
    current_avail_path = DATA / "current_player_availability.csv"
    current_roles_path = DATA / "roles_current_production_eligible_v1.csv"
    current_roles_status_path = DATA / "roles_current_production_eligible_v1_status.json"

    source_cert_path = source_root / "data/current_player_availability_game_certification.csv"
    source_meta_path = source_root / "data/current_player_availability_game_certification.json"
    source_avail_path = source_root / "data/current_player_availability.csv"
    source_roles_path = source_root / "data/roles_current_production_eligible_v1.csv"

    cur_cert = _read(current_cert_path)
    src_cert = _read(source_cert_path)
    cur_avail = _read(current_avail_path)
    src_avail = _read(source_avail_path)
    cur_roles = _read(current_roles_path)
    src_roles = _read(source_roles_path)

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

    for idx, row in cur_cert.iterrows():
        state = str(row.get("certification_state", ""))
        eligible = bool(_bool_series(pd.Series([row.get("production_eligible")])).iloc[0])
        if state != "REQUIRED_MISSING_FAIL_CLOSED" or eligible:
            continue

        kickoff = pd.to_datetime(row["kickoff_utc"], utc=True, errors="coerce")
        if pd.isna(kickoff) or kickoff <= now:
            continue

        key = _game_key(row)
        src = src_by_game.get(key)
        if src is None:
            raise RuntimeError(f"withheld game absent from pinned source certification: {key}")
        src_eligible = bool(_bool_series(pd.Series([src.get("production_eligible")])).iloc[0])
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

        preserved = src.copy()
        preserved["certification_state"] = "PRESERVED_PRE_T75_REPLAY"
        preserved["production_eligible"] = True
        preserved["failure_reason"] = ""
        preserved["replay_source_run_id"] = int(source_run_id)
        preserved["replay_source_certification_state"] = src_state
        preserved["replay_restored_at_utc"] = now.isoformat()
        for col in preserved.index:
            if col not in cur_cert.columns:
                cur_cert[col] = pd.NA
        cur_cert.loc[idx, preserved.index] = preserved.values

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

    # Restore the exact football-only player availability and eligible-role rows
    # for every restored team from the pinned pre-T75 run.
    cur_avail = _replace_team_rows(cur_avail, src_avail, restored_teams, label="availability")
    cur_roles = _replace_team_rows(cur_roles, src_roles, restored_teams, label="eligible roles")

    # Whole-game invariant: both teams must now be present and every restored
    # game must be production eligible.
    role_teams = set(cur_roles["team"].map(canon_team).dropna().astype(str))
    if not restored_teams.issubset(role_teams):
        raise RuntimeError(f"restored teams missing from eligible roles: {sorted(restored_teams-role_teams)}")
    elig = _bool_series(cur_cert["production_eligible"])
    expected_team_count = int(elig.sum()) * 2
    cert_teams = set(cur_cert.loc[elig, "away_team"]) | set(cur_cert.loc[elig, "home_team"])
    if len(cert_teams) != expected_team_count:
        raise RuntimeError("patched certification is not exactly two teams per eligible game")
    if role_teams != cert_teams:
        raise RuntimeError(
            f"patched eligible role teams differ from certification: "
            f"roles_only={sorted(role_teams-cert_teams)} cert_only={sorted(cert_teams-role_teams)}"
        )

    cur_cert.to_csv(current_cert_path, index=False)
    cur_avail.to_csv(current_avail_path, index=False)
    cur_roles.to_csv(current_roles_path, index=False)

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

    role_status = json.loads(current_roles_status_path.read_text(encoding="utf-8")) if current_roles_status_path.exists() else {}
    role_status.update({
        "output_active_rows": int(len(cur_roles)),
        "eligible_games": int(elig.sum()),
        "withheld_games": int((~elig).sum()),
        "eligible_teams": sorted(role_teams),
        "withheld_teams": sorted(withheld),
        "certification_state_counts": {str(k): int(v) for k, v in counts.items()},
        "replay_preserved_pre_t75_games": int(len(restored)),
        "replay_source_run_id": int(source_run_id),
        "sportsbook_inputs_used": 0,
    })
    current_roles_status_path.write_text(json.dumps(role_status, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = {
        "disposition": "PRESERVED_PRE_T75_GAME_STATE_RESTORED",
        "season": season,
        "week": week,
        "source_run_id": int(source_run_id),
        "restored_games": int(len(restored)),
        "restored_teams": sorted(restored_teams),
        "games": restored,
        "sportsbook_inputs_used": 0,
        "sportsbook_files_read": [],
        "player_availability_source": "pinned_pre_t75_full_slate_artifact",
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
