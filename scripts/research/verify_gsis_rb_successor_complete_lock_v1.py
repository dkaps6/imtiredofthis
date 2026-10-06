#!/usr/bin/env python3
"""Fail-closed completeness verifier for a GSIS RB successor pregame lock.

This verifier is intentionally private-file aware but public-output safe.
It must be run only before the target games kick off, after both:
1. the private allocation lock exists; and
2. the private three-arm projection lock exists.

It emits hashes/counts only. It never emits raw GSIS cells or player identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

FORBIDDEN_PUBLIC = {
    "player", "player_clean_key", "successor_player_clean_key",
    "unavailable_player", "unavailable_player_clean_key",
}


def sha256(path: Path) -> str:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty required private file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path, label: str) -> pd.DataFrame:
    sha256(path)
    x = pd.read_csv(path, low_memory=False)
    if x.empty:
        raise RuntimeError(f"{label} has zero rows")
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allocation-lock", type=Path, required=True)
    ap.add_argument("--event-audit", type=Path, required=True)
    ap.add_argument("--projection-lock", type=Path, required=True)
    ap.add_argument("--projection-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expected-season", type=int, required=True)
    ap.add_argument("--expected-week", type=int, required=True)
    args = ap.parse_args()

    alloc = read_csv(args.allocation_lock, "allocation lock")
    events = read_csv(args.event_audit, "event audit")
    proj = read_csv(args.projection_lock, "projection lock")

    if not args.projection_manifest.exists() or args.projection_manifest.stat().st_size <= 0:
        raise RuntimeError("projection public-safe manifest missing/empty")
    manifest = json.loads(args.projection_manifest.read_text(encoding="utf-8"))
    if manifest.get("disposition") != "GSIS_RB_SUCCESSOR_LINEUP_V1_THREE_ARM_PROJECTION_LOCKED":
        raise RuntimeError(f"unexpected projection disposition: {manifest.get('disposition')}")
    if int(manifest.get("sportsbook_inputs_used", -1)) != 0:
        raise RuntimeError("projection manifest indicates sportsbook inputs")
    if int(manifest.get("target_game_outcomes_attached", -1)) != 0:
        raise RuntimeError("projection manifest indicates target outcomes were attached")
    if not bool(manifest.get("private_rows_must_not_be_committed", False)):
        raise RuntimeError("projection manifest missing private-row protection")

    for label, frame in (("allocation", alloc), ("projection", proj)):
        for col, expected in (("target_season", args.expected_season), ("target_week", args.expected_week)):
            if col not in frame.columns:
                raise RuntimeError(f"{label} missing {col}")
            vals = set(pd.to_numeric(frame[col], errors="raise").astype(int).tolist())
            if vals != {int(expected)}:
                raise RuntimeError(f"{label} {col} mismatch: {sorted(vals)} != {expected}")

    req_alloc = {"event_id", "team", "successor_player_clean_key"}
    req_proj = {"event_id", "team", "player_clean_key", "market"}
    if req_alloc - set(alloc.columns):
        raise RuntimeError(f"allocation missing {sorted(req_alloc-set(alloc.columns))}")
    if req_proj - set(proj.columns):
        raise RuntimeError(f"projection missing {sorted(req_proj-set(proj.columns))}")

    successor_keys = set(
        zip(
            alloc["event_id"].astype(str),
            alloc["team"].astype(str),
            alloc["successor_player_clean_key"].astype(str),
        )
    )
    projected_players = set(
        zip(
            proj["event_id"].astype(str),
            proj["team"].astype(str),
            proj["player_clean_key"].astype(str),
        )
    )
    missing = successor_keys - projected_players
    if missing:
        raise RuntimeError(
            f"allocation successors missing from projection lock: count={len(missing)}"
        )

    markets = set(proj["market"].astype(str).str.lower())
    if markets != {"rush_att", "rush_yards"}:
        raise RuntimeError(f"unexpected projection markets: {sorted(markets)}")

    event_ids = set(events["event_id"].astype(str)) if "event_id" in events.columns else set()
    alloc_events = set(alloc["event_id"].astype(str))
    proj_events = set(proj["event_id"].astype(str))
    if event_ids and (event_ids != alloc_events or event_ids != proj_events):
        raise RuntimeError(
            f"event coverage mismatch event_audit={len(event_ids)} allocation={len(alloc_events)} projection={len(proj_events)}"
        )

    receipt = {
        "disposition": "GSIS_RB_SUCCESSOR_LINEUP_V1_COMPLETE_PREGAME_LOCK_READY",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "target_season": int(args.expected_season),
        "target_week": int(args.expected_week),
        "allocation_lock_sha256": sha256(args.allocation_lock),
        "event_audit_sha256": sha256(args.event_audit),
        "projection_lock_sha256": sha256(args.projection_lock),
        "projection_manifest_sha256": sha256(args.projection_manifest),
        "events": int(len(proj_events)),
        "allocation_rows": int(len(alloc)),
        "projection_rows": int(len(proj)),
        "locked_players": int(proj[["event_id","team","player_clean_key"]].drop_duplicates().shape[0]),
        "markets": sorted(markets),
        "arms": manifest.get("arms"),
        "sportsbook_inputs_used": 0,
        "target_outcomes_used": 0,
        "raw_private_rows_emitted": False,
        "private_files_required": True,
    }

    # Public-safe receipt may never grow player-identity fields.
    leak = FORBIDDEN_PUBLIC.intersection(receipt)
    if leak:
        raise RuntimeError(f"public receipt leaked forbidden fields: {sorted(leak)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
