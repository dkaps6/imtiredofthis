#!/usr/bin/env python3
"""Prepare the current-slate availability-aware football universe before opportunity.

This candidate-only orchestrator performs the frozen availability sequence after
raw roster/schedule/injury artifacts exist:
  Ourlads status -> official inactives -> T-75 certification -> availability ->
  reconciled active roles -> timing-eligible current roles.
It reads no sportsbook data and changes no scientific model parameters.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.runtime_context import resolve_week


def run(*args: str) -> None:
    cmd = [sys.executable, *args]
    print("[availability-candidate]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--asof-utc", default="")
    a = ap.parse_args()
    week = int(a.week if a.week is not None else resolve_week())

    run("scripts/providers/ourlads_depth_status_v1.py")
    run("scripts/providers/nfl_official_inactives_v1.py")
    timing = ["scripts/validate_current_player_availability_timing_v1.py", "--season", str(a.season), "--week", str(week)]
    if a.asof_utc:
        timing += ["--asof-utc", a.asof_utc]
    run(*timing)
    run("scripts/build/build_current_player_availability_v1.py")
    run("scripts/build/build_reconciled_active_roles_v1.py")
    run("scripts/build/build_production_eligible_active_roles_v1.py")

    required = [
        Path("data/roles_ourlads_status_v1.csv"),
        Path("data/official_inactives_v1.csv"),
        Path("data/current_player_availability_game_certification.csv"),
        Path("data/current_player_availability.csv"),
        Path("data/roles_ourlads_active_v1.csv"),
        Path("data/roles_current_production_eligible_v1.csv"),
    ]
    missing = [str(p) for p in required if not p.exists() or p.stat().st_size <= 0]
    if missing:
        raise RuntimeError(f"availability candidate prep missing artifacts: {missing}")
    status = json.loads(Path("data/roles_current_production_eligible_v1_status.json").read_text(encoding="utf-8"))
    if int(status.get("sportsbook_inputs_used", 1)) != 0:
        raise RuntimeError("availability candidate prep reports sportsbook input")
    print(json.dumps({"status":"AVAILABILITY_CANDIDATE_PREP_COMPLETE","season":a.season,"week":week,**status}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
