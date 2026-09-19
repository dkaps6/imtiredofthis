#!/usr/bin/env python3
"""Rehydrate canonical multi-season history and execute role/room qualification.

This is a deterministic execution wrapper around already-qualified repository builders.
It intentionally ends before predictive science and never reads sportsbook data.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from scripts.backtest.historical_inputs import build_schedule_history, build_team_weekly_from_pbp
from scripts.backtest.historical_player_logs import build_historical_player_logs


def parse_seasons(value: str) -> list[int]:
    out: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(token))
    return sorted(set(out))


def run(cmd: list[str]) -> None:
    print("[context_rehydrate]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", default="2019-2025")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--builder-commit", default="unknown")
    p.add_argument("--python", default=sys.executable)
    a = p.parse_args()
    seasons = parse_seasons(a.seasons)
    if not seasons:
        raise RuntimeError("no seasons requested")

    root = Path(__file__).resolve().parents[2]
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    schedule_path = out / "schedule_history.csv"
    team_weekly_path = out / "team_weekly_history.csv"
    player_logs_path = out / "player_game_logs_history.csv"
    manifest_path = out / "historical_base_manifest_v1.json"
    qualification_dir = out / "role_room_qualification"

    schedule = build_schedule_history(seasons)
    schedule.to_csv(schedule_path, index=False)
    team_weekly = build_team_weekly_from_pbp(seasons)
    team_weekly.to_csv(team_weekly_path, index=False)
    player_logs = build_historical_player_logs(seasons=seasons, schedule_history=schedule)
    player_logs.to_csv(player_logs_path, index=False)

    run([
        a.python, str(root / "scripts/research/build_historical_base_manifest_v1.py"),
        "--player-logs", str(player_logs_path),
        "--team-weekly", str(team_weekly_path),
        "--schedule", str(schedule_path),
        "--out", str(manifest_path),
        "--builder-commit", a.builder_commit,
        "--source-lineage", "deterministic canonical repo rehydration for football-context qualification",
    ])
    run([
        a.python, str(root / "scripts/research/run_role_room_qualification_pipeline.py"),
        "--historical", str(player_logs_path),
        "--out-dir", str(qualification_dir),
    ])
    print(f"[context_rehydrate] complete seasons={seasons} out={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
