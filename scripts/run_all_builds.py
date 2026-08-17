#!/usr/bin/env python3
"""Run optional enrichment builders under one authoritative season context.

Production Full Slate calls the important builders directly; this remains a
manual convenience orchestrator and now fails on required step errors instead
of silently continuing through a broken feature chain.
"""
from __future__ import annotations

import os
import subprocess
import sys

from scripts.runtime_context import resolve_season


def run(cmd: list[str], *, required: bool = True) -> bool:
    print("\n$", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        msg = f"step failed ({result.returncode}): {' '.join(cmd)}"
        if required:
            raise RuntimeError(msg)
        print(f"[WARN] {msg}", file=sys.stderr)
        return False
    return True


def main() -> int:
    season = resolve_season()
    os.makedirs("data", exist_ok=True)
    py = sys.executable

    # Web enrichments can change independently and are optional.
    optional = [
        [py, "scripts/build/build_cb_coverage_team.py"],
        [py, "scripts/build/build_cb_coverage_player.py"],
        [py, "scripts/build/build_weather_week.py", "--season", str(season)],
        [py, "scripts/build/build_injuries_weekly.py", "data/injuries.csv"],
        [py, "scripts/build/build_wr_cb_exposure.py"],
    ]
    for cmd in optional:
        run(cmd, required=False)

    # All nflverse features are built together so they share exactly one season.
    run([py, "scripts/build/pbp_features.py", "--season", str(season)], required=False)

    # Refresh live opponent map after props are present.
    run([py, "scripts/build/build_opponent_map_from_props.py"], required=False)

    merge_script = "scripts/utils/merge_opponent_into_player_form.py"
    if os.path.exists(merge_script) and os.path.exists("data/player_form_consensus.csv"):
        run([py, merge_script], required=False)

    if os.path.exists("model/features/build.py"):
        run([
            py,
            "-c",
            "from model.features.build import build_matchup_frame as bm; "
            "df=bm(); df.to_csv('outputs/matchup_features.csv', index=False); "
            "print(df.shape, '-> outputs/matchup_features.csv')",
        ], required=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
