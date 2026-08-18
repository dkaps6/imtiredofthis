#!/usr/bin/env python3
"""Run the existing PlayerForm v2 orchestration with the maintained nflreadpy loader."""
from __future__ import annotations

import scripts.run_player_form_v2 as runner
from scripts.player_stats_loader_v2 import load_weekly_player_stats


def main() -> int:
    # player_form_v2 keeps the build/blend semantics; only replace the broken
    # provider adapter that still passed the removed stat_type argument.
    runner.pf._load_weekly = load_weekly_player_stats
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
