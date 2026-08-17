#!/usr/bin/env python3
"""Run make_team_form under the shared runtime season context.

This adapter protects the 2026 migration from stale 2025 PBP reloads that still
exist inside the legacy TeamForm implementation. It can be removed once those
internal literals are refactored directly.
"""

from __future__ import annotations

import sys

from scripts.runtime_context import log_runtime_context, resolve_season
import scripts.make_team_form as make_team_form


def _install_pbp_season_guard(active_season: int) -> None:
    """Force accidental legacy 2025 PBP reloads onto the active season.

    TeamForm intentionally may load prior-season *participation* for box-count
    fallback; this guard touches only NFLV.load_pbp, so that historical
    participation behavior remains available.
    """
    nflv = make_team_form.NFLV
    if not hasattr(nflv, "load_pbp"):
        return

    original = nflv.load_pbp

    def guarded_load_pbp(*args, **kwargs):
        requested = kwargs.get("seasons")
        if requested is None and args:
            requested = args[0]

        normalized = []
        if isinstance(requested, (list, tuple, set)):
            try:
                normalized = [int(v) for v in requested]
            except Exception:
                normalized = []
        elif requested is not None:
            try:
                normalized = [int(requested)]
            except Exception:
                normalized = []

        if active_season != 2025 and normalized == [2025]:
            print(
                f"[run_team_form_context] intercepted stale PBP season 2025 -> {active_season}"
            )
            if "seasons" in kwargs:
                kwargs["seasons"] = [active_season]
            elif args:
                args = ([active_season],) + args[1:]
            else:
                kwargs["seasons"] = [active_season]

        return original(*args, **kwargs)

    nflv.load_pbp = guarded_load_pbp


def main() -> None:
    season = resolve_season()
    log_runtime_context()
    _install_pbp_season_guard(season)

    # Preserve any extra CLI flags supplied by the workflow, but guarantee the
    # active season is explicit even if a caller omits --season.
    argv = list(sys.argv[1:])
    if "--season" not in argv:
        argv = ["--season", str(season), *argv]
    sys.argv = [sys.argv[0], *argv]

    make_team_form.main()


if __name__ == "__main__":
    main()
