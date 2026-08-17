#!/usr/bin/env python3
"""Run make_team_form under the shared runtime season context.

The adapter intercepts any accidental PBP request for a season different from
the active runtime season.  Historical participation fallbacks remain untouched
because only ``NFLV.load_pbp`` is guarded.
"""
from __future__ import annotations

import sys

from scripts.runtime_context import log_runtime_context, resolve_season
import scripts.make_team_form as make_team_form


def _install_pbp_season_guard(active_season: int) -> None:
    nflv = make_team_form.NFLV
    if not hasattr(nflv, "load_pbp"):
        return
    original = nflv.load_pbp

    def guarded_load_pbp(*args, **kwargs):
        requested = kwargs.get("seasons")
        if requested is None and args:
            requested = args[0]

        normalized: list[int] = []
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

        if normalized and normalized != [int(active_season)]:
            print(
                f"[run_team_form_context] intercepted PBP season request {normalized} -> {[int(active_season)]}"
            )
            if "seasons" in kwargs:
                kwargs["seasons"] = [int(active_season)]
            elif args:
                args = ([int(active_season)],) + args[1:]
            else:
                kwargs["seasons"] = [int(active_season)]
        return original(*args, **kwargs)

    nflv.load_pbp = guarded_load_pbp


def main() -> None:
    season = resolve_season()
    log_runtime_context()
    _install_pbp_season_guard(season)
    argv = list(sys.argv[1:])
    if "--season" not in argv:
        argv = ["--season", str(season), *argv]
    sys.argv = [sys.argv[0], *argv]
    make_team_form.main()


if __name__ == "__main__":
    main()
