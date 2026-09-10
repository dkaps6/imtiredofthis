#!/usr/bin/env python3
"""Week-1 operational wrapper for the existing TeamForm runtime context.

At target Week 1 there are, by definition, no completed current-season games
strictly before the target week. Some nflverse feeds may nevertheless return
non-empty active-season rows (for example preseason/partial feed material),
which causes the legacy runtime guard to treat them as usable current-season
history and later fail the strict `week < 1` cutoff.

This wrapper changes no TeamForm feature formula. For Week 1 only, it forces the
already-declared PRIOR_SEASON PBP fallback before invoking the canonical
`run_team_form_context` pipeline, and stamps provenance through that pipeline.
For any target week other than 1 it refuses to run so it cannot silently alter
in-season semantics.
"""
from __future__ import annotations

import sys

from scripts.runtime_context import resolve_prior_season, resolve_season, resolve_week
import scripts.run_team_form_context as base


def _set_seasons_arg(args, kwargs, season: int):
    args = tuple(args)
    kwargs = dict(kwargs)
    if "seasons" in kwargs:
        kwargs["seasons"] = [int(season)]
        return args, kwargs
    if args:
        return ([int(season)],) + args[1:], kwargs
    kwargs["seasons"] = [int(season)]
    return args, kwargs


def _week1_prior_guard(active_season: int, prior_season: int):
    week = int(resolve_week(season=active_season))
    if week != 1:
        raise RuntimeError(f"Week-1 prior wrapper is only valid for target week 1; got week={week}")
    nflv = base.make_team_form.NFLV
    if not hasattr(nflv, "load_pbp"):
        raise RuntimeError("TeamForm nflverse loader has no load_pbp")
    original = nflv.load_pbp
    def prior_only_load_pbp(*args, **kwargs):
        p_args, p_kwargs = _set_seasons_arg(args, kwargs, int(prior_season))
        return original(*p_args, **p_kwargs)
    nflv.load_pbp = prior_only_load_pbp
    print(f"[team_form_week1_prior] target week=1; forcing declared prior-season PBP {prior_season} for active season {active_season}")
    return {"pbp_feature_season": int(prior_season),"used_prior": True,"fallback_reason": "target_week_1_has_no_legal_current_season_pre_target_games"}


def main() -> None:
    season = int(resolve_season()); prior = int(resolve_prior_season()); week = int(resolve_week(season=season))
    if week != 1: raise RuntimeError(f"refusing Week-1 wrapper outside week 1: week={week}")
    if prior >= season: raise RuntimeError(f"invalid prior season: prior={prior} active={season}")
    base._install_pbp_season_guard = _week1_prior_guard
    base.main()

if __name__ == "__main__":
    main()
