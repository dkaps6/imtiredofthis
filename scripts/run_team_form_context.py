#!/usr/bin/env python3
"""Run make_team_form under the shared runtime season context.

During preseason / before current-season nflverse PBP exists, active-season PBP
loads explicitly fall back to PRIOR_SEASON. The resulting TeamForm remains a
2026 slate-context artifact, but provenance columns identify that PBP-derived
features came from the prior season. Explicit prior-season requests are allowed
through unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from scripts.runtime_context import (
    log_runtime_context,
    resolve_prior_season,
    resolve_season,
)
import scripts.make_team_form as make_team_form

TEAM_FORM_PATH = Path("data/team_form.csv")


def _normalize_requested(requested) -> list[int]:
    if isinstance(requested, (list, tuple, set)):
        try:
            return [int(v) for v in requested]
        except Exception:
            return []
    if requested is not None:
        try:
            return [int(requested)]
        except Exception:
            return []
    return []


def _set_seasons_arg(args, kwargs, seasons: list[int]):
    if "seasons" in kwargs:
        kwargs["seasons"] = seasons
        return args, kwargs
    if args:
        args = (seasons,) + tuple(args[1:])
        return args, kwargs
    kwargs["seasons"] = seasons
    return args, kwargs


def _install_pbp_season_guard(active_season: int, prior_season: int) -> dict[str, object]:
    """Allow active PBP when available; explicitly fall back to prior otherwise."""
    state: dict[str, object] = {
        "pbp_feature_season": int(active_season),
        "used_prior": False,
        "fallback_reason": "",
    }
    nflv = make_team_form.NFLV
    if not hasattr(nflv, "load_pbp"):
        return state

    original = nflv.load_pbp

    def guarded_load_pbp(*args, **kwargs):
        requested = kwargs.get("seasons")
        if requested is None and args:
            requested = args[0]
        normalized = _normalize_requested(requested)

        # Intentional prior-season requests remain prior-season requests.
        if normalized == [int(prior_season)]:
            return original(*args, **kwargs)

        # Any stale request for an unrelated season is redirected to active.
        if normalized and normalized != [int(active_season)]:
            print(
                f"[run_team_form_context] redirected stale PBP season request "
                f"{normalized} -> {[int(active_season)]}"
            )
            args, kwargs = _set_seasons_arg(args, kwargs, [int(active_season)])

        try:
            result = original(*args, **kwargs)
            # Some loaders return an empty frame instead of raising when a future
            # season is unavailable. Treat that the same as an unavailable season.
            empty = False
            if hasattr(result, "height"):
                empty = int(result.height) == 0
            elif hasattr(result, "empty"):
                empty = bool(result.empty)
            if not empty:
                state["pbp_feature_season"] = int(active_season)
                return result
            raise RuntimeError("active-season PBP returned 0 rows")
        except Exception as exc:
            print(
                f"[run_team_form_context] active-season PBP unavailable for {active_season}: {exc}"
            )
            print(
                f"[run_team_form_context] using explicit prior-season PBP fallback: {prior_season}"
            )
            prior_args, prior_kwargs = _set_seasons_arg(args, kwargs, [int(prior_season)])
            result = original(*prior_args, **prior_kwargs)
            state["pbp_feature_season"] = int(prior_season)
            state["used_prior"] = True
            state["fallback_reason"] = str(exc)
            return result

    nflv.load_pbp = guarded_load_pbp
    return state


def _stamp_provenance(active_season: int, prior_season: int, state: dict[str, object]) -> None:
    if not TEAM_FORM_PATH.exists() or TEAM_FORM_PATH.stat().st_size == 0:
        raise RuntimeError("TeamForm did not write data/team_form.csv")
    df = pd.read_csv(TEAM_FORM_PATH)
    if df.empty:
        raise RuntimeError("TeamForm wrote 0 rows")

    # team_form is a current-slate context table. PBP-derived fields may still be
    # prior-season priors, so retain both concepts explicitly.
    df["season"] = int(active_season)
    df["team_form_active_season"] = int(active_season)
    df["team_form_prior_season"] = int(prior_season)
    df["pbp_feature_season"] = int(state.get("pbp_feature_season", active_season))
    df["pbp_prior_used"] = int(bool(state.get("used_prior", False)))
    df["pbp_feature_source"] = (
        f"prior_{prior_season}_pbp"
        if bool(state.get("used_prior", False))
        else f"current_{active_season}_pbp"
    )
    df.to_csv(TEAM_FORM_PATH, index=False)
    print(
        f"[run_team_form_context] team_form rows={len(df)} active={active_season} "
        f"pbp_feature_season={df['pbp_feature_season'].iloc[0]} "
        f"prior_used={df['pbp_prior_used'].iloc[0]}"
    )


def main() -> None:
    season = resolve_season()
    prior = resolve_prior_season()
    log_runtime_context()
    state = _install_pbp_season_guard(season, prior)

    argv = list(sys.argv[1:])
    if "--season" not in argv:
        argv = ["--season", str(season), *argv]
    sys.argv = [sys.argv[0], *argv]

    make_team_form.main()
    _stamp_provenance(season, prior, state)


if __name__ == "__main__":
    main()
