#!/usr/bin/env python3
"""Run make_team_form under the shared runtime season context.

During preseason / before current-season nflverse PBP exists, active-season PBP
loads explicitly fall back to PRIOR_SEASON. The resulting TeamForm remains a
current-slate context artifact, but provenance columns identify which PBP season
fed each PBP-derived field. Explicit prior-season requests are allowed through
unchanged.

The legacy make_team_form module contains a late 2025-specific success/explosive
PBP derivation. Until Team Context v3 replaces that module, this wrapper repairs
those four fields after the legacy build using the guarded runtime PBP source and
a strict current-season `week < target_week` cutoff.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.runtime_context import (
    log_runtime_context,
    resolve_prior_season,
    resolve_season,
    resolve_week,
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

        # Any stale request for an unrelated season (including a hard-coded 2025
        # after 2026) is redirected to the active season first.
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
                state["used_prior"] = False
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


def _to_pandas(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if hasattr(obj, "to_pandas") and callable(getattr(obj, "to_pandas")):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def _repair_success_explosive_context(
    active_season: int,
    target_week: int,
    state: dict[str, object],
) -> None:
    """Replace legacy hard-coded success/explosive fields with runtime PBP.

    For current-season PBP, only completed games from weeks strictly before the
    target week are eligible. During preseason/Week 1, the guarded loader falls
    back to PRIOR_SEASON and uses the full prior regular season as a declared
    prior rather than pretending it is current data.
    """
    if not TEAM_FORM_PATH.exists() or TEAM_FORM_PATH.stat().st_size == 0:
        raise RuntimeError("TeamForm missing before success/explosive repair")

    raw = make_team_form.load_pbp(int(active_season))
    pbp = _to_pandas(raw)
    if pbp.empty:
        raise RuntimeError("runtime PBP source empty during TeamForm success/explosive repair")
    pbp.columns = [str(c).strip().lower() for c in pbp.columns]

    source_season = int(state.get("pbp_feature_season", active_season))
    if "season_type" in pbp.columns:
        reg = pbp.loc[pbp["season_type"].astype(str).str.upper().eq("REG")].copy()
        if not reg.empty:
            pbp = reg
    elif "game_type" in pbp.columns:
        reg = pbp.loc[pbp["game_type"].astype(str).str.upper().eq("REG")].copy()
        if not reg.empty:
            pbp = reg

    if source_season == int(active_season) and "week" in pbp.columns:
        week = pd.to_numeric(pbp["week"], errors="coerce")
        pbp = pbp.loc[week.lt(int(target_week))].copy()
        if pbp.empty:
            raise RuntimeError(
                f"active-season PBP has no completed pre-target rows for season={active_season} week={target_week}"
            )

    if "play_type" in pbp.columns:
        play_type = pbp["play_type"].astype(str).str.lower()
        is_pass = play_type.eq("pass")
        is_run = play_type.eq("run")
    else:
        is_pass = pd.to_numeric(pbp.get("pass", 0), errors="coerce").fillna(0).eq(1)
        is_run = pd.to_numeric(pbp.get("rush", 0), errors="coerce").fillna(0).eq(1)
    eligible = is_pass | is_run
    pbp = pbp.loc[eligible].copy()
    is_pass = is_pass.loc[pbp.index]
    is_run = is_run.loc[pbp.index]
    if pbp.empty:
        raise RuntimeError("runtime PBP contains no run/pass plays for success/explosive repair")

    off_col = next((c for c in ("posteam", "offense_team", "club_code") if c in pbp.columns), None)
    def_col = next((c for c in ("defteam", "def_team") if c in pbp.columns), None)
    if not off_col or not def_col:
        raise RuntimeError("runtime PBP missing offense/defense team columns")

    pbp["off_team"] = pbp[off_col].map(make_team_form.canon_team)
    pbp["def_team"] = pbp[def_col].map(make_team_form.canon_team)
    pbp = pbp.loc[pbp["off_team"].isin(make_team_form.VALID) & pbp["def_team"].isin(make_team_form.VALID)].copy()
    is_pass = is_pass.loc[pbp.index]
    is_run = is_run.loc[pbp.index]

    epa = pd.to_numeric(pbp.get("epa"), errors="coerce")
    yards = pd.to_numeric(pbp.get("yards_gained"), errors="coerce")
    pbp["is_success"] = epa.gt(0).astype(int)
    pbp["is_explosive"] = ((is_pass & yards.ge(15)) | (is_run & yards.ge(10))).astype(int)

    off = pbp.groupby("off_team", as_index=False).agg(
        off_plays=("is_success", "size"),
        off_success=("is_success", "sum"),
    )
    off["success_rate_off"] = off["off_success"] / off["off_plays"]
    defense = pbp.groupby("def_team", as_index=False).agg(
        def_plays=("is_success", "size"),
        def_success_allowed=("is_success", "sum"),
        def_explosive_allowed=("is_explosive", "sum"),
    )
    defense["success_rate_def"] = defense["def_success_allowed"] / defense["def_plays"]
    defense["explosive_play_rate_allowed"] = defense["def_explosive_allowed"] / defense["def_plays"]

    off_map = off.set_index("off_team")["success_rate_off"]
    def_success_map = defense.set_index("def_team")["success_rate_def"]
    def_explosive_map = defense.set_index("def_team")["explosive_play_rate_allowed"]

    tf = pd.read_csv(TEAM_FORM_PATH)
    tf["team"] = tf["team"].map(make_team_form.canon_team)
    if tf["team"].eq("").any() or tf.duplicated("team").any():
        raise RuntimeError("TeamForm has invalid/duplicate teams before runtime context repair")

    repaired_off = tf["team"].map(off_map)
    repaired_def = tf["team"].map(def_success_map)
    repaired_explosive = tf["team"].map(def_explosive_map)

    # In active-season mode every team should have prior-week evidence after
    # Week 1. In prior-season fallback mode the regular-season prior should cover
    # all 32 teams. Missing teams are therefore a source-integrity failure.
    if repaired_off.isna().any() or repaired_def.isna().any() or repaired_explosive.isna().any():
        missing = tf.loc[
            repaired_off.isna() | repaired_def.isna() | repaired_explosive.isna(), "team"
        ].tolist()
        raise RuntimeError(
            f"runtime success/explosive context missing teams source_season={source_season}: {missing}"
        )

    tf["success_rate_off"] = repaired_off.to_numpy(float)
    tf["success_rate_def"] = repaired_def.to_numpy(float)
    tf["success_rate_diff"] = tf["success_rate_off"] - tf["success_rate_def"]
    tf["explosive_play_rate_allowed"] = repaired_explosive.to_numpy(float)
    tf["success_explosive_source_season"] = int(source_season)
    tf["success_explosive_source"] = (
        f"prior_{source_season}_pbp_runtime_guarded"
        if source_season != int(active_season)
        else f"current_{active_season}_pregame_pbp"
    )
    tf["success_explosive_target_week"] = int(target_week)
    tf.to_csv(TEAM_FORM_PATH, index=False)
    print(
        f"[run_team_form_context] repaired success/explosive context "
        f"source_season={source_season} target_week={target_week} teams={len(tf)}"
    )


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
    week = resolve_week(season=season)
    log_runtime_context()
    state = _install_pbp_season_guard(season, prior)

    argv = list(sys.argv[1:])
    if "--season" not in argv:
        argv = ["--season", str(season), *argv]
    sys.argv = [sys.argv[0], *argv]

    make_team_form.main()
    _repair_success_explosive_context(season, week, state)
    _stamp_provenance(season, prior, state)


if __name__ == "__main__":
    main()
