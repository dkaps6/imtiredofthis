#!/usr/bin/env python3
"""Week-2+ TeamForm operations compatibility runner.

The legacy TeamForm builder still treats Sharp's legacy man/zone coverage table
as a hard prerequisite even though canonical Coverage-v2 is built separately
later in Full Slate and takes precedence downstream. If (and only if) the legacy
builder exits after publishing an otherwise valid 32-team TeamForm whose sole
Sharp-source gap is man/zone coverage, this runner accepts that intermediate
artifact and continues the existing runtime strict-prior repairs.

No coverage values are created, filled, or substituted here. Any failure in a
core TeamForm field remains fatal. Week 1 retains its dedicated existing runner.
"""
from __future__ import annotations

import sys

import pandas as pd

import scripts.make_team_form as make_team_form
import scripts.run_team_form_context as base
from scripts.runtime_context import log_runtime_context, resolve_prior_season, resolve_season, resolve_week


def _validate_coverage_only_legacy_exit() -> None:
    path = base.TEAM_FORM_PATH
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError("legacy TeamForm failed without publishing a recoverable artifact")

    tf = pd.read_csv(path, low_memory=False)
    if tf.empty:
        raise RuntimeError("legacy TeamForm failed and published zero rows")

    team_col = "team" if "team" in tf.columns else "team_abbr" if "team_abbr" in tf.columns else None
    if team_col is None:
        raise RuntimeError("legacy TeamForm recovery artifact has no team identity column")
    teams = tf[team_col].map(make_team_form.canon_team)
    if teams.eq("").any() or teams.nunique() != 32 or len(tf) != 32:
        raise RuntimeError(
            f"legacy TeamForm recovery artifact invalid team universe rows={len(tf)} teams={teams.nunique()}"
        )

    # Re-run the legacy core required-metric contract explicitly. This ensures
    # we never use this path to mask a pace/EPA/PROE/red-zone/air-yard/box failure.
    make_team_form._validate_required(tf, allow_missing_box=False)

    for col in ("neutral_pace", "pass_rate_over_expected"):
        if col not in tf.columns or pd.to_numeric(tf[col], errors="coerce").dropna().empty:
            raise RuntimeError(f"legacy TeamForm recovery missing real required Sharp field: {col}")

    coverage_available = False
    for col in ("coverage_man_rate", "coverage_zone_rate"):
        if col in tf.columns and pd.to_numeric(tf[col], errors="coerce").notna().any():
            coverage_available = True
    if coverage_available:
        raise RuntimeError(
            "legacy TeamForm exited despite usable legacy coverage; refusing to mask a non-coverage failure"
        )

    print(
        "[team_form_week2_compat] LEGACY_COVERAGE_ONLY_FAILURE_ACCEPTED "
        "core_teamform=PASS teams=32 coverage_values_fabricated=0 downstream_authority=Coverage-v2"
    )


def main() -> None:
    season = resolve_season()
    prior = resolve_prior_season()
    week = resolve_week(season=season)
    if int(week) <= 1:
        raise RuntimeError(
            "run_team_form_context_week2_compat is Week-2+ only; Week 1 must use its frozen dedicated runner"
        )

    log_runtime_context()
    state = base._install_pbp_season_guard(season, prior)

    argv = list(sys.argv[1:])
    if "--season" not in argv:
        argv = ["--season", str(season), *argv]
    sys.argv = [sys.argv[0], *argv]

    try:
        make_team_form.main()
    except SystemExit as exc:
        code = int(exc.code or 0)
        if code == 0:
            pass
        elif code == 1:
            _validate_coverage_only_legacy_exit()
        else:
            raise

    base._repair_success_explosive_context(season, week, state)
    base._stamp_provenance(season, prior, state)


if __name__ == "__main__":
    main()
