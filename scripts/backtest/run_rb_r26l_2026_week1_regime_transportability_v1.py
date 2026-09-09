#!/usr/bin/env python3
"""Mechanical compatibility runner for frozen R26L.

The frozen evaluator asks the production-safe R9 identity runtime for history
through TARGET_SEASON=2026. For a 2026 Week-1 pregame audit there are no prior
2026 games, and nflverse has no stats_player_week_2026 asset yet. Strict-prior
identity must therefore end at PRIOR_SEASON=2025.

This wrapper changes no R26L feature, threshold, gate, distance rule, source
population, prediction, R9 fit, R22 state, or production parameter.
"""
from __future__ import annotations

from scripts.backtest import audit_rb_r26l_2026_week1_regime_transportability_v1 as audit
from scripts.modeling.rb_receiving_identity_runtime_v1 import identity_atlas as _identity_atlas


def _strict_prior_identity_atlas(history_start: int, through_season: int):
    if int(through_season) != int(audit.TARGET_SEASON):
        raise RuntimeError(
            f"R26L compatibility runner expected through_season={audit.TARGET_SEASON}, got {through_season}"
        )
    return _identity_atlas(int(history_start), int(audit.PRIOR_SEASON))


audit.identity_atlas = _strict_prior_identity_atlas


if __name__ == "__main__":
    raise SystemExit(audit.main())
