#!/usr/bin/env python3
"""WR-R18 Stage A corrected mechanics wrapper, pre-result.

This file applies two pre-result mechanical corrections to the frozen WR-R18 V1
implementation without changing the scientific hypothesis, cohort, thresholds,
or promotion gates:

1. The last-8 receiver history is selected from all prior target-bearing games,
   exactly as the frozen plan states; CPOE validity is evaluated inside that
   already-selected game window.
2. A positive tail rate over a zero comparison rate is treated as an infinite
   ratio rather than NaN. A 0/0 tail comparison remains undefined.

No 2024 scoring is added here. The base evaluator still owns the frozen Stage-A
logic and output contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.research import evaluate_wr_r18_receiver_target_cpoe_stage_a_v1 as base


def _ratio(high: float, low: float) -> float:
    """Frozen high/low tail ratio with explicit zero-denominator semantics."""
    if not np.isfinite(high) or not np.isfinite(low):
        return np.nan
    if low > 0:
        return float(high / low)
    if low == 0 and high > 0:
        return float("inf")
    return np.nan


def receiver_state(
    targets: pd.DataFrame,
    rosters: pd.DataFrame,
    authority_name_key: str,
    authority_team: str,
    season: int,
    week: int,
) -> dict:
    """Compute receiver CPOE on the literal last-8 target-bearing-game window."""
    history_all, audit = base.resolve_prior_receiver_history(
        targets, rosters, authority_name_key, authority_team, season, week
    )

    # Freeze the history window from every target-bearing game first, regardless
    # of whether an individual target has a defined CPOE value.
    h8_all = base._last_games(history_all, base.PRIOR_GAMES)
    games8 = (
        h8_all[["season", "week", "game_id"]].drop_duplicates()
        if not h8_all.empty else pd.DataFrame()
    )
    h8_valid = h8_all.loc[h8_all["cpoe_num"].notna()].copy() if not h8_all.empty else h8_all.copy()

    cpoe = base._num(h8_valid["cpoe_num"]).dropna() if not h8_valid.empty else pd.Series(dtype=float)
    air_valid = base._num(h8_valid["air"]).dropna() if not h8_valid.empty else pd.Series(dtype=float)
    null_mask = h8_all["cpoe_num"].isna() if not h8_all.empty else pd.Series(dtype=bool)

    out = {
        **audit,
        "prior_target_games": int(len(games8)),
        "prior_valid_cpoe_targets": int(len(cpoe)),
        "prior_otherwise_eligible_targets": int(len(h8_all)),
        "prior_null_cpoe_targets": int(null_mask.sum()) if len(h8_all) else 0,
        "prior_null_cpoe_rate": float(null_mask.mean()) if len(h8_all) else np.nan,
        "WR_TARGET_CPOE_MEAN8": np.nan,
        "mean_air_yards_per_target8": np.nan,
        "history_max_season": np.nan,
        "history_max_week": np.nan,
    }
    if len(games8):
        latest = games8.sort_values(["season", "week", "game_id"], kind="mergesort").iloc[-1]
        out["history_max_season"] = int(latest["season"])
        out["history_max_week"] = int(latest["week"])
    if audit["identity_mode"] in {"ambiguous", "unmatched"}:
        return out
    if len(games8) < base.MIN_PRIOR_TARGET_GAMES or len(cpoe) < base.MIN_VALID_CPOE_TARGETS:
        return out

    out["WR_TARGET_CPOE_MEAN8"] = float(cpoe.mean())
    out["mean_air_yards_per_target8"] = float(air_valid.mean()) if len(air_valid) else np.nan
    return out


# Patch only the two mechanical functions used dynamically by the frozen base
# evaluator. All scientific constants and Stage-A/mediation logic remain base V1.
base._ratio = _ratio
base.receiver_state = receiver_state

# Re-export the base API so synthetic tests and workflow callers can point at v1b.
AUTHORITY_VARIANT = base.AUTHORITY_VARIANT
EXPECTED_ROWS = base.EXPECTED_ROWS
DEV_SEASON = base.DEV_SEASON
PRIOR_GAMES = base.PRIOR_GAMES
MIN_PRIOR_TARGET_GAMES = base.MIN_PRIOR_TARGET_GAMES
MIN_VALID_CPOE_TARGETS = base.MIN_VALID_CPOE_TARGETS
MIN_COVERAGE = base.MIN_COVERAGE
MIN_SPEARMAN = base.MIN_SPEARMAN
MIN_RESIDUAL_GAP = base.MIN_RESIDUAL_GAP
MIN_TAIL_RATIO = base.MIN_TAIL_RATIO
MIN_SLICE_N = base.MIN_SLICE_N
MEDIATION_MIN_SPEARMAN = base.MEDIATION_MIN_SPEARMAN
MEDIATION_MIN_RESIDUAL_GAP = base.MEDIATION_MIN_RESIDUAL_GAP

_num = base._num
_clean_id = base._clean_id
_name_key = base._name_key
_regular_only = base._regular_only
_to_pandas = base._to_pandas
_first = base._first
load_authority = base.load_authority
load_pbp = base.load_pbp
load_roster_identity = base.load_roster_identity
prepare_pbp_sources = base.prepare_pbp_sources
_prior = base._prior
_last_games = base._last_games
resolve_roster_player_id = base.resolve_roster_player_id
resolve_prior_receiver_history = base.resolve_prior_receiver_history
team_state = base.team_state
build_development_panel = base.build_development_panel
_spearman = base._spearman
raw_stage_a = base.raw_stage_a
mediation_robustness = base.mediation_robustness
cpoe_missingness_audit = base.cpoe_missingness_audit
identity_audit = base.identity_audit
main = base.main


if __name__ == "__main__":
    raise SystemExit(main())
