#!/usr/bin/env python3
"""WR-R18 Stage A corrected mechanics v1c, pre-result.

Supersedes v1/v1b for execution. Keeps the frozen WR-R18 science unchanged.
Changes are mechanical only:
- receiver history uses the literal last 8 prior target-bearing games;
- high>0 versus low==0 tail rates are represented as +inf;
- the tail gate explicitly accepts positive +inf as satisfying the frozen >=1.20
  ratio requirement, while 0/0 remains undefined and non-passing.

No real-data result was exposed before these corrections.
"""
from __future__ import annotations

import numpy as np

from scripts.research import evaluate_wr_r18_receiver_target_cpoe_stage_a_v1 as base
from scripts.research import evaluate_wr_r18_receiver_target_cpoe_stage_a_v1b as v1b

# Preserve the unwrapped raw scorer before replacing the module-global function.
_BASE_RAW_STAGE_A = base.raw_stage_a


def _ratio(high: float, low: float) -> float:
    return v1b._ratio(high, low)


def _tail_ratio_pass(value: float) -> bool:
    """Frozen ratio >=1.20, with +inf accepted and NaN/negative inf rejected."""
    if np.isposinf(value):
        return True
    return bool(np.isfinite(value) and value >= base.MIN_TAIL_RATIO)


def raw_stage_a(panel):
    """Run frozen scorer, correcting only the zero-denominator tail predicate."""
    out, quartiles = _BASE_RAW_STAGE_A(panel)
    tail_ok = _tail_ratio_pass(out.get("actual100_rate_ratio", np.nan)) or _tail_ratio_pass(
        out.get("underproj30_rate_ratio", np.nan)
    )
    out["supported_raw"] = bool(
        out.get("coverage", 0.0) >= base.MIN_COVERAGE
        and np.isfinite(out.get("spearman", np.nan))
        and out.get("spearman", np.nan) >= base.MIN_SPEARMAN
        and np.isfinite(out.get("q4_minus_q1_residual_gap", np.nan))
        and out.get("q4_minus_q1_residual_gap", np.nan) >= base.MIN_RESIDUAL_GAP
        and tail_ok
        and (
            out.get("wr1_n", 0) < base.MIN_SLICE_N
            or (np.isfinite(out.get("wr1_gap", np.nan)) and out.get("wr1_gap", np.nan) > 0)
        )
        and (
            out.get("wr2plus_n", 0) < base.MIN_SLICE_N
            or (np.isfinite(out.get("wr2plus_gap", np.nan)) and out.get("wr2plus_gap", np.nan) > 0)
        )
    )
    out["tail_gate_zero_denominator_semantics"] = "high>0_low==0_is_positive_infinity;_0_over_0_undefined"
    return out, quartiles


# Patch the base module used by main() and by the original synthetic suite.
base._ratio = _ratio
base.receiver_state = v1b.receiver_state
base.raw_stage_a = raw_stage_a

# Re-export stable public/mechanical API.
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
receiver_state = v1b.receiver_state
team_state = base.team_state
build_development_panel = base.build_development_panel
_spearman = base._spearman
mediation_robustness = base.mediation_robustness
cpoe_missingness_audit = base.cpoe_missingness_audit
identity_audit = base.identity_audit
main = base.main


if __name__ == "__main__":
    raise SystemExit(main())
