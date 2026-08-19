"""Canonical football-context rules for the v2 projection stack.

This module preserves the empirically developed rules from the legacy
``rules_engine.py``, ``elite_rules.py`` and ``agent_based.py`` while removing
schema duplication and correcting pressure-matchup semantics.

It is intentionally pure and side-effect free. Production integration happens
in a later migration after parity/backtest checks.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple

from .contracts import TeamContext


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _num(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return float(default)
        return out
    except Exception:
        return float(default)


@dataclass(frozen=True)
class GameScriptProjection:
    projected_plays: float
    projected_pass_attempts: float
    projected_rush_attempts: float
    lead_prob: float
    neutral_prob: float
    trail_prob: float
    pressure_mismatch: bool
    blowout_risk: bool
    shootout_risk: bool


@dataclass(frozen=True)
class MatchupMultipliers:
    wr1_target_mult: float = 1.0
    wr1_5_target_mult: float = 1.0
    slot_target_mult: float = 1.0
    te_target_mult: float = 1.0
    rb_rec_target_mult: float = 1.0
    rb_rush_eff_mult: float = 1.0
    pass_eff_mult: float = 1.0
    sack_mult: float = 1.0
    int_mult: float = 1.0
    volatility_mult: float = 1.0


def success_diff(offense: TeamContext, defense: TeamContext) -> float:
    return _num(offense.success_rate_off) - _num(defense.success_rate_def)


def offensive_pressure_mismatch(offense: TeamContext, defense: TeamContext) -> float:
    """Positive means the opposing pass rush has an advantage over protection."""
    return _num(defense.pressure_rate_generated) - _num(offense.pressure_rate_allowed)


def script_distribution(diff: float) -> Tuple[float, float, float]:
    scaled = _clamp(diff * 20.0, -3.0, 3.0)
    lead = math.exp(scaled)
    trail = math.exp(-scaled)
    neutral = 1.0
    total = lead + neutral + trail
    return lead / total, neutral / total, trail / total


def estimate_plays(offense: TeamContext) -> float:
    prior = _num(offense.plays_est, 64.0)
    pace = _num(offense.neutral_pace, 0.0)
    sec_last5 = _num(offense.sec_per_play_last5, 0.0)
    recent = _num(offense.neutral_pace_last5, 0.0)

    # Pace here is seconds per play for one offense, not both teams combined.
    # A team owns roughly half of the 3600-second game clock, so converting with
    # 3600 / pace doubles the implied opportunity pool and routinely hits the
    # 80-play ceiling. Use the same 1800-second team possession basis as the
    # Monte Carlo fallback instead.
    if sec_last5 > 5.0:
        recent_play_est = 1800.0 / sec_last5
        recent_play_est = _clamp(recent_play_est, 50.0, 80.0)
    elif recent > 20.0:
        recent_play_est = _clamp(1800.0 / recent, 50.0, 80.0)
    elif pace > 20.0:
        recent_play_est = _clamp(1800.0 / pace, 50.0, 80.0)
    else:
        recent_play_est = prior
    return _clamp(0.5 * prior + 0.5 * recent_play_est, 48.0, 82.0)


def project_game_script(offense: TeamContext, defense: TeamContext) -> GameScriptProjection:
    diff = success_diff(offense, defense)
    pressure = offensive_pressure_mismatch(offense, defense)
    lead, neutral, trail = script_distribution(diff)
    plays = estimate_plays(offense)

    pass_share = 0.55 + _clamp(_num(offense.proe), -0.10, 0.10)
    # Trailing teams pass more; leading teams run more. Keep the adjustment modest.
    pass_share += 0.08 * (trail - lead)
    pass_share = _clamp(pass_share, 0.42, 0.70)

    pass_attempts = plays * pass_share
    rush_attempts = plays - pass_attempts
    return GameScriptProjection(
        projected_plays=plays,
        projected_pass_attempts=pass_attempts,
        projected_rush_attempts=rush_attempts,
        lead_prob=lead,
        neutral_prob=neutral,
        trail_prob=trail,
        pressure_mismatch=abs(pressure) >= 0.05,
        blowout_risk=abs(diff) >= 0.06,
        shootout_risk=abs(diff) < 0.03 and plays >= 68.0,
    )


def matchup_multipliers(offense: TeamContext, defense: TeamContext) -> MatchupMultipliers:
    zone = _num(defense.coverage_zone_rate)
    man = _num(defense.coverage_man_rate)
    middle = _num(defense.middle_open_rate)
    pressure = offensive_pressure_mismatch(offense, defense)
    light = _num(defense.light_box_rate)
    heavy = _num(defense.heavy_box_rate)

    values: Dict[str, float] = {
        "wr1_target_mult": 1.0,
        "wr1_5_target_mult": 1.0,
        "slot_target_mult": 1.0,
        "te_target_mult": 1.0,
        "rb_rec_target_mult": 1.0,
        "rb_rush_eff_mult": 1.0,
        "pass_eff_mult": 1.0,
        "sack_mult": 1.0,
        "int_mult": 1.0,
        "volatility_mult": 1.0,
    }

    # Preserve the tested legacy thresholds/magnitudes until the 2025 walk-forward
    # backtest recalibrates them.
    if zone >= 0.60:
        values["te_target_mult"] *= 1.15
        values["rb_rec_target_mult"] *= 1.20
    if man >= 0.50:
        values["wr1_target_mult"] *= 0.95
        values["wr1_5_target_mult"] *= 1.15
        values["slot_target_mult"] *= 1.05
    if middle >= 0.50:
        values["slot_target_mult"] *= 1.10
        values["te_target_mult"] *= 1.10

    # Opponent pressure advantage: suppress pass efficiency, increase checkdowns,
    # sacks/INT risk and uncertainty. This corrects the old defense-vs-defense diff.
    if pressure > 0.05:
        values["pass_eff_mult"] *= 0.94
        values["rb_rec_target_mult"] *= 1.25
        values["sack_mult"] *= 1.10
        values["int_mult"] *= 1.10
        values["volatility_mult"] *= 1.10
    elif pressure < -0.05:
        values["pass_eff_mult"] *= 1.03

    # Preserve legacy box-count efficiency rules.
    if light >= 0.60:
        values["rb_rush_eff_mult"] *= 1.07
    if heavy >= 0.60:
        values["rb_rush_eff_mult"] *= 0.94

    return MatchupMultipliers(**{k: _clamp(v, 0.50, 1.80) for k, v in values.items()})


def redistribute_alpha_usage(
    alpha_share: float,
    wr2_share: float,
    slot_te_share: float,
    rb_share: float,
    *,
    alpha_limited: bool,
) -> tuple[float, float, float, float]:
    """Preserve the legacy 60/30/10 redistribution rule for an unavailable alpha."""
    if not alpha_limited:
        return alpha_share, wr2_share, slot_te_share, rb_share
    give = max(0.0, alpha_share) * 0.50
    return (
        alpha_share - give,
        wr2_share + give * 0.60,
        slot_te_share + give * 0.30,
        rb_share + give * 0.10,
    )


def coverage_penalty(
    yards_per_target: float,
    target_share: float,
    *,
    tough_shadow: bool = False,
    heavy_man: bool = False,
    heavy_zone: bool = False,
) -> tuple[float, float]:
    """Preserve the legacy individual coverage adjustment rules."""
    ypt = float(yards_per_target)
    share = float(target_share)
    if tough_shadow or heavy_man:
        ypt *= 0.94
        share *= 0.92
    if heavy_zone:
        ypt *= 1.04
        share *= 1.06
    return ypt, share


def widen_volatility(sigma: float, *, pressure_mismatch: bool = False, qb_inconsistent: bool = False) -> float:
    widen = 0.10 * int(bool(pressure_mismatch)) + 0.10 * int(bool(qb_inconsistent))
    return float(sigma) * (1.0 + widen)
