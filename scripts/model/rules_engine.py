"""
scripts/model/rules_engine.py

Non-invasive scaffolding for the NFL player-prop rules engine.

This module encodes the game-level, matchup-level, and player-usage rules
we derived from our post-mortems, but *does not* change any existing
pipeline behavior yet.

Key design goals:
- Use existing schema/labels only (no new position strings).
- Keep everything as pure functions (easy to test & safe to import).
- Provide clear entrypoints for later integration.

Position conventions (as they exist TODAY):
- QB, RB, TE, FB
- LWR, RWR, SWR

Conceptual mapping:
- "WR1"   = primary perimeter WR (one of {LWR, RWR})
- "WR1.5" = other perimeter WR (the opposite of WR1 in {LWR, RWR})
- "SLOT"  = SWR
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math

# --------------------------------------------------------------------------------------
# Basic containers
# --------------------------------------------------------------------------------------


@dataclass
class TeamScriptFeatures:
    """Minimal team-level features the rules engine expects.

    All fields map directly to *existing* team_form columns.
    If a column is missing at runtime, you should derive/populate it upstream,
    NOT rename things here.
    """

    team: str

    # Success rates
    success_rate_off: float
    success_rate_def: float

    # Pressure rates (off = allowed, def = generated)
    pressure_rate_off: float
    pressure_rate_def: float

    # Pace / volume
    neutral_pace: float                    # plays per minute or similar
    neutral_pace_last5: float              # recent neutral pace
    sec_per_play_last5: float              # seconds per play (recent)
    plays_est: float                       # your existing estimated plays per game

    # Passing tendency
    pass_rate_over_expected: float         # PROE

    # Explosiveness / coverage
    explosive_play_rate_allowed: float
    coverage_man_rate: float               # between 0 and 1
    coverage_zone_rate: float              # between 0 and 1
    middle_open_rate: float                # between 0 and 1


@dataclass
class GameScriptProjection:
    """Outputs from the game-level engine."""
    projected_plays: float
    projected_pass_attempts: float
    projected_rush_attempts: float

    # Simple probabilities for script buckets
    lead_prob: float
    neutral_prob: float
    trail_prob: float

    # Flags for downstream usage
    pressure_mismatch: bool
    blowout_risk: bool
    shootout_risk: bool


# --------------------------------------------------------------------------------------
# Helpers for success-rate and pressure rules
# --------------------------------------------------------------------------------------


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_success_diff(our: TeamScriptFeatures, opp: TeamScriptFeatures) -> float:
    """Success rate differential (offense vs opponent defense).

    This is the primary predictor of who controls the script.
    """
    return our.success_rate_off - opp.success_rate_def


def compute_pressure_diff(our: TeamScriptFeatures, opp: TeamScriptFeatures) -> float:
    """Pressure differential from our perspective.

    Positive values mean *we* generate more pressure on their QB
    than they do on ours.
    """
    return our.pressure_rate_def - opp.pressure_rate_def


def estimate_script_distribution(success_diff: float) -> Tuple[float, float, float]:
    """Convert success-rate differential into (lead, neutral, trail) probs.

    This is intentionally simple and monotonic; we can refine later with
    empirical calibration.
    """
    # Scale diff to a -3..+3 range
    scaled = _clamp(success_diff * 20.0, -3.0, 3.0)

    # Softmax-ish: favor lead when scaled >> 0, trail when << 0
    lead_score = math.exp(scaled)
    trail_score = math.exp(-scaled)
    neutral_score = 1.0

    total = lead_score + neutral_score + trail_score
    return (
        lead_score / total,
        neutral_score / total,
        trail_score / total,
    )


def estimate_plays(our: TeamScriptFeatures, opp: TeamScriptFeatures) -> float:
    """Base estimate of plays using neutral pace & recent pace.

    We weight recent pace heavier to respect "pace last 5" rule.
    """
    # Lower sec_per_play = faster pace; invert for plays-per-minute-ish factor
    recent_pace_factor = 60.0 / max(our.sec_per_play_last5, 1e-6)

    # Blend season neutral pace with recent pace factor
    blended_pace = 0.6 * our.neutral_pace + 0.4 * recent_pace_factor

    # Scale by your existing plays_est as a sanity prior
    return 0.5 * our.plays_est + 0.5 * blended_pace * 60.0 / 100.0  # 60 min game, rough


def project_game_script(our: TeamScriptFeatures, opp: TeamScriptFeatures) -> GameScriptProjection:
    """Core game-level engine implementing Section 1 & parts of Section 4.

    Rules baked in:
    - Success rate diff is the baseline script predictor.
    - Pressure diff modulates volatility and sack/INT expectations.
    - Recent pace (neutral_pace_last5, sec_per_play_last5) adjusts total volume.
    """
    success_diff = compute_success_diff(our, opp)
    pressure_diff = compute_pressure_diff(our, opp)

    lead_prob, neutral_prob, trail_prob = estimate_script_distribution(success_diff)
    base_plays = estimate_plays(our, opp)

    # PROE drives pass/run split
    # PROE ~ -0.1 .. +0.1; convert to pass share tweak
    base_pass_share = 0.55 + _clamp(our.pass_rate_over_expected, -0.1, 0.1)
    base_pass_share = _clamp(base_pass_share, 0.45, 0.65)

    projected_pass = base_plays * base_pass_share
    projected_rush = base_plays - projected_pass

    # Flags
    pressure_mismatch = abs(pressure_diff) >= 0.05
    blowout_risk = success_diff >= 0.06  # rough ~+6% success rate diff
    shootout_risk = (
        success_diff > -0.03 and success_diff < 0.03 and base_plays > 120
    )

    return GameScriptProjection(
        projected_plays=base_plays,
        projected_pass_attempts=projected_pass,
        projected_rush_attempts=projected_rush,
        lead_prob=lead_prob,
        neutral_prob=neutral_prob,
        trail_prob=trail_prob,
        pressure_mismatch=pressure_mismatch,
        blowout_risk=blowout_risk,
        shootout_risk=shootout_risk,
    )


# --------------------------------------------------------------------------------------
# WR role logic: WR1 / WR1.5 / Slot using existing LWR/RWR/SWR
# --------------------------------------------------------------------------------------

PERIMETER_POS = {"LWR", "RWR"}
SLOT_POS = {"SWR"}


@dataclass
class PlayerRoleRow:
    """Minimal view of a row from roles_ourlads.csv or similar."""
    team: str
    position: str      # "LWR", "RWR", "SWR", "RB", "TE", ...
    player: str        # already canonicalized full name
    depth: int         # 1 = starter, 2 = WR2 on that side, etc.


def identify_wr_roles(
    team: str,
    roles: List[PlayerRoleRow],
    usage_hint: Optional[Dict[str, float]] = None,
) -> Dict[str, Optional[PlayerRoleRow]]:
    """
    Identify conceptual WR1, WR1.5, and SLOT for a team using existing roles.

    - WR1     = perimeter starter (LWR/RWR) with highest usage_hint or lowest depth.
    - WR1_5   = the *other* perimeter starter.
    - SLOT    = SWR starter (depth 1), if present.

    usage_hint: optional dict player_name -> target_share or route_share
    that lets us pick WR1 more intelligently when we have data.

    Returns a dict with keys: "WR1", "WR1_5", "SLOT".
    Any missing roles are mapped to None instead of raising.
    """
    team_roles = [r for r in roles if r.team == team]

    perims = [r for r in team_roles if r.position in PERIMETER_POS and r.depth == 1]
    slots = [r for r in team_roles if r.position in SLOT_POS and r.depth == 1]

    # Choose WR1 among perimeter starters
    if usage_hint:
        # Prefer highest usage
        perims_sorted = sorted(
            perims,
            key=lambda r: usage_hint.get(r.player, 0.0),
            reverse=True,
        )
    else:
        # Fall back to arbitrary but stable ordering by position then depth
        perims_sorted = sorted(perims, key=lambda r: (r.position, r.depth))

    wr1 = perims_sorted[0] if perims_sorted else None

    wr1_5 = None
    if wr1:
        others = [r for r in perims_sorted if r.player != wr1.player]
        wr1_5 = others[0] if others else None

    slot = slots[0] if slots else None

    return {"WR1": wr1, "WR1_5": wr1_5, "SLOT": slot}


# --------------------------------------------------------------------------------------
# Matchup / funnel rules (Section 2)
# --------------------------------------------------------------------------------------


@dataclass
class MatchupMultipliers:
    """How much to bump usage/efficiency for each archetype."""
    wr1_target_mult: float = 1.0
    wr1_5_target_mult: float = 1.0
    slot_target_mult: float = 1.0
    te_target_mult: float = 1.0
    rb_rec_target_mult: float = 1.0
    rb_rush_eff_mult: float = 1.0
    sack_mult: float = 1.0
    int_mult: float = 1.0


def compute_matchup_multipliers(
    our: TeamScriptFeatures,
    opp: TeamScriptFeatures,
    pressure_diff: float,
) -> MatchupMultipliers:
    """
    Apply the model rules that depend on coverage, pressure, and trenches.
    (We keep it generic here; OL/DL metrics can be threaded in later.)

    Rules implemented:
    - Zone-heavy defenses funnel RB/TE receiving.
    - Man-heavy defenses isolate WR1 and open up WR1.5.
    - High pressure_diff boosts RB receiving, sacks, and INTs.
    - Success-rate / run metrics for RB rush efficiency can be layered in later.
    """
    zone = opp.coverage_zone_rate
    man = opp.coverage_man_rate
    middle_open = opp.middle_open_rate

    m = MatchupMultipliers()

    # Zone coverage: bump TE + RB receiving
    if zone >= 0.6:
        m.te_target_mult *= 1.15
        m.rb_rec_target_mult *= 1.20

    # Man coverage: WR1 is stressed, WR1.5 + slot gain leverage
    if man >= 0.5:
        m.wr1_target_mult *= 0.95
        m.wr1_5_target_mult *= 1.15
        m.slot_target_mult *= 1.05

    # Middle open: slot + TE in the crosshairs
    if middle_open >= 0.5:
        m.slot_target_mult *= 1.10
        m.te_target_mult *= 1.10

    # Pressure differential: RB receiving, sacks, INTs
    if pressure_diff > 0.05:
        # We pressure *them* more
        m.sack_mult *= 1.25
        m.int_mult *= 1.20
    elif pressure_diff < -0.05:
        # They pressure *us* more: our RB receiving goes up
        m.rb_rec_target_mult *= 1.25
        m.sack_mult *= 1.10
        m.int_mult *= 1.10

    # Clamp everything to reasonable bounds
    for field in m.__dataclass_fields__:
        val = getattr(m, field)
        setattr(m, field, _clamp(val, 0.5, 1.8))

    return m


# --------------------------------------------------------------------------------------
# Public entrypoints we will eventually call from the projection layer
# --------------------------------------------------------------------------------------


def compute_game_and_matchup(
    our: TeamScriptFeatures,
    opp: TeamScriptFeatures,
) -> Tuple[GameScriptProjection, MatchupMultipliers]:
    """
    Convenience function that runs both the game-level and matchup-level logic.

    This is the function we will eventually call from the projection engine
    once the pipeline is fully wired. For now, it's safe to import but unused.
    """
    script = project_game_script(our, opp)
    pressure_diff = compute_pressure_diff(our, opp)
    multipliers = compute_matchup_multipliers(our, opp, pressure_diff)
    return script, multipliers
