# Model Rulebook Engine – Phase 1 Scaffold

This document explains how the new `scripts/model/rules_engine.py` module
connects the conceptual rulebook to our existing data model.

## Position Mapping

We **do not** introduce new position labels.

Existing positions from `roles_ourlads.csv`:

- `QB`, `RB`, `TE`, `FB`
- `LWR`, `RWR`, `SWR`

Conceptual roles used in the rulebook:

- **WR1**   = primary perimeter WR (one of `LWR` or `RWR`)
- **WR1.5** = other perimeter WR (the opposite side)
- **SLOT**  = `SWR` starter

The helper `identify_wr_roles(team, roles, usage_hint)` identifies these
conceptual roles **without** changing the underlying `position` values.

## Game-Level Engine

`TeamScriptFeatures` is a typed view of the columns we expect in `team_form`.

The `project_game_script(our, opp)` function implements:

- Success rate differential as primary script predictor.
- Pressure differential as volatility and sack/INT driver.
- Recent pace (`neutral_pace_last5`, `sec_per_play_last5`) as volume override.
- PROE as pass/run split driver.

It returns a `GameScriptProjection` with:

- `projected_plays`, `projected_pass_attempts`, `projected_rush_attempts`
- `lead_prob`, `neutral_prob`, `trail_prob`
- flags: `pressure_mismatch`, `blowout_risk`, `shootout_risk`

## Matchup Engine

`compute_matchup_multipliers(our, opp, pressure_diff)` implements:

- Zone-heavy defenses → TE + RB receiving bumps.
- Man-heavy defenses → WR1.5 + slot bumps, small WR1 nerf.
- Middle-open coverage → slot + TE bumps.
- Pressure differential → RB receiving / sacks / INT multipliers.

Output is a `MatchupMultipliers` dataclass with per-archetype scalars that
we can later apply to:

- target shares
- yards / target
- sack / INT pricing

## Integration Plan (Phase 2 – future patch)

In a later patch we will:

1. Construct `TeamScriptFeatures` from each row of `team_form.csv`.
2. Construct `PlayerRoleRow` objects from `roles_ourlads.csv`.
3. For each game:
   - Call `compute_game_and_matchup(our, opp)` to get script + multipliers.
   - Use `identify_wr_roles(team, roles, usage_hint)` to map `LWR/RWR/SWR`
     into conceptual WR1 / WR1.5 / SLOT.
   - Feed script + multipliers + WR roles into the projection engine that
     builds per-player usage (targets, rush attempts, etc.) and, ultimately,
     per-prop distributions.

For now, this module is **non-invasive**: it is safe to import, but the
GitHub Actions workflows do not call it yet.
