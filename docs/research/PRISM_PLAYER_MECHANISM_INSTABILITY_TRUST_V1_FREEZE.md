# PRISM Player Mechanism-Instability Trust V1 — Frozen Test

Date: 2026-10-17

Status: **FROZEN BEFORE SCORING / HISTORICAL CONFIRMATION / TRUST ONLY**

Model:
**PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_PLAYER_MECHANISM_INSTABILITY_TRUST_V1`

Source authority:
- source audit run `39061606501`
- exact head `5f24be29f91345e4181cea7095bf7402233d4bb6`
- disposition `PRISM_PLAYER_MECHANISM_INSTABILITY_TRUST_V1_SOURCE_READY_WR_TE`

## Scientific question

The Week-4/5 forensic replay showed that the same named player can fail through a
different mechanism from week to week.

This test asks:

> Does a named player's strictly-prior tendency to switch between
> opportunity-dominant, efficiency-dominant and mixed failure states identify a
> future receiving-yard projection as more fragile, **after controlling for the
> already-known magnitude of that player's opportunity difficulty and efficiency
> difficulty**?

This is a reliability/trust test only.

It may not change the mean, width, sportsbook price or selection signal.

## Candidate families

Primary families:
- WR receiving yards
- TE receiving yards

Excluded:
- RB: source not proven for this exact V1
- QB: excluded by anti-retest / prior reliability closure

## Exact component-state label

Use the already-frozen Week-4/5 dominance convention without alteration.

For each historical player-week:

- `O = abs(current_opportunity_error)`
- `E = abs(current_efficiency_error)`

Label:
- `OPPORTUNITY_DOMINANT` if `O >= 1.5 * E`
- `EFFICIENCY_DOMINANT` if `E >= 1.5 * O`
- otherwise `MIXED_OPPORTUNITY_EFFICIENCY`

No label threshold search is permitted.

The exact identity remains:

`current_total_error = current_opportunity_error + current_efficiency_error`.

## Strictly-prior mechanism-instability state

For each target player-week:
- same player
- same season
- rows strictly before target week
- latest up to 8 prior component-state rows
- minimum 4 prior rows

Primary candidate:

`mechanism_switch_rate = number of adjacent prior-state changes / (prior_rows - 1)`

Examples:
- O,O,O,O -> 0.00
- O,E,O,E -> 1.00

The target-week mechanism label is never used to construct the predictor.

Secondary diagnostics only:
- prior dominant-state share
- prior count of each mechanism state
- current target-week mechanism label

No alternate entropy/Gini/transition weighting may rescue V1.

## Existing-trust controls

To prove novelty beyond already-known player difficulty, all variables are
ranked within exact season/week and the primary statistic is:

`partial_rank_corr(mechanism_switch_rate, current_abs_total_error |
                   prior8_abs_opportunity_error_mean,
                   prior8_abs_efficiency_error_mean)`

This is an association test, not a predictive fit.

Secondary:
- unadjusted within-week rank association
- partial association with current absolute opportunity component
- partial association with current absolute efficiency component

Those secondaries cannot rescue a failed primary result.

## Chronology

WR:
- 2023 development
- 2024 independent confirmation

TE:
- 2024 development
- 2025 independent confirmation

No pooled-season rescue if the independent confirmation fails.

## Frozen qualification gates — per position

A position qualifies only if ALL pass:

1. source-ready support already certified;
2. development primary partial rank association > 0;
3. confirmation primary partial rank association > 0;
4. confirmation player-cluster bootstrap 95% CI lower bound > 0;
5. confirmation team-week-cluster bootstrap 95% CI lower bound > 0;
6. chronology violations = 0;
7. decomposition identity gap <= 1e-10.

Bootstrap:
- 5,000 deterministic replicates
- fixed seed 20261017 for player clusters
- fixed seed 20261018 for team-week clusters

No minimum effect-size threshold beyond positive dependence-aware confirmation.

## Anti-rescue

If a position fails:
- do not change 1.5 dominance ratio;
- do not change 8-game window;
- do not lower 4-prior-row minimum;
- do not pool seasons to rescue;
- do not switch to entropy/Gini/majority-share;
- do not carve WR1/WR2+/TE role subsets;
- do not remove existing-difficulty controls;
- do not use 2026 Week-4/5 outcomes to refit the candidate.

If it passes:
- qualification is historical reliability evidence only;
- no production change;
- any 2026 use requires a separately frozen prospective pregame shadow;
- do not combine with the fixed WR dual reliability score without a separate
  preregistered test.

## Integrity

At freeze:
- candidate scored: 0
- 2026 outcomes used: 0
- Week-6 outcomes used: 0
- sportsbook inputs: 0
- OddsAPI calls: 0
- predictive models fit: 0
- projection/distribution changes: 0
- production changes: false

Disposition:
`PRISM_PLAYER_MECHANISM_INSTABILITY_TRUST_V1_FROZEN`.
