# WR Player Mechanism Persistence V1 — Frozen Diagnostic Plan

**STATUS: FROZEN BEFORE RESULT. DIAGNOSTIC ONLY. NO PRODUCTION CHANGE.**

## Purpose

Apply the same individual-player mechanism decomposition used for TE to the exact promoted WR architecture.

Question:

> After M38 WR1 anchoring and WR-R15 individualized WR2+ entitlement are already applied, does an individual WR still show stable same-player opportunity error and/or efficiency difficulty?

This tests the user's player-centric hypothesis at the WR level. It does not treat WRs as one homogeneous position beyond using the promoted WR stack as the parent projection.

## Exact parent authority

WR-R15 OOS production-certification authority:
- run `34238301577` — SUCCESS
- artifact `10061328722`
- digest `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`
- file `backtests/wr_r15_wr1_anchor_v1/wr_r15_confirmation_predictions.csv`
- required variant `WR_R15_WR1_ANCHORED_PARTICIPATION`

Frozen authority invariants:
- 4,193 candidate player-games;
- 2023: 2,076;
- 2024: 2,117;
- receiving-yard MAE: `22.52635823935656`;
- receiving-yard RMSE: `31.988194283296206`;
- target MAE: `2.0438856296229853`;
- 273 distinct WR identities;
- zero duplicate season/week/team/player rows;
- no 2025 confirmation rows.

The promoted architecture is:
- WR1 = M38 anchor;
- WR2+ = WR-R15 participation entitlement inside the conserved WR room.

## Protected / closed WR science

Preserve:
- M38 WR1 hierarchy;
- WR-R15 production entitlement;
- WR-R3 same-player error persistence finding;
- WR-R3 combined calibration = CLOSED;
- WR-R3 GBM/Ridge rescue family = CLOSED;
- exact QB-C2 -> WR1 shared-tail selector = CLOSED;
- generic coverage-rate WR efficiency feature = already tested / near-null;
- current production weights and simulation.

This diagnostic may explain the WR-R3 persistence signal but does not reopen its failed calibration candidate.

## Exact decomposition

For each candidate row:

`pred_targets = pred_targets`

`actual_targets = actual_targets`

`pred_ypt = mc_rec_yards / pred_targets`

When actual targets > 0:

`actual_ypt = actual_rec_yards / actual_targets`

When actual targets = 0:
- `actual_ypt = 0`;
- efficiency error is mechanically zero because it is multiplied by actual targets.

Require finite positive predicted targets.

Define:

`opportunity_error = (pred_targets - actual_targets) * pred_ypt`

`efficiency_error = actual_targets * (pred_ypt - actual_ypt)`

Identity:

`opportunity_error + efficiency_error = mc_rec_yards - actual_rec_yards`

must hold within `1e-9` on every scoreable authority row.

This is accounting decomposition, not causal attribution.

## Same-player pregame history

Target seasons:
- 2023
- 2024

For every target row:
- use only the same WR's exact promoted-authority rows strictly before the target row;
- history may cross seasons;
- latest up to 8 prior WR authority games;
- minimum 4 prior games.

Features:
- prior8 opportunity signed-error mean;
- prior8 absolute opportunity-error mean;
- prior8 efficiency signed-error mean;
- prior8 absolute efficiency-error mean.

No target/future row may enter its own feature.

## Frozen mechanism tests

### O1 — Opportunity signed persistence
Spearman(prior opportunity error mean, current opportunity error).

PASS only if:
- rho > 0 in 2023;
- rho > 0 in 2024;
- pooled player-cluster bootstrap P(rho > 0) >= 0.95.

### O2 — Opportunity difficulty persistence
Spearman(prior mean absolute opportunity error, current absolute opportunity error).

Same gates.

### E1 — Efficiency signed persistence
Spearman(prior efficiency error mean, current efficiency error).

Same gates.

### E2 — Efficiency difficulty persistence
Spearman(prior mean absolute efficiency error, current absolute efficiency error).

Same gates.

## Support

Each target season must have:
- >= 800 scoreable player-games;
- >= 100 distinct WR identities.

If this misses, disposition is source-limited rather than null.

## Player-cluster bootstrap

- cluster = `player_clean_key`;
- 5,000 replicates;
- seed `20261007`;
- resample WR identities with replacement;
- retain all eligible rows for each sampled WR;
- separately evaluate O1/O2/E1/E2.

## Role diagnostics

Report separately but do not use as rescue gates:
- WR1 (`wr_rank == 1`);
- WR2+ (`wr_rank >= 2`).

These are interpretive diagnostics only. If the pooled mechanism fails, a role subgroup may not rescue it under V1.

## Error-mass diagnostics

Per season and pooled report:
- mean absolute total receiving-yard error;
- mean absolute opportunity component;
- mean absolute efficiency component;
- normalized opportunity mass;
- normalized efficiency mass;
- component sign-cancellation rate.

## Disposition

`WR_PLAYER_PERSISTENCE_EFFICIENCY_DOMINANT`
if support passes, at least one E family passes, and neither O family passes.

`WR_PLAYER_PERSISTENCE_OPPORTUNITY_DOMINANT`
if support passes, at least one O family passes, and neither E family passes.

`WR_PLAYER_PERSISTENCE_MIXED_MECHANISM`
if support passes and at least one O family plus at least one E family pass.

`WR_PLAYER_PERSISTENCE_MECHANISM_UNRESOLVED`
if support passes and none pass.

`WR_PLAYER_PERSISTENCE_MECHANISM_SOURCE_LIMITED`
if support fails.

## Boundaries

A positive result does not authorize:
- direct same-player residual correction;
- target-share patch;
- YPT patch;
- width multiplier;
- WR1/WR2 carveout;
- sportsbook conditioning;
- reopening WR-R3 combined calibration.

A positive opportunity result authorizes only a later search for genuine pregame player-specific role/opportunity state missing from M38/WR-R15.

A positive efficiency-difficulty result authorizes only a later search for genuine player-specific football characteristics that explain outcome dispersion.

Models fit: **0**  
Sportsbook inputs: **0**  
2026 outcomes: **0**  
Production mutations: **0**
