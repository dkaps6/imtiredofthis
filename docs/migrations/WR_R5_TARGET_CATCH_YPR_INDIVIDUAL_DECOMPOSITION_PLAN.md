# WR-R5 — Target / Catch / YPR Individual Mechanism Decomposition

## Status

`PREREGISTERED / DIAGNOSTIC ONLY`

## Why this exists

WR-R4 established that, among 337 WRs with at least eight scoreable games in the 2020-2025 paired M38 casebook, 226 were reception-dominant, 83 mixed, and only 28 YPR-dominant. That result says *reception volume* is the dominant recurring yardage-error family for most qualifying WRs, but it does not distinguish whether the reception error is driven primarily by target opportunity or target-to-catch conversion.

WR-R5 answers that narrower question at the player-game and individual-player level.

This is not a new broad WR feature hunt. It is a mechanical decomposition of the already-frozen M38 projection error.

## Frozen inputs

1. Canonical WR-ND5 artifact from run `34055534002`, containing the exact 2025 M38 target reconstruction casebook (`wr_nd5_casebook.csv`).
2. Canonical WR-R1 multi-season artifact from run `34058453941`, containing the paired M38 receptions and receiving-yards casebook (`wr_r1_paired_wr_casebook.csv`).
3. Filter WR-R1 to 2025 WR rows and merge to the ND5 2025 WR casebook by season/week/team/player identity.

No sportsbook data. No target-game feature is used as a predictor. Actual outcomes are used only for postgame decomposition.

## Frozen integrity requirements

- ND5 target rows must equal `2130` before the canonical Bond anomaly exclusion behavior already inherited in the ND5 artifact.
- WR-R1 2025 receptions and receiving-yards rows must merge one-to-one on player-game identity for the available ND5 cohort.
- Projected yards must reconcile exactly as:
  `pred_targets * projected_catch_rate * projected_ypr = projected_receiving_yards`
  where projected catch rate and projected YPR are algebraically derived from the frozen M38 projected receptions/yards.
- Actual yards must reconcile exactly as:
  `actual_targets * actual_catch_rate * actual_ypr = actual_receiving_yards`
  with zero-opportunity states handled deterministically.
- The sum of the three Shapley components must reconstruct the receiving-yard residual to numerical tolerance `1e-6`.

## Frozen decomposition

For each player-game define:

- `T` = targets
- `C` = catch rate = receptions / targets
- `Y` = yards per reception = receiving yards / receptions

Receiving yards are `T * C * Y`.

Use the exact three-factor Shapley decomposition of:

`actual_yards - projected_yards`

across the three factors:

1. `TARGET_COMPONENT`
2. `CATCH_COMPONENT`
3. `YPR_COMPONENT`

The Shapley value for each factor is the average marginal contribution of replacing that factor from projected to actual over all six permutations of the three factors. This gives an exact additive decomposition and avoids arbitrary ordering.

## Individual-player profiles

Minimum qualifying sample: `8` merged 2025 games.

For each qualifying WR report:

- games
- receiving-yard MAE and bias
- target MAE and bias
- reception MAE and bias
- mean and mean-absolute TARGET component
- mean and mean-absolute CATCH component
- mean and mean-absolute YPR component
- component shares of total mean-absolute decomposed error
- miss rates at 20/30/40 receiving yards
- dominant mechanism

Frozen dominant-mechanism rule:

- `TARGETS` if target-component absolute mean is >= `1.25x` each other component.
- `CATCH` if catch-component absolute mean is >= `1.25x` each other component.
- `YPR` if YPR-component absolute mean is >= `1.25x` each other component.
- otherwise `MIXED`.

This classification is descriptive. It does not authorize a player-specific hand-tuned correction.

## Research interpretation rule

The purpose is to identify *which football mechanism repeatedly drives each player's errors* so later pregame research can be conditioned on the relevant mechanism and role rather than treating all WRs as one homogeneous population.

A player profile may guide which already-supported football feature families deserve testing for that mechanism, but:

- no per-player constants are fit;
- no postgame outcome becomes an upstream feature;
- no feature is promoted merely because it explains one famous player;
- any later feature must be leakage-safe, predeclared, and evaluated out of sample in the full WR stack.

## Position independence

WR-R5 advances the WR model only. QB-WR cross-position work remains a separate bridge lane and may not change WR-R5 results or thresholds.

## Production rule

`production_changed = false`

A decomposition result is not a model win. It can only authorize a later frozen pregame signal/integration experiment at the natural football layer (targets, catch conversion, or YPR/distribution).