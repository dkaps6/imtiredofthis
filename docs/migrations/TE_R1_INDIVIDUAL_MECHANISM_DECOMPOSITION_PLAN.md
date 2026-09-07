# TE-R1 Individual Receiving Mechanism Decomposition — Frozen Plan

## Status
Frozen before results. Diagnostic only. Production unchanged.

## Purpose
Start the dedicated TE research lane by decomposing individual TE receiving-yard projection error into football mechanisms rather than copying WR coefficients. Use the exact 2020-2025 player casebook from the validated Joint Pass/Receiving Conservation V1 run.

## Lineage
- Source run: Joint Pass/Receiving Conservation V1 canonical run `34081764151`.
- Source artifact: `10004223287`, digest `sha256:753aa191e6c80a059918553d8567499c9cdbce82b12a26b4d5b19b21225764ac`.
- Baseline receiving architecture: B0, preserving promoted M38 WR work and current TE Bayesian/rules inputs.
- C2 conservation is not used to define the mechanism decomposition because TE-R1 is diagnosing the current TE point-projection mechanics first.
- Sportsbook inputs: 0.

## Cohort
- Seasons 2020-2025 regular season from `joint_v1_paired_player_casebook.csv`.
- `position_group == TE` only.
- Player-games must have finite B0 expected targets, B0 receptions, B0 receiving yards, actual targets, actual receptions, and actual receiving yards.
- No target-game outcome may enter any projection-side input.

## Three-factor football identity
Represent receiving yards as:

`receiving_yards = targets * catch_rate * yards_per_reception`

Pregame baseline components:
- `pred_targets = b0_expected_targets`
- `pred_catch_rate = b0_receptions / b0_expected_targets` when expected targets > 0; otherwise bounded neutral fallback from the observed B0 player projection identity.
- `pred_ypr = b0_rec_yards / b0_receptions` when projected receptions > 0; otherwise neutral fallback that preserves zero-projection semantics.

Actual components:
- `actual_targets = targets`
- `actual_catch_rate = receptions / targets` when targets > 0; when targets == 0, actual receiving yards must be 0 and the row is retained for opportunity error but catch/YPR components are defined by exact permutation accounting with zero opportunity.
- `actual_ypr = rec_yards / receptions` when receptions > 0; otherwise zero-outcome semantics.

Use an exact three-factor Shapley decomposition across all 3! permutations. For each player-game, compute the contribution to `actual_rec_yards - b0_rec_yards` from:
1. TARGETS
2. CATCH_RATE
3. YPR

The three components must sum to the exact receiving-yard residual within numerical tolerance.

## Frozen outputs
Pooled and by season:
- B0 TE target / reception / receiving-yard MAE, RMSE, bias, correlation;
- median / p75 / p90 receiving-yard absolute error;
- 20+ / 30+ / 40+ receiving-yard miss rates;
- mean absolute Shapley contribution by mechanism;
- fraction of absolute mechanism mass by mechanism.

Player profiles:
- require >= 20 scoreable TE games across 2020-2025;
- report mean/median absolute TARGETS, CATCH_RATE, YPR components;
- assign dominant mechanism only when one mechanism contributes >=45% of that player's total absolute Shapley mass; otherwise MIXED;
- report player MAE, p90 error, 30+/40+ miss rates, and seasons represented.

Also report the dominant mechanism among the highest-error quartile of TE player-games, using the absolute Shapley component largest for that game.

## Frozen interpretation gates
This is a diagnostic, not a candidate-model pass/fail. It supports a next predictive lane only if:
1. at least 4,000 TE player-games are scoreable across 2020-2025;
2. all six seasons are represented with >=500 TE player-games each;
3. Shapley reconstruction max absolute error <= 1e-6 yards;
4. at least 40 TE players have >=20 scoreable games;
5. one mechanism accounts for >=45% of pooled absolute Shapley mass OR the top two mechanisms together account for >=75%, creating a sufficiently concentrated next research target.

If those gates pass: `TE_MECHANISM_DECOMPOSITION_ACTIONABLE` and the next TE experiment must target the identified mechanism(s) with leakage-safe pregame information.
If not: `TE_MECHANISM_DECOMPOSITION_TOO_DIFFUSE`; do not invent a coefficient search.

## Production
No production change authorized.
