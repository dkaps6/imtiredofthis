# RB R26D — Exit Significance × R26 Effect Atlas V1 — Frozen Plan

Status: FROZEN BEFORE EXIT-SIGNIFICANCE OUTCOME SLICING
Date frozen: 2026-09-09
Parent production authority: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Research governance: `docs/research/RESEARCH_COMPONENT_PRESERVATION_DOCTRINE.md`

## Purpose

R26 V1 passed 19/20 frozen gates and improved vacancy-incumbent reception MAE in 5/6 seasons, but failed the no-season-worsens-more-than-2% protection because 2023 worsened 4.54%.

R26B localized the failure primarily to within-room allocation after Week 1 rather than RB-room receiving volume. R26C proved that the strictly-prior receiving significance of the departed back can be reconstructed without target-game participation/outcomes, sportsbook inputs, or same-week historical depth.

R26D is therefore a **no-refit diagnostic**. It asks whether the already-existing R26 V1 effect is systematically different when the departed RB represented meaningful receiving opportunity versus low/no receiving opportunity.

R26D does not create, refit, or score a new prediction model.

## Immutable parent evidence

R26 V1:
- workflow run: `34356222339`
- artifact ID: `10106271075`
- artifact name: `rb-r26-vacancy-gated-r9-retrospective-v1`
- digest: `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`

R26C source audit:
- workflow run: `34361409319`
- artifact ID: `10108036449`
- artifact name: `rb-r26c-exit-significance-source-audit-v1`
- digest: `sha256:1c7ee705179d91b7e3e53c39b57783314bd87876df6c784d5d1930ee5cf79156`
- source disposition: `EXIT_SIGNIFICANCE_SOURCE_READY`

Only files from those immutable artifacts may supply R26 predictions, labels already used by R26, or exited-player significance state.

## Parent mechanics that are frozen / not under test

R26D may not alter:

1. R9 feature set, coefficient fitting, reliability shrinkage, or residual clipping.
2. R26 V1 baseline or candidate predictions.
3. Non-vacancy behavior.
4. The finite RB target pool.
5. Team entitlement mass.
6. Non-RB entitlement.
7. Receiving-yard production means.
8. R22 authority.
9. The strict-prior/as-of identity contract.
10. Sportsbook separation.
11. The original R26 20-gate scientific contract.

R26 V1 remains `RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW` regardless of R26D findings.

## Primary predeclared exit-significance classifier

At the vacancy **team-week** level, use the R26C aggregate only.

A team-week is `MEANINGFUL_RECEIVING_EXIT` when at least one departed RB/FB with positive prior history satisfies either:

- `max_exit_prior_targets_pg > 1.0`, **OR**
- `max_exit_prior_rb_room_share >= 0.25`.

A team-week is `LOW_RECEIVING_EXIT` when exited-player prior history is available but neither condition is met.

A team-week is `UNKNOWN_EXIT_HISTORY` when no departed player has positive prior receiving history.

These values are frozen before R26D outcome grading. R26D may not search alternative target/share thresholds to improve the result.

### Why these cut points are football-natural

- More than one prior target per game indicates a recurring passing-game role rather than essentially unused roster depth.
- A quarter or more of the RB room target share indicates meaningful room responsibility even when total team pass volume is low.
- The OR rule allows an established passing-down role to qualify on either absolute usage or within-room responsibility.

## Secondary predeclared descriptive bins

These are reported for mechanism understanding only and may not be used to retroactively redefine the primary classifier:

Prior targets/game:
- `0`
- `(0,1]`
- `(1,2]`
- `>2`
- no prior history

Prior RB-room target share:
- `<0.10`
- `[0.10,0.25)`
- `[0.25,0.50)`
- `>=0.50`
- no prior history

Last-8 targets/game uses the same target bins as a corroborating recency view.

Phase:
- Week 1
- Weeks 2+

Role:
- RB1 incumbent
- RB2+ incumbent

## Analysis population

Primary population: R26 V1 rows satisfying:
- `vacancy_active == 1`
- `continuing_same_team == 1`

Join R26 rows to R26C team-week aggregates on exact `(season, week, team)`.

All six R26 test seasons remain visible: 2020, 2021, 2022, 2023, 2024, 2025.

No season may be removed because of its result.

## Effect metrics

For receptions and targets separately, calculate for baseline and frozen R26 candidate:

- n
- MAE
- RMSE
- signed bias
- p90 absolute error where sample size permits

Also calculate per-player:

`effect_delta_abs_error = abs(candidate - actual) - abs(baseline - actual)`

Negative values mean R26 improved the player-game; positive values mean R26 worsened it.

Report pooled and season-level results for:
- all vacancy incumbents
- primary significance class
- RB1/RB2+
- W1/W2+
- secondary bins.

## Predeclared replication test for child-candidate eligibility

R26D itself cannot authorize production or shadow use. It may only authorize a **separately frozen child-candidate design** if the primary classifier shows a reproducible interaction with the frozen R26 effect.

`SIGNIFICANCE_ROUTER_HYPOTHESIS_SUPPORTED` requires all of:

1. `MEANINGFUL_RECEIVING_EXIT` pooled vacancy-incumbent receptions MAE is improved by frozen R26 versus baseline.
2. The meaningful-exit reception MAE direction improves in at least **4 of 6** seasons.
3. Meaningful-exit pooled support is nontrivial: at least **200 player-games**, with at least **20 player-games in 4+ seasons**.
4. `LOW_RECEIVING_EXIT` is less favorable to R26 than `MEANINGFUL_RECEIVING_EXIT` in pooled mean absolute-error delta.
5. That meaningful-vs-low ordering appears in at least **3 of 6** season comparisons where both strata each have at least 15 player-games.
6. Evidence is not 2023-only: the meaningful-vs-low ordering must occur in at least one season from 2020-2022 and at least one season from 2024-2025.
7. Neither RB1 nor RB2+ meaningful-exit pooled receptions MAE may worsen by more than 1% versus baseline.
8. Week-1 meaningful-exit receptions MAE may not worsen by more than 0.5% versus baseline.

If all eight hold, disposition is:
- `SIGNIFICANCE_ROUTER_HYPOTHESIS_SUPPORTED`

Otherwise:
- `EXIT_SIGNIFICANCE_EFFECT_MIXED_NO_ROUTER`

Insufficient joined/source support yields:
- `EXIT_SIGNIFICANCE_EFFECT_INSUFFICIENT`

Even a supported R26D disposition **does not change R26 V1's failed disposition** and does not authorize production/shadow. It only permits a new frozen child candidate (expected name: R26E) that preserves R26's supported mechanics while replacing only the coarse activation gate.

## Child-candidate constraint if supported

Any R26E design must:
- inherit the original R26 parent mechanics unchanged;
- change only the vacancy activation eligibility layer;
- retain the original 20 R26 gates or stronger protections;
- score all six seasons, including 2023;
- remain retrospective evidence unless separately authorized prospectively;
- preserve failed/null evidence;
- never tune the significance threshold after R26D outcome results.

## Prohibited actions

- no new R9 fit;
- no R26 prediction regeneration;
- no coefficient shrink/rescue;
- no alternate significance threshold search;
- no season removal;
- no sportsbook input;
- no target-game participation as a feature;
- no same-week historical depth;
- no receiving-yard mean change;
- no R22 change;
- no production-file edits.
