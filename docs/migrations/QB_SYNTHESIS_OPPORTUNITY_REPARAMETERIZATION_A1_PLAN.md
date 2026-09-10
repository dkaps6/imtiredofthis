# QB Synthesis Opportunity Reparameterization A1 — Frozen Diagnostic Plan

## Purpose

Test whether the already-promoted M89/M90 **pregame passing-yard synthesis mean** contains reusable team-pass-opportunity information that can be re-expressed upstream without changing the QB passing-yard mean.

This is a deterministic architecture diagnostic. It fits no model, changes no production code, and uses no sportsbook input.

The immediate motivation is the failed `QB_TEAM_PASS_OPPORTUNITY_SCHEDULE_REST_D1`: raw team pass opportunity and QB attempts improved, but adding that correction beneath the unchanged M89/M90 synthesis adjustment double-counted downstream correction and worsened passing-yard accuracy.

A1 asks a narrower question:

> If the final M89/M90 passing-yard mean is held fixed and current predicted YPA is held fixed, what QB attempt/team-pass-opportunity volume is already implied by the promoted synthesis mean?

## Canonical lineage

- D1 result commit: `27d4b44d16dd321346af6919a5c6fb8e0c7cb6f6`
- D1 run: `34533794810`
- D1 artifact: `10174577290`
- D1 disposition: `SCHEDULE_REST_D1_FAIL_NO_CONFIRMATION`
- Opportunity-chain run: `34523313743`
- Opportunity-chain artifact: `10170531084`
- Opportunity-chain digest: `sha256:75cd32198caf7d9cbf193d5769e5762acce53352f2b2626cd92a9b117e75dafd`
- Shared QB/WR source run: `34066549394`
- Shared QB/WR artifact: `9999119623`
- Shared QB/WR digest: `sha256:4e148f982d2f8db8a8e19cd2bbe2dbeb775dee296efc01aaf37193d98db2ddc6`

## Frozen deterministic reparameterization

For every M89-aligned QB-game:

- current promoted QB passing-yard mean: `YARDS_M89 = football_synthesis`
- current pregame predicted YPA: `YPA0 = pred_ypa`
- current pregame predicted attempt conversion: `C0 = pred_C`
- current pregame predicted primary-QB attempt share: `S0 = pred_S`
- current predicted attempts: `A0 = pred_attempts`
- current predicted team pass opportunity: `D0 = pred_D`

Define exactly one implied-volume state:

`A1_IMPLIED_ATTEMPTS = YARDS_M89 / YPA0`

`D1_IMPLIED_PASS_OPPORTUNITY = A1_IMPLIED_ATTEMPTS / (C0 * S0)`

No clipping, shrinkage, blend, coefficient, threshold, or alternate YPA denominator is authorized.

The implied attempt correction is:

`IMPLIED_ATTEMPT_CORRECTION = A1_IMPLIED_ATTEMPTS - A0`

The implied team-pass-opportunity correction is:

`IMPLIED_D_CORRECTION = D1_IMPLIED_PASS_OPPORTUNITY - D0`

QB passing-yard mean must remain exactly unchanged by construction:

`A1_IMPLIED_ATTEMPTS * YPA0 = football_synthesis`

within floating-point tolerance.

## Cohorts

Primary architecture diagnostic:
- exact 884 M89/M90 aligned QB-games, 2024-2025.
- report 2024, 2025, and pooled.

Shared receiver diagnostic:
- exact 440-row 2025 WR target-mass cohort from the immutable shared-volume artifact.
- exact 884-row 2024-2025 WR reception-mass replication cohort.

No row exclusions may be chosen after results are visible.

## Frozen metrics

### QB attempts
Compare `A0` vs `A1_IMPLIED_ATTEMPTS` against actual official QB attempts:
- MAE
- RMSE
- bias
- correlation
- p90 absolute error
- 8+ miss rate
- 10+ miss rate

### Team pass opportunity
Compare `D0` vs `D1_IMPLIED_PASS_OPPORTUNITY` against corrected actual team pass opportunities:
- MAE
- RMSE
- bias
- correlation
- p90 absolute error

### Correction information content
For the implied attempt correction versus actual attempt residual (`actual_attempts - A0`), report by 2024, 2025, pooled:
- Pearson
- Spearman
- same-sign rate
- mean correction
- mean absolute correction
- p90 absolute correction

### Shared receiving opportunity
On the exact 2025 WR target-mass cohort, correlate `IMPLIED_ATTEMPT_CORRECTION` with `wr_target_mass_residual`:
- Pearson
- Spearman
- same-sign rate
- signed correction Q4-minus-Q1 WR residual gap

Repeat on the 2024-2025 WR reception-mass replication cohort with pooled and season splits.

## Integrity gates

Scientific interpretation stops unless all pass:

1. exact 884 M89 QB rows;
2. exact 440 primary WR-target rows and 884 secondary WR-reception rows;
3. zero sportsbook inputs;
4. zero model fitting;
5. no production change;
6. all `pred_ypa > 0` and all `pred_C * pred_S > 0`;
7. current attempt identity `D0*C0*S0 = A0` max abs error <= `1e-6`;
8. implied mean identity `A1_IMPLIED_ATTEMPTS*YPA0 = football_synthesis` max abs error <= `1e-6`;
9. implied-D identity `D1_IMPLIED_PASS_OPPORTUNITY*C0*S0 = A1_IMPLIED_ATTEMPTS` max abs error <= `1e-6`;
10. exact shared-volume source residuals reproduce prior canonical total correlations within `1e-9`;
11. target outcomes are diagnostic labels only and never inputs to the implied state.

## Frozen support gates

Disposition `M89_SYNTHESIS_CONTAINS_REUSABLE_OPPORTUNITY_SIGNAL` requires all:

1. pooled QB attempt MAE improves by >= `0.25` attempts;
2. QB attempt MAE is non-worse in both 2024 and 2025;
3. pooled team-pass-opportunity MAE improves by >= `0.25` opportunities;
4. team-pass-opportunity MAE is non-worse in both 2024 and 2025;
5. pooled implied-correction Spearman vs actual attempt residual >= `0.20`;
6. implied-correction Spearman vs actual attempt residual >= `0.10` in both 2024 and 2025;
7. 2025 implied-correction Spearman vs WR target-mass residual >= `0.20`;
8. pooled 2024-2025 implied-correction Spearman vs WR reception-mass residual >= `0.15`;
9. QB 10+ attempt miss rate does not worsen pooled;
10. QB passing-yard mean preservation identity passes exactly;
11. all integrity gates pass.

If any support gate fails, disposition is:

`M89_SYNTHESIS_OPPORTUNITY_REALLOCATION_NOT_SUPPORTED`

## What a pass would mean

A pass would **not** authorize production integration by itself.

It would authorize one subsequent mean-neutral shared-opportunity integration test in which:

- QB passing-yard mean remains exactly M89/M90;
- the implied M89 attempt/pass-opportunity state becomes the finite team passing-opportunity pool;
- existing player entitlement models allocate that pool;
- receiver targets/receptions/yards are evaluated under existing M38 / WR-R15 / TE-R5P / RB-R26 logic;
- no new QB mean correction is added.

## Stopping rule

Run this single deterministic reparameterization once.

Do not:
- test alternate YPA anchors;
- blend A0 and A1;
- add D1 schedule/rest correction;
- fit an allocation coefficient;
- tune a cap;
- use sportsbook fields;
- change M89/M90 mean.
