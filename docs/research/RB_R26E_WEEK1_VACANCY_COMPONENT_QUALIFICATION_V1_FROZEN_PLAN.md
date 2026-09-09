# RB R26E — Week 1 Vacancy Component Qualification V1 — Frozen Plan

Status: FROZEN BEFORE SEASON-BY-SEASON WEEK-1 GRADING
Date frozen: 2026-09-09
Production authority: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Research governance: `docs/research/RESEARCH_COMPONENT_PRESERVATION_DOCTRINE.md`
Parent candidate: `RB_R26_VACANCY_GATED_R9_RETROSPECTIVE_V1`

## Purpose

R26 V1 passed 19/20 all-season gates and improved pooled Week-1 receptions materially. R26B/R26D showed that the primary failure is later-season vacancy/churn allocation rather than Week 1. R26D's predeclared phase atlas showed the frozen R26 mechanism improved Week-1 receptions in low-significance, meaningful-significance, and unknown-exit-history classes.

Under the component-preservation doctrine, R26E does not redesign R26. It qualifies only the **Week-1 component of the original frozen R26 V1 mechanism** across the six already-fixed seasons.

R26E is a no-refit/no-regeneration qualification. It cannot alter the R26 model, vacancy definition, R9 weights, predictions, or thresholds.

## Immutable parent evidence

R26 V1:
- run: `34356222339`
- artifact ID: `10106271075`
- artifact name: `rb-r26-vacancy-gated-r9-retrospective-v1`
- digest: `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`
- full candidate disposition remains `RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW`

R26D is diagnostic context only:
- run: `34364089085`
- artifact: `10109078127`
- disposition: `EXIT_SIGNIFICANCE_EFFECT_MIXED_NO_ROUTER`

R26E must grade Week 1 directly from immutable R26 V1 prediction and structural artifacts.

## Component being qualified

Exactly the original R26 V1 Week-1 behavior:

- non-vacancy RB rooms remain production-exact;
- any canonical ACT/INA RB/FB exit from the strict-prior roster snapshot activates the already-frozen R9 redistribution;
- R9 mechanics remain unchanged;
- only the already-finite RB target pool is redistributed;
- non-RB entitlement remains exact;
- team entitlement remains exact;
- receiving-yard point means remain exact;
- R22 remains exact;
- sportsbook inputs remain upstream zero;
- no target/future outcome is used as a feature.

No R26C/R26D exit-significance threshold is introduced into R26E.

## Test seasons and population

Seasons are fixed:
- 2020
- 2021
- 2022
- 2023
- 2024
- 2025

Primary Week-1 population:
- `week == 1`
- `vacancy_active == 1`
- `continuing_same_team == 1`

Secondary safety populations:
- all Week-1 RB/FB rows
- Week-1 vacancy RB1 incumbents
- Week-1 vacancy RB2+ incumbents

No season may be removed or downweighted after grading.

## Metrics

For receptions and targets where applicable:
- n
- MAE
- RMSE
- signed bias
- p90 absolute error

## Frozen Week-1 qualification gates

All gates must pass for `WEEK1_COMPONENT_RETROSPECTIVE_SUPPORT_FOR_2026_PROSPECTIVE_SHADOW`.

### Integrity / inheritance
1. Immutable R26 artifact digest matches exactly.
2. R26 predictions are not regenerated and R9 is not refit.
3. Sportsbook inputs added = 0; production parameters changed = false.
4. Parent R26 structural audits remain exact for RB-room mass, non-RB entitlement, receiving-yard means, and R22.

### Primary Week-1 vacancy-incumbent quality
5. Pooled Week-1 vacancy-incumbent receptions MAE improves.
6. Pooled Week-1 vacancy-incumbent receptions RMSE is non-worse.
7. Pooled Week-1 vacancy-incumbent absolute bias is non-worse.
8. Pooled Week-1 vacancy-incumbent receptions p90 may worsen by no more than 2%.
9. Pooled Week-1 vacancy-incumbent target MAE improves.

### Temporal replication
10. Week-1 vacancy-incumbent receptions MAE improves in at least **4 of 6 seasons**.
11. No individual Week-1 season may worsen vacancy-incumbent receptions MAE by more than **2%**.
12. At least **4 of 6 seasons** must contain at least 20 Week-1 vacancy-incumbent player rows; pooled Week-1 vacancy-incumbent support must be at least 150 rows.

### Role protection
13. Week-1 vacancy RB1 incumbent receptions MAE may worsen by no more than 1%.
14. Week-1 vacancy RB2+ incumbent receptions MAE may worsen by no more than 1%.
15. At least one of Week-1 vacancy RB1 or RB2+ receptions MAE improves.

### Global Week-1 safety
16. All Week-1 RB/FB receptions MAE may worsen by no more than 0.5%.
17. All Week-1 RB/FB receptions RMSE may worsen by no more than 0.5%.
18. All Week-1 RB/FB absolute reception bias is non-worse.

## Dispositions

If all frozen gates pass:
- `WEEK1_COMPONENT_RETROSPECTIVE_SUPPORT_FOR_2026_PROSPECTIVE_SHADOW`

If integrity fails:
- `WEEK1_COMPONENT_INTEGRITY_FAILURE`

Otherwise:
- `WEEK1_COMPONENT_MIXED_OR_FAIL_NO_SHADOW`

A passing R26E authorizes only the design/execution of a **2026 Week-1 prospective shadow** of this preserved R26 component. It does not authorize production promotion, all-season use, or changing the canonical MC/receiving-yard/R22 paths.

## If R26E passes

The next step must be a separately frozen prospective shadow plan that:
- uses 2026 pregame data only;
- produces RB receptions/target-entitlement shadow outputs without altering production authority;
- verifies current roster/identity coverage and exact finite-room conservation;
- preserves R22 receiving-yard distribution/mean behavior;
- does not use sportsbook lines to construct projections;
- records the prospective lock time before Week-1 outcomes.

## If R26E fails

Preserve all Week-1 subcomponents that passed and isolate the exact season/role/safety failure under the component-preservation doctrine. Do not discard the R26 parent mechanics wholesale and do not relax these gates after viewing the results.

## Prohibited actions

- no new model fit;
- no R9 coefficient/reliability change;
- no significance-router threshold;
- no season removal;
- no Week-1 threshold tuning;
- no sportsbook input;
- no production write;
- no receiving-yard mean change;
- no R22 change;
- no changing the original R26 full-candidate failed disposition.
