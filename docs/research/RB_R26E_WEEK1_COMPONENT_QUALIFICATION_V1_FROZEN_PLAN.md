# RB R26E — Week-1 Component Qualification V1 Frozen Plan

Status: FROZEN BEFORE SCORING
Date: 2026-09-09
Parent production authority: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Parent R26 workflow run: `34356222339`
Parent R26 artifact: `10106271075`
Parent artifact digest: `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`
Parent R26 disposition remains: `RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW`

## Scientific purpose

R26D showed that the original frozen R26 broad-vacancy/R9 mechanism behaves materially differently in Week 1 than in Weeks 2+. R26E asks one narrow question:

> Does the already-existing, already-frozen R26 Week-1 component replicate strongly enough across historical seasons to justify a 2026 Week-1 prospective shadow candidate?

R26E is a qualification study, not a new fitted model.

## Immutable candidate

R26E MUST NOT regenerate, refit, recalibrate, or alter R26 predictions.

The candidate is exactly the Week-1 slice of the immutable parent R26 artifact. The following parent mechanics are inherited unchanged:
- original broad vacancy gate: at least one canonical ACT/INA RB/FB room exit;
- original R9 receiving-identity redistribution;
- original R9 coefficients/reliability;
- fixed RB receiving target-pool mass;
- non-vacancy baseline-exact behavior;
- non-RB exactness;
- receiving-yard production mean exactness;
- R22 exactness;
- sportsbook inputs upstream = 0;
- target/future outcome leakage = 0.

The failed R26D all-season significance router is NOT part of R26E.

## Evaluation population

Historical test seasons: 2020, 2021, 2022, 2023, 2024, 2025.

Primary population:
- Week = 1 only;
- vacancy-active, same-team incumbent RB/FB rows (`vacancy_active == 1` and `continuing_same_team == 1`).

Secondary safety populations:
- all Week-1 RB/FB rows;
- Week-1 vacancy-active RB1 incumbents;
- Week-1 vacancy-active RB2+ incumbents.

No Week-2+ row may enter any R26E metric.

## Frozen metrics

For receptions and targets, compute:
- MAE;
- RMSE;
- signed bias;
- p90 absolute error.

For each test season, compute Week-1 vacancy-incumbent receptions MAE for baseline and frozen R26 candidate.

## Frozen qualification gates

All gates must pass for the disposition `WEEK1_COMPONENT_QUALIFIED_FOR_2026_PROSPECTIVE_SHADOW`.

Integrity / invariance:
1. parent artifact digest matches exactly;
2. all scored rows are Week 1;
3. sportsbook inputs remain zero;
4. future-outcome feature use remains zero;
5. no receiving-yard mean change is present in parent structural audit;
6. no R22 authority delta is present;
7. no protected production files are changed on the R26E branch.

Primary football performance:
8. pooled Week-1 vacancy-incumbent receptions MAE improves;
9. pooled Week-1 vacancy-incumbent receptions RMSE is non-worse;
10. pooled Week-1 vacancy-incumbent absolute reception bias is non-worse;
11. pooled Week-1 vacancy-incumbent reception p90 absolute error worsens by no more than 2%;
12. pooled Week-1 vacancy-incumbent target MAE improves.

Temporal replication:
13. Week-1 vacancy-incumbent reception MAE improves in at least 4 of 6 seasons;
14. no individual season's Week-1 vacancy-incumbent reception MAE worsens by more than 5%.

Role safety:
15. Week-1 vacancy RB1-incumbent reception MAE worsens by no more than 1%;
16. Week-1 vacancy RB2+-incumbent reception MAE worsens by no more than 1%;
17. at least one of the two role cohorts improves reception MAE.

Global Week-1 safety:
18. all-RB Week-1 reception MAE improves;
19. all-RB Week-1 reception RMSE worsens by no more than 0.25%;
20. all-RB Week-1 reception p90 absolute error worsens by no more than 1%.

## Interpretation

If all 20 gates pass:
- authorize only a `2026 Week-1 prospective shadow candidate`;
- do NOT promote to production from this retrospective qualification alone;
- preserve the original R26 full-candidate failure and R26D mixed result exactly.

If any gate fails:
- disposition is `WEEK1_COMPONENT_NOT_QUALIFIED_NO_SHADOW`;
- preserve any independently supported Week-1 subcomponents under the component-preservation doctrine;
- do not weaken gates, change cohorts, adjust R9, or introduce R26D significance thresholds to rescue the result.

## Production boundary

R26E may add only research docs, research evaluators, research workflow wiring, and output evidence. It may not modify production runtime/model/configuration files.
