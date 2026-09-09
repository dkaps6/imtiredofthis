# RB R26G — Week 1 Balanced-Turnover Guard V1 — Frozen Plan

Status: FROZEN BEFORE R26G CHILD-PREDICTION GRADING
Date frozen: 2026-09-09
Production authority: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Research governance: `docs/research/RESEARCH_COMPONENT_PRESERVATION_DOCTRINE.md`
Parent mechanism: `RB_R26_VACANCY_GATED_R9_RETROSPECTIVE_V1`
Forensic authorization: `RB_R26F_2020_WEEK1_VACANCY_FAILURE_FORENSIC_ATLAS_V1`

## Purpose

R26E passed 17/18 Week-1 qualification gates but failed because 2020 Week-1 vacancy-incumbent receptions MAE worsened 8.97%. R26F then established that the 2020 failure was **within-room allocation dominant**, while the room-total reception error actually improved.

R26F also identified a football-grounded pregame state that met the predeclared cross-season replication rule:

`room_exits_n == room_entrants_n`

Balanced turnover accounted for about 51% of the 2020 net worsening and showed the same harmful R26 direction in 2021 and 2023. This can represent a replacement/churn state rather than true net-open RB-room opportunity.

R26G is the first child candidate under the component-preservation doctrine. It preserves every supported R26 Week-1 mechanism except the activation behavior in this one replicated guard state.

## Immutable evidence

R26 V1:
- run `34356222339`
- artifact `10106271075`
- digest `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`
- disposition remains `RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW`

R26E:
- run `34364872300`
- artifact `10109398212`
- digest `sha256:4d136cf20bccebbdd874b24392d95669d11ad01e55b4ac4af6cdee06e7e99651`
- disposition remains `WEEK1_COMPONENT_MIXED_OR_FAIL_NO_SHADOW`

R26F:
- run `34365496225`
- artifact `10109658591`
- digest `sha256:bce3f1a81ff516f962109dcadc9ad1e6efa2273c3a96851e94abbe9e4c332022`
- disposition `WEEK1_FAILURE_MECHANISM_IDENTIFIED_REPLICATED`
- child-candidate design authorized `true`

## Frozen R26G child rule

Apply only to Week 1.

At the complete RB/FB room `(season, week=1, team)` level:

1. `vacancy_active == 0`
   - child = production baseline exactly.

2. `vacancy_active == 1` and `room_exits_n != room_entrants_n`
   - child = original frozen R26 candidate exactly.

3. `vacancy_active == 1` and `room_exits_n == room_entrants_n`
   - child = production baseline exactly for **every RB/FB row in that team-week room**.

The fallback must be room-complete, not player-specific. Mixing R26 and baseline players within a guarded room is prohibited because it could break finite RB target-pool conservation.

No R9 coefficient, reliability rule, feature, clip, prediction, or identity score is changed.

## Why this is a legitimate child delta

The guard was not selected from an unrestricted search. It came from a predeclared R26F forensic dimension and met a predeclared cross-season replication requirement. It is also football-interpretable: balanced exits/entrants often describe direct room replacement rather than an unfilled opportunity vacuum.

The alternative replicated R26F state, R9 residual-magnitude Q3, is deliberately **not** used in R26G because a model-internal quantile magnitude is less football-natural and carries more overfitting risk than roster-room balance.

## Scientific label

R26G is a **retrospective child-candidate qualification on previously exposed historical seasons**.

A pass cannot be called independent historical confirmation and cannot authorize production. At most it may authorize a separately frozen **2026 Week-1 prospective shadow** before outcomes.

## Test population

Fixed seasons:
- 2020
- 2021
- 2022
- 2023
- 2024
- 2025

Primary:
- Week 1
- vacancy-active
- same-team incumbents

Safety:
- all Week-1 RB/FB rows
- vacancy RB1 incumbents
- vacancy RB2+ incumbents
- guarded balanced-turnover rooms
- unguarded vacancy rooms

No season may be removed or downweighted.

## Frozen qualification gates

### Inheritance / structural integrity
1. R26 immutable artifact digest matches exactly.
2. R26F immutable artifact digest and authorized forensic disposition match exactly.
3. R26 predictions are not regenerated; R9 is not refit.
4. Sportsbook inputs added = 0; production parameters changed = false; receiving-yard means and R22 remain unchanged.
5. Every balanced-turnover Week-1 RB room is baseline-exact for all RB/FB target and reception child values.
6. Every unbalanced vacancy Week-1 RB room is R26-exact for all RB/FB target and reception child values.
7. Every non-vacancy Week-1 RB room is baseline-exact.
8. Maximum child-minus-baseline RB-room target-mass gap remains <= `1e-10`.

### Original R26E quality gates — unchanged
9. Pooled Week-1 vacancy-incumbent receptions MAE improves versus baseline.
10. Pooled Week-1 vacancy-incumbent receptions RMSE is non-worse versus baseline.
11. Pooled Week-1 vacancy-incumbent absolute reception bias is non-worse versus baseline.
12. Pooled Week-1 vacancy-incumbent reception p90 worsens by no more than 2% versus baseline.
13. Pooled Week-1 vacancy-incumbent target MAE improves versus baseline.
14. Week-1 vacancy-incumbent receptions MAE improves in at least 4 of 6 seasons.
15. No individual Week-1 season worsens vacancy-incumbent receptions MAE by more than 2%.
16. At least 4 of 6 seasons have >=20 labeled Week-1 vacancy incumbents and pooled support >=150.
17. Week-1 vacancy RB1 incumbent receptions MAE may worsen by no more than 1% versus baseline.
18. Week-1 vacancy RB2+ incumbent receptions MAE may worsen by no more than 1% versus baseline.
19. At least one of vacancy RB1 or RB2+ receptions MAE improves versus baseline.
20. All Week-1 RB/FB receptions MAE may worsen by no more than 0.5% versus baseline.
21. All Week-1 RB/FB receptions RMSE may worsen by no more than 0.5% versus baseline.
22. All Week-1 RB/FB absolute reception bias is non-worse versus baseline.

### Component-preservation gates versus original R26 Week-1 candidate
23. R26G pooled Week-1 vacancy-incumbent receptions MAE may be no more than **0.5% worse** than original R26.
24. R26G pooled Week-1 vacancy-incumbent target MAE may be no more than **0.5% worse** than original R26.
25. Across seasons 2021-2025 pooled together, R26G Week-1 vacancy-incumbent receptions MAE may be no more than **0.5% worse** than original R26.
26. R26G global Week-1 RB/FB receptions MAE may be no more than **0.5% worse** than original R26.

These preservation gates ensure the child cannot "fix 2020" by erasing the broader R26 gains that motivated component preservation.

## Dispositions

If all 26 frozen gates pass:
- `WEEK1_BALANCED_TURNOVER_GUARD_RETROSPECTIVE_SUPPORT_FOR_2026_SHADOW`

If integrity/inheritance fails:
- `WEEK1_BALANCED_TURNOVER_GUARD_INTEGRITY_FAILURE`

Otherwise:
- `WEEK1_BALANCED_TURNOVER_GUARD_MIXED_NO_SHADOW`

A passing result authorizes only a separately frozen 2026 Week-1 prospective shadow design. It does not authorize production, all-season use, or a receiving-yard/R22 change.

## If R26G fails

Apply the component-preservation doctrine again:
- preserve every gate/component that remains supported;
- identify the exact residual failure;
- do not discard the R26 parent architecture wholesale;
- do not add the R9-Q3 guard automatically;
- do not tune another balance threshold after seeing R26G results.

## Prohibited actions

- no R9 refit or coefficient change;
- no player-specific partial fallback inside a guarded room;
- no alternative exit/entrant ratio threshold;
- no R9 residual Q3 guard in this version;
- no season exclusion;
- no sportsbook feature;
- no same-week historical depth;
- no receiving-yard mean change;
- no R22 change;
- no production write;
- no weakening of the original R26E season-protection gate.
