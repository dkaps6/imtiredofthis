# RB R26D — Exit Significance × R26 Effect Atlas V1 Result

Status: **EXIT_SIGNIFICANCE_EFFECT_MIXED_NO_ROUTER**
Date: 2026-09-09
Production authority protected: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Research head: `59b4183f75a354e1c65ee8bbc28f22142f678b17`

## Authoritative lineage

- Workflow: `RB R26D Exit Significance Effect Atlas V1`
- Run: `34364089085`
- Job: `102508218446`
- Artifact: `10109078127`
- Artifact digest: `sha256:827d45611a0ff588093f2ab6a5b74eb739dd5602afa9461a713083a8c44e8756`
- Parent R26 artifact: `10106271075`, digest `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`
- Parent R26C artifact: `10108036449`, digest `sha256:1c7ee705179d91b7e3e53c39b57783314bd87876df6c784d5d1930ee5cf79156`
- Workflow conclusion: mechanically `success`
- Diagnostic disposition: **EXIT_SIGNIFICANCE_EFFECT_MIXED_NO_ROUTER**

R26D regenerated no predictions, refit no R9 model, changed no production parameters, changed no receiving-yard means, changed no R22 behavior, and added no sportsbook inputs.

## Full-candidate disposition

The predeclared significance router did **not** earn child-candidate authorization. Seven of eight frozen replication gates passed. The sole failed gate was:

- meaningful-exit R26 reception MAE improved in only **3 of 6** seasons, below the frozen 4-of-6 requirement.

R26D therefore does not authorize a significance-gated router or any production/shadow change.

## Supported components to preserve

### 1. Exit significance is a real effect modifier pooled

Meaningful receiving exits (`n=936`):
- receptions MAE: `1.304162 -> 1.280611` (**1.81% better**)
- RMSE: `1.774268 -> 1.714271`
- bias: `-0.572401 -> -0.382160`
- p90 abs error: `2.815412 -> 2.773593`
- targets MAE: `1.550143 -> 1.518895` (**2.02% better**)

Low receiving exits (`n=342`):
- receptions MAE: `1.169478 -> 1.180126` (**0.91% worse**)
- targets MAE: `1.401539 -> 1.414367` (**0.92% worse**)

Thus the broad `any exit` gate remains too coarse for all-season use.

### 2. Role behavior remains supported inside meaningful exits

Meaningful-exit RB1 incumbents (`n=314`):
- receptions MAE `1.609010 -> 1.565520` (**2.70% better**)

Meaningful-exit RB2+ incumbents (`n=622`):
- receptions MAE `1.150268 -> 1.136782` (**1.17% better**)

The frozen role safety gate passed.

### 3. Week-1 broad vacancy behavior is especially strong

The predeclared phase diagnostic shows R26 improves Week-1 receptions MAE in **all three** exit-history classes:

- low receiving exit (`n=32`): `1.687823 -> 1.566025` (**7.22% better**)
- meaningful receiving exit (`n=197`): `1.364287 -> 1.260364` (**7.62% better**)
- unknown exit history (`n=17`): `1.626378 -> 1.436539` (**11.67% better**)

By contrast, Weeks 2+ show the expected churn problem:
- low receiving exit: `1.115972 -> 1.140291` (**2.18% worse**)
- meaningful receiving exit: `1.288134 -> 1.286008` (essentially flat / **0.17% better**)

This supports treating **Week 1 broad vacancy redistribution** and **Weeks 2+ vacancy routing** as different research components rather than forcing one gate across both regimes.

### 4. Predeclared fine bins support the football interpretation

Prior target-volume bins:
- departed RB with `0` prior targets/game: MAE worsened about **5.37%**
- `(0,1]`: improved about **0.61%**
- `(1,2]`: improved about **2.54%**
- `>2`: improved about **1.02%**

Prior RB-room target-share bins:
- `<0.10`: worsened about **2.31%**
- `0.10-0.25`: improved about **1.43%**
- `0.25-0.50`: improved about **2.03%**
- `>=0.50`: improved about **1.27%**

Last-8 target-volume bins:
- `0`: worsened about **5.37%**
- `(0,1]`: worsened about **0.91%**
- `(1,2]`: improved about **3.66%**
- `>2`: improved about **1.30%**

These are diagnostic evidence only and may not be used to retune R26D's frozen threshold after the fact.

## Failed / unsupported component

The simple all-season primary classifier:

`meaningful = max prior targets/game > 1 OR max prior RB-room share >= 0.25`

is not temporally robust enough to authorize an all-season router. Meaningful-exit R26 reception MAE improved in 2021, 2022, and 2024, but worsened in 2020, 2023, and 2025.

Season meaningful MAE deltas (candidate minus baseline):
- 2020: `+0.02146`
- 2021: `-0.06223`
- 2022: `-0.06140`
- 2023: `+0.07588`
- 2024: `-0.17021`
- 2025: `+0.02167`

The significance concept is useful but insufficient by itself outside Week 1.

## Next child-candidate delta

Following the component-preservation doctrine, do **not** discard R26.

Split the research into two components:

1. **Week-1 component qualification**
   - inherit the original frozen R26 broad-vacancy/R9 behavior unchanged;
   - test season-by-season Week-1 replication under a separately frozen qualification contract;
   - do not introduce the failed all-season significance router;
   - if robust, this may justify a 2026 Week-1 prospective shadow only, never an all-season promotion.

2. **Weeks 2+ vacancy router research**
   - preserve R26/R9 identity and conservation mechanics;
   - investigate strictly-prior recency/significance/current-role state to distinguish real midseason receiving vacancies from roster churn;
   - any new threshold/router requires a new frozen plan and remains retrospective until separately validated.

## Do-not-change list

Preserve:
- R9 identity mechanics;
- R26 finite RB target-pool conservation;
- production-exact non-vacancy behavior;
- non-RB exactness;
- receiving-yard mean exactness;
- R22;
- sportsbook separation;
- strict-prior leakage protections;
- the original R26 full-candidate failed disposition;
- all negative R26D evidence.

No production files were changed by R26D.
