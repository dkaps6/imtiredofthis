# RB R26 — Vacancy-Gated R9 Retrospective V1 Result

Status: **RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW**
Date: 2026-09-09
Production authority protected: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Frozen candidate head: `9b9d49099ec0e62a168e37679cbd56e6b767131d`

## Authoritative lineage

- Workflow: `RB R26 Vacancy Gated R9 Retrospective V1`
- Run: `34356222339`
- Job: `102481464805`
- Artifact: `10106271075`
- Artifact digest: `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`
- Workflow conclusion: mechanically `success`
- Scientific disposition: **RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW**
- Frozen gates passed: **19 / 20**

A successful GitHub workflow does not override the scientific disposition. R26 V1 is a preserved scientific failure and does not authorize a 2026 prospective shadow or production change.

## Frozen mechanism tested

R26 V1 left non-vacancy RB rooms on the production baseline exactly. In rooms where one or more ACT/INA RB/FB identities disappeared from the prior leakage-safe roster snapshot, the existing R9 receiving-identity mechanism redistributed only the already-finite production RB target pool. The test did not change team target mass, non-RB entitlement, receiving-yard point means, R22, sportsbook independence, or production runtime.

Fixed folds were:

- 2019 -> 2020
- 2020 -> 2021
- 2021 -> 2022
- 2022 -> 2023
- 2023 -> 2024
- 2024 -> 2025

## Structural integrity

All integrity gates passed:

- sportsbook inputs upstream = `0`
- future/target-game outcomes in features = `0`
- strict-prior state/fitting contract = pass
- minimum player/room transition-state coverage = `1.0`
- maximum RB-room target-mass gap = `5.55e-17`
- maximum non-RB entitlement delta = `0`
- team entitlement gap = `0`
- receiving-yard production mean delta = `0`
- R22 authority delta = `0`
- protected production authority remained clean

## Pooled result

### All RB/FB receptions

| Metric | Baseline | R26 candidate |
|---|---:|---:|
| MAE | 1.205761 | 1.201000 |
| RMSE | 1.685379 | 1.673546 |
| Bias | -0.374325 | -0.348658 |
| p90 absolute error | 2.566292 | 2.558989 |
| Pearson | 0.46035 | 0.45385 |
| Spearman | 0.44301 | 0.44899 |

Global receptions MAE improved about **0.40%** while RMSE, absolute bias, p90, and Spearman were also non-worse/improved under the frozen safety gates.

### Week 1 receptions

| Metric | Baseline | R26 candidate |
|---|---:|---:|
| MAE | 1.312603 | 1.235284 |
| RMSE | 1.837827 | 1.758467 |
| Bias | -0.537235 | -0.401004 |
| p90 absolute error | 2.857038 | 2.791584 |
| Spearman | 0.43686 | 0.48932 |

Week-1 MAE improved about **5.9%**, which is consistent with the pre-existing diagnostic that Week 1 is especially transition-heavy. This remains retrospective evidence only.

## Vacancy-incumbent pooled result

For same-team incumbents in vacancy-active rooms (`n=1444`):

### Receptions

| Metric | Baseline | R26 candidate |
|---|---:|---:|
| MAE | 1.298022 | 1.277966 |
| RMSE | 1.798473 | 1.732017 |
| Bias | -0.530895 | -0.352552 |
| p90 absolute error | 2.84244 | 2.78927 |
| Pearson | 0.43352 | 0.45847 |
| Spearman | 0.37839 | 0.43369 |

### Targets

- MAE: `1.532246 -> 1.507269`
- RMSE: `2.09650 -> 2.02930`
- bias: `-0.6495 -> -0.42085`

The vacancy mechanism therefore had meaningful pooled support.

## Role result

### Vacancy RB1 incumbents (`n=503`)

- receptions MAE: `1.640317 -> 1.601164`
- RMSE: `2.240168 -> 2.083073`
- bias: `-0.857710 -> -0.140032`
- p90: `3.63661 -> 3.33270`
- Spearman: `0.27692 -> 0.29922`

### Vacancy RB2+ incumbents (`n=941`)

- receptions MAE: `1.115053 -> 1.105204`
- RMSE: `1.510289 -> 1.511281`
- bias: `-0.3562 -> -0.46615`
- p90: `2.21738 -> 2.35219`
- Spearman: `0.17662 -> 0.27137`

The pooled RB1 result is a clear correction of underprojection. RB2+ MAE improves slightly, but bias and p90 deteriorate, which is important evidence that the unresolved problem is allocation/order rather than simply whether vacancy exists.

## Season replication

Vacancy-incumbent receptions MAE change:

- 2020: `1.339070 -> 1.333151` (**0.44% better**)
- 2021: `1.397842 -> 1.391478` (**0.46% better**)
- 2022: `1.498794 -> 1.441995` (**3.79% better**)
- 2023: `1.174518 -> 1.227870` (**4.54% worse**)
- 2024: `1.284315 -> 1.185275` (**7.71% better**)
- 2025: `0.994816 -> 0.979801` (**1.51% better**)

R26 improved **5 of 6 seasons**. The sole failed frozen gate was:

> `15_no_season_worsens_more_than_2pct = false`

because 2023 worsened by about **4.54%**. This failure is decisive under the predeclared plan.

## 2023 failure shape already known from the frozen grader

2023 vacancy RB1:

- MAE `1.411798 -> 1.499771` (~6.23% worse)
- bias `-0.6317 -> +0.0785`

2023 vacancy RB2+:

- MAE `1.047280 -> 1.082067` (~3.32% worse)
- bias `-0.3884 -> -0.5244`

The same R9 redistribution therefore **overfed RB1 while further underfeeding RB2+ in 2023**. That is not evidence that vacancy is absent. It is evidence that the candidate can mis-rank or misallocate the inherited opportunity in a specific regime.

## Scientific interpretation

R26 V1 remains a failure and may not be rescued by changing the vacancy threshold, hand-selecting a smaller R9 reliability weight, redefining cohorts, or relaxing the 2% season gate.

What is supported diagnostically:

1. Vacated RB receiving opportunity is a real and repeatable pregame state worth modeling.
2. The existing R9 receiving identity is generally useful inside that state.
3. The remaining failure is concentrated in **who inherits the opportunity / role ordering**, not total RB-room receiving mass.
4. 2023 is the necessary forensic case because it is the only season that breaks an otherwise 5-of-6 replicated mechanism.
5. The next step must be a no-refit forensic atlas using already-produced R26 predictions and timing-safe pregame state. Only after that atlas is frozen and inspected may a materially different candidate/router be specified.

No production files or `main` authority are changed by this result.