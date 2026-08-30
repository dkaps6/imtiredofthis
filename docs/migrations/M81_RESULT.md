# Migration 81 — Authoritative Result

## Disposition

`NO_FTN_DEVELOPMENT_SURVIVOR`

Migration 81 completed its frozen 2024-only development screen successfully. No M80-qualified FTN information family improved the canonical-v3 football-only QB passing projection enough to survive the preregistered development gates. The 2025 target outcomes remained sealed and therefore no M82 confirmation candidate exists from M81.

## Authoritative run

- GitHub Actions workflow: `Migration 81 QB FTN Novel Mechanism Development`
- Run: `33318290201` (Run #1)
- Artifact: `m81-qb-ftn-development`
- Artifact ID: `9734132689`
- Artifact SHA256: `6b79ff0ed9c768458edd65473959c61f152fc9d89d297eca73cc37fa88f6021b`
- Canonical snapshot SHA256: `c4a481a760657bb7516d52b9aa9ba84af096063a4e78660b12a541f59fd7b742`
- Source contract: PASS
- QB identity map rate: `1.000`
- Sportsbook features used: `False`
- Target-game FTN used: `False`
- 2025 target outcomes accessed: `False`
- Production actionable: `False`

## Frozen development cohort

- Fit: 2024 Weeks 1-9
- Holdout: 2024 Weeks 10-18
- Holdout QB-games: `209`

Canonical-v3 holdout baseline:

- passing MAE: `65.170617`
- passing RMSE: `82.103250`
- passing correlation: `-0.014397`
- 100+ yard misses: `43`
- attempts MAE: `6.524020`
- YPA MAE: `1.461737`

## Family results

| Family | Corrected pass MAE | MAE gain vs canonical | Corr gain | RMSE gain | 100+ misses | Attempt MAE gain | YPA MAE gain | Bootstrap P(MAE gain > 0) | Survivor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TACTICAL_CALL_STRUCTURE | 65.322998 | -0.152381 | -0.010356 | -0.975654 | 45 | -0.261442 | -0.080124 | 0.4025 | No |
| PRESSURE_RESPONSE | 66.311314 | -1.140697 | +0.035403 | -1.203902 | 47 | -0.167824 | -0.068174 | 0.2075 | No |
| THROW_DECISION_QUALITY | 65.285609 | -0.114992 | +0.030848 | +0.258528 | 44 | -0.343876 | -0.120575 | 0.4710 | No |
| RECEIVER_ERROR_ATTRIBUTION | 65.192894 | -0.022277 | -0.005142 | -0.047359 | 42 | -0.053097 | -0.027023 | 0.4575 | No |

Positive `MAE gain`, `RMSE gain`, `Attempt MAE gain`, and `YPA MAE gain` mean improvement; negative values mean regression.

## Interpretation

No family improved passing-yard MAE. More importantly, every family worsened both attempt MAE and YPA MAE, so there is no hidden component-level point-prediction gain to preserve.

- `PRESSURE_RESPONSE` improved passing correlation but materially worsened MAE, RMSE, tails, attempts, and YPA.
- `THROW_DECISION_QUALITY` improved correlation and RMSE, but worsened MAE, 100+ misses, attempts, and YPA.
- `RECEIVER_ERROR_ATTRIBUTION` reduced 100+ misses from 43 to 42, but was otherwise essentially flat/slightly worse and did not improve either component.
- `TACTICAL_CALL_STRUCTURE` worsened every principal point-forecast metric.

Because zero independent families survived, the preregistered survivor stack was not tested.

## Anti-loop consequence

Do not open another migration that changes Ridge alpha, swaps to HGB/XGB/random forest/neural network, selects FTN subsets after seeing these results, or recombines the same four FTN information families as a new standalone hypothesis. M81 tested the incremental point-prediction value of these qualified FTN history families under the frozen development contract and they failed.

A future revisit requires materially new pregame information, not a different transformation/model over the same information.

## M82 boundary

There is no M82 confirmation candidate from M81. The 2025 outcomes remain unused by M81 and should not be opened merely to see whether a rejected M81 family happens to look better there.
