# RB STACK6K — Results

## Canonical run

- branch: `research-rb-stack6k-within-state-tendency-oracle`
- run: `33638739235`
- job: `100276119195`
- SHA: `2a6775bacaa1e6b7d1ec5a0b0f5ca302c66a7417`
- artifact ID: `9849896540`
- artifact digest: `b2c9e670f313ac5fe60458fad762a24494799bbd7b8656e7c17891cf8884ca6d`

## Integrity

PASS.

- M94C rows: `544`
- STACK6H rows: `544`
- W6-18 n: `388`
- M94C base MAE reproduced exactly: `6.2034547805`
- STACK6J occupancy MAE reproduced exactly: `5.5183819623`
- fitted models/search/sportsbook inputs: `0`

## PBP / weekly rushing-truth bridge

PASS under the frozen bridge requirements.

- n `388`
- MAE `0.286082`
- RMSE `1.303247`
- bias `+0.286082`
- corr `0.987069`
- exact match `88.92%`
- absolute difference >1: `3.35%`
- absolute difference >2: `3.09%`

The PBP source is sufficiently aligned with the weekly-stat team-rush target for this oracle attribution.

## Overall

| Arm | MAE | RMSE | Bias | Corr | Recovery vs M94C |
|---|---:|---:|---:|---:|---:|
| M94C | 6.203455 | 7.741320 | +0.179588 | 0.250307 | 0.000000 |
| Perfect state occupancy | 5.518382 | 6.968764 | +0.119826 | 0.477745 | +0.685073 |
| Perfect occupancy + realized rushing tendency | 3.450328 | 4.399124 | +0.433598 | 0.850749 | +2.753127 |

Incremental tendency recovery after occupancy = **`+2.068054` attempts MAE**.

Remaining STACK6I rate headroom after occupancy was `2.555622`; realized within-state tendency explains **80.92%** of that remaining headroom.

## Current-P3 extreme pool-error bins

### Pool overprojection >=5

- occupancy-only MAE `7.481507`
- occupancy+tendency MAE `3.968024`
- incremental tendency recovery **`+3.513483`**

### Pool underprojection >=5

- occupancy-only MAE `7.890213`
- occupancy+tendency MAE `4.561592`
- incremental tendency recovery **`+3.328622`**

### Absolute pool miss >=5

- incremental tendency recovery **`+3.420568`**

### Non-extreme pool miss <3

- incremental tendency recovery `+0.587529`

Unlike the state-occupancy oracle, the realized-tendency oracle also improves the already-near-centered population.

## Frozen disposition

`WITHIN_STATE_TENDENCY_DOMINANT`

No production change and no predictive model authorized.

## Research meaning

The dominant remaining M94C total-rush error is the offense's rushing tendency / play selection within its realized game environment, not total offensive plays and not state occupancy alone.

The next diagnostic should attribute the tendency error across lead, neutral and trail states before fitting. With only three states, an exact no-fit subset/Shapley attribution can quantify each state's contribution without order dependence.

P3 remains the RB point champion.
