# RB STACK6J — Results

## Canonical run

- branch: `research-rb-stack6j-state-occupancy-oracle`
- run: `33638294183`
- job: `100274619508`
- SHA: `660c489bd3545d086e0de2f8d359b206d447ab3f`
- artifact ID: `9849711171`
- artifact digest: `23534fd343a818c356b917f2cba45dfcc538ecddd2b8cd0c8b030e281e4a344e`

## Integrity

PASS.

- M94C rows: `544`
- STACK6H rows: `544`
- joined rows: `544`
- W6-18 n: `388`
- structured rebuild max abs error: `1.42e-14`
- candidate rebuild max abs error: `1.42e-14`
- predicted state shares sum to one to numerical tolerance
- actual state shares sum to one to numerical tolerance
- frozen M94C W6-18 MAE reproduced exactly: `6.2034547805`
- fitted models/search/sportsbook inputs: `0`

## State-share prediction quality

| State | MAE | RMSE | Bias | Corr |
|---|---:|---:|---:|---:|
| Lead | 0.249691 | 0.292919 | +0.006480 | 0.256088 |
| Neutral | 0.205442 | 0.245884 | -0.004649 | -0.003156 |
| Trail | 0.304229 | 0.347249 | -0.001830 | 0.227740 |

Neutral-state occupancy is especially weak in ranking terms.

## Overall oracle

- M94C total-rush MAE: `6.203455`
- perfect state occupancy MAE: `5.518382`
- MAE recovery: `+0.685073`
- correlation: `0.250307 -> 0.477745`

STACK6I perfect-rush-rate headroom was `3.240694`, so perfect state occupancy explains only `0.211397` (21.14%) of that headroom.

## Current-P3 extreme pool-error bins

### P3 pool overprojection >= 5

- n `95`
- occupancy recovery `+1.411448`

### P3 pool underprojection >= 5

- n `96`
- occupancy recovery `+1.218698`

### Absolute P3 pool miss >= 5

- n `191`
- occupancy recovery `+1.314568`

### Non-extreme pool residual < 3

- n `117`
- occupancy recovery `-0.275146`

Thus correct game-state occupancy helps the large misses in both directions but slightly damages already-well-centered games.

## Frozen disposition

`STATE_OCCUPANCY_PARTIAL`

No production change and no predictive model authorized.

## Research meaning

STACK6J confirms that M94C's lead/neutral/trail mapper is one real source of team-rush error, but it is not the primary remaining rush-rate bottleneck. Only about one-fifth of the STACK6I rate headroom is attributable to state occupancy.

The next diagnostic should therefore quantify the incremental headroom from **within-state rushing tendency / play-selection mechanics**, after actual state occupancy is already supplied. Do not fit another margin/state mapper yet.

P3 remains the RB point champion.
