# RB STACK6I — Results

## Canonical corrected run

- branch: `research-rb-stack6i-team-rush-mechanics-oracle`
- run: `33637785836`
- job: `100272894018`
- SHA: `343312c47be50bd450b0cea69361aeea3e2f52fa`
- artifact: `9849508420`
- artifact digest: `e22f2188774c2186f3a4bfb7273e5b19ea1195e9528924495d77f57cffa487ad`

The first run failed mechanically on a pandas `Series.corr` namespace collision before outputs were written. `RB_STACK6I_IMPLEMENTATION_CORRECTION.md` documents the correction. Frozen equations and gates were unchanged.

## Integrity

PASS.

- M94C rows: `544`
- STACK6H rows: `544`
- joined rows: `544`
- W6-18 team-games: `388`
- base factorization max absolute error: `3.55e-15`
- actual factorization max absolute error: `7.11e-15`
- frozen M94C W6-18 score reproduced exactly:
  - MAE `6.2034547805`
  - RMSE `7.7413204361`
  - bias `+0.1795877880`
  - corr `0.2503068367`
- fitted models: `0`
- search: `0`
- sportsbook inputs: `0`

## Component accuracy

### Offensive plays

- MAE `6.850906`
- RMSE `8.706236`
- bias `+0.907347`
- corr `0.178516`

### Effective rush rate

- MAE `0.085286`
- RMSE `0.105400`
- bias `-0.003207`
- corr `0.307351`

Raw units differ, so the frozen oracle recoveries are the attribution authority.

## Frozen oracle results — W6-18

| Arm | MAE | RMSE | Bias | Corr | MAE recovery vs M94C |
|---|---:|---:|---:|---:|---:|
| M94C total rush | 6.203455 | 7.741320 | +0.179588 | 0.250307 | 0.000000 |
| Perfect offensive plays | 5.242041 | 6.564061 | -0.237204 | 0.562432 | +0.961414 |
| Perfect effective rush rate | 2.962760 | 3.829166 | +0.365221 | 0.877015 | **+3.240694** |
| Perfect both | ~0 | ~0 | ~0 | 1.000000 | +6.203455 |

Perfect effective rush rate recovers `2.279280` more MAE than perfect play volume.

## Current-P3 extreme pool-error bins

### P3 RB pool overprojection >= 5 carries

- n `95`
- M94C total-rush MAE `8.892955`
- perfect plays recovery `+2.192760`
- perfect rush-rate recovery `+6.142490`

### P3 RB pool underprojection >= 5 carries

- n `96`
- M94C total-rush MAE `9.108912`
- perfect plays recovery `+1.715559`
- perfect rush-rate recovery `+5.584543`

### Absolute P3 RB pool miss >= 5

- n `191`
- M94C total-rush MAE `9.001498`
- perfect plays recovery `+1.952910`
- perfect rush-rate recovery `+5.862056`

The rush-rate oracle dominates in both error directions, satisfying the frozen attribution rule.

## Disposition

`RUSH_RATE_DOMINANT`

No production change. No predictive model authorized by STACK6I.

## Research meaning

STACK6H established that current P3 team-RB-pool error is primarily an upstream total-team-rushing problem. STACK6I now localizes that total-rush problem primarily to **effective run/pass allocation per offensive play**, not offensive play volume.

The next research step must therefore localize the rate miss inside M94C's football mechanics before fitting another candidate. In particular, the next diagnostic should distinguish errors in:

1. predicted lead/neutral/trail game-state occupancy; versus
2. rushing tendency conditional on those states / the remaining structured-rate mechanics.

P3 remains the RB point champion.
