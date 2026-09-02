# RB STACK6H — Current P3 Team-RB-Pool Oracle Results

## Status

Authoritative corrected execution completed successfully. No production change. No predictive model authorized.

## Authoritative run

- Workflow: `RB STACK6H Current Pool Oracle`
- Run: `33632678179`
- Job: `100255745683`
- Branch: `research-rb-stack6h-current-pool-oracle`
- Tested SHA: `268362dc386e7f3952a86e6d44e714c3379d1d2e`
- Artifact: `rb-stack6h-current-pool-oracle`
- Artifact ID: `9847470353`
- Artifact SHA256: `c1ba164627d405cbac80a1f43aa3d8c30ca1115b8d86028adfa38629f7cec6e0`
- Disposition: **`TOTAL_TEAM_RUSHING_DOMINANT`**

The first execution `33632461834` failed mechanically before outputs because pandas attribute access `t.T` resolved to DataFrame transpose instead of the frozen `T` column. The correction is documented in `RB_STACK6H_IMPLEMENTATION_CORRECTION.md`; no oracle formula, population, threshold, or attribution rule changed.

## Integrity

- STACK6 player rows: `1,393`
- P3 team-games: `544`
- actual RB team-games: `544`
- M94C team-games: `544`
- joined team-games: `544`
- invalid/missing denominator rows: `0`
- W6-18 team-games: `388`
- ORACLE_BOTH max absolute identity error: `3.55e-15`
- expected STACK6F P3 W6-18 pool MAE: `5.7419459771`
- observed BASE P3 pool MAE: `5.7419459771`
- integrity: **PASS**

No fitted models, hyperparameter search, feature search, threshold search, or sportsbook inputs were used. Actual total rush attempts and actual RB share were oracle grading variables only.

## Overall W6-18 oracle decomposition

| Arm | MAE | RMSE | Bias | Corr | MAE recovery vs P3 |
|---|---:|---:|---:|---:|---:|
| BASE_P3_POOL | 5.741946 | 7.198068 | -0.468984 | 0.175421 | 0.000000 |
| ORACLE_TOTAL_RUSH | **2.261455** | **3.126775** | -0.669928 | **0.900501** | **3.480491** |
| ORACLE_RB_SHARE | 5.181572 | 6.551384 | +0.185033 | 0.373507 | 0.560374 |
| ORACLE_BOTH | ~0 | ~0 | ~0 | 1.000000 | 5.741946 |

The total-team-rush oracle recovers **3.4805 carries MAE**, while the RB-share oracle recovers only **0.5604**. The difference is **+2.9201 carries** in favor of total team rushing.

## Large-error bins

### P3 RB pool over by 5+ carries (`n=95`)

- BASE MAE: `8.247745`
- ORACLE_TOTAL_RUSH MAE: `1.727707`
- total-rush recovery: **`6.520038`**
- ORACLE_RB_SHARE MAE: `7.162135`
- RB-share recovery: `1.085610`

### P3 RB pool under by 5+ carries (`n=96`)

- BASE MAE: `9.901523`
- ORACLE_TOTAL_RUSH MAE: `3.400525`
- total-rush recovery: **`6.500997`**
- ORACLE_RB_SHARE MAE: `7.848230`
- RB-share recovery: `2.053292`

### Absolute pool miss 5+ carries (`n=191`)

- BASE MAE: `9.078963`
- ORACLE_TOTAL_RUSH MAE: `2.568495`
- total-rush recovery: **`6.510468`**
- ORACLE_RB_SHARE MAE: `7.506979`
- RB-share recovery: `1.571984`

### Non-extreme absolute miss <3 carries (`n=117`)

- BASE MAE: `1.491021`
- ORACLE_TOTAL_RUSH MAE: `1.917723`
- ORACLE_RB_SHARE MAE: `2.405573`

This non-extreme behavior is not contradictory: when BASE already lands close because two component errors partially cancel, replacing only one component with truth can worsen the result. The large-miss bins are the important bottleneck attribution, and those overwhelmingly favor total team rushing.

## Frozen attribution decision

The precommitted TOTAL_TEAM_RUSHING_DOMINANT rule required:

1. total-rush recovery exceed RB-share recovery by >= `0.50` carry overall;
2. total-rush recovery be >= RB-share recovery in `POOL_OVER_5`;
3. total-rush recovery be >= RB-share recovery in `POOL_UNDER_5`.

Observed:

- overall difference: **`+2.920117`**
- POOL_OVER_5: `6.520038` vs `1.085610`
- POOL_UNDER_5: `6.500997` vs `2.053292`

All conditions pass decisively.

## Durable conclusion

The remaining current-P3 team-RB-pool problem is **not primarily RB-vs-QB/non-RB rushing mass**. It is overwhelmingly the prediction of **total team rushing attempts**.

This materially narrows the RB frontier:

- Do not spend the next research cycle on another M95E-style RB-room-share model.
- Do not use STACK6G's QB/playcaller ideas as point corrections.
- Do not return to RB1/RB2 redistribution as the primary explanation of secondary-back false highs.
- Focus the next diagnostic on the internal mechanics of total team rushing opportunity.

The next no-fit decomposition should determine whether M94C total-rush error is principally:

1. **offensive play-volume error**, or
2. **rush-rate / run-pass tendency error given the play volume**.

Only after that attribution should a new predictive source/model family be frozen. P3 remains the point champion.