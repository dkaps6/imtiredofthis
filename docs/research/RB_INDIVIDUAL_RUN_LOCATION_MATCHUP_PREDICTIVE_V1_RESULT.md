# RB PLAYER ROLE MATCHUP — INDIVIDUAL RUN-LOCATION PREDICTIVE V1 RESULT

Date: 2026-10-12

Status: **COMPLETE — EXACT V1 CANDIDATE FAILED CLOSED. NO PRODUCTION CHANGE.**

Frozen contract:
- `docs/research/RB_INDIVIDUAL_RUN_LOCATION_MATCHUP_PREDICTIVE_V1_FREEZE.md`

Canonical authority:
- workflow run: `38258940285`
- exact scoring head: `20230872a13729f8c5269b653c46ffec030c1c78`
- artifact: `11752168196`
- artifact digest: `sha256:476000f0f6814af60f956f71bf96708442057f6a4b55a37e975bdc0bfac77ed2`
- workflow conclusion: `SUCCESS`

Source authority:
- source run: `38189983578`
- source artifact: `11720966759`
- source digest: `sha256:dec7d48d1a2c39ef255a6391b0014f75779ceb5939dfc7a90feca384074caad5`
- source-ready rows: **2,654**

Frozen candidate:
`RB-INDIVIDUAL-RUN-LOCATION-YPC-V1`

Disposition:
`RB_RUN_LOCATION_MATCHUP_HISTORICAL_FAIL_CLOSED`

## Baseline integrity

Historical football-only baseline reproduced exactly:
- rows: **38,258**
- max absolute gap: `1.4210854715202004e-13`
- frozen tolerance: <=1e-8
- result: **PASS**

No contamination:
- sportsbook inputs: 0
- OddsAPI calls: 0
- 2026 outcomes: 0
- fitted coefficients: 0
- threshold searches: 0
- target-game usage: false
- production change authorized: false

## Primary source-ready rushing-yard result

### 2024

- rows: **1,063**
- games: **256**
- players: **104**
- baseline MAE: **23.718100**
- candidate MAE: **23.651992**
- MAE improvement: **+0.066108 yards**
- baseline RMSE: **32.965763**
- candidate RMSE: **32.891659**
- game-cluster 95% CI: **[-0.037042, +0.169096]**
- player-cluster 95% CI: **[-0.030315, +0.167714]**
- supported improvement: **false**

The point estimate improved, but neither frozen clustered confidence interval
cleared zero.

### 2025

- rows: **1,018**
- games: **256**
- players: **108**
- baseline MAE: **23.228504**
- candidate MAE: **23.250639**
- MAE change: **-0.022136 yards**
- baseline RMSE: **33.677352**
- candidate RMSE: **33.677807**
- game-cluster 95% CI: **[-0.151492, +0.106300]**
- player-cluster 95% CI: **[-0.131980, +0.092754]**
- supported improvement: **false**
- RMSE non-worse: **false**

The candidate therefore fails the required cross-season replication gate before
any implementation-isolation question is considered.

## Broad all-RB protection

2024:
- rows: 1,314
- baseline MAE: 21.410878
- candidate MAE: 21.357398
- point improvement: +0.053480
- unsupported

2025:
- rows: 1,308
- baseline MAE: 20.680148
- candidate MAE: 20.697376
- point change: -0.017228
- RMSE slightly worse

Broad protection therefore did not pass the frozen two-season rule.

## Opportunity / input isolation

The direct simulation audit passed the important football isolation checks for
every scored week:
- input non-`rules_ypc` max delta: **0.0**
- RB rush-attempt mean max delta: **0.0**
- RB receiving-yard mean max delta: **0.0**
- RB reception mean max delta: **0.0**

Thus the candidate did preserve the intended carry opportunity and receiving
state at the simulation-input level.

The final-output audit reported:
- RB rec-yard projection max delta: **0.0**
- RB receptions projection max delta: **0.0**
- RB rush-att projection max delta: **0.0**
- untreated final projection max delta: **21.966971**

That final untreated-output condition fails the frozen isolation gate. The scorer
recomputed all Monte Carlo market means from the raw simulator, while the
canonical baseline has additional market-specific post-processing (notably QB
pass-yard adjustments). This is an implementation-routing defect in the
historical candidate harness, not evidence that the RB YPC change moved RB
opportunity.

Do **not** repair/re-score this V1 to seek promotion because the independent
primary scientific gate already failed: 2025 worsened and 2024 was unsupported.

## Interpretation

The source result remains useful:
- individual RB left/middle/right usage is reconstructable;
- opponent run defense by the same locations is reconstructable;
- the interaction is genuinely player-specific and leakage-safe.

But the exact direct YPC multiplier:
- does not replicate across 2024 and 2025;
- does not clear clustered support;
- therefore does not earn a future shadow or production integration.

This result does not imply matchup is irrelevant. It rejects this exact
transmission formula.

## No-rescue boundary

Do not:
- tune left/middle/right definitions;
- tune multiplier bounds;
- add success-rate blending after seeing this result;
- select inside/outside runners or bellcow subgroups;
- add shrinkage;
- combine with M95A/M95B;
- alter carry opportunity;
- repair the scorer and reinterpret the same V1 as a new scientific chance.

A genuinely new RB matchup mechanism requires new information or a separately
justified football mechanism, not V1 parameter rescue.
