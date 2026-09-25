# Receiver Targetable-Dropback V1 — 2024-2025 Confirmation Result

Date: 2026-09-25

Disposition: **RECEIVER_TARGETABLE_DROPBACK_V1_2024_2025_CONFIRMED**

This is a team-opportunity confirmation of the exact 2022-2023-supported
targetable-dropback candidate. It does not authorize production or player-level
integration by itself.

## Authority

- branch: `research-receiver-targetable-dropback-v1-confirm-2024-2025`
- authoritative run: `36145789611`
- job: `108106504880`
- tested head: `0fdc8727a035603a5a0346a2e75e360ff467fdec`
- artifact: `10869935285`
- digest: `sha256:2add0a5706ad6398f058e7566aa3c36eed2e6132b7e6bba0e2d991d2c4e80cb6`
- parameters fit: `0`
- candidate variants scored: `1`
- sportsbook inputs: `0`
- target-game outcomes used upstream: `0`

## Exact unchanged candidate

For each target team-game:

`R_T = sum(strict-prior team targets) / sum(strict-prior team dropbacks)`

Eligible history:
- all regular-season games from the prior season;
- only completed current-season games strictly before target week.

Baseline receiver opportunity:

`B = projected_plays * 0.57`

Candidate receiver targetable pool:

`C = B * R_T`

No shrinkage, recency weighting, minimum-games rule, threshold, team/player/QB
carveout or fitted coefficient was introduced.

## 2024

- team-target MAE: `6.505208 -> 6.394324`
- RMSE: `8.140594 -> 8.186393`
- p90 AE: `13.287450 -> 13.460497`
- absolute bias: `2.569570 -> 2.820775`
- changed-row candidate closer rate: ~`50.37%`

The frozen season p90 guard passed despite the small 2024 p90 regression.

## 2025

- team-target MAE: `6.671869 -> 6.056585`
- RMSE: `8.193002 -> 7.753547`
- p90 AE: `13.117668 -> 12.870110`
- absolute bias: `3.155552 -> 2.169727`
- changed-row candidate closer rate: ~`54.78%`

## Pooled 2024-2025

- team-target MAE: `6.588539 -> 6.225454`
- improvement: **0.363085 targets/team**
- RMSE: `8.166840 -> 7.972908`
- p90 AE: `13.150607 -> 13.124031`
- absolute bias: `2.862561 -> 2.495251`
- changed-row candidate closer rate: **52.57%**
- targetable-rate median: ~`0.84405`
- rate range: ~`0.72605 .. 0.92179`
- league fallback rows: **0**

## Frozen confirmation gates

All passed:

- 2024 MAE improves
- 2025 MAE improves
- pooled MAE improves
- pooled RMSE nonworse
- pooled p90 nonworse
- pooled absolute bias improves
- candidate closer rate >50%
- 2024 p90 guard
- 2025 p90 guard
- >=99% strict-prior team conversion source
- <=1% league fallback
- conversion values finite/in-bounds
- target-game outcomes upstream = 0
- sportsbook inputs = 0
- parameters fit = 0
- one candidate only

## Four-season replication

The exact cumulative-count targetable-dropback mechanism has now improved
team-target MAE independently in:

- **2022**
- **2023**
- **2024**
- **2025**

The first temporal screen (2022-2023) also passed all frozen gates:
- pooled MAE `6.629337 -> 6.159377`
- pooled RMSE `8.147865 -> 7.890454`
- pooled p90 `13.480627 -> 12.640581`
- pooled absolute bias `2.653981 -> 2.398329`
- candidate closer rate `54.14%`

This is materially stronger evidence than the failed official-attempt conversion
or the later duplicate arithmetic-mean targetable-rate formulation.

## Interpretation

The evidence now supports the receiver-opportunity statement:

> receiver opportunity should be anchored to targetable dropbacks rather than
> all simulated dropbacks.

This does **not** mean the broader team dropback forecast is solved.

The existing stack still underprojects team dropback volume on average, while
the current receiver simulator overincludes non-targetable dropbacks. Those two
errors partially offset each other. Therefore the first player-level integration
must isolate the targetable-pool correction and must not simultaneously alter
the 0.57 team dropback partition.

## Next authorized step

Freeze a separate player-level/full-stack candidate before any scoring.

Initial scope:
- keep fixed 0.57 team dropback partition unchanged;
- derive strict-prior cumulative targetable-dropback rate exactly as above;
- thin receiver opportunity from the dropback pool to the targetable pool;
- keep explicit player target entitlement unchanged;
- keep M38 unchanged;
- keep WR-R15 unchanged;
- keep TE-R5P unchanged;
- keep catch rates unchanged;
- keep YPT unchanged;
- keep QB M89/M90 means unchanged;
- keep QB C2 unchanged;
- keep all rushing arrays unchanged;
- keep ATD unchanged;
- rebuild/evaluate dependent RB rush+receiving through current RB V2;
- no sportsbook input.

Historical player-level qualification should use the current full stack on
2024-2025 with fold-safe specialist authority, followed by prospective 2026
shadow capture before any production promotion.
