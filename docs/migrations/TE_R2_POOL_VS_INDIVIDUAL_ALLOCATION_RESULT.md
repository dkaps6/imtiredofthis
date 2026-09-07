# TE-R2 Team TE Pool vs Individual Allocation — Result

## Disposition

`TE_TARGET_POOL_FIRST`

TE-R2 passed every frozen integrity gate. The primary TE target-error layer is **team TE target-pool volume**, not individual within-room allocation. This route is stable pooled, in every individual season, and becomes substantially stronger in the highest-error quartile.

No sportsbook inputs were used. No model was fit. Production is unchanged.

## Canonical evidence
- Branch: `research-te-r2-pool-vs-individual-allocation`
- Frozen plan commit: `8cfd2e20955e1fa160660929adfa889be2a334ba`
- Implementation commit: `72d5e48a78c1e0ee832180603d3f6c4ed32a1257`
- Canonical run: `34126026512`
- Job: `101754774701`
- Tested SHA: `234d58129a2d43126ed91715df9477c00c1d11f4`
- Artifact: `10020131404` (`te-r2-pool-vs-individual-allocation`)
- Artifact digest: `sha256:54309eda0886cd0b327333ea89bde12a2fc506d141e2b72ddc3b1d1d778b874d`
- Parent TE-R1 run: `34123413402`
- Joint source run: `34081764151`

## Integrity
- TE player-games: **6,371**
- team-games: **3,197**
- players with >=20 scoreable games: **108**
- seasons: all 2020-2025, each >=1,000 player-games except none below gate
- max player target identity reconstruction gap: `1.7763568394002505e-15`
- max two-factor Shapley reconstruction gap: `3.552713678800501e-15`
- sportsbook inputs: 0

All frozen integrity gates passed.

## Pooled player target baseline
- MAE: **1.691897 targets**
- RMSE: 2.427971
- bias: **-0.912188**
- correlation: 0.606595
- median absolute error: 1.111161
- p75: 2.150350
- p90: **4.066575**
- 2+ miss rate: 27.0915%
- 4+ miss rate: 10.3751%
- 6+ miss rate: 3.4845%

## Exact target-error layer decomposition
Pooled absolute mechanism mass:
- **TEAM_TE_POOL: 60.4050%**
- INDIVIDUAL_ALLOCATION: 39.5950%

Mean absolute component per player-game:
- TEAM_TE_POOL: **1.309432 targets**
- INDIVIDUAL_ALLOCATION: 0.858323 targets

Same-direction rate with final player target residual:
- TEAM_TE_POOL: **74.31%**
- INDIVIDUAL_ALLOCATION: 68.44%

The frozen >=55% routing threshold is therefore passed by the team-pool layer.

## Temporal stability
Every season independently routes `TE_TARGET_POOL_FIRST`:
- 2020: pool 61.3349% / allocation 38.6651%
- 2021: 59.3007% / 40.6993%
- 2022: 58.6544% / 41.3456%
- 2023: 61.3876% / 38.6124%
- 2024: 62.0815% / 37.9185%
- 2025: 59.5540% / 40.4460%

This is not a one-season artifact.

## Highest-error quartile
Threshold: absolute player target error >= **2.150350 targets**; n=1,593.

Absolute mechanism mass:
- **TEAM_TE_POOL: 72.0726%**
- INDIVIDUAL_ALLOCATION: 27.9274%

Game-level dominant component counts:
- **TEAM_TE_POOL: 1,262**
- INDIVIDUAL_ALLOCATION: 331

The biggest TE misses are therefore even more clearly driven by getting the **team's overall TE opportunity environment** wrong.

## Team TE target-pool baseline
Across 3,197 team-games:
- MAE: **2.609443 targets**
- RMSE: 3.448559
- bias: **-1.817814 targets**
- correlation: 0.528812
- median absolute error: 1.980176
- p90: **5.735330**
- 3+ miss rate: 33.2499%
- 5+ miss rate: 14.4823%

The negative bias is persistent in every season. This is a football modeling problem at the TE room/team usage layer, not a reason for a blanket post-hoc target lift.

## Scientific conclusion
TE-R1 found TARGETS are the largest TE receiving-yard error mechanism. TE-R2 now locates most of that target error one layer higher: **the model is mis-estimating how much target opportunity a team will allocate to its TE room in that game**.

That means the next predictive work should not begin with a per-player residual correction. It should model the team TE pool from pregame football information first, then preserve a distinct individual-allocation layer beneath it.

This is aligned with the player-centric architecture:

`pregame game environment -> team pass opportunity -> team TE target pool -> individual TE entitlement -> catch conversion -> yard generation`

A better team TE pool is expected to improve the individual TE projection because the player's opportunity budget becomes more accurate before within-room allocation.

## Authorized TE-R3
Freeze a walk-forward **team TE target-pool model** using only pregame football information appropriate to the pool layer. Candidate feature families must include, subject to leakage-safe historical availability:
- strict-prior team TE usage/tendency;
- strict-prior team pass volume/pass rate/pace;
- expected game environment/game-script proxies available pregame;
- opponent TE target/coverage allowance from prior games only;
- opponent pass-volume/coverage context;
- current TE-room availability/participation/competition when historical source integrity permits.

Individual player history, role and within-room share remain a separate downstream allocation layer and must not be discarded.

No generic target lift, sportsbook feature, or post-result cap tuning is authorized.
