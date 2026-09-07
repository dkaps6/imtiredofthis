# TE-R3 Target-Pool Context Model — Result

## Disposition

`TE_TARGET_POOL_CONTEXT_MODEL_FAIL`

This is a **scientific failure**, not an integrity/source failure. No production change.

The model substantially improved the team TE target pool and individual TE target projections, but it failed the frozen individual receiving-yard gates and failed the frozen differential-correction gate because the learned correction was almost universally positive.

## Canonical evidence
- Branch: `research-te-r3-target-pool-context-model`
- Frozen plan commit: `a95a9f66a4615a56c6f9a5c86174b8f20aef2da9`
- Implementation commit: `41ff49a856c0acf249af1fa3d3fd97e50ad0be26`
- Canonical run: `34126813280`
- Job: `101757309151`
- Tested SHA: `c9d1cd1858e34900e33fabc07848fc3f9f86bd1e`
- Artifact: `10020423842` (`te-r3-target-pool-context-model`)
- Artifact digest: `sha256:99699c7e4745ab8cf678b527fc57325c5f402a5c7be930ff9cb90dba96e63ff8`
- OOS team-games: **2,134**
- OOS TE player-games: **4,288**
- OOS seasons: 2022-2025
- sportsbook inputs: 0
- same/future outcome uses: 0
- B0 player parity max gap: `8.881784197001252e-16`

All frozen integrity gates passed.

## Team TE target-pool result
This layer improved clearly:
- MAE: **2.604965 -> 2.293122** (-0.311843 targets)
- RMSE: 3.448480 -> 2.896628
- bias: **-1.789603 -> -0.102031**
- correlation: 0.533345 -> 0.557307
- p90 absolute error: **5.812586 -> 4.641891**
- 3+ miss rate: 32.8960% -> 28.3974%
- 5+ miss rate: 14.6673% -> 7.8257%

The team-pool MAE improved in **all 4 OOS seasons**, passing its frozen gate.

## Individual TE target result
The better team pool propagated into better player target forecasts:
- MAE: **1.666440 -> 1.605626** (-0.060814 targets)
- RMSE: 2.418847 -> 2.142149
- bias: **-0.890628 -> -0.050777**
- correlation: 0.610239 -> 0.625815
- p90: **4.024803 -> 3.333590**
- 4+ target miss rate: 10.0979% -> 6.1567%
- 6+ target miss rate: 3.7080% -> 1.8657%

This passed the frozen individual-target MAE gate.

## Individual TE receiving-yard result
Despite the target improvements, the candidate failed the main player-yard objective:
- MAE: **16.183617 -> 16.339728** (+0.156111 yards; worse)
- RMSE: **23.247476 -> 21.911693** (better)
- bias: **-6.163127 -> +0.061917**
- median absolute error: **11.368911 -> 13.057228** (worse)
- p90: **37.185429 -> 33.143535** (better)
- 20+ miss rate: **24.5103% -> 27.1455%** (worse)
- 30+ miss rate: **13.4795% -> 12.2901%** (better)
- 40+ miss rate: **8.7687% -> 6.4132%** (better)

Receiving-yard MAE improved in only **1 of 4** OOS seasons. Combined 2024-2025 MAE worsened **16.178397 -> 16.344061**.

The candidate therefore cannot be promoted merely because RMSE/p90 and large-miss tails improved. The project objective requires accurate individual point projections as well as tail control.

## High-opportunity Q4 finding
The highest B0 TE opportunity quartile did improve materially:
- receiving-yard MAE: **23.993853 -> 22.605126**
- 30+ miss rate: 26.5858% -> 26.5858%
- 40+ miss rate: **18.5634% -> 15.0187%**

However, this was not enough to rescue the frozen full-population candidate, and no Q4-only exception may be promoted post hoc.

## Correction-behavior failure
The candidate correction had enough dispersion (`SD=0.636609`) but was not genuinely two-sided:
- mean correction: **+1.687572 targets**
- p10: +0.864750
- median: +1.684119
- p90: +2.530322
- positive: **99.4845%** of OOS team-games
- negative: **0.5155%**
- cap-hit rate: 2.8116%

Frozen differential gate required at least 15% positive and 15% negative corrections. It failed decisively.

As with WR-R11, the model primarily learned to erase a broad baseline underprojection rather than identify sufficiently game-specific positive **and negative** TE opportunity environments.

## Scientific interpretation
TE-R2 remains valid: team TE pool is the dominant target-error layer. TE-R3 also demonstrates that football context/history can forecast that pool better. But simply correcting pool volume and then preserving existing per-target receiving-yard conversion is not enough for the full individual projection problem.

The evidence points to a more structured player model:

`game/pass environment -> TE-room opportunity -> individual TE participation/role -> individual target entitlement -> catch/depth/YAC efficiency -> matchup-conditioned yard distribution`

The next TE work must bring genuinely new **individual participation/role and matchup information** rather than retuning the pool model or applying a post-hoc Q4-only exception.

## Do not
- retune Ridge alpha or the +/-3 target cap;
- recenter the corrections after seeing the result;
- create a Q4-only promotion from this run;
- lower the receiving-yard MAE gate;
- call the target-pool improvement a production win by itself.

## Authorized next direction
Audit and, if source-eligible, test strict-prior TE participation/role information (offensive snap share and related room competition) plus opponent coverage/personnel context as a separate individual-allocation/efficiency layer. The validated team-pool insight may remain as research evidence, but any new model must be frozen independently and win the complete individual projection scorecard.
