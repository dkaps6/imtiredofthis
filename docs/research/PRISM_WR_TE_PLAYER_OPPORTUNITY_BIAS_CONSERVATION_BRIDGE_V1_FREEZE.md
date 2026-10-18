# PRISM WR/TE Player Opportunity-Bias Conservation Bridge V1 — Frozen Action Contract

Date: 2026-10-18

Status: **FROZEN BEFORE ANY BRIDGE OUTCOME SCORE**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_WR_TE_PLAYER_OPPORTUNITY_BIAS_CONSERVATION_BRIDGE_V1`

## 1. Scientific question

The player-specific signed component-bias signal has already qualified for:
- WR opportunity;
- TE opportunity.

Can that strictly-prior named-player signal be translated into the existing
opportunity-allocation mechanisms, with exact finite-opportunity conservation,
and improve **named-player target opportunity and final receiving-yard
projections** out of sample?

This bridge is allowed to change opportunity entitlement only.

Efficiency stays frozen.

## 2. Preconditions

The bridge may score only if all required source/deployment authorities are
valid.

### Both positions
The source-recovery audit must be:
`PRISM_WR_TE_OPPORTUNITY_BIAS_CONSERVATION_BRIDGE_SOURCE_RECOVERED_READY_ALL`.

### WR
The separately frozen topology-eligibility result determines which WR-R15
topology classes are allowed to use player-specific bias.

No WR action may be scored before that result exists.

### TE
No additional topology split is required. TE-R5P is already a whole-TE-pool
softmax and the qualified TE opportunity signal is the authority.

## 3. Frozen player modifier

For an eligible supported player:

`bias_factor = exp(-hierarchical_signed_bias / parent_predicted_receiving_yards)`.

Where:
- `hierarchical_signed_bias` is the frozen player signed opportunity-bias
  signal;
- positive bias means the parent overprojects the player's opportunity;
- parent predicted receiving yards is strictly the frozen parent projection
  for that row.

Therefore:
- positive historical overprojection -> factor <1;
- negative historical underprojection -> factor >1.

Unsupported players use:
`bias_factor = 1`.

No clipping.
No winsorization.
No fitted scale.
No threshold.
No alternate transform.

This exact exponential factor was frozen during source planning before bridge
outcomes were inspected.

## 4. WR action — respect actual R15 hierarchy

Parent:
`WR_R15_WR1_ANCHORED_PARTICIPATION`.

Parent allocation state:
the exact candidate `entitlement_tgt_share` produced by WR-R15 before the
existing explicit simulator.

Topology eligibility controls factor activation.

### Case A — ANCHOR closed, SECONDARY eligible
- anchor factor is forced to 1 and anchor entitlement stays **bit-exact**;
- supported secondary WRs receive their frozen bias factor;
- unsupported secondary WRs receive factor 1;
- multiply only secondary parent entitlement by factors;
- renormalize secondary entitlement back to the exact original secondary-pool
  mass;
- anchor entitlement is unchanged;
- total WR-room mass and total team-player mass remain exact.

### Case B — ANCHOR eligible
This rule applies whether SECONDARY is eligible or closed.

- supported anchor receives its frozen bias factor;
- if SECONDARY eligible, supported secondary players receive their frozen bias
  factors;
- if SECONDARY closed, all secondary players use factor 1;
- unsupported players always use factor 1;
- multiply the parent WR entitlements by active factors;
- renormalize the **entire WR room** back to its exact original WR-room mass.

Thus if only ANCHOR is eligible, secondary players absorb the conservation
offset proportionally without using a failed secondary player-specific signal.

### Case C — both WR topology classes closed
No WR bridge is scored or promoted.

### WR downstream propagation
After entitlement modification:
- inject the modified WR entitlement state into the exact authority-era parent
  bundle;
- leave every non-WR input unchanged;
- run the existing authority `explicit_simulate`;
- use the same deterministic simulation iteration count and the same
  season/week seeds as the frozen WR-R15 authority;
- no direct hand-built target-count or receiving-yard patch is permitted.

## 5. TE action — exact R5P TE pool

Parent:
`TE_R5P_PRODUCTION_CONTRACT_REFIT_V1`.

Parent TE-R5P mechanics:
- `entitlement_score = log(b0_te_room_share + EPS) + predicted_residual`;
- team softmax produces `candidate_room_share_r5p`;
- `candidate_targets_r5p = b0_te_pool * candidate_room_share_r5p`.

Bridge:
- start from the parent TE-R5P room weights/shares;
- supported TE receives frozen `bias_factor`;
- unsupported TE receives factor 1;
- multiply parent TE room weights by factor;
- renormalize within the exact team TE room to 1;
- multiply by the **unchanged** parent `b0_te_pool`;
- single-TE rooms are mechanically unchanged;
- preserve parent predicted receptions-per-target and receiving-yards-per-target;
- no efficiency field changes.

Equivalent score-space representation:
`bridge_score = parent_score + log(bias_factor)`.

No other TE feature or coefficient changes.

## 6. Frozen development / confirmation windows

WR:
- development 2023;
- independent confirmation 2024.

TE:
- development 2024;
- independent confirmation 2025.

No 2026 outcome.
No Week-6 2026 outcome.
No sportsbook input.

## 7. Frozen row-level outcome metrics

For target opportunity:
`target_AE_gain = abs(parent_targets - actual_targets) - abs(bridge_targets - actual_targets)`.

For final receiving yards:
`rec_yards_AE_gain = abs(parent_rec_yards - actual_rec_yards) - abs(bridge_rec_yards - actual_rec_yards)`.

Positive means the bridge improves the named-player projection.

Report per position/season:
- target MAE / RMSE / p90 absolute error;
- receiving-yard MAE / RMSE / p90 absolute error;
- mean target AE gain;
- mean receiving-yard AE gain;
- target and receiving-yard signed bias;
- Spearman of predicted vs actual targets;
- Spearman of predicted vs actual receiving yards.

## 8. Frozen player-aware inference

Primary inferential quantity:
`rec_yards_AE_gain`.

Exactly:
- 5,000 bootstrap replicates;
- seed `20261018`;
- player-cluster bootstrap;
- team-week-cluster bootstrap;
- equal-player bootstrap;
- equal-player diagnostics require >=5 scored rows for a named player.

Also report equal-player target AE gain.

Report:
- equal-player mean / median receiving-yard gain;
- named-player receiving-yard improved share;
- p10 / p25 / p50 / p75 / p90 player receiving-yard gain;
- worst named-player receiving-yard mean gain;
- equal-player target mean / median gain;
- named-player target improved share.

## 9. Frozen conservation / integrity gates

Every scored position/season must satisfy:
- exact source authority;
- signal join coverage 100%;
- zero duplicate identity rows;
- all parent/bridge predictions finite;
- all bridge target entitlements nonnegative;
- all bridge target projections nonnegative;
- maximum room-mass conservation gap <=1e-12;
- maximum protected non-position input delta =0;
- efficiency inputs unchanged;
- sportsbook inputs =0;
- same/future feature violations =0.

WR when ANCHOR is closed additionally requires:
- maximum anchor entitlement delta <=1e-12.

TE additionally requires:
- maximum TE-pool target mass gap <=1e-12.

Any integrity failure blocks the position.

## 10. Frozen qualification gates

WR and TE qualify independently.
No cross-position pooling may rescue a failed position.

### Development season
All required:
1. source/support/integrity gates pass;
2. mean target AE gain >0;
3. bridge target RMSE < parent target RMSE;
4. bridge target p90 absolute error <= parent target p90 absolute error;
5. mean receiving-yard AE gain >0;
6. bridge receiving-yard RMSE < parent receiving-yard RMSE.

### Independent confirmation season
All required:
7. source/support/integrity gates pass;
8. mean target AE gain >0;
9. bridge target RMSE < parent target RMSE;
10. bridge target p90 absolute error <= parent target p90 absolute error;
11. mean receiving-yard AE gain >0;
12. player-cluster 95% CI lower bound for receiving-yard AE gain >0;
13. team-week-cluster 95% CI lower bound >0;
14. equal-player receiving-yard mean gain >0;
15. equal-player 95% CI lower bound >0;
16. median named-player receiving-yard gain >=0;
17. named-player receiving-yard improved share >0.50;
18. bridge receiving-yard RMSE < parent receiving-yard RMSE;
19. bridge receiving-yard p90 absolute error <= parent receiving-yard p90 absolute error;
20. confirmation tail guard:
    `p10_player_rec_yards_gain >= -0.25 * parent_equal_player_rec_yards_MAE`;
21. equal-player target mean gain >0;
22. named-player target improved share >0.50.

Spearman is reported as diagnostic and must not be used to rescue a failed
error-distribution gate.

## 11. Frozen support

A position/season must retain:
- >=200 bridge-effect rows;
- >=40 distinct named players.

A bridge-effect row is a row whose room contains at least two players and for
which the player or another player in that room receives an active factor, so
conservation can actually change the projection.

Single-player TE rooms are not bridge-effect rows.

For WR, rows in rooms with no active eligible factor are not bridge-effect
rows.

No support floor may be changed after outcomes are scored.

## 12. Promotion meaning

Qualification means only:
- this player-specific opportunity-bias mechanism can improve the existing
  research parent when translated through the real conserved opportunity
  mechanism.

It does **not** automatically authorize production mutation.

Production remains unchanged unless separately authorized.

If a position fails:
- close this exact bridge for that position;
- do not tune the factor;
- do not change support floors;
- do not rescue by player subsets, positive/negative bias, star/non-star,
  role labels, team/opponent, or sportsbook data.

## 13. Integrity at freeze

- bridge outcomes scored: **0**;
- target corrections applied: **0**;
- room renormalizations applied: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 / Week-6 outcomes used: **0**;
- sportsbook / OddsAPI used: **0**;
- production changed: **false**.

Disposition:
`PRISM_WR_TE_PLAYER_OPPORTUNITY_BIAS_CONSERVATION_BRIDGE_V1_FROZEN_BEFORE_SCORING`.
