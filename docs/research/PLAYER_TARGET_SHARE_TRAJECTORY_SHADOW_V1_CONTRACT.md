# Player Target Share Trajectory Shadow V1 — Frozen Week-5 Prospective Contract

**STATUS: FROZEN BEFORE WEEK-5 OUTCOMES. SHADOW ONLY. NO PRODUCTION CHANGE.**

Parent signal:
- `PLAYER_TARGET_SHARE_TRAJECTORY_SIGNAL_CONFIRMED`
- run `37638269235`
- artifact `11490707699`
- digest `sha256:2f032cb5e68f5e097e30ca56a2ebe30e505f90233bc60774e4318158fad3fafd`

## Purpose

Prospectively test whether the confirmed individual-player target-share trajectory signal improves WR/TE target entitlement when added **on top of** the current promoted M38 / WR-R15 / TE-R5P architecture.

This is a complement, not a replacement.

## Protected architecture

Keep unchanged:
- team pass volume;
- M38 WR1 anchor;
- WR-R15 WR2+ conserved pool;
- TE-R5P conserved TE room;
- QB mean/distribution science;
- receiving efficiency;
- simulation/distribution formulas;
- sportsbook separation.

The shadow changes only the within-room allocation of existing target entitlement.

## Pregame target

- season: 2026
- first lock: Week 5
- all source data must be strictly before Week 5
- no Week-5 outcome/stat/PBP row may be read

## Baseline entitlement

Build the current public football universe using the same production-order components:

`PlayerForm/Bayesian/rules -> M38 explicit entitlement -> TE-R5P -> WR-R15`

Use committed current production TE-R5P and WR-R15 model assets.

No sportsbook input is permitted.

The baseline target share for the shadow is the final current production `entitlement_tgt_share`.

## Trajectory feature

For each current WR/TE player on his target team:

1. use same-season completed offensive games strictly before target week;
2. require at least 4 prior target-team games;
3. RECENT2 = latest two completed target-team games;
4. EARLIER = all completed target-team games before RECENT2;
5. require EARLIER >=2 team games;
6. compute:
   - `recent2_share = player RECENT2 targets / team RECENT2 targets`
   - `earlier_share = player EARLIER targets / team EARLIER targets`
   - `trajectory_delta = recent2_share - earlier_share`

If a player does not satisfy the feature-support rules, set `trajectory_delta = 0` for shadow allocation only and mark him `TRAJECTORY_UNAVAILABLE_NO_CHANGE`.

No imputation from position, teammate, sportsbook, or target-game information.

## Frozen transformation

No coefficient is fitted.

For every eligible room player:

`weight_i = baseline_entitlement_i * exp(trajectory_delta_i)`

Then renormalize weights within the protected room to preserve the original room pool exactly.

### TE

Room:
- all current TE rows for the event/team.

Preserve:
- exact baseline TE-room target mass.

### WR

M38 WR1 anchor:
- frozen exactly at baseline entitlement;
- trajectory is diagnostic only for the anchor and does not alter it.

Room eligible for shadow:
- WR2+ rows only.

Preserve:
- exact WR1 entitlement;
- exact WR2+ pool;
- exact total WR room mass.

No other position changes.

## Why exponentiation

The signal is directional and continuous.

`exp(delta)` gives:
- positive monotonic reweighting;
- no fitted coefficient;
- no threshold;
- exact pool conservation after normalization;
- zero feature -> exact no-change weight.

This V1 does not search alternate scales.

## Immutable lock outputs

Before target outcomes, persist:

1. row-level baseline and shadow entitlement:
   - season/week/event/team/opponent
   - player identity/display name
   - position
   - WR anchor flag
   - baseline entitlement
   - recent2 share
   - earlier share
   - trajectory delta
   - trajectory availability
   - raw trajectory weight
   - shadow entitlement
   - entitlement delta

2. room-level audit:
   - baseline pool
   - shadow pool
   - max conservation gap
   - number of changed players
   - max player share change

3. lock metadata:
   - source chronology max week
   - stable-ID mapping coverage
   - number of target teams
   - row CSV SHA256
   - sportsbook inputs = 0
   - Week-5 outcomes read = 0
   - fitted parameters = 0
   - production changed = false

Week-5 lock is immutable after creation.

## Week-5 scoring

When Week-5 outcomes become available, score only the locked player rows.

Actual target share:

`actual_team_target_share = player target events / team target events`

Primary:
- absolute target-share error baseline vs shadow.

Secondary:
- RMSE;
- signed bias;
- player rank correlation within team-position room;
- top target-earner accuracy within TE room and WR2+ room;
- receiving-yard translation diagnostic using the existing baseline per-target yard rate with baseline vs shadow target share.

The receiving-yard translation is diagnostic only and does not alter production efficiency.

## Prospective accumulation

Week 5 is lock #1.

No final PASS/FAIL until at least:
- 4 future locked weeks;
- 400 scoreable WR/TE player-games;
- >=120 distinct WR/TE player identities;
- >=80 scoreable team-position rooms.

No formula, exponent scale, support rule, room definition, or grading rule may change during accumulation.

## Final PASS gate

`PLAYER_TARGET_SHARE_TRAJECTORY_SHADOW_CONFIRMED` requires:

1. support floors met;
2. pooled shadow target-share MAE < baseline;
3. pooled shadow RMSE <= baseline;
4. player-cluster bootstrap P(MAE improvement > 0) >= 0.80;
5. WR pooled MAE improves;
6. TE pooled MAE improves;
7. shadow non-worse in at least 3 of first 4 qualified weeks;
8. room-rank correlation >= baseline;
9. no room-mass conservation violation >1e-12;
10. zero chronology violations;
11. zero sportsbook inputs;
12. production unchanged.

Otherwise:
`PLAYER_TARGET_SHARE_TRAJECTORY_SHADOW_CLOSED`

No rescue.

## Anti-rescue

Do not try after Week-5 outcomes:
- different exponent coefficients;
- clipping/capping trajectory;
- recent1/recent3/recent4;
- rising-only routing;
- falling-only routing;
- WR-only rescue;
- TE-only rescue;
- WR1 unfreezing;
- target-count thresholds;
- model-residual features;
- sportsbook-conditioned variants.

A materially new design requires a new future prospective lock.

Production mutations authorized: **0**
Sportsbook inputs authorized: **0**
Week-5 outcomes before lock: **0**
Fitted parameters: **0**
