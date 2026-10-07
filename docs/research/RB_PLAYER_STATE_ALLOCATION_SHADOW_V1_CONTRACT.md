# RB Player State Allocation Shadow V1 — Frozen Prospective Contract

**STATUS: FROZEN BEFORE WEEK-5 OUTCOMES. SHADOW ONLY. NO PRODUCTION CHANGE.**

Parent source authority:
- `PLAYER_STATE_LIVE_READY_FOR_PROSPECTIVE_SHADOW`
- run `37560311001`
- artifact `11456556226`
- digest `sha256:a39b958e492a781e310de0f14d34153e21ca589a76cd226479b3b39e10f9328e`
- target source snapshot: 2026 Week 5
- Week-5 outcomes read by parent: 0
- sportsbook inputs: 0

## Purpose

Test one narrow player-centric hypothesis prospectively:

> Within a finite NFL backfield, strictly-prior individual participation state contains incremental information about next-game RB carry allocation beyond recent carry allocation alone.

This is an **allocation** test, not a rushing-yard model.

It directly tests the open seam identified by the full-stack player-individualization audit while preserving:
- team rushing opportunity;
- M94/P3/M96 historical conclusions;
- RB Rush+Receiving Conservation V2;
- player efficiency;
- opponent defense;
- Monte Carlo;
- sportsbook separation.

## Standing RB prohibition

The M96 terminal stop remains binding.

This contract does not:
- fit on 2025;
- retune an M96 router;
- search thresholds;
- reopen YPC / efficiency families;
- use historical target outcomes to choose weights;
- use sportsbook data.

All evidence begins with untouched prospective 2026 target games.

## Target

For each locked team-game, predict each locked RB/HB/FB player's:

`actual_rb_room_carry_share = player_rush_attempts / sum(RB/HB/FB rush_attempts for team)`

QB and WR rushing are excluded from the denominator.

The shadow does not predict team rush attempts. It predicts only allocation of the finite RB room.

## Locked pregame state

Use only the frozen parent Week-5 row artifact.

For every target-team RB/HB/FB:

1. `last3_room_opportunity_share`
   - strictly-prior same-team RB-room carry share from the latest up-to-three completed games.

2. `last3_room_snap_fraction`
   - strictly-prior last-three offensive participation, normalized inside the current RB room.

Both fields were constructed before target outcomes and have chronology < target week.

Target-week injury status is **not** a V1 feature because the public Week-5 injury source contained zero rows at the parent capture. Do not add it after this contract merely because it becomes available later.

GSIS/private successor data is not used.

## Locked row universe

A player is V1-lockable when:
- target roster position is RB/HB/FB;
- stable target team exists;
- both `last3_room_opportunity_share` and `last3_room_snap_fraction` are finite.

A team is lockable when:
- at least two RBs are lockable.

At outcome grading, a locked team-game is scoreable only if **every RB/HB/FB who records at least one target-game rush attempt was present in the pregame locked player set**.

If an un-locked RB records a carry, fail that team-game closed as `UNSCORED_NEW_OR_MISSING_CARRIER`. Never assign that player zero pregame share after the fact.

No team or player may be added to the Week-5 lock after target outcomes are available.

## Control

The control is deliberately simple and player-specific:

`CONTROL_RECENT_CARRY`

Within the locked player set for each team:

1. take each player's raw `last3_room_opportunity_share`;
2. renormalize those values to sum to 1.0 among locked players.

This represents “recent carries alone.”

## Shadow

Candidate:

`RB_PLAYER_STATE_ALLOC_V1`

For each locked team:

1. renormalize `last3_room_opportunity_share` among locked players:
   `carry_state_i`;

2. renormalize `last3_room_snap_fraction` among locked players:
   `snap_state_i`;

3. fixed no-fit blend:
   `shadow_share_i = 0.50 * carry_state_i + 0.50 * snap_state_i`.

The 50/50 weight is an equal-information structural blend frozen prospectively. It is not fit to any target outcome.

Because both components are normalized, shadow shares sum exactly to 1.0 inside each locked backfield.

No intercept, cap, threshold, depth rank, player name, hand tuning, opponent feature, efficiency feature, or sportsbook input is permitted.

## Week-5 lock outputs

Before outcomes, persist:
- target season/week;
- team/opponent;
- player stable identity and display name;
- raw source fields;
- control share;
- shadow share;
- room size;
- row-universe membership;
- source authority IDs;
- generated timestamp;
- SHA256 of canonical row CSV;
- proof sportsbook inputs = 0;
- proof target outcomes read = 0.

The Week-5 lock is immutable after creation.

## Prospective accumulation

Week 5 is lock #1.

Before each later target week, the same formula may be applied to a newly captured strictly-prior public state snapshot.

No formula, weight, row rule, or grading rule may change while the shadow is accumulating.

Scientific disposition is forbidden until at least:
- **4 distinct future locked weeks**;
- **80 scoreable team-games**;
- **200 scoreable RB player-games**.

A weekly diagnostic may be recorded, but it cannot change V1.

## Frozen evaluation

Primary:
- player-level absolute error of RB-room carry share.

For each scoreable team-game:
- control player-share MAE;
- shadow player-share MAE.

Pooled:
- player-share MAE;
- RMSE;
- signed bias;
- team-game mean paired improvement.

Dependence-aware inference:
- team-game cluster bootstrap;
- 10,000 replicates;
- seed `20261007`;
- statistic = control AE minus shadow AE;
- positive favors shadow.

Secondary:
- predicted top-back accuracy vs actual top RB carry share;
- within-team Spearman rank correlation between predicted and actual carry share;
- season-week summaries;
- room-size summaries.

Ties use average rank; top-back ties count correct if the actual top carrier is among tied predicted leaders.

## PASS gate after minimum support

`RB_PLAYER_STATE_ALLOC_V1_CONFIRMED` requires all:

1. support floors met;
2. pooled shadow player-share MAE < control MAE;
3. pooled shadow RMSE <= control RMSE;
4. cluster-bootstrap P(mean AE improvement > 0) >= 0.80;
5. top-back accuracy >= control;
6. pooled within-room rank correlation >= control;
7. shadow MAE is non-worse in at least 3 of the first 4 qualified locked weeks;
8. every lock has zero chronology violations;
9. every lock has zero sportsbook inputs;
10. no target-week outcome entered before its lock;
11. production remains unchanged.

Otherwise:

`RB_PLAYER_STATE_ALLOC_V1_CLOSED`

No rescue.

## Explicit anti-rescue

After Week-5 outcomes exist, do not try:
- 60/40 or 40/60 weights;
- last-1 vs last-3;
- snap-only;
- carry-only as a new candidate;
- top-2 RB carveouts;
- depth rank;
- minimum carry thresholds;
- injury additions;
- high/low room concentration;
- opponent splits;
- favorite/underdog splits;
- YPC/efficiency;
- rushing-yard outcome tuning.

A materially new idea requires a new prospective contract and a new future lock.

## Promotion boundary

A PASS would establish only that current player participation state improves **within-backfield workload allocation** prospectively.

It would not authorize production.

Production integration would still require a separate frozen test proving:
- compatibility with finite team rushing opportunity;
- no QB/WR rushing mass corruption;
- RB Rush+Receiving V2 conservation;
- downstream rush-yard and combo-yard non-harm;
- player-level and aggregate error gates;
- no sportsbook input upstream.

Production mutations authorized: **0**  
Sportsbook inputs authorized: **0**  
Week-5 outcomes authorized before lock: **0**
