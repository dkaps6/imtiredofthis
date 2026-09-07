# RB-R1 Room Volume vs Individual Allocation — Frozen Plan

## Question
The prior RB individual decomposition established that rushing-yard error is CARRIES-dominant for 35 qualifying RBs, YPC-dominant for 25, and mixed for 26. A later role-context test showed that simple depth-chart/current-role states do not broadly explain the carry-dominant errors.

RB-R1 asks the next mechanical football question:

**When the model misses an individual RB's carries, how much of the miss comes from projecting the wrong total RB-room carry volume versus distributing that room volume to the wrong individual player?**

This is a diagnostic routing study. It does not modify production.

## Frozen evidence
- Source run: `34065409969`
- Source artifact name: `rb-individual-mechanism-decomposition`
- Primary input: `rb_individual_mechanism_casebook.csv`
- Exact expected player-game rows: **1,393**
- No sportsbook inputs.

## Identity / room definition
For each `(season, week, team)` represented in the frozen casebook:
- projected RB-room carries = sum of `pred_att` across all included RB-family player rows;
- actual RB-room carries = sum of `actual_att` across the same frozen player universe;
- projected player room share = `pred_att / projected RB-room carries`;
- actual player room share = `actual_att / actual RB-room carries`.

If an actual room total is zero, actual player shares are defined as zero so that `actual room total × actual share = 0` exactly. No QB rushes or other non-RB rushing volume are added after the fact.

## Exact two-factor Shapley decomposition
Individual carries are factorized as:

`RB-room carry volume × player within-room carry share`.

For projected state `(V0, S0)` and actual state `(V1, S1)`:
- room-volume component = `(V1 - V0) × (S0 + S1) / 2`
- individual-allocation component = `(S1 - S0) × (V0 + V1) / 2`

The components must reconcile to:

`actual_att - pred_att`

within floating-point tolerance on every row. If reconciliation fails, no scientific result is allowed.

## Primary summaries
Report for:
1. all 1,393 player-games;
2. player-games belonging to the previously identified **CARRIES-dominant** qualifying players;
3. YPC-dominant players as a diagnostic contrast;
4. mixed players as a diagnostic contrast.

For each slice report:
- rows;
- mean absolute room-volume component;
- mean absolute individual-allocation component;
- allocation/volume absolute-component ratio;
- MAE of individual carries;
- signed carry bias.

## Player profiles
Use the exact player mechanism labels inherited from the frozen source profile/casebook. Qualifying player for RB-R1 requires >= 8 player-games.

For every qualifying player calculate mean absolute room-volume and allocation components. Classify the player's carry-miss submechanism:
- `ROOM_VOLUME` if room-volume absolute mean >= 1.25 × allocation absolute mean;
- `INDIVIDUAL_ALLOCATION` if allocation absolute mean >= 1.25 × room-volume absolute mean;
- otherwise `MIXED`.

No alternate dominance ratio will be tried after results.

## Frozen routing disposition
A strong routing disposition requires all integrity gates plus consistent evidence in the carry-dominant population.

`RB_CARRY_ERRORS_ROUTE_TO_INDIVIDUAL_ALLOCATION` only if:
- carry-dominant slice allocation/volume ratio >= 1.20; and
- >= 50% of qualifying CARRIES-dominant players classify `INDIVIDUAL_ALLOCATION`; and
- at least 20 qualifying CARRIES-dominant players are available.

`RB_CARRY_ERRORS_ROUTE_TO_ROOM_VOLUME` only if:
- carry-dominant slice volume/allocation ratio >= 1.20; and
- >= 50% of qualifying CARRIES-dominant players classify `ROOM_VOLUME`; and
- at least 20 qualifying CARRIES-dominant players are available.

Otherwise disposition is:
`RB_CARRY_ERRORS_REMAIN_MIXED_ROOM_AND_ALLOCATION`.

## Scientific meaning
- A room-volume route sends next research toward team rushing opportunity, game script, pace, score-state, QB/rush interaction, and team run tendency.
- An individual-allocation route sends next research toward current depth/role, personnel competition, injuries, coaching rotation, prior snap/carry role, and player-specific workload entitlement.
- A mixed result means both must remain explicit in the generative stack rather than forcing one universal fix.

No production change is authorized by this diagnostic alone. Any correction must be separately frozen and tested full-stack out of sample.
