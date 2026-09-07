# RB-R2 Allocation-Subgroup Role Context — Frozen Plan

## Why this is new
The earlier carry-role-context audit tested all 35 CARRIES-dominant RBs together and found no strong global role-context concentration. RB-R1 then supplied genuinely new player-level information by decomposing carry misses into RB-room volume versus within-room individual allocation:
- 17 CARRIES-dominant players classified `INDIVIDUAL_ALLOCATION`;
- 12 mixed;
- 6 `ROOM_VOLUME`.

RB-R2 does not reopen the failed global role-context gate. It asks the mechanism-conditioned question:

**Inside the 17 players whose carry misses are specifically allocation-dominant, do pregame role/depth/transition states concentrate the individual allocation error?**

No production change is authorized.

## Frozen evidence
- RB-R1 run `34070566812`, artifact `10000309225`, artifact name `rb-r1-room-volume-vs-individual-allocation`, digest `sha256:daf7ace3bfc256b71d63930558322593bf0d88f046886848489f1877bf1bb116`.
- Exact source casebook: `rb_r1_room_allocation_casebook.csv`.
- Exact player classifications: `rb_r1_player_submechanisms.csv`.
- Expected qualifying allocation-dominant parent-CARRIES players: **17**.
- Sportsbook inputs prohibited.

## Conditioned population
Rows for players satisfying both:
- parent mechanism = `CARRIES`;
- RB-R1 carry submechanism = `INDIVIDUAL_ALLOCATION`.

The primary error quantity is exact RB-R1 `individual_allocation_component`; carry MAE is the secondary outcome.

## Frozen pregame states
Primary state:
1. `state_depth_vs_carry_order_mismatch`

Secondary transition states, each tested separately:
2. `state_injury_created_context`
3. `state_no_prior_same_team_game`
4. `state_rookie`
5. `state_limited_prior_history`

No interaction states, combinations, alternate definitions, or threshold searches.

## Candidate scoring
For state=1 versus state=0 report:
- positive-state N and negative-state N;
- mean absolute individual-allocation component;
- state1/state0 allocation-component ratio;
- carry MAE and state1/state0 carry-MAE ratio;
- signed actual-minus-projected carry residual;
- W2-18 allocation-component ratio;
- W13-18 allocation-component ratio.

## Frozen state pass gate
A state passes only if every condition holds:
1. state=1 N >= 25;
2. state=0 N >= 50;
3. allocation-component absolute ratio >= 1.25;
4. carry-MAE ratio >= 1.15;
5. state1 minus state0 mean absolute allocation-component difference >= 0.75 attempts;
6. W2-18 allocation-component ratio > 1.00;
7. W13-18 allocation-component ratio > 1.00.

No gate may be lowered after results.

## Disposition
- `RB_ALLOCATION_SUBGROUP_DEPTH_SIGNAL_PASS` if the primary depth/carry-order mismatch state passes all gates.
- Otherwise `RB_ALLOCATION_SUBGROUP_TRANSITION_SIGNAL_PASS` only if at least **2 of the 4** secondary states independently pass all gates.
- Otherwise `NO_ACTIONABLE_RB_ALLOCATION_SUBGROUP_ROLE_CONTEXT_SIGNAL`.

A pass is diagnostic only. It would authorize a separately frozen full-stack allocation-layer experiment, not a hand adjustment to individual RB projections.
