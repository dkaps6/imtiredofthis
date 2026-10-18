# PRISM RB Short-Yardage Carry-Type Identity V1 — Source / Support Audit Plan

Date: 2026-10-17

Status: **FROZEN BEFORE SOURCE AUDIT / NO PREDICTIVE SCORING / RESEARCH ONLY**

Version:
`PRISM_RB_SHORT_YARDAGE_CARRY_TYPE_IDENTITY_V1_SOURCE_AUDIT`

## Scientific object

PRISM projects the named player in his actual football role.

This lane asks an opportunity-**type** question while preserving finite total
rushing opportunity:

> Conditional on a named RB receiving a carry, is his strictly-prior same-team
> tendency to receive high-leverage short-yardage carries source-valid and
> sufficiently supported to justify a predictive mechanism test?

This does **not** change projected team rush volume or the player's total carry
entitlement. It asks what kind of carries compose that entitlement.

## Why this is distinct

Protected closures remain protected:
- generic recent carry-share replacement is closed;
- RB individual run-location matchup V1 is closed;
- M95A/M95B generic role x run-defense is closed;
- scoring-area high-value share persistence is closed;
- direct historical scoring-area TD conversion is closed;
- RB depth-role transition and teammate-vacancy source paths remain blocked.

This lane is not another team-pool share allocator.

The outcome is the named player's **within-player carry-type composition**:
short-yardage carry versus ordinary carry, conditional on that player carrying
the ball.

That composition may later matter differently for rushing yards, carries and
TD markets, but no translation is authorized in the source stage.

## Frozen short-yardage definition

A qualifying RB rushing event is:
- native `rush_attempt == 1`;
- stable named rusher family RB/FB/HB;
- regular season;
- nonmissing `down` and `ydstogo`.

Frozen short-yardage flag:

`down in {3,4} AND 0 < ydstogo <= 2`.

Do not add 2nd-and-short, goal-to-go, yardline bands, score state or clock
filters after support is observed.

QB rushes/sneaks are outside this V1.

## Historical source

- canonical nflverse/nflfastR PBP through repo `get_pbp`;
- canonical weekly-roster GSIS identity bridge;
- 2022-2025 regular seasons;
- no 2026 outcomes.

Required fields:
- season;
- week;
- game_id;
- posteam;
- rush_attempt;
- down;
- ydstogo;
- rusher stable player identity.

## Strict-prior support object

Aggregate every named RB player-game into:
- total RB carries;
- short-yardage carries.

For every target player-game, construct same-player **same-team** history from
completed earlier games only.

Frozen minimum support for a later candidate:
**>=10 prior same-team carries**.

The source audit does not choose a coefficient, shrinkage strength or recency
window.

## Frozen source gates

For every season 2022-2025:
1. regular-season PBP nonempty;
2. stable rusher identity coverage >= 99.0%;
3. weekly-roster RB-family bridge >= 98.0%;
4. `down` coverage on named-RB rushes >= 99.5%;
5. `ydstogo` coverage on named-RB rushes >= 99.5%;
6. short-yardage events > 0 in every regular-season week;
7. duplicate player-play contradictions = 0;
8. short-yardage flag is deterministic/binary.

For both 2024 and 2025:
9. >=500 target player-games with >=10 strictly-prior same-player same-team
   carries;
10. >=60 distinct supported named RBs.

All gates are AND gates.

Do not lower support gates after inspection.

## What source readiness would authorize

Only one separately frozen predictive mechanism:

conditional on a target carry, compare:
- strictly-prior same-team **team-RB** short-yardage carry probability;
- versus a shrunk same-player same-team short-yardage carry-type identity.

Total carries remain unchanged on both sides.

Any later translation into rushing yards or touchdowns must be separately
frozen and must preserve the canonical finite-opportunity architecture.

## Anti-rescue

If source-blocked, do not rescue with:
- fewer than 10 prior carries;
- all-team player history;
- position pooling;
- 2nd-and-short;
- alternate ydstogo cutoffs;
- yardline/goal-line bands;
- lead-back subsets;
- sportsbook/OddsAPI;
- 2026 outcomes.

## Integrity

Required:
- predictive candidates scored = 0;
- 2026 outcomes read = 0;
- sportsbook inputs = 0;
- OddsAPI calls = 0;
- production mutation = false.
