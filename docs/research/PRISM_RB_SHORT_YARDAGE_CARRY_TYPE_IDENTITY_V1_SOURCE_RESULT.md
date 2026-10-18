# PRISM RB Short-Yardage Carry-Type Identity V1 — Source / Support Result

Date: 2026-10-17

Frozen plan:
`docs/research/PRISM_RB_SHORT_YARDAGE_CARRY_TYPE_IDENTITY_V1_SOURCE_PLAN.md`

Canonical source authority:
- run `39094323921` — **SUCCESS**
- exact head `5dbfe88491a7398bb9ce34450a5dffb22bda7a84`
- artifact `12123703725`
- digest `sha256:c99b5462dd5c501b5d5ebfbed4c63f7bc9030cd9ed6a23182beebae953a9b989`

Disposition:

`PRISM_RB_SHORT_YARDAGE_CARRY_TYPE_IDENTITY_SOURCE_READY`

## Frozen opportunity type

A named RB carry is short-yardage iff:

`down in {3,4} AND 0 < ydstogo <= 2`.

No field-position, score, clock, goal-line or second-down filter is included.

## Source quality

Across 2022-2025 regular seasons:
- stable rusher identity coverage: 100%;
- `down` coverage on named-RB rushes: 99.78%-99.84%;
- `ydstogo` coverage on named-RB rushes: 100%;
- stable roster-family coverage: 99.88%-99.93%;
- duplicate named player-play rows: 0;
- short-yardage events occurred in all 18 regular-season weeks each year.

Short-yardage RB carry events:
- 2022: 759
- 2023: 727
- 2024: 755
- 2025: 836

## Frozen support gate

Minimum history:
**10 strictly-prior same-player same-team carries**.

2024:
- supported target player-games: **1,147**
- supported named RBs: **109**

2025:
- supported target player-games: **1,136**
- supported named RBs: **115**

Both predeclared gates passed by wide margins:
- >=500 player-games;
- >=60 named players.

## Scientific boundary

This result authorizes only a separately frozen test of **within-player carry
type**:

`TOTAL PLAYER CARRY EXISTS -> SHORT-YARDAGE TYPE OR ORDINARY TYPE`.

It does not authorize:
- more/fewer total player carries;
- a new team rush-volume model;
- a new recent carry-share allocator;
- rushing-yards or TD translation;
- production promotion.

The finite-opportunity architecture remains unchanged.

## Integrity

- predictive candidates scored: 0
- 2026 outcomes read: 0
- sportsbook inputs: 0
- OddsAPI calls: 0
- production mutation: false
