# PRISM RB Short-Yardage Carry-Type Identity V1 — Predictive Mechanism Freeze

Date: 2026-10-17

Status: **FROZEN BEFORE SCORING / RESEARCH ONLY / NO PRODUCTION CHANGE**

Source authority:
- run `39094323921` — SUCCESS;
- exact source head `5dbfe88491a7398bb9ce34450a5dffb22bda7a84`;
- artifact `12123703725`;
- disposition `PRISM_RB_SHORT_YARDAGE_CARRY_TYPE_IDENTITY_SOURCE_READY`.

## Exact scientific question

> Conditional on the named RB receiving a carry, does his own strictly-prior
> same-team short-yardage carry-type history improve the probability that the
> carry is a 3rd/4th-and-1/2 carry over the team's strictly-prior RB carry-type
> baseline?

This is **opportunity type conditional on total player carries**.

It does not alter:
- team rush volume;
- player total carry entitlement;
- conservation;
- teammate carry allocation.

## Frozen short-yardage event

`down in {3,4} AND 0 < ydstogo <= 2`.

No alternate down/distance or field-position definition may be tried after
scoring.

## Chronology

- 2022-2023: history seed;
- 2024: development;
- 2025: independent confirmation;
- no 2026 outcomes.

Every prior quantity excludes the entire target game.

Player history is same stable player + same team only.

## Frozen baseline

For target team `T`, before target week:

`p_team = (prior_team_RB_short + 0.5) / (prior_team_RB_carries + 1)`

using all named-RB carries for the same team from completed earlier games.

This is a strictly-prior Jeffreys-smoothed team RB carry-type baseline.

## Frozen named-player candidate

Minimum history:
**>=10 prior same-team carries**.

Fixed shrinkage:
**20 team-baseline pseudo-carries**.

`p_player = (prior_player_short + 20 * p_team) / (prior_player_carries + 20)`

The value 20 is fixed before scoring. No search or fit chooses it.

This is not a recent-share replacement; it is a conditional carry-type
composition state.

## Frozen outcomes

### Primary — carry-level type probability

Every target named-RB carry is binary:
- 1 = frozen short-yardage carry;
- 0 = ordinary carry.

Primary metric:
**Brier improvement = team-baseline Brier - player-candidate Brier**.

Protection:
log-loss improvement.

### Secondary — conditional player-game short-yardage exposure

For a target player-game with realized total carries `H`:

`p_any_short = 1 - (1 - p)^H`.

Outcome:
1 if the named RB received >=1 frozen short-yardage carry in that game.

This is conditional mechanism grading only; target-game `H` is not a
pregame predictor.

Secondary:
- player-game any-short Brier improvement;
- player-game any-short log-loss improvement.

## Frozen inference

For 2025 independent confirmation use 5,000 deterministic bootstrap replicates
(seed 20261017).

Report:
- carry-level player-cluster CI;
- carry-level target-game-cluster CI;
- carry-level equal-player CI;
- conditional-game player-cluster CI;
- conditional-game target-game-cluster CI;
- equal-player mean carry-level gain;
- median supported-player gain;
- share of supported players improved.

## Frozen qualification gates

The mechanism qualifies only if **all** are true:

2024 development:
1. carry-level Brier improvement > 0;
2. conditional-game Brier improvement > 0.

2025 independent confirmation:
3. carry-level Brier improvement > 0;
4. carry-level log-loss improvement >= 0;
5. conditional-game Brier improvement > 0;
6. conditional-game log-loss improvement >= 0;
7. carry-level player-cluster CI lower bound > 0;
8. carry-level game-cluster CI lower bound > 0;
9. carry-level equal-player CI lower bound > 0;
10. conditional-game player-cluster CI lower bound > 0;
11. conditional-game game-cluster CI lower bound > 0.

No pooled-season rescue.

## Anti-rescue

If this mechanism fails, do not rescue with:
- another pseudo-carry strength;
- fewer/more minimum player carries;
- recent windows;
- all-team player history;
- second-down inclusion;
- 1-yard-only or 3-yard cutoffs;
- field-position/goal-line subsets;
- lead-back/starter subsets;
- score/clock/game-script subsets;
- opponent interactions;
- yardage/TD outcomes;
- sportsbook/OddsAPI;
- 2026 outcomes.

A failure closes this exact direct persistent short-yardage carry-type identity
mechanism.

## Translation boundary

Qualification would establish only:

`RB_SHORT_YARDAGE_CARRY_TYPE_IDENTITY_SUPPORTED`.

Any use in:
- rushing-yards translation;
- TD probability;
- reliability/trust;
- production

requires a separately frozen downstream mechanism.

## Integrity at freeze

- predictive rows scored: 0
- 2026 outcomes read: 0
- sportsbook inputs: 0
- OddsAPI calls: 0
- production mutation: false
