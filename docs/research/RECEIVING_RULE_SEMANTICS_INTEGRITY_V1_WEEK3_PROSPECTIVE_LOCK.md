# Receiving Rule Semantics Integrity V1 — Week-3 Prospective Accuracy Lock

Date locked: 2026-09-26
Status: **PREGAME LOCK — OUTCOMES FORBIDDEN UNTIL GAMES FINAL**

## Authority

Frozen structural run:
- run `36276736046`
- head `52238e3aa8e129db4a632c6e8fe1f80c6a109a52`
- artifact `10917254062`
- digest `sha256:c7ff51d17adfaf0f96e18808cec5948fa314f126f5c8b0392c8bed5568fa66fa`

Baseline parity:
- A0 entitlement vs accepted authority max absolute gap: ~`9.99e-17`

No Week-3 outcome was read when the four cells were materialized.

## Frozen cells

- A0B0 = current production semantics
- A1B0 = middle-open unit canonicalization only
- A0B1 = slot-alignment carry only
- A1B1 = combined repair

No additional variant may be introduced after outcomes.

## Frozen structural cohort

Week-3 Stage 1 recorded:
- PlayerForm WR rows: 161
- preserved SWR rows: 56
- A0 current SLOT labels: 0
- A1B0 target-share changed rows: 59 TE
- A0B1 target-share changed rows: 56 WR
- A1B1 target-share changed rows: 77 total
- non-target rule max delta: 0.0

## Postgame scoring contract

Only after all target games are final:

1. join actual target-week outcomes by canonical player/team identity;
2. compute actual team target share where team target totals are available;
3. grade A0B0/A1B0/A0B1/A1B1 without recomputing or modifying any pregame cell;
4. report:
   - target-share absolute error;
   - receptions MAE;
   - receiving-yards MAE;
   - p90 absolute error;
   - pooled WR/TE;
   - WR only;
   - TE only;
   - frozen SWR/slot subgroup for B1 cells;
   - changed-row-only diagnostics;
   - team/game concentration to ensure result is not one-game driven.

## Directional qualification gates

A production-facing semantic repair must still satisfy the parent plan:
- no material pooled WR/TE MAE regression;
- targeted subgroup improves;
- targeted subgroup p90 non-worse;
- no protected-market contamination;
- result not dependent on one game/team.

If Week 3 is insufficient, disposition remains prospective and the same unchanged semantic cells continue into later weeks.

## No-go

Do not:
- change the 0.50 middle-open threshold;
- change target multipliers;
- redefine SLOT;
- rescue by team/player after outcomes;
- alter M38 / WR-R15 / TE-R5P;
- combine with RB Vacancy V1;
- use sportsbook lines to determine football truth.
