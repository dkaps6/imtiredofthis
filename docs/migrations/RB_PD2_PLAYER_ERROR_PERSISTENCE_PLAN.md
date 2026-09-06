# RB-PD2 — Walk-Forward Individual RB Error Persistence — Frozen Plan

## Purpose

The retrospective 2025 individual scorecard showed large player-to-player differences in carry and rushing-yard error. It also showed that direct depth-order mismatch is not the broad error concentration and that Role-Order Remap V1 must remain rejected.

RB-PD2 tests whether the individual error history is actually usable **before** the next game:

**Does an RB's own prior carry/yards error predict the next carry/yards miss?**

No sportsbook data, no model fitting, and no production change.

## Canonical evidence

- Parent/result commit: `634bb3abd6fee55269a4845c3d6b9d8491ea48e9`
- STACK1 production-equivalent source run: `33535308110`
- STACK2 timestamp-safe metadata source run: `33538770934` (rookie flag only; no candidate projection value comes from STACK2)
- exact 2025 canonical player rows: `1,393`.

## Frozen history construction

For every 2025 player-game in chronological order:

- same player only;
- strictly completed prior canonical games;
- last **8** eligible prior games;
- minimum **4** prior games for scoring;
- no alternate windows after results.

Full-season player MAE/bias from the previous diagnostic is not used as a feature.

## Prior-only quantities

Carries:
- `PRIOR8_CARRY_BIAS` = mean(projected carries - actual carries)
- `PRIOR8_CARRY_MAE` = mean absolute carry error

Rushing yards:
- `PRIOR8_YARD_BIAS` = mean(projected rushing yards - actual)
- `PRIOR8_YARD_MAE` = mean absolute rushing-yard error

Target outcomes are next-game signed / absolute errors only.

## Frozen diagnostics

### A. Carry directional persistence
Pass only if all:
- scoreable rows >=700
- Spearman(prior carry bias, target carry error) >=+.08
- high-low quartile target carry-error gap >=+1.0 carry
- sign agreement >=55% where |prior carry bias|>=0.5
- quartile gap >0 in Weeks 5-12 and Weeks 13-18

### B. Carry difficulty persistence
Pass only if all:
- scoreable rows >=700
- Spearman(prior carry MAE, target absolute carry error) >=+.08
- high-low absolute-error gap >=+0.75 carry
- gap >0 in Weeks 5-12 and Weeks 13-18

### C. Yard directional persistence
Pass only if all:
- scoreable rows >=700
- Spearman(prior yard bias, target yard error) >=+.08
- high-low target yard-error gap >=+6 yards
- sign agreement >=55% where |prior yard bias|>=3 yards
- gap >0 in Weeks 5-12 and Weeks 13-18

### D. Yard difficulty persistence
Pass only if all:
- scoreable rows >=700
- Spearman(prior yard MAE, target absolute yard error) >=+.08
- high-low target absolute-error gap >=+5 yards
- gap >0 in Weeks 5-12 and Weeks 13-18

Rookie versus non-rookie results will be reported descriptively using the already-existing pregame rookie flag, but rookie status is not an excuse to lower gates or create a post-hoc boost.

## Dispositions

- none pass: `NO_ACTIONABLE_RB_PLAYER_ERROR_PERSISTENCE`
- >=1 pass: `RB_PLAYER_ERROR_PERSISTENCE_DETECTED`
- integrity failure: `RB_PLAYER_ERROR_PERSISTENCE_INTEGRITY_FAILURE`

A pass authorizes only a separately frozen calibration/full-stack test. No direct player correction is promoted here.
