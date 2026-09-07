# Cross-Position Catastrophic Game-Spot Overlay V1 — Frozen Plan

## Purpose

Phase A showed where the largest QB/WR/TE/RB errors come from. Phase B separated those errors into postgame mechanisms. Phase C asks a narrower pregame question:

> Were the catastrophic underprojections already in favorable football game spots, and were catastrophic overprojections already in adverse football game spots?

This is diagnostic only. It does not change production, coefficients, projections, or distributions.

## Frozen evidence

- Phase A exact frozen model rows from QB M89/M90, WR M38/C2-B0, TE-R5, RB P3/STACK3.
- Phase B exact catastrophic PBP casebook.
- nflverse regular-season PBP, 2019-2025, used only to construct leakage-safe strict-prior team/opponent environment histories.
- Sportsbook features are prohibited.

## Leakage contract

For a season/week/team game, every environment feature must be computed only from that team's or opponent defense's prior games. Rolling histories are `shift(1)` then trailing 5 games and may cross the prior-season boundary. No same-game or future observations may enter the spot score.

## Frozen spot architecture

Two independent axes are computed before an equal-weight reporting composite.

### Pass opportunity spot — QB/WR/TE
Equal-weight mean of weekly standardized strict-prior:
- offense pass attempts/game;
- offense pass rate;
- offense plays/game;
- opponent pass attempts faced/game.

Higher = more favorable passing opportunity.

### Pass efficiency spot — QB/WR/TE
Equal-weight mean of weekly standardized strict-prior opponent defense:
- pass YPA allowed;
- pass EPA allowed;
- pass success rate allowed;
- 20+ yard pass rate allowed;
- YAC per completion allowed.

Higher = more favorable passing efficiency/explosive environment.

### Rush opportunity spot — RB
Equal-weight mean of weekly standardized strict-prior:
- offense non-QB-scramble rush attempts/game;
- offense rush rate;
- offense plays/game;
- opponent non-QB-scramble rush attempts faced/game.

Higher = more favorable rushing opportunity.

### Rush efficiency spot — RB
Equal-weight mean of weekly standardized strict-prior opponent defense:
- RB-style rush YPC allowed;
- rush EPA allowed;
- rush success rate allowed;
- 10+ yard rush rate allowed;
- 20+ yard rush rate allowed.

Higher = more favorable rushing efficiency/explosive environment.

### Composite game spot
`0.50 * opportunity_spot + 0.50 * efficiency_spot`.

The composite is reporting/stratification only; opportunity and efficiency remain separately reported because Phase A/B showed different causal layers.

Within each season/week, team-games are ranked on the position-appropriate composite:
- top quartile = `FAVORABLE`;
- middle 50% = `NEUTRAL`;
- bottom quartile = `ADVERSE`.

No thresholds or weights may be changed after seeing results.

## Primary diagnostics

For each position and direction:
1. catastrophic rate by FAVORABLE / NEUTRAL / ADVERSE spot;
2. catastrophic absolute-error-mass share by spot;
3. mean opportunity and efficiency spot scores;
4. same analysis restricted to opportunity Q4 players;
5. Phase-B mechanism-by-spot cross-tab.

Directional alignment classification for catastrophic rows:
- `ALIGNED`: underprojection in FAVORABLE spot OR overprojection in ADVERSE spot;
- `OPPOSITE`: underprojection in ADVERSE spot OR overprojection in FAVORABLE spot;
- `NEUTRAL`: neutral spot.

The important test is magnitude, not a binary promotion gate. A real signal can be useful without explaining every tail.

## Interpretation rules

- If underprojection catastrophes concentrate in FAVORABLE spots, the current model/distribution is underreacting to favorable environment for that mechanism.
- If overprojection catastrophes concentrate in ADVERSE spots, it is underreacting to negative environment.
- If large tails are common in opposite spots, generic matchup adjustment is not the answer; role/entitlement, game-script, efficiency mixture, or low-predictability explosive events dominate.
- Phase-B low-predictability single-play/YAC/early-exit cases are never converted into deterministic mean corrections.
- A game-spot signal may be used later only in the causal layer it supports (opportunity vs efficiency) and only after a separate frozen prospective candidate.

## Integrity requirements

- sportsbook inputs used = 0;
- same/future PBP observations used in spot construction = 0;
- all 1,911 Phase-B catastrophic rows retained;
- Phase-A all-row counts unchanged;
- >= 98% game-spot coverage overall and by position, otherwise mechanical/integrity failure;
- opponent mapping must be one-to-one by season/week/team.

## Output

- all-row game-spot casebook;
- catastrophic game-spot + Phase-B casebook;
- spot bucket scorecards;
- Q4 spot scorecards;
- Phase-B mechanism x spot tables;
- directional alignment table;
- JSON disposition.

Final Phase-C disposition is diagnostic only: `PHASE_C_GAME_SPOT_OVERLAY_COMPLETE` or mechanical/integrity failure.