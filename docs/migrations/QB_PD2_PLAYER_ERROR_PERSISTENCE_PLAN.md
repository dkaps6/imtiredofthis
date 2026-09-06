# QB-PD2 — Walk-Forward Individual Player Error Persistence — Frozen Plan

## Purpose

Test the user's individual-QB accuracy hypothesis without leaking retrospective player scorecards into the model.

The prior Week-1 pathology audit showed large differences in historical QB-specific MAE, bias and synthesis reliability. QB-PD2 asks the only safe next question:

**Before a target game, does that QB's own prior model-error history predict the direction or magnitude of the next model error?**

This is a diagnostic/calibration study, not a new football-feature hunt and not a production change.

## Canonical evidence

- QB pathology result parent: `d1b446f1902a7841cb02d8f9bded549c7b750ae4`
- Historical source run: `33331073376`
- Historical source artifact: `9737913528` (`m89-qb-data-integrity-casebook-synthesis`)
- Exact untouched M89/M90 validation trace: 884 rows across 2024-2025.

No sportsbook information is used.

## Frozen player-history construction

For every target QB-game in chronological order:

- use only that QB's completed rows strictly before the target game;
- retain the last **8** eligible prior QB-games;
- require at least **4** prior games for a row to be scoreable;
- do not use later games, full-season player averages, the Week-1 2026 market, or target-game outcomes in the prior features.

No alternate history windows will be searched after results.

## Prior-only diagnostic quantities

From the prior 8 games:

1. `PRIOR8_SYNTH_BIAS` = mean(synthesis projection - actual passing yards)
2. `PRIOR8_SYNTH_MAE` = mean absolute synthesis error
3. `PRIOR8_BASE_MAE` = mean absolute corrected-base error
4. `PRIOR8_SYNTH_ADVANTAGE` = PRIOR8_BASE_MAE - PRIOR8_SYNTH_MAE

Target-game outcomes used only for evaluation:

- target synthesis signed error
- target synthesis absolute error
- target base absolute error
- target synthesis advantage = target base absolute error - target synthesis absolute error

## Frozen diagnostics / gates

### A. Directional bias persistence
Actionable diagnostic only if all:
- scoreable rows >= 500
- Spearman(PRIOR8_SYNTH_BIAS, target synthesis signed error) >= +0.08
- top-vs-bottom quartile target signed-error gap >= +15 yards
- prior-bias sign agrees with target-error sign in >= 55% of rows where |prior bias| >= 5 yards
- signed-error quartile gap > 0 in both 2024 and 2025 when evaluable

### B. Individual difficulty persistence
Actionable diagnostic only if all:
- scoreable rows >= 500
- Spearman(PRIOR8_SYNTH_MAE, target synthesis absolute error) >= +0.08
- top-vs-bottom quartile target absolute-error gap >= +8 yards
- absolute-error gap > 0 in both 2024 and 2025 when evaluable

### C. Synthesis-reliability persistence
Actionable diagnostic only if all:
- scoreable rows >= 500
- Spearman(PRIOR8_SYNTH_ADVANTAGE, target synthesis advantage) >= +0.08
- top-vs-bottom quartile target synthesis-advantage gap >= +5 yards
- target synthesis-advantage gap > 0 in both 2024 and 2025 when evaluable

No threshold lowering.

## Frozen dispositions

- none pass: `NO_ACTIONABLE_QB_PLAYER_ERROR_PERSISTENCE`
- one/more pass: `QB_PLAYER_ERROR_PERSISTENCE_DETECTED`
- source/integrity failure: `QB_PLAYER_ERROR_PERSISTENCE_INTEGRITY_FAILURE`

A persistence pass does **not** authorize a player-specific correction by itself. It only authorizes a separately frozen calibration candidate, with shrinkage/minimum-sample protection and a full walk-forward production-stack comparison.
