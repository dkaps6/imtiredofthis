# WR-R3 — Walk-Forward Individual WR Error Persistence — Frozen Plan

## Purpose

Test whether M38's individual-player errors are persistent enough to be useful pregame.

WR-R1 confirmed M38 across 2020-2025. WR-R2 showed no actionable NGS tracking signal but produced a 486-player retrospective error scorecard. WR-R3 converts the user's individual-MAE idea into a leakage-safe test:

**Does a WR's own prior M38 error history predict the next M38 miss?**

No model is fit, no sportsbook data is used, and M38 is not changed in this branch.

## Canonical evidence

- Parent/result commit: `1ddb482f860152264b7500ea1a0a22e7d126b7e8`
- Exact WR-R1 source run: `34058453941`
- Exact WR-R1 artifact: `9997412312` (`wr-r1-multiseason-2020-2025`)
- Canonical M38 receiving-yard rows: `12,396`, target seasons 2020-2025.

## Frozen player-history construction

For every WR target game:

- use only completed canonical M38 rows for the same player strictly before the target game;
- retain the last **8** eligible prior games, crossing seasons when available;
- require at least **4** prior games for scientific scoring;
- no alternate windows after results.

## Prior-only quantities

1. `PRIOR8_M38_BIAS` = mean(M38 projection - actual receiving yards)
2. `PRIOR8_M38_MAE` = mean absolute M38 error
3. `PRIOR8_M38_MISS30_RATE` = fraction of prior errors with |error| >= 30 yards

Target outcomes used only for evaluation:

- target signed M38 error
- target absolute M38 error
- target 30+ absolute miss indicator

## Frozen diagnostics

### A. Directional player-bias persistence
Pass only if all:
- scoreable rows >= 7,000
- Spearman(PRIOR8_M38_BIAS, target signed error) >= +0.08
- top-vs-bottom quartile target signed-error gap >= +6 yards
- sign agreement >=55% where |prior bias| >=3 yards
- quartile gap >0 in at least 4 of 6 target seasons
- quartile gap >0 in 2024 and 2025

### B. Individual difficulty persistence
Pass only if all:
- scoreable rows >= 7,000
- Spearman(PRIOR8_M38_MAE, target absolute error) >= +0.08
- top-vs-bottom quartile target absolute-error gap >= +4 yards
- absolute-error gap >0 in at least 4 of 6 seasons
- gap >0 in 2024 and 2025

### C. Extreme-miss persistence
Pass only if all:
- scoreable rows >= 7,000
- prior top-quartile MISS30 rate has target 30+ miss enrichment >=1.25x versus the scoreable population
- enrichment >1.0 in at least 4 of 6 seasons
- enrichment >1.0 in 2024 and 2025

No threshold lowering.

## Dispositions

- no passes: `NO_ACTIONABLE_WR_PLAYER_ERROR_PERSISTENCE`
- >=1 pass: `WR_PLAYER_ERROR_PERSISTENCE_DETECTED`
- source/integrity failure: `WR_PLAYER_ERROR_PERSISTENCE_INTEGRITY_FAILURE`

A pass authorizes only a separately frozen full-stack calibration/integration test. Directional bias would naturally target the mean layer; difficulty/extreme-miss persistence would naturally target MC uncertainty/tails. Full-sample retrospective player MAE is never an upstream feature.
