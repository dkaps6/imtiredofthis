# M95S — Population-Mass vs Player-Ranking Decomposition Audit Results

## Authoritative execution

- Workflow: `M95S RB Mass Ranking Decomposition`
- Run: `33454116869`
- Job: `99690286310`
- Tested SHA: `78f08a5fdb8e5ff34143e1a6dc72d6d901daa2f2`
- Artifact: `9780815195`
- Artifact SHA256: `aa1c29ea8f665699278a484d5d0e41fc768ab337c0e06b9eabd58e7500f4eb88`
- Execution: SUCCESS
- Disposition: `M95S_DECOMPOSITION_SUPPORTED_ADVANCE_TO_CONSTRAINED_M95T`
- Model fit: 0
- Feature search: 0
- Coefficient search: 0
- Sportsbook inputs: 0
- Production change: 0

The earlier run `33454080389` executed the same frozen diagnostic successfully. Run `33454116869` is authoritative because it is the later workflow-file-triggered execution; the workflow change between the two was comment-only and did not alter diagnostic logic.

## Headline finding

M95S supports separating stable-workhorse tail modeling into two distinct jobs:

1. **Population mass / calibration:** how much 20+ carry probability should exist in the current environment. This needs contemporaneous within-season context and cannot safely be driven by a slow cross-season residual correction.
2. **Player ordering / allocation:** which stable workhorses should receive more or less of that mass. Historical feed/ceiling information can add ordering value in some regimes, but it is not universal across seasons.

This is the strongest narrowing result since M95L because it explains why M95K could be useful in 2025 yet fail 2023, and why M95R could repair 2023 calibration yet badly damage 2025.

## 2025 made M95R's error visible pregame

Across Weeks 2-9 in 2025:

- M95F's mean weekly overprediction of the realized stable-workhorse 20+ rate: **15.3812 percentage points**.
- M95R nevertheless added another **13.4238 percentage points** of probability mass on average.
- The pregame league prior-four-week lead-RB 20+ rate averaged only **13.2826%**.

Thus the current environment was already giving evidence inconsistent with a large positive cross-season tail-mass correction. M95R's slow residual layer was stale.

By late 2025 (Weeks 13-18), M95F's mean absolute weekly calibration gap was **9.5590 percentage points**. Weekly samples remain noisy, but the slow M95R correction was clearly not the right mechanism.

## Player-ranking decomposition

The frozen M95K architecture already preserved stable-workhorse 20+ probability mass in the authoritative traces, so mass normalization did not alter its ordering or metrics. This makes the cross-season ranking contrast especially clean.

### 2023 W13-18 — frozen M95K via M95L

- n = 73, events = 24
- M95F AUC: `.727041`
- frozen M95K/M95L AUC: `.545068`
- AUC gain: **`-0.181973`**
- M95F Brier: `.233221`
- frozen ranking Brier: `.244446`

### 2025 full — M95K authoritative

- n = 237, events = 52
- M95F AUC: `.581185`
- M95K AUC: `.641164`
- AUC gain: **`+0.059979`**
- M95F Brier: `.186593`
- M95K Brier: `.171528`

The ranking direction flips across seasons. Therefore feed/ceiling ranking is **conditionally useful, not universal**.

## Pregame workload-anchor audit

Across the exact available 2020-2025 panel (`n=564`):

- league season-to-date lead20 vs M95F residual: Spearman `rho=.112369`, `p=.007559`
- league prior-four-week lead20 vs residual: `rho=.098229`, `p=.019634`
- league prior-one-week lead20 vs residual: `rho=.073312`, `p=.081939`
- league prior-two-week lead20 vs residual: `rho=.036358`, not significant

The effect is modest, not strong enough to justify a direct formula by itself, but it supports contemporaneous population-state information.

Team recent workload measures were much stronger predictors of the raw 20+ outcome than of M95F residual error. Across the exact panel, team prior-2/prior-4 lead-carry measures had raw-outcome correlations around `.21`, while their residual correlations were slightly negative. In 2025 full, team recent workload residual correlations were materially negative (roughly `-.15` to `-.25`).

Interpretation: **team/player workload context is already substantially represented by the M95F/player-game baseline. Reusing it as a blanket population-mass booster risks double counting.** Population mass should be anchored primarily by contemporaneous league/population state, while player-level ranking should be handled separately and conditionally.

## What M95S changes

M95S does not promote a model. It narrows the next candidate architecture.

The next candidate must not be another generic additive residual model. It should:

- keep M94C central carries unchanged;
- keep M95F as the base stable-workhorse tail probability;
- estimate/anchor current **population tail mass** using a fast, pregame within-season population signal with strong shrinkage and fail-safe behavior;
- separately use player/feed ranking only through a precommitted conditional gate or bounded reranking mechanism;
- preserve total probability mass after player reranking;
- avoid double-counting team workload variables already captured by M95F;
- keep vacancy/transition separate under M95I semantics;
- use 20+ as primary and 25+ as secondary diagnostic;
- require season-by-season non-regression, not pooled-only gains.

## Stopping rule

M95T is the final historical-development candidate in this RB tail sequence.

- If M95T cannot improve the expanded historical panel without material season-specific regression, stop new RB-tail candidate development and retain M94C/M95F as the research-safe architecture for 2026 while moving to WR research.
- If M95T passes, freeze it before 2026 regular-season outcomes and run it in prospective/shadow confirmation. No further retrospective coefficient search is permitted before that confirmation.
