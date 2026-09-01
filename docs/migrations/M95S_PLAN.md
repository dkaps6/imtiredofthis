# M95S — Population-Mass vs Player-Ranking Decomposition Audit

## Purpose

M95S is a diagnostic/postmortem migration, not a tuned candidate. It tests whether M95R failed because a slow cross-season population-mass correction became stale while player-level ranking information remained conditionally useful.

## Frozen questions

1. How quickly does M95F stable-workhorse 20+ population calibration move by season/week?
2. Could strictly pregame within-season workload anchors (1-, 2-, 4-week and season-to-date) have detected the 2025 shift before M95R injected large positive tail mass?
3. After removing population-mass differences, does frozen M95K/feed ordering still add player-level discrimination in the seasons where authoritative traces exist?
4. Is ranking value conditional rather than universal across seasons?

## Frozen design

- Diagnostic only; no fitted M95S candidate.
- Exact stable-workhorse panel: M95R/M95Q/M95P authoritative traces, seasons 2020-2025.
- Frozen ranking traces: M95K authoritative 2025 and M95L authoritative opened 2023 confirmation.
- M94C central carries unchanged.
- M95F remains the baseline.
- Primary target: stable-workhorse 20+ carries.
- 25+ is diagnostic only.
- Pregame workload-anchor grid is predeclared: prior 1 week, prior 2 weeks, prior 4 weeks, and season-to-date; all shifted so target-week outcomes are unavailable.
- League anchors and team anchors are audited separately.
- Player ranking is evaluated both in native frozen probabilities and after a common logit intercept shift that matches the M95F mean probability mass within the evaluated scope. The intercept shift does not change ordering; it isolates discrimination from mass calibration.
- No feature selection.
- No coefficient search.
- No hyperparameter search.
- No sportsbook input.
- No production change.
- No retuning M95K or M95R against exposed outcomes.

## Predeclared diagnostic outputs

- weekly/season M95F population calibration gaps;
- pregame 1/2/4/STD anchor correlations with actual 20+ and M95F residual error;
- 2025 week-by-week M95F vs M95R mass alongside pregame league anchors;
- M95K/M95L base vs frozen-ranking vs mass-normalized-ranking AUC/Brier/logloss for 2023 and 2025;
- 25+ event-count audit;
- method/disposition audit.

## Interpretation rules

- A ranking signal is not considered universal if its AUC direction flips between 2023 and 2025 after mass normalization.
- Evidence that 2025's M95R failure was detectable pregame requires the contemporaneous workload anchors to be materially lower than the positive R mass shift during periods where M95F was already overpredicting the realized stable-workhorse tail rate.
- This migration cannot promote a model. It can only narrow the architecture for M95T or recommend stopping with M94C/M95F.
