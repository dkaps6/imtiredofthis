# Migration 86 — QB Full-Stack Error-Floor / Recoverability Audit

## Status

`PREREGISTERED / FORENSIC DIAGNOSTIC ONLY`

M86 does not fit or tune a pregame QB projection. It forensically decomposes the authoritative M82 full-stack errors to estimate which remaining misses look structurally recoverable from pregame information and which are heavily associated with target-game stochastic events.

## Authoritative baseline

M82 OOS ensemble on exact canonical-v3 identities:

- 884 QB games: 444 in 2024, 440 in 2025
- MAE `56.749517`
- RMSE `72.303902`
- correlation `0.149475`
- 100+ yard misses `123`
- hindsight existing-model-library oracle `41.103131` MAE (nondeployable)

M86 must consume the frozen M82 common-cohort trace from authoritative artifact `9734973786`; it may not rebuild or retune ensemble weights.

## Purpose

Answer four questions:

1. Among the 123 100+ yard ensemble misses, how often is canonical component error primarily attempt-volume, YPA-efficiency, or mixed?
2. How strongly are catastrophic misses associated with target-game events that are inherently difficult or impossible to know pre-kickoff?
3. Which catastrophic misses remain comparatively "clean" — no major post-snap chaos marker — and therefore deserve future pregame information research?
4. What conservative practical error-floor range is suggested by removing only explicitly event-associated error as a diagnostic oracle, without claiming those events were predictable?

## Component attribution

Canonical-v3 provides leakage-safe `pred_attempts`, `implied_pred_ypa`, actual attempts, and actual YPA.

For every row define:

- attempt residual = actual attempts - predicted attempts
- YPA residual = actual YPA - predicted YPA
- attempt contribution magnitude = `abs(attempt_residual * predicted_YPA)`
- YPA contribution magnitude = `abs(YPA_residual * predicted_attempts)`

Frozen classification:

- `VOLUME_DOMINANT` if attempt contribution >= 1.25 × YPA contribution
- `EFFICIENCY_DOMINANT` if YPA contribution >= 1.25 × attempt contribution
- otherwise `MIXED`

This is diagnostic attribution, not a causal decomposition.

## Target-game event markers

M86 may read 2024/2025 play-by-play **only after predictions are frozen** and only for forensic attribution. These fields may never be fed into a pregame model.

At team-game offensive grain, record at minimum:

- longest completed pass / pass gain;
- 40+ yard pass count;
- 60+ yard pass count;
- 30+ YAC completion count where available;
- maximum YAC where available;
- interceptions thrown;
- sacks taken;
- QB scramble count;
- overtime indicator;
- fourth-down attempts / conversions;
- offensive turnover count where reconstructable.

Because canonical-v3 uses stable-primary QBs (>=80% official QB attempts), team-game PBP events are allowed as contextual forensic markers but must be labeled team-context rather than exact QB causality when player attribution is ambiguous.

## Frozen event-associated flag

A catastrophic row is `HIGH_EVENT_CHAOS` if at least one of these occurs:

- a 60+ yard completed pass;
- at least two 40+ yard completed passes;
- a 30+ YAC completion;
- overtime;
- 4+ sacks;
- 2+ interceptions;
- 5+ QB scrambles;
- 4+ fourth-down attempts.

Otherwise it is `LOW_EVENT_CHAOS`.

These thresholds are frozen before outcome inspection and are descriptive, not predictive.

## Required outputs

- row-level 884-game forensic trace;
- 123-tail component classification table;
- event-marker prevalence for tail vs non-tail games;
- underprojection vs overprojection tail split;
- high-event-chaos vs low-event-chaos tail MAE and counts;
- model-library oracle choice distribution within each tail type where available;
- conservative diagnostic floor calculations;
- prioritized future-research subset consisting of low-event-chaos catastrophic misses.

## Error-floor diagnostics

M86 may report two explicitly nondeployable bounds:

1. `MODEL_LIBRARY_ORACLE_FLOOR`: frozen M82 hindsight oracle (`41.103131`).
2. `EVENT_EXCLUSION_DIAGNOSTIC`: MAE on games not flagged `HIGH_EVENT_CHAOS`, plus the share of total absolute error carried by high-event-chaos rows.

M86 must **not** claim that excluding high-event-chaos games is a deployable forecast or that all flagged events are unpredictable.

## Anti-leakage / anti-loop boundary

- no postgame event feature may enter a pregame prediction;
- no thresholds may be tuned after seeing which ones best explain misses;
- no new QB projection is fit;
- no sportsbook variables;
- no claim of causality from postgame correlations;
- no reopening M83/M84/M85 same-information mechanisms.

## Decision boundary

M86 should recommend the next research direction based on where low-event-chaos catastrophic errors concentrate:

- volume/opportunity surprise;
- efficiency/explosive surprise;
- mixed/regime-selection surprise;
- or evidence that remaining free-data pregame headroom is narrow relative to stochastic/error-floor effects.

`production_actionable = false`
