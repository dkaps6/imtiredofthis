# RB R26K Week-1 Allocation Mechanism Atlas V1 — Implementation Lock

Status: LOCKED BEFORE R26K OUTCOME EXECUTION
Date: 2026-09-09

Frozen plan:
`docs/research/RB_R26K_WEEK1_ALLOCATION_MECHANISM_ATLAS_V1_FROZEN_PLAN.md`

Evaluator:
`scripts/backtest/audit_rb_r26k_week1_allocation_mechanism_atlas_v1.py`

## Immutable parents

R26:
- run `34356222339`
- artifact `10106271075`
- digest `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`

R26J:
- run `34374987828`
- artifact `10113466373`
- digest `sha256:7d886f228487e052ee1617cc5f42b974cd9da6a9d448f3a86b1de82cb4fc4f46`
- required parent disposition `2020_SOURCE_REGIME_DISTINCT_FOR_MECHANISM_FOLLOWUP`

Production comparison base:
`f8417f55b04ce0e19baf260e9d532765034c47f1`

## Locked implementation semantics

- Target seasons: 2020-2025 exactly.
- Target phase: Week 1 exactly.
- Room population: R26 vacancy-active rooms only.
- Primary player population: continuing same-team incumbents with finite actual/baseline/candidate receptions.
- R26 predictions are read only; no prediction is regenerated.
- R26J room states are joined exactly by `(season, week, team)`.
- Parent artifact digests are verified before execution.
- R26J source integrity/disposition is checked before outcome grading.

Primary states are exactly the five definitions in the frozen plan. No data-driven threshold search occurs in the evaluator.

Metrics reuse R26F definitions:
- MAE
- RMSE
- signed bias
- p90 absolute error
- mean/summed change in absolute error.

Room-total versus summed-player diagnosis uses the exact frozen 25% allocation-dominance rule from the plan.

Ordering diagnostic uses deterministic player-key tie breaking and is diagnostic only.

A primary state can authorize child-candidate design only if all eight frozen requirements pass. No partial-pass state is promoted by the evaluator.

## Required fail-closed protections

The workflow must fail if:
- R26 or R26J artifact digest mismatches;
- R26J parent disposition/integrity mismatches;
- protected production runtime/model paths differ from the production base;
- sportsbook/future-outcome feature counters in R26 are nonzero;
- the R26J room join is incomplete;
- required seasons are absent;
- evaluator output claims shadow or production authority.

## Authority ceiling

Even a positive R26K result authorizes only a separately frozen future child-design study.

R26K itself can never authorize:
- prospective shadow;
- production promotion;
- exclusion of 2020;
- R9 refit;
- R22 or receiving-yard-mean changes.
