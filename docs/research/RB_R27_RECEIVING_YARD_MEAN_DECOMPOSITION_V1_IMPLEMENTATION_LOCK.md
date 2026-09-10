# RB R27 Receiving-Yard Mean Decomposition V1 — Implementation Lock

Status: **IMPLEMENTATION LOCKED BEFORE FIRST SCIENTIFIC EXECUTION**  
Date: 2026-09-09

## Frozen scientific plan

- plan commit: `5333d7e1cc33dcb567d03c924c774afb6877e932`
- plan blob SHA: `4ad4801dbe600238dd7f1090df0a6596ff8a6a46`
- production-code parent: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`
- continuity/handoff parent: `d63ba0216d43e4763954e0be7351ece06f8fa6b4`

## Exact staged historical parent code

R27 stages these historical research files byte-identically from their prior authoritative branches rather than rewriting the R26 mechanism:

- `scripts/backtest/evaluate_rb_r25_receptions_specialist_v1.py`
  - blob SHA `93da760073059d68c6481b212a906c68734089e3`
- `scripts/backtest/evaluate_rb_r26_vacancy_gated_r9_v1.py`
  - blob SHA `17170baa072bcc3157dee03c7a0de75297060e71`
- `scripts/backtest/audit_rb_r26_safe_transition_state_v1.py`
  - blob SHA `7459dc8a48a0c5e58e14b358ec4a9b0cd19eb959`

Staging commit: `62e1b285f930e4cc2c437d85710c4579c0941faa`.

## R27 implementation

Implementation commit before this lock: `2ceb6b372bf04156cf61ffc23234b12c87b6352a`.

- `scripts/backtest/evaluate_rb_r27_receiving_yard_mean_decomposition_v1.py`
  - blob SHA `a284592a0b2f948b8f12f973ca9eb7d9c08d6b23`
- `scripts/backtest/finalize_rb_r27_receiving_yard_mean_decomposition_v1.py`
  - blob SHA `4f8bf99b25210e8c88a6c8c4511ff9d565d47652`

## Locked behavior

The implementation is locked to the frozen V1 question:

1. Execute the exact historical R26 vacancy-gated R9 mechanism for each 2020-2025 fold using the immediately prior season for fitting.
2. Consume exact R26 `baseline_targets` and `candidate_targets`.
3. Rebuild the same leakage-safe target-game production context.
4. Hold production receiving efficiency fixed with the exact existing `rules_ypt → bayes_ypt` fallback.
5. Define baseline receiving yards as `baseline_targets × production_ypt`.
6. Define the sole R27 V1 candidate as `R26_candidate_targets × the exact same production_ypt`.
7. Audit the equivalent receptions path using the exact production catch-rate fallback and implied production YPR.
8. Use actual receiving yards only as post-prediction labels/diagnostics.
9. Aggregate all six seasons and apply the exact 27 gates frozen in the plan.
10. Make no production or R22 changes.

## No-retuning lock

After the first scientific execution, the following may not change under V1:

- R26 vacancy definition or formula;
- R8/R9 features, Ridge alpha, clips or reliability construction;
- historical seasons/folds;
- production YPT/catch-rate fallback hierarchy;
- candidate formula;
- cohort definitions;
- thresholds or gate logic;
- R22 boundary;
- sportsbook separation.

Only a documented value-neutral mechanical repair may be rerun under V1. A scientific failure must remain a failure and inform a separately frozen R27B study.
