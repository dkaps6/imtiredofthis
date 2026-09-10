# RB R27D Run2 — Outcome-Null Scoring Mechanical Repair

Status: `INTEGRITY/SCORING FAILURE / NO SCIENTIFIC DECISION`

## Preserved execution

- Run: `34435897671`
- Job: `102740771822`
- Head: `775d9bae7f264ae343209a75cd4947bc88994da9`
- Artifact: `10136169669`
- Artifact digest: `sha256:d00eab566113e0c43502bf75f07fe477295dbef6a7ce20389e23ea941ec6d845`
- Frozen plan blob: `d0c2b0ff2de154e52fa21fb9ce19b739039633f3`
- Builder blob: `2465ae781b55a33ba44098c9b3a3d99f9cf47e01`
- Evaluator blob: `f34ef4069910d17cae9faf313f58a170c6b4b7c4`

## What passed

Run2 repaired the Run1 Week1 column collision and successfully completed:
- exact immutable parent verification;
- protected production/R22/R26 boundary verification;
- strict-prior xYAC/YACOE feature construction;
- all six outer folds;
- model fitting and candidate materialization;
- exact outside-scope and RB2+ parity;
- correction-cap enforcement;
- artifact upload.

## Why the emitted scientific disposition is invalid

The immutable V2 parent includes projection rows without authoritative `actual_rec_yards` outcomes. The evaluator's `metrics()` function converted the full masked cohort directly to NumPy arrays without first restricting to rows where both outcome and prediction are non-null.

Consequently the primary MAE/RMSE/bias/p90 values were `NaN`; the emitted summary contained:
- `rb1_mae_improvement_pct_vs_b1: NaN`
- `rb1_2023_mae_improvement_pct_vs_b1: NaN`
- `week1_pct_change_vs_b1: NaN`
- season RB1 B1/C1 MAEs all `NaN`

A frozen scientific gate cannot be interpreted from a non-finite primary metric. Therefore the workflow's emitted `R27D_STRICT_PRIOR_YACOE_RESIDUAL_MIXED_OR_FAIL_NO_INTEGRATION` string is **not accepted as a scientific result**. Run2 is preserved as an integrity/scoring failure with no scientific decision.

This classification does not depend on whether the candidate would ultimately pass or fail after valid scoring; the failure is mechanically visible from the NaN scorecard itself.

## Minimum authorized repair

Only the evaluator scoring function may change:
- for each reported cohort/prediction, construct a temporary two-column frame `(actual_rec_yards, prediction)`;
- drop rows where either value is non-finite/missing;
- compute n/MAE/RMSE/bias/median AE/p90 AE/30+ miss rate on that common observed subset.

No prediction, model fit, training row, feature, label, fold, alpha, K, correction cap, application scope, baseline, threshold, or scientific gate changes.

The repaired evaluator must be newly pinned in the workflow and implementation lock before another execution. Run2 artifact remains immutable evidence of the scoring bug.
