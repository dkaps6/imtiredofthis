# RB R27D — Strict-Prior YACOE Residual V1 Implementation Lock

Status: `RELOCKED AFTER RUN1/RUN2 MECHANICAL-INTEGRITY FAILURES / BEFORE FIRST VALID SCIENTIFIC RESULT`

This lock authorizes the first valid scientific execution of the already-frozen R27D plan after two preserved value-neutral implementation repairs. It does not authorize production integration.

## Frozen authority

- Protected production-code authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`
- Frozen plan commit: `69edc12a16c691e3838eadcd75559b85dbba7865`
- Frozen plan blob: `d0c2b0ff2de154e52fa21fb9ce19b739039633f3`
- Immutable R27B V2 parent run: `34428917229`
- Immutable R27B V2 parent artifact: `10134023092`
- Immutable parent digest: `sha256:1df1a14c9900b890fe86e04269849dc90d76fc03a9215ad8612c1820dee5341f`

## Preserved Run1 mechanical failure

- Run: `34435661834`
- Job: `102740069405`
- Head: `1757d7549c0fc8c44ad207f55971fdd8c3eab754`
- Failure: parent/features merge produced identical `week1_x` / `week1_y`, while evaluator required canonical `week1`.
- Model fit reached: false; candidate scored: false; scientific result: none.
- Repair record: `docs/research/RB_R27D_RUN1_WEEK1_COLUMN_COLLISION_MECHANICAL_REPAIR.md`

## Preserved Run2 integrity/scoring failure

- Run: `34435897671`
- Job: `102740771822`
- Head: `775d9bae7f264ae343209a75cd4947bc88994da9`
- Artifact: `10136169669`
- Artifact digest: `sha256:d00eab566113e0c43502bf75f07fe477295dbef6a7ce20389e23ea941ec6d845`
- Failure: cohort metric function included parent rows with null `actual_rec_yards`, propagating NaN through primary scientific metrics and season scorecards.
- Emitted mixed/fail label is invalid because the scorecard itself had non-finite primary metrics; scientific decision: none.
- Repair record: `docs/research/RB_R27D_RUN2_OUTCOME_NULL_SCORING_MECHANICAL_REPAIR.md`

## Exact implementation blobs for next execution

- strict-prior xYAC/YACOE dataset builder: `scripts/backtest/build_rb_r27d_yacoe_dataset_v1.py`
  - blob `2465ae781b55a33ba44098c9b3a3d99f9cf47e01`
- immutable-parent normalization adapter: `scripts/backtest/prepare_rb_r27d_parent_v1.py`
  - blob `ed021ce01201920072e70d66a4b07af00c8e0bc9`
- frozen evaluator / 31-gate scorecard: `scripts/backtest/run_rb_r27d_yacoe_residual_v1.py`
  - repaired blob `2c11de8d716d080628ee85c3a6e3914f4fadfcc5`
  - original Run1/Run2 evaluator blob `f34ef4069910d17cae9faf313f58a170c6b4b7c4` preserved in lineage
- workflow: `.github/workflows/research-rb-r27d-yacoe-residual-v1.yml`
  - blob `1bd498d9a313cd500ee774405ba3002138e244be`

## Value-neutral repair boundaries

The Run1 builder repair only collapses duplicate deterministic Week1 fields after asserting both equal `(week == 1)`.

The Run2 evaluator repair only defines each reported metric on the common subset where `actual_rec_yards` and the compared prediction are finite/non-null, which is required for MAE/RMSE/bias/quantile metrics to be mathematically defined. It additionally fails integrity if the primary scientific scalar metrics are non-finite.

Neither repair changes a prediction, training row, feature definition, label, fold, K, alpha, correction cap, application scope, baseline, candidate arithmetic, metric formula on observed rows, scientific threshold, gate, production file, R26 component, or R22 component.

## Execution boundary

The first valid run must:
- verify the exact parent artifact digest before use;
- build all xYAC/YACOE predictors strict-prior with full-week materialization before state updates;
- train only on seasons strictly before each test season;
- retain all six 2020–2025 test folds;
- apply C1 only to vacancy-active incumbent RB1 rows;
- force every other row, including vacancy RB2+, to exact B1 parity;
- compute frozen outcome metrics only where outcomes are actually observed;
- evaluate all 31 gates exactly as frozen;
- preserve the first valid PASS/MIXED/FAIL result without retuning.

Production, R26, R22 and Full Slate code remain unchanged during this scientific run.
