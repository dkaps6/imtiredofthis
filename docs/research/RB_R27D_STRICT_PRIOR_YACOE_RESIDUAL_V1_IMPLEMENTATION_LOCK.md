# RB R27D — Strict-Prior YACOE Residual V1 Implementation Lock

Status: `RELOCKED AFTER RUN1 MECHANICAL FAILURE / BEFORE FIRST VALID SCIENTIFIC RESULT`

This lock authorizes the first valid scientific execution of the already-frozen R27D plan after one preserved value-neutral plumbing repair. It does not authorize production integration.

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
- Failure: evaluator required canonical `week1`, while parent/features merge produced identical `week1_x` / `week1_y` columns.
- Model fit reached: false.
- Candidate scored: false.
- Scientific result: none.
- Repair record: `docs/research/RB_R27D_RUN1_WEEK1_COLUMN_COLLISION_MECHANICAL_REPAIR.md`

## Exact repaired implementation blobs

- strict-prior xYAC/YACOE dataset builder: `scripts/backtest/build_rb_r27d_yacoe_dataset_v1.py`
  - repaired blob `2465ae781b55a33ba44098c9b3a3d99f9cf47e01`
  - original Run1 blob `1a1f379a66ba016afedede24a3880348d89a0351` preserved in the repair record
- immutable-parent normalization adapter: `scripts/backtest/prepare_rb_r27d_parent_v1.py`
  - blob `ed021ce01201920072e70d66a4b07af00c8e0bc9`
- frozen evaluator / 31-gate scorecard: `scripts/backtest/run_rb_r27d_yacoe_residual_v1.py`
  - blob `f34ef4069910d17cae9faf313f58a170c6b4b7c4`
- repaired workflow: `.github/workflows/research-rb-r27d-yacoe-residual-v1.yml`
  - blob `b8034f947eb740666c19980d3bc2f3cbcff094c7`

## Mechanical repair boundary

The repaired builder only collapses duplicate deterministic Week1 fields after asserting both copies equal `(week == 1)` on every row. It does not change any historical xYAC/YACOE observation, feature value, label, fold, model, K, alpha, cap, application scope, baseline, candidate arithmetic, metric, threshold, or gate.

The immutable-parent adapter remains value-neutral: exact B0/R27 and B1/R27 aliases must reproduce within `1e-10`, and `production_implied_ypr` is exactly `production_ypt / production_catch_rate`.

## Execution boundary

The first valid run must:
- verify the exact parent artifact digest before use;
- build all xYAC/YACOE predictors strict-prior with full-week materialization before state updates;
- train only on seasons strictly before each test season;
- retain all six 2020–2025 test folds;
- apply C1 only to vacancy-active incumbent RB1 rows;
- force every other row, including vacancy RB2+, to exact B1 parity;
- evaluate all 31 gates exactly as frozen;
- preserve the first valid PASS/MIXED/FAIL result without retuning.

Production, R26, R22 and Full Slate code remain unchanged during this scientific run.
