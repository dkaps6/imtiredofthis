# RB R27D — Strict-Prior YACOE Residual V1 Implementation Lock

Status: `IMPLEMENTATION LOCKED BEFORE FIRST SCIENTIFIC RESULT`

This lock authorizes exactly one first scientific execution of the already-frozen R27D plan. It does not authorize production integration.

## Frozen authority

- Protected production-code authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`
- Frozen plan commit: `69edc12a16c691e3838eadcd75559b85dbba7865`
- Frozen plan blob: `d0c2b0ff2de154e52fa21fb9ce19b739039633f3`
- Immutable R27B V2 parent run: `34428917229`
- Immutable R27B V2 parent artifact: `10134023092`
- Immutable parent digest: `sha256:1df1a14c9900b890fe86e04269849dc90d76fc03a9215ad8612c1820dee5341f`

## Exact implementation blobs

- strict-prior xYAC/YACOE dataset builder: `scripts/backtest/build_rb_r27d_yacoe_dataset_v1.py`
  - blob `1a1f379a66ba016afedede24a3880348d89a0351`
- immutable-parent normalization adapter: `scripts/backtest/prepare_rb_r27d_parent_v1.py`
  - blob `ed021ce01201920072e70d66a4b07af00c8e0bc9`
- frozen evaluator / 31-gate scorecard: `scripts/backtest/run_rb_r27d_yacoe_residual_v1.py`
  - blob `f34ef4069910d17cae9faf313f58a170c6b4b7c4`
- workflow: `.github/workflows/research-rb-r27d-yacoe-residual-v1.yml`
  - blob `d6837488b260062aaf8cf7f7fe6ed628d33514c9`

## Value-neutral parent adapter boundary

The immutable V2 final CSV already contains exact `b0_rec_yards` and `b1_rec_yards` plus identical `r27_baseline_rec_yards` and `r27_candidate_rec_yards`. The adapter may only:
1. assert those pairs reproduce within `1e-10`; and
2. materialize `production_implied_ypr = production_ypt / production_catch_rate` exactly as specified in the frozen plan.

It may not change a prediction, cohort, opportunity value, result target, feature, gate, threshold, shrinkage constant, model alpha, correction cap, or scope.

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
