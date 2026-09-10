# RB R27D Run1 — Week1 Column Collision Mechanical Repair

Status: `MECHANICAL FAILURE / NO SCIENTIFIC RESULT`

## Preserved failed execution

- Run: `34435661834`
- Job: `102740069405`
- Head: `1757d7549c0fc8c44ad207f55971fdd8c3eab754`
- Frozen plan blob: `d0c2b0ff2de154e52fa21fb9ce19b739039633f3`
- Original builder blob: `1a1f379a66ba016afedede24a3880348d89a0351`
- Evaluator blob: `f34ef4069910d17cae9faf313f58a170c6b4b7c4`

## What passed before failure

- frozen implementation verification
- protected production/R22/R26 boundary
- exact R27B V2 parent artifact/digest verification
- value-neutral parent normalization
- strict-prior xYAC/YACOE dataset construction for 2019–2025
- 13,086/13,086 evaluation rows received legal prior league state
- target-week features used: 0
- future features used: 0
- sportsbook inputs: 0

## Exact failure

The evaluator stopped before model fitting with:

`RuntimeError: missing columns hist=[] parent=['week1']`

The immutable V2 parent already contains a deterministic `week1` column. The strict-prior feature table also contains the frozen `week1` control. Their one-to-one merge therefore created `week1_x` and `week1_y`, while the evaluator correctly required the frozen canonical name `week1`.

No model was fit, no candidate projection was scored, no 31-gate scientific scorecard was produced, and no result artifact was uploaded. Therefore Run1 is not a scientific PASS/MIXED/FAIL.

## Minimum authorized repair

Only `scripts/backtest/build_rb_r27d_yacoe_dataset_v1.py` is changed. After the parent/features merge it must:

1. detect `week1_x` and `week1_y`;
2. assert both equal the deterministic indicator `(week == 1)` on every row;
3. materialize that identical value under canonical name `week1`; and
4. drop the two suffixed duplicate columns.

Repaired builder blob: `2465ae781b55a33ba44098c9b3a3d99f9cf47e01`.

No feature definition, history window, label, training population, fold, K, alpha, cap, application scope, candidate arithmetic, metric, gate, threshold, production file, R26 component, or R22 component changes.

The repaired builder must be newly pinned in the workflow and implementation lock before the first valid scientific execution.
