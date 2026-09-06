# RB Role-Order Remap V1 — Result

## Disposition

`ROLE_ORDER_REMAP_V1_NOT_ACTIONABLE`

No production change.

## Canonical evidence

- Workflow run: `34063904515`
- Job: `101569200535`
- Tested SHA: `a13020f4fe098ef5f51df886b6ae4dd78f751b7e`
- Artifact: `9998334300` (`rb-role-order-remap-v1`)
- Artifact digest: `sha256:55ce439daef26637ec38b68144d4eb5b0dc48d4ec5d7114ef6fa01cb46c04594`

Mechanical lineage before the canonical run is preserved. The final repair only canonicalized artifact team aliases (`JAC -> JAX`, `LA -> LAR`) and relaxed copied-value parity to the actual non-null overlap while retaining STACK1 as the frozen opportunity/yards source. It recovered exact 1393-row metadata identity and reproduced inherited depth coverage `0.949748743718593`. No scientific candidate, threshold, gate, or outcome changed.

## Frozen candidate

The candidate did not fit a model and did not create or remove team opportunity. Within each team-week it sorted the already-existing STACK1 carry means onto timestamp-safe current RB/HB depth order, preserving exact team-week player carry mass. Yards followed the remapped carries at each player's existing implied YPC.

This was deliberately a hard test of the hypothesis that current depth ordering should directly control carry ordering.

## Parent parity

Canonical STACK1 parity reproduced exactly:

- rush attempts: `n=1393`, MAE `3.482575936421331`, RMSE `4.741739937918672`, bias `-0.8627247216838725`, corr `0.7353960735701256`
- rush yards: `n=1393`, MAE `20.4241632527228`, RMSE `30.06980647091156`, bias `-5.537127660834512`, corr `0.6168472895165803`
- Week 1 rush-yards MAE: `20.09082962816558`

Integrity:

- depth coverage: `0.949748743718593`
- timestamp violations: `0`
- max absolute team-week carry-mass delta: `3.55e-15`
- sportsbook inputs: `false`
- model fitting: `false`
- production changed: `false`

## Results

| Slice | Market | Baseline MAE | Candidate MAE | Candidate - baseline |
|---|---|---:|---:|---:|
| ALL_RB | rush_att | 3.482576 | 4.106428 | +0.623852 |
| ALL_RB | rush_yards | 20.424163 | 22.836638 | +2.412474 |
| WEEK1 | rush_att | 3.643282 | 3.494627 | -0.148655 |
| WEEK1 | rush_yards | 20.090830 | 20.651841 | +0.561012 |
| W2_18 | rush_att | 3.472132 | 4.146186 | +0.674053 |
| W13_18 | rush_att | 3.573496 | 4.456570 | +0.883074 |
| RB1 | rush_att | 3.915618 | 4.415211 | +0.499592 |
| RB2 | rush_att | 3.485085 | 4.005469 | +0.520384 |
| RB3 | rush_att | 2.621897 | 2.962211 | +0.340315 |
| BASELINE_ROLE_MISALIGNED | rush_att | 3.658285 | 4.344670 | +0.686384 |

Only the frozen Week-1 carry slice improved. Every RB1/RB2/RB3 carry slice worsened; the role-misaligned subset itself worsened; overall carry MAE worsened 17.9% and overall rush-yards MAE worsened 11.8%.

Frozen gates passed only source/integrity, exact team mass, Week-1 carry improvement, and absolute-bias noninferiority. The candidate failed the overall, W2-18, W13-18, role-slice, yards, and Week-1 yards gates.

## Scientific conclusion

The prior production audit remains valid: current depth state is preserved in the data but is not a direct rushing-allocation input. However, **forcing current depth order to own the existing carry magnitudes is not a valid correction**. Current depth rank contains useful role information, but it cannot simply replace historical/usage-derived allocation.

Do not:

- promote this remap;
- rescue it by lowering gates;
- promote a Week-1-only exception after observing that slice;
- search blend weights between depth order and STACK1 after seeing these results;
- infer that depth charts are useless.

The next legitimate RB work should diagnose *where* the current model is repeatedly wrong at the individual-player level and whether those errors are systematically associated with pregame role transition, limited prior same-team history, rookie/new-team state, injury-created opportunity, or depth/model-role mismatch. That diagnostic should be frozen before any new correction is proposed.
