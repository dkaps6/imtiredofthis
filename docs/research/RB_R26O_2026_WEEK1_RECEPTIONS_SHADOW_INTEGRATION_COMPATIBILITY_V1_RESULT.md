# RB R26O 2026 Week-1 Receptions Shadow Integration Compatibility V1 — Result

## Canonical execution

- Branch: `research-rb-r26o-2026-week1-receptions-shadow-integration-compatibility-v1`
- Frozen plan: `96e4a6977458b55ec00cb29fbc9de05c56c89b88`
- Frozen evaluator implementation: `6a171cb32e77c7d6691ceb75da9861a5d6dd7bc4`
- Run-1 selector-staging repair note: `e5ac65c8dcb2b73f59706880e893c9f72f5d115d`
- Run-2 dtype-repair note: `2c5fc54c0c88aff28e581a901331d175adc87adf`
- Dtype compatibility wrapper: `960fbb0bc6a29724b3d0b7901c92ac905bd69f0e`
- Canonical execution head: `550d3d532e9f27c34b90ecd027f3811292674862`
- Workflow run: `34398759284`
- Job: `102625073624`
- Artifact: `10122672501`
- Artifact name: `rb-r26o-2026-week1-receptions-shadow-integration-compatibility-v1`
- Artifact digest: `sha256:6d72de1eb8902fd8d956feb01b4d1a3cb5847aa58417ad4f416f3e1102e96e82`
- Workflow conclusion: `SUCCESS` (the workflow completed and emitted a scientific disposition)

## Scientific disposition as executed

`R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_FAIL_NO_SHADOW`

The disposition must be preserved exactly. R26O did **not** authorize prospective sealing, live shadow activation, or production promotion.

## Gate result

- Frozen gates passed: **37 / 38**
- Sole failed gate: `15_r22_baseline_mean_parity`
- Recorded evidence for Gate 15: `{"mean_parity": null, "max_mean_delta": null}`

All other structural and governance gates passed.

### Important successful compatibility findings

The actual R26N receptions integration behaved exactly as intended outside Gate 15 evidence extraction:

- current population: 468 players / 32 teams / 16 games
- RB/FB rows: 107
- R26N-changed vacancy-active RB/FB rows: 104
- exactly 104 shadow arrays changed
- forbidden array changes: 0
- CIN non-vacancy RB/FB receptions: exact baseline
- non-RB/FB receptions: exact baseline
- all non-reception arrays: exact baseline
- all RB/FB `rec_yards`: exact R22 baseline
- all RB/FB `rush_rec_yards`: exact R22 baseline
- all RB/FB rushing arrays: exact baseline
- all QB pass-yard arrays: exact baseline
- full simulation key universe: 2,892 keys before/after
- candidate replay deterministic
- candidate reception arrays finite, nonnegative, integer-valued
- max baseline MC reception-mean gap vs sealed R26N: `0.017563893432533284`
- max candidate MC reception-mean gap vs sealed R26N: `0.018766081922851008`
- max MC reception-delta gap vs sealed R26N: `0.023037323394851983`
- frozen tolerance for each: `0.05 receptions`
- 2026 outcomes used: 0
- sportsbook football inputs used: 0
- same-week depth used: false
- R9 refit: false
- R22 changed by R26O splice: false
- receiving-yard means changed: false
- production parameters changed: false

## Gate-15 forensic anomaly

The failed Gate 15 is a **null-evidence anomaly**, not a measured mean-parity miss.

The protected R22 adapter at production base `f8417f55b04ce0e19baf260e9d532765034c47f1` returns:

`(adapted_result, trace_dataframe, audit_payload)`

The frozen R26O evaluator assigned:

`baseline_v4, r22_audit, _ = apply_rb_receiving_tail_production(...)`

and later expected:

- `r22_audit.get("gates", {}).get("mean_parity")`
- `r22_audit.get("max_mean_delta")`

Therefore the evaluator queried the **trace DataFrame** as though it were the audit payload. That mechanically produces the observed `null/null` evidence.

Independent immutable R22 parent evidence from artifact `10084118525` records:

- adapter disposition: `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_ADAPTER_PASS`
- `gates.mean_parity = true`
- `max_mean_delta = 5.329070518200751e-15`
- `gates.receptions_exact = true`

This does **not** retroactively convert R26O to PASS. The executed disposition remains `FAIL_NO_SHADOW` until a separately frozen forensic governance step determines whether the Gate-15 evidence wiring is a mechanical implementation defect eligible for an exact corrected rerun.

## Authorized next step

Freeze and execute a read-only Gate-15 contract forensic (R26P-style) that may inspect only immutable R26O/R22 artifacts and protected source contracts.

The forensic may authorize only a mechanical evidence-wiring correction and exact R26O rerun if it proves all of the following without changing any scientific threshold or candidate:

1. frozen Gate 15 requires R22 baseline receiving-yard mean parity;
2. protected R22 return order is `(adapted, trace, payload)`;
3. frozen R26O evaluator bound the second return value to `r22_audit`;
4. the second return value is not the audit payload;
5. canonical R22 parent payload passes `gates.mean_parity` with finite `max_mean_delta <= 1e-8`;
6. all R26O gates/thresholds/candidate/seed/iterations remain frozen;
7. no 2026 outcome or sportsbook input is introduced.

No production change is authorized.
