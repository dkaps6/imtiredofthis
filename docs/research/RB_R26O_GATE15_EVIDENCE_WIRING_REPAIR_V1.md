# RB R26O Gate-15 Evidence-Wiring Repair V1

## Authority

This repair is authorized only by canonical R26P:

- R26P run: `34399525657`
- job: `102627643995`
- artifact: `10122862934`
- digest: `sha256:2e596f3a3bcbef156983664d332ded94ef41dae1402d1a96ab0b61d185c68d47`
- disposition: `R26P_GATE15_EVIDENCE_WIRING_DEFECT_CONFIRMED_MECHANICAL_RERUN_AUTHORIZED`
- 17 / 17 forensic tests passed

The prior R26O scientific execution remains preserved as:
`R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_FAIL_NO_SHADOW`.

## Exact defect

Protected R22 returns:

`(adapted_result, trace_dataframe, audit_payload)`

Frozen R26O evaluator expects its local variable `r22_audit` to support the protected audit-payload contract, but assigned:

`baseline_v4, r22_audit, _ = apply_rb_receiving_tail_production(...)`

Thus the local `r22_audit` received the trace DataFrame. The sole failed Gate 15 consequently recorded `mean_parity=null` and `max_mean_delta=null`.

## Authorized correction

Do not modify the frozen R26O evaluator or protected R22 code.

Add a hash-tracked runtime wrapper that:

1. preserves/reuses the already-authorized R22 identity dtype-only compatibility seam;
2. intercepts only the R26O-local call to `apply_rb_receiving_tail_production`;
3. calls the original protected R22 function unchanged;
4. receives the protected return `(adapted, trace, payload)`;
5. verifies `trace` is a DataFrame and `payload` is a dictionary with:
   - disposition `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_ADAPTER_PASS`,
   - boolean `gates.mean_parity`,
   - finite `max_mean_delta`,
   - boolean `gates.receptions_exact`;
6. returns `(adapted, payload, trace)` **only to the frozen R26O evaluator call site**, so its pre-existing `baseline_v4, r22_audit, _` binding receives the intended audit payload;
7. changes no adapted arrays, trace rows, payload values, football inputs, candidate values, gates, thresholds, seeds, iterations, or production files.

## Required rerun invariants

The corrected R26O execution must retain:

- frozen plan commit `96e4a6977458b55ec00cb29fbc9de05c56c89b88`;
- frozen evaluator commit `6a171cb32e77c7d6691ceb75da9861a5d6dd7bc4`;
- all 38 frozen gates;
- 25,000 iterations;
- seed 42;
- 0.05 reception compatibility tolerances;
- exact R26N/R22/current-production parents;
- all earlier selector/key/dtype mechanical repairs;
- zero 2026 outcomes;
- zero sportsbook football inputs;
- no R9 refit;
- no production change.

The rerun disposition must be accepted exactly as emitted. A PASS would still authorize only prospective research-shadow sealing, not live production or promotion.
