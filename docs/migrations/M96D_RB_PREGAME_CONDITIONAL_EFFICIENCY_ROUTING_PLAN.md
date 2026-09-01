# M96D — Pregame Conditional Efficiency Routing Audit

Status: **FROZEN BEFORE OUTCOME SCORING**

## Question
Can leakage-safe pregame workload/role state identify when the M96C opponent-defense efficiency expert D should be active, preserving its low/mid-workload rushing-yard gains without recreating its high-workload damage?

## Frozen parents
- C = M94C central rush carries / rushing-yard point. No carry adjustment.
- D = exact M96C opponent run-efficiency/resistance residual prediction (`pred_D`) from the authoritative M96C OOF trace.
- E/P are controlled M96C alternative experts only; they cannot become the selected winner in M96D.
- X remains rejected as an isolated separable tail increment.
- M95F/M95I semantics remain workload/vacancy evidence; M96D does not reopen their coefficient search.

## Temporal / leakage contract
M96C predictions are strict 2025 expanding-week OOF (Weeks 6-18), trained only on earlier weeks. M96D performs **no new outcome-trained routing model**. The router is deterministic from target-week pregame variables only.

Actual carries/yards are evaluation diagnostics only and cannot enter the router.
No sportsbook inputs.

## Primary router — frozen
Define `entrenched_workhorse` as:

- `role_is_workhorse == 1`, AND
- pregame `rb_rb_share_avg5 >= 0.65`.

Turn D ON only when:

- frozen M94C `candidate_rush_att < 15.0`, AND
- `entrenched_workhorse == 0`.

Otherwise use C unchanged.

This is the **only arm eligible for M96D retention**. Thresholds are football-structural and frozen before scoring; they will not be tuned after results.

## Controlled diagnostics — not selection candidates
To diagnose which side of the gate matters without creating a threshold search:
- `R_D_CARRY_ONLY`: D if M94C projected carries < 15.
- `R_D_ROLE_ONLY`: D if not entrenched_workhorse.
- `R_E_PRIMARY`: E under the exact primary router.
- `R_P_PRIMARY`: P under the exact primary router.

These arms may support/reject the routing hypothesis but cannot supersede `R_D_PRIMARY` in this migration.

## Metrics
Report Weeks 6-18 and Weeks 13-18:
- MAE, RMSE, bias, correlation;
- evaluation-only actual-carry slices: 0-5, 6-10, 11-14, 15-19, 20+, 25+;
- 75+/100+ rushing-yard AUC from point ranking;
- router activation rate overall and by pregame workload/role strata.

## Frozen retention gate for R_D_PRIMARY
All must pass:
1. Weeks 6-18 all-RB MAE improvement >= 0.10 yard vs C.
2. Weeks 6-18 RMSE cannot regress > 0.15 yard.
3. Absolute bias cannot worsen by > 1.0 yard.
4. Actual 15-19, 20+, and 25+ evaluation slices may each regress by at most 0.50 MAE yard vs C.
5. 75+ and 100+ AUC may each regress by at most 0.005.
6. Weeks 13-18 all-RB MAE may regress by at most 0.10 yard.

No gate can be weakened after scoring.

## Decision
- If `R_D_PRIMARY` passes: freeze it research-only for prospective 2026 confirmation; no production promotion.
- If it fails but diagnostics clearly show a different *type* of pregame state is required, record that scientific lesson and allow at most one newly precommitted architecture migration if justified by existing evidence.
- If routing does not safely separate regimes, retain C/M94C and stop retrospective RB efficiency refinement rather than opening unlimited variants.
