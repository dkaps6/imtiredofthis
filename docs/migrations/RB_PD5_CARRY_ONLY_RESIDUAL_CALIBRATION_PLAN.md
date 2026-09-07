# RB-PD5 Carry-Only Residual Calibration — Frozen Plan

## Status
Frozen before results. Production unchanged.

## Lineage
- Production RB baseline: RB P3.
- RB-PD3 established that player-specific residual calibration improved carry MAE and rushing-yard MAE but failed pooled rushing-yard p90.
- RB-PD4 restricted the same residual mechanism to pregame role-stable rows. It still failed pooled and late-season rushing-yard p90, while carry p90 remained protected and stable-role central accuracy improved.
- Repository search found no prior carry-only residual calibration experiment.

## Scientific hypothesis
Recent player-specific carry residuals contain transportable information about workload entitlement, while recent rushing-efficiency residuals are materially less stable and are the likely source of the rushing-yard tail deterioration observed in PD3/PD4. Therefore, a carry-only residual correction should retain the workload signal without injecting unstable efficiency corrections into rushing-yard tails.

## Frozen candidate
Use the exact PD3 carry residual history definition and carry correction parameters:
- strictly prior player-games only;
- last 8 prior games;
- minimum 4 prior games;
- alpha = 0.25;
- carry correction cap = +/-2.0 carries.

Do **not** use any player-specific efficiency-yard residual correction.

Candidate rushing yards must be recomputed from the corrected carry projection using the baseline P3 implied yards-per-carry / yard-per-carry mapping already present in the canonical RB evidence. No new efficiency feature, cap, shrinkage parameter, week rule, depth-chart boost, rookie adjustment, sportsbook input, or future information may be introduced.

Outside rows with sufficient prior carry history, candidate equals RB P3 exactly.

## Evaluation cohort
Use the same canonical 2020-2025 RB evidence and same scoreable-row definitions as PD3/PD4, preserving all walk-forward and source-integrity checks.

## Primary metrics
Evaluate pooled and by-season:
- carry MAE, RMSE, bias, correlation;
- rushing-yard MAE, RMSE, bias, correlation;
- median / p75 / p90 absolute error;
- carry 3+/5+/7+ miss rates;
- rushing-yard 20+/30+/40+ miss rates.

Also report Weeks 5-12, Weeks 13-18, highest projected-carry quartile, and prior-history-eligible rows.

## Frozen scientific gates
All must pass:
1. carry MAE improves by at least 0.03 vs RB P3;
2. rushing-yard MAE improves by at least 0.10 yards vs RB P3;
3. pooled carry p90 does not worsen;
4. pooled rushing-yard p90 does not worsen;
5. Weeks 13-18 rushing-yard p90 does not worsen;
6. carry 5+ miss rate does not worsen by more than 0.5 percentage points;
7. rushing-yard 30+ miss rate does not worsen by more than 0.5 percentage points;
8. rushing-yard 40+ miss rate does not worsen by more than 0.5 percentage points;
9. highest projected-carry quartile carry MAE does not worsen by more than 0.05;
10. highest projected-carry quartile rushing-yard MAE does not worsen by more than 0.20;
11. no walk-forward leakage violations;
12. sportsbook_inputs_used == false.

## Interpretation rule
- Passing supports a separate full-stack integration test; it does not directly replace RB P3.
- Failure is scientific evidence. Do not relax thresholds, tune alpha/caps/history window, or reintroduce efficiency residuals in response to a near miss.

## Production
No production changes are authorized by this plan.
