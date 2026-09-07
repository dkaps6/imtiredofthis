# RB-PD6 Positive-Entitlement Residual Calibration — Frozen Plan

## Status
Frozen before results. Production unchanged.

## Lineage
- Production RB baseline: RB P3.
- RB-PD3 showed that player-specific residual calibration materially improved central carry and rushing-yard accuracy but failed pooled rushing-yard p90.
- RB-PD4 added a pregame role-stability gate and still failed pooled / late-season rushing-yard p90.
- RB-PD5 removed efficiency-residual corrections entirely. It fixed the late-season p90 failure and improved carry MAE, rushing-yard MAE, bias, catastrophic-miss rates, and top-workload tails, but still narrowly failed pooled eligible rushing-yard p90 (48.783 -> 49.098).
- Repository search found no prior one-sided positive-entitlement carry-residual experiment.

## Scientific hypothesis
RB P3 has a persistent underprojection bias in carries. Recent positive player carry residuals (actual carries above projection) may represent durable workload entitlement that the team/player model has not fully absorbed yet. Recent negative residuals (actual carries below projection) are more likely to mix injury limitation, game script, short-term rotation, and other nonpersistent suppression states.

Therefore the player-specific residual mechanism should be asymmetric: use prior residual evidence only to increase a player's projected workload when history says the model has repeatedly underprojected him; do not decrease a player's workload from negative historical residuals.

This is a football-mechanism test, not an alpha/cap/window retune of PD5.

## Frozen candidate
Use the exact RB-PD5 walk-forward history definition and correction parameters:
- strictly prior player-games only;
- last 8 prior games;
- minimum 4 prior games;
- alpha = 0.25;
- carry correction cap = +/-2.0 carries;
- no player-specific efficiency residual correction;
- baseline P3 implied yards-per-carry mapping for rushing yards.

Change only the residual application rule:
- if prior8_carry_bias < 0 (baseline projection minus actual is negative, meaning historical actual carries exceeded projection), apply the exact PD5 carry correction;
- if prior8_carry_bias >= 0, apply zero carry correction;
- outside rows with fewer than 4 prior games, candidate equals RB P3 exactly.

Equivalently, the candidate may only move projected carries upward relative to RB P3. No downward player-residual correction is permitted.

No new efficiency feature, role threshold, week filter, depth-chart boost, rookie adjustment, sportsbook input, future information, alpha search, cap search, or history-window search may be introduced.

## Evaluation cohort
Use the exact same canonical 2020-2025 RB evidence and scoreable-row definitions as PD5, preserving all source-integrity and walk-forward checks.

## Primary metrics
Evaluate pooled and by-season:
- carry MAE, RMSE, bias, correlation;
- rushing-yard MAE, RMSE, bias, correlation;
- median / p75 / p90 absolute error;
- carry 3+/5+/7+ miss rates;
- rushing-yard 20+/30+/40+ miss rates.

Also report:
- prior-history-eligible rows;
- positive-entitlement-applied rows;
- eligible rows where correction is suppressed;
- Weeks 5-12;
- Weeks 13-18;
- highest projected-carry quartile.

## Frozen scientific gates
All must pass:
1. eligible carry MAE improves by at least 0.03 vs RB P3;
2. eligible rushing-yard MAE improves by at least 0.10 yards vs RB P3;
3. pooled eligible carry p90 does not worsen;
4. pooled eligible rushing-yard p90 does not worsen;
5. Weeks 13-18 rushing-yard p90 does not worsen;
6. carry 5+ miss rate does not worsen by more than 0.5 percentage points;
7. rushing-yard 30+ miss rate does not worsen by more than 0.5 percentage points;
8. rushing-yard 40+ miss rate does not worsen by more than 0.5 percentage points;
9. highest projected-carry quartile carry MAE does not worsen by more than 0.05;
10. highest projected-carry quartile rushing-yard MAE does not worsen by more than 0.20;
11. applied-row carry MAE must improve vs RB P3;
12. applied-row rushing-yard MAE must improve vs RB P3;
13. suppressed eligible rows must remain exactly RB P3;
14. no walk-forward leakage violations;
15. sportsbook_inputs_used == false.

## Interpretation rule
- Passing supports a separate full-stack integration test; it does not directly replace RB P3.
- Failure is scientific evidence. Do not convert this into a two-sided threshold search, retune alpha/caps/history window, or inspect near-miss thresholds to rescue the candidate.

## Production
No production changes are authorized by this plan.