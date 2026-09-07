# RB-PD4 — Role-Stability-Gated Player Residual Calibration — Frozen Plan

## Purpose

Test a new football hypothesis motivated by the independently frozen RB-PD3 result without retuning PD3: same-player residual history is transportable only when the target-game workload role is demonstrably stable using pregame information. RB P3 remains the production anchor. This migration is research only.

## Lineage

- Production rushing anchor: RB P3.
- RB-PD2 established strictly-prior same-player error persistence.
- RB-PD3 run: `34089486097`.
- RB-PD3 job: `101639902906`.
- RB-PD3 launch SHA: `fc24d344960b672cc3ede6b3779e77922566e35b`.
- RB-PD3 artifact: `10006329196`.
- Frozen RB-PD3 disposition: `RB_PD3_PLAYER_RESIDUAL_CALIBRATION_FAIL`.
- RB-PD3 improved pooled carry MAE and rushing-yard MAE and improved 30+/40+ yard miss rates, but failed its frozen pooled rushing-yard p90 guard. The failure was concentrated in late-season tail behavior while the highest projected workload quartile improved.
- No PD3 coefficient, cap, history window, minimum sample, or scientific threshold may be altered here.
- Sportsbook inputs: **0**.

## Frozen population and history

Use the exact canonical 2025 STACK1 production-equivalent RB/HB/FB rows and STACK2 identity metadata used by PD2/PD3.

For every target player-game, construct same-player history strictly before the target week. The PD3 correction remains defined from the last **8** eligible games with a minimum **4** prior games. Role-stability features use only the last **4** strictly prior games.

No target-game outcome may enter a target projection or gate classification.

## Frozen football hypothesis

Historical player residuals should be acted on only when recent workload evidence indicates that the player's role has persisted into the target game. Abrupt role changes, low-volume committee roles, highly volatile workloads, or meaningful participation gaps make the historical residual less transportable and should leave the production baseline untouched.

This is a regime-gating hypothesis, not a parameter search around PD3.

## Frozen stable-role definition

A target player-game is `stable_role = true` only when **all** conditions are met using information available before kickoff:

1. at least **4** strictly prior same-player games exist;
2. the last prior appearance is no more than **2 NFL weeks** before the target week;
3. mean actual carries over the last 4 prior appearances is at least **8.0**;
4. coefficient of variation of actual carries over those four appearances is at most **0.35** (`std / mean`, population std);
5. the current production-baseline carry projection is within **4.0 carries** of the mean actual carries over those four prior appearances.

These values are frozen before candidate results. No alternate stability thresholds or combinations may be tried in this migration.

## Frozen candidate

For `stable_role = false`, the candidate equals RB P3 exactly for both carries and rushing yards.

For `stable_role = true`, apply the exact already-frozen PD3 correction:

### Carry layer

`prior8_carry_bias = mean(predicted carries - actual carries)` over the last 8 strictly prior games.

`carry_adjustment = clip(0.25 * prior8_carry_bias, -2.0, +2.0)`

`candidate_carries = max(0, baseline_carries - carry_adjustment)`

### Yard layer

For each prior game:

`baseline_ypc = baseline_rush_yards / baseline_carries` when baseline carries > 0.

`efficiency_residual_error = (baseline_rush_yards - actual_rush_yards) - (baseline_carries - actual_carries) * baseline_ypc`

At the target game:

1. propagate candidate carries through target baseline YPC;
2. subtract `0.25 * mean(last8 efficiency_residual_error)`;
3. cap the efficiency-only correction at **±8 rushing yards**;
4. floor candidate rushing yards at zero.

No rookie boost, depth-order boost, injury hindsight, target-game outcome, sportsbook information, alternate alpha, alternate cap, or alternate history window is allowed.

## Frozen outputs

Report baseline and candidate for the full paired scoreable cohort and the stable-role subset:

- carries: MAE, RMSE, bias, correlation, median/p75/p90 absolute error, 3+/5+/7+ miss rates;
- rushing yards: MAE, RMSE, bias, correlation, median/p75/p90 absolute error, 20+/30+/40+ miss rates;
- Weeks 5-12 and Weeks 13-18;
- top baseline carry-volume quartile;
- stable-role prevalence by week;
- individual-player scorecard for players with at least 6 scoreable games;
- exact correction-size and role-gate counts.

## Integrity gates

Scientific interpretation stops unless all pass:

1. exact canonical source rows = `1,393`;
2. full paired scoreable cohort >= `700`;
3. stable-role rows >= `200`;
4. no target game contributes to its own prior history;
5. all role-gate inputs use only strictly prior outcomes plus the current pregame baseline projection;
6. candidate equals baseline exactly outside stable-role rows;
7. no sportsbook inputs;
8. production baseline values reproduce canonical STACK1 exactly.

## Scientific gates

`RB_PD4_ROLE_STABILITY_GATED_CALIBRATION_PASS` requires all:

1. pooled carry MAE improves by **>= 0.03 carries**;
2. pooled rushing-yard MAE improves by **>= 0.15 yards**;
3. pooled carry p90 absolute error does not worsen;
4. pooled rushing-yard p90 absolute error does not worsen;
5. pooled carry 5+ miss rate does not worsen by more than **0.5 percentage points**;
6. pooled rushing-yard 30+ miss rate does not worsen by more than **0.5 percentage points**;
7. pooled rushing-yard 40+ miss rate does not worsen by more than **0.5 percentage points**;
8. Weeks 13-18 rushing-yard p90 absolute error does not worsen;
9. Weeks 13-18 rushing-yard MAE does not worsen by more than **0.25 yards**;
10. top baseline carry-volume quartile carry MAE does not worsen by more than **0.10 carries**;
11. top baseline carry-volume quartile rushing-yard MAE does not worsen by more than **0.50 yards**;
12. within the stable-role subset, carry MAE is better than baseline;
13. within the stable-role subset, rushing-yard MAE is better than baseline;
14. all integrity gates pass.

If any gate fails, preserve RB P3 and record `RB_PD4_ROLE_STABILITY_GATED_CALIBRATION_FAIL`. Do not relax thresholds or try nearby stability definitions, coefficients, caps, or windows in this migration.

## Production rule

A pass authorizes only a separately frozen multi-season/full-stack confirmation and production-integration test. It does not directly change production.
