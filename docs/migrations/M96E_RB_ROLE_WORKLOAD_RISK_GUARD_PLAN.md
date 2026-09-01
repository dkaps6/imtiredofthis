# M96E — Role Router with Frozen Workload-Risk Guard

Status: **FROZEN BEFORE OUTCOME SCORING**

## Question
Can the stronger M96D non-entrenched role-based D router keep its global rushing-yard improvement while frozen M95F/M95I pregame workload-risk evidence suppresses D in the rare non-entrenched games where a 20+/25+ workload spike or transition is plausible?

## Frozen parents
- C = M94C central rush carries / rush-yard point.
- D = exact M96C opponent-defense efficiency residual prediction, carried through the authoritative M96D trace.
- Role-only insight = M96D controlled diagnostic: D was useful broadly for non-entrenched backs but still damaged rare unexpected high-workload games.
- W = M95F frozen workload distribution/calibration outputs. Safety guard only; no refit.
- V = M95I frozen vacancy/transition state. Safety guard only; no refit.

## No-reopen contract
- No carry adjustment.
- No new workload model.
- No recalibration of M95F or M95I.
- No threshold grid/search after scoring.
- No sportsbook inputs.
- No actual carries/yards or postgame features in routing.

## Frozen role definition
`entrenched_workhorse = role_is_workhorse == 1 AND rb_rb_share_avg5 >= 0.65`.

## Frozen workload-risk guard
For a non-entrenched back, define `workload_risk = 1` if **any** of:
1. M95F frozen calibrated probability of 20+ carries `cal_prob_20 >= 0.25`;
2. M95F frozen 90th-percentile workload `m95f_p90 >= 20`;
3. M95I frozen `prior_top1_unavailable == 1` (vacancy/transition state).

These thresholds are frozen for football interpretation, not selected from M96E outcomes: a one-in-four calibrated 20+ chance is materially nontrivial, p90>=20 explicitly places a 20-carry game inside the modeled upper workload distribution, and vacancy is an independently validated role-transition regime.

## Primary router — only retention-eligible arm
Turn D ON iff:
- `entrenched_workhorse == 0`, AND
- `workload_risk == 0`.

Otherwise use C unchanged.

## Controlled diagnostics — not eligible to replace primary
- `R_D_ROLE_ONLY`: M96D role-only arm.
- `R_D_ROLE_W_ONLY`: role gate plus M95F risk guard, without vacancy guard.
- `R_D_ROLE_V_ONLY`: role gate plus vacancy guard, without M95F risk guard.

Diagnostics identify which guard contributes but cannot be selected over the frozen primary after results.

## Evaluation
2025 M96C/M96D strict expanding-week OOF rows, Weeks 6-18 primary and Weeks 13-18 secondary. Actual workload slices are evaluation-only diagnostics.

Report:
- MAE, RMSE, bias, correlation;
- actual 0-5, 6-10, 11-14, 15-19, 20+, 25+ diagnostic slices;
- 75+/100+ AUC from point ranking;
- activation and workload-risk rates;
- how many evaluation-only actual 20+/25+ games are protected by each pregame guard.

## Frozen retention gate for primary M96E
All must pass:
1. Weeks 6-18 all-RB MAE gain >= `0.15` yard vs C.
2. Weeks 6-18 RMSE regression <= `0.10` yard.
3. Absolute bias may worsen <= `1.0` yard.
4. Actual 15-19, 20+, 25+ diagnostic slice MAE regression each <= `0.50` yard vs C.
5. 75+ and 100+ AUC regression each no worse than `-0.005`.
6. Weeks 13-18 all-RB MAE may not regress vs C.

No gate can be weakened after scoring.

## Stopping rule
M96E is the final retrospective RB efficiency-router migration.
- If it passes: freeze the architecture research-only for genuinely prospective 2026 confirmation. Write `AUTONOMOUS_RB_RESEARCH_STOP` because retrospective refinement is complete.
- If it fails: retain C/M94C as conservative point architecture, preserve M95F workload and M95I vacancy as separate diagnostics, reject retrospective D routing for promotion, and write `AUTONOMOUS_RB_RESEARCH_STOP`. No further retrospective RB efficiency variants.
