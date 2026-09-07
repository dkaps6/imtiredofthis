# RB-PD3 — Walk-Forward Player Residual Calibration — Frozen Plan

## Purpose

Convert the independently passed RB-PD2 player-error persistence finding into one conservative, leakage-safe predictive calibration test. This is research only and does not change RB P3 production unless a later promotion migration is separately authorized.

## Lineage

- Parent branch: `research-rb-pd2-player-error-persistence`
- Parent result SHA: `a4173a7a06cd72c93c1064cd377977c9fd404c16`
- RB-PD2 run: `34064637295`
- RB-PD2 job: `101571138337`
- RB-PD2 artifact: `9998549410`
- Frozen RB-PD2 disposition: `RB_PLAYER_ERROR_PERSISTENCE_DETECTED`
- Production rushing anchor remains RB P3.
- Sportsbook inputs: **0**.

RB-PD2 found that an RB's strictly prior same-player error history predicts both the direction and difficulty of the next carry and rushing-yard miss. PD3 tests whether a fixed, deliberately small correction improves next-game projections.

## Frozen population and history

Use the exact canonical 2025 STACK1 production-equivalent RB/HB/FB rows and STACK2 identity/rookie metadata used by PD2.

For every target player-game:

- history is same player only;
- only games completed strictly before the target game;
- last **8** eligible games;
- minimum **4** prior games;
- no full-season player bias or future result may enter a target projection.

Primary scientific evaluation is paired on the exact scoreable PD2 cohort.

## Frozen candidate

### Carry layer

`prior8_carry_bias = mean(predicted carries - actual carries)` over the last 8 strictly prior games.

Use a fixed shrinkage coefficient of **0.25** and a fixed maximum absolute correction of **2.0 carries**:

`carry_adjustment = clip(0.25 * prior8_carry_bias, -2.0, +2.0)`

`candidate_carries = max(0, baseline_carries - carry_adjustment)`

No alternate coefficient, history window, minimum sample, or cap may be searched after results.

### Yard layer

The yard correction must not blindly duplicate the carry correction.

For each strictly prior game define:

`baseline_ypc = baseline_rush_yards / baseline_carries` when baseline carries > 0.

`efficiency_residual_error = (baseline_rush_yards - actual_rush_yards) - (baseline_carries - actual_carries) * baseline_ypc`

At the target game:

1. mechanically propagate the carry candidate through the target baseline YPC;
2. subtract **0.25** times the mean prior-8 efficiency residual error;
3. cap the efficiency-only yard correction at **±8 rushing yards**;
4. floor candidate rushing yards at zero.

Thus the candidate can learn persistent player-specific carry bias and persistent yardage error not already explained by carry error, while keeping both corrections small.

No rookie boost, depth-order boost, role remap, target-game outcome, or sportsbook information is allowed.

## Frozen outputs

For baseline and candidate, pooled and Weeks 5-12 / Weeks 13-18:

- carries: MAE, RMSE, bias, correlation, median/p75/p90 absolute error, 3+/5+/7+ carry miss rates;
- rushing yards: MAE, RMSE, bias, correlation, median/p75/p90 absolute error, 20+/30+/40+ yard miss rates;
- top baseline carry-volume quartile versus remaining players;
- rookie versus veteran descriptive slices;
- individual player scorecard for players with at least 6 scoreable games.

Also report correction-size distributions and exact walk-forward integrity counts.

## Integrity gates

Scientific interpretation stops unless all pass:

1. exact canonical source rows = `1,393`;
2. scoreable rows >= `700`;
3. no target game contributes to its own prior history;
4. all candidate features use strictly prior same-player games only;
5. no sportsbook inputs;
6. no target-game outcome used to select coefficient/window/cap;
7. baseline predictions reproduce the canonical STACK1 values exactly.

## Scientific gates

`RB_PD3_PLAYER_RESIDUAL_CALIBRATION_PASS` requires all:

1. pooled carry MAE improves by **>= 0.05 carries**;
2. pooled rushing-yard MAE improves by **>= 0.25 yards**;
3. carry p90 absolute error does not worsen;
4. rushing-yard p90 absolute error does not worsen;
5. carry 5+ miss rate does not worsen by more than **0.5 percentage points**;
6. rushing-yard 30+ miss rate does not worsen by more than **0.5 percentage points**;
7. rushing-yard 40+ miss rate does not worsen by more than **0.5 percentage points**;
8. Weeks 5-12 carry MAE does not worsen by more than **0.10 carries**;
9. Weeks 13-18 carry MAE does not worsen by more than **0.10 carries**;
10. Weeks 5-12 rushing-yard MAE does not worsen by more than **0.50 yards**;
11. Weeks 13-18 rushing-yard MAE does not worsen by more than **0.50 yards**;
12. top baseline carry-volume quartile carry MAE does not worsen by more than **0.10 carries**;
13. top baseline carry-volume quartile rushing-yard MAE does not worsen by more than **0.50 yards**;
14. all integrity gates pass.

If any scientific gate fails, preserve RB P3 and record `RB_PD3_PLAYER_RESIDUAL_CALIBRATION_FAIL`. Do not relax a threshold or try a nearby alpha/cap/window in this migration.

## Production rule

A pass authorizes only a separate multi-season/full-stack confirmation and production-integration test. It does not directly change production.
