# WR Targeted-Route Matchup Full-Stack Integration V1 — Frozen Plan

Date: 2026-10-14

Status: **FROZEN BEFORE FULL-STACK CANDIDATE OUTPUT / RESEARCH ONLY**

Parents:
- `docs/research/PLAYER_TARGETED_ROUTE_MATCHUP_V1_FREEZE.md`
- `docs/research/PLAYER_TARGETED_ROUTE_MATCHUP_V1_STAGE_B_FREEZE.md`
- `docs/research/PLAYER_TARGETED_ROUTE_MATCHUP_V1_RESULT.md`
- standing WR protocol: `docs/migrations/WR_FULL_STACK_INTEGRATION_PROTOCOL.md` on the WR research lineage

Confirmed source signal:
- WR YPT route-matchup family: CONFIRMED
- WR catch-rate route-matchup family: CONFIRMED
- WR direct receiving-yard residual: CLOSED
- all TE families: CLOSED

Corrected Stage-B authority:
- canonical source run: `38637901016`;
- canonical Stage-B rerun: `38641891257`;
- exact scored head: `fe1d8ff52bd7cfa322e095a7a6f58752acb0b55f`;
- artifact: `11929999088`;
- digest: `sha256:0ff2c393e0daa0fa5163d34ce35e38eb25fb7c907cdf287019d609b804ca5128`;
- this authority correction changes no frozen integration mechanics.

This experiment tests only the two confirmed WR mechanisms inside the historical production-order WR stack.

## 1. Authoritative integration year

Primary full-stack integration year: **2024 only**.

Reason:
- exact fold-safe WR-R15 authority exists for `train_season=2023 -> test_season=2024`;
- WR-R15's own frozen science forbids treating 2025 as an OOS confirmation fold.

The 2024 stack must preserve production order:

`historical pregame state -> Bayesian/rules -> M38 -> explicit finite target entitlement -> TE-R5P 2024 fold -> WR-R15 2024 fold -> route-matchup efficiency arm -> joint MC -> frozen ensemble weights`

2025 may not be presented as a fold-safe WR-R15 integration result in V1.

## 2. Frozen development-to-integration transmission

Use the already-frozen corrected-source Stage-B casebook from run `38641891257` (artifact `11929999088`, digest `sha256:0ff2c393e0daa0fa5163d34ce35e38eb25fb7c907cdf287019d609b804ca5128`).

Fit **only 2023 development rows**, WR only.

YPT transmission:
- x = `matchup_delta_ypt`
- y = `ypt_residual`
- fit ordinary least squares with intercept;
- retain only the slope;
- `beta_ypt = clip(OLS_slope, 0.0, 1.0)`.

Catch transmission:
- x = `matchup_delta_catch`
- y = `catch_residual`
- fit ordinary least squares with intercept;
- retain only the slope;
- `beta_catch = clip(OLS_slope, 0.0, 1.0)`.

The intercept is discarded because V1 is a matchup mechanism, not a global recalibration layer.

No 2024 outcome may enter either coefficient.

No hand-selected coefficient, grid search, cap search, threshold search, or post-result shrinkage is allowed.

## 3. Candidate seams

Only source-valid 2024 WR player-games present in the Stage-B casebook are eligible.

Map stable GSIS player identity to the historical canonical `player_clean_key` through the exact 2024 player-log identity bridge.

### YPT arm

After TE-R5P + WR-R15 entitlement has been materialized and **after target shares are final**, but before simulation:

`candidate_rules_ypt = base_rules_ypt + beta_ypt * matchup_delta_ypt`

Use the canonical simulator's valid YPT safety range:
- floor at 2.0;
- ceiling at 20.0.

No target share, pass rate, plays, catch rate, volatility, TE input, RB input, or QB input changes in this arm.

### Catch arm

`candidate_rules_catch_rate = base_rules_catch_rate + beta_catch * matchup_delta_catch`

Clip only to the simulator probability domain `[0.001, 0.999]`.

No YPT, target share, pass rate, plays, volatility, TE input, RB input, or QB input changes in this arm.

### Combined arm

Apply both frozen adjustments to the same WR state.

Because the canonical simulator can consume RNG differently when binomial probabilities change, the **combined arm is diagnostic only in V1**. It cannot rescue either primary arm. Primary scientific decisions are made on:
- YPT arm -> receiving yards;
- Catch arm -> receptions.

A later joint production candidate must use the repository's governed RNG-isolation path if both arms pass.

## 4. Exact baseline

The baseline is the fold-safe 2024 production-order historical replay:
- M38 target hierarchy;
- explicit finite target entitlement;
- TE-R5P 2024 OOS fold;
- WR-R15 2024 OOS fold;
- canonical `simulation_v2` / explicit-entitlement MC;
- unchanged `data/model_ensemble_weights.csv`.

Required before interpretation:
- legacy -> explicit entitlement parity;
- TE pool conservation <=1e-10;
- WR1 M38 anchor preserved <=1e-10;
- WR2+ pool preserved <=1e-10;
- total WR room mass preserved <=1e-10;
- modeled team target mass preserved <=1e-10;
- no same/future participation usage;
- route-matchup candidate target-share max absolute delta <=1e-12;
- route-matchup candidate non-WR efficiency-field max absolute delta <=1e-12.

## 5. Evaluation populations

Primary rows are 2024 WR player-games where:
- the promoted historical stack has a valid projection;
- the Stage-B route-matchup casebook has a source-valid strictly-prior feature;
- actual outcome is available.

Report WR1 and WR2+ separately using the production-order WR-R15 anchor flag.

Minimum coverage:
- >=150 named WRs across the eligible 2024 integration population;
- >=1,500 eligible WR player-games.

If coverage misses either floor, disposition is `WR_ROUTE_MATCHUP_INTEGRATION_SOURCE_LIMITED`.

## 6. Frozen primary gates — YPT arm

Market: receiving yards.

All hard gates must pass:
1. source / baseline / conservation gates pass;
2. direct MC receiving-yard MAE improves by >=0.50%;
3. direct MC RMSE non-worse;
4. full-stack ensemble receiving-yard MAE improves (>0);
5. full-stack RMSE non-worse;
6. absolute full-stack bias non-worse;
7. full-stack correlation non-worse within tolerance 0.005;
8. Weeks 5-12 full-stack MAE non-worse;
9. Weeks 13-18 full-stack MAE non-worse;
10. WR1 MAE non-worse;
11. WR2+ MAE non-worse;
12. 20+, 30+, and 40+ yard absolute-error rates all non-worse;
13. empirical MC CRPS non-worse;
14. absolute 80% interval coverage gap from 0.80 non-worse;
15. absolute 90% interval coverage gap from 0.90 non-worse;
16. median qualifying-player full-stack MAE delta <0 for players with >=4 eligible games;
17. fraction of those qualifying players with lower MAE >0.50.

Disposition:
- `WR_ROUTE_MATCHUP_YPT_FULL_STACK_CONFIRMED` only if every gate passes;
- otherwise `NO_ACTIONABLE_WR_ROUTE_MATCHUP_YPT_INTEGRATION`.

## 7. Frozen primary gates — Catch arm

Market: receptions.

All hard gates must pass:
1. source / baseline / conservation gates pass;
2. direct MC receptions MAE improves by >=0.50%;
3. direct MC RMSE non-worse;
4. full-stack ensemble receptions MAE improves (>0);
5. full-stack RMSE non-worse;
6. absolute full-stack bias non-worse;
7. full-stack correlation non-worse within tolerance 0.005;
8. Weeks 5-12 full-stack MAE non-worse;
9. Weeks 13-18 full-stack MAE non-worse;
10. WR1 MAE non-worse;
11. WR2+ MAE non-worse;
12. absolute-error >=2 and >=3 reception rates both non-worse;
13. empirical MC CRPS non-worse;
14. absolute 80% interval coverage gap from 0.80 non-worse;
15. absolute 90% interval coverage gap from 0.90 non-worse;
16. median qualifying-player full-stack MAE delta <0 for players with >=4 eligible games;
17. fraction of those qualifying players with lower MAE >0.50.

Disposition:
- `WR_ROUTE_MATCHUP_CATCH_FULL_STACK_CONFIRMED` only if every gate passes;
- otherwise `NO_ACTIONABLE_WR_ROUTE_MATCHUP_CATCH_INTEGRATION`.

## 8. Overall V1 disposition

- If both primary arms pass:
  `WR_ROUTE_MATCHUP_FULL_STACK_BOTH_ARMS_CONFIRMED`.
- If exactly one passes:
  `WR_ROUTE_MATCHUP_FULL_STACK_PARTIAL_CONFIRMATION`.
- If neither passes:
  `NO_ACTIONABLE_WR_ROUTE_MATCHUP_FULL_STACK_INTEGRATION`.

The combined arm never changes the primary disposition.

## 9. Player-first reporting

For each primary arm report:
- named WR count;
- eligible player-games;
- median per-player baseline MAE;
- median per-player candidate MAE;
- median per-player MAE delta;
- fraction of qualifying players improved;
- distribution of per-player deltas;
- WR1 vs WR2+ results.

Position-wide averages are safety / system gates, not the sole definition of success.

## 10. Integrity / stop rules

- sportsbook inputs = 0;
- OddsAPI calls = 0;
- 2026 outcomes = 0;
- production changes = false;
- no TE transport;
- no direct receiving-yard residual correction;
- no target-share modification from route matchup;
- no tuning after results;
- no alternate history windows;
- no subgroup rescue;
- no coefficient refit on 2024;
- no 2025 WR-R15 pseudo-confirmation.

A V1 pass authorizes only prospective shadow / governed integration work. It does not itself alter production.
