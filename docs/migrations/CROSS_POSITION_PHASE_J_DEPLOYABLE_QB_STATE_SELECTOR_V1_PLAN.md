# CROSS-POSITION PHASE J — DEPLOYABLE QB STATE SELECTOR V1

## Purpose
Phase I validated a mean-neutral QB distribution selector, but its original Phase-E state model included WR/TE predicted target-pool features that are not cleanly materialized on the no-odds production path before pricing.

Phase J tests a deliberately reduced **deployment-portable** selector using only inputs already available or straightforwardly buildable pregame in production:
- four strict-prior Phase-C team spot scores,
- promoted M89/M90 predicted QB attempts,
- week number.

This is a deployment portability test, not a broad feature search. It does not reinterpret Phase E/F/H failures and does not alter M89/M90 means, C2 distributions, WR/TE/RB projections, or production.

## Frozen lineage
- Phase C authoritative all-row game-spot artifact: run `34147777341`.
- C2 authoritative QB distribution casebook: run `34142510405`.
- Phase I authoritative mean-neutral distribution validation: run `34151186485`, disposition `MEAN_NEUTRAL_QB_DISTRIBUTION_STATE_ELIGIBLE`.

## Frozen feature set
Exactly:
1. `pass_opportunity_spot`
2. `pass_efficiency_spot`
3. `rush_opportunity_spot`
4. `rush_efficiency_spot`
5. `pred_qb_attempts`
6. `week`

No WR target pool, TE target pool, RB carry pool, sportsbook field, same-game result, or future-game result is allowed as a predictor.

## Frozen model
- target: `actual_qb_attempts - pred_qb_attempts`
- `StandardScaler`
- `Ridge(alpha=20.0)`
- walk-forward 2025 evaluation exactly as Phase E: for each scored 2025 week, train only on chronologically earlier eligible team-games; require at least 128 training rows, otherwise emit delta 0.
- `delta_up = max(predicted_delta, 0)` is retained only for the opportunity diagnostics.

## Distribution selector
For each aligned 2025 C2 QB game:
- if predicted `delta_pass_attempts > 0`, select the existing C2 QB distribution;
- otherwise select B0.
- the M89/M90 mean remains exactly unchanged.
- no receiver C2 output is activated.

## Frozen scorecard
Report:
### Opportunity portability
- QB attempt MAE/RMSE/bias/correlation baseline vs positive-only attempt correction;
- p90 QB attempt absolute error;
- residual sign accuracy;
- PASS_STATE_HIGH / PASS_STATE_LOW recall using realized Phase-D definitions for diagnostic interpretation only.

### QB distribution
B0 vs Phase-J candidate vs ALL_C2 reference:
- mean MAE and mean-anchor parity;
- CRPS and paired improvement vs B0;
- 10,000-resample paired bootstrap probability, seed `5610`;
- 50/80/90 interval coverage + calibration error;
- 50/80/90 interval widths;
- p90 mean absolute error;
- 100+ yard mean-miss rate;
- candidate-vs-B0 CRPS win/loss/tie counts.

## Frozen eligibility gates
`DEPLOYABLE_QB_STATE_SELECTOR_ELIGIBLE` only if all are true:
1. positive-only QB attempt MAE improves by >= **0.40 attempts**;
2. QB attempt p90 absolute error does not worsen;
3. candidate mean-anchor max gap <= **0.01 yard**;
4. candidate mean MAE differs from B0 by <= **0.01 yard**;
5. candidate CRPS improves vs B0 by >= **0.75 yard**;
6. paired bootstrap probability of CRPS improvement >= **0.95**;
7. absolute 80% coverage error does not worsen vs B0 by > **0.02**;
8. absolute 90% coverage error does not worsen vs B0 by > **0.02**;
9. p90 mean absolute error unchanged within **0.01 yard**;
10. 100+ yard mean-miss rate unchanged within **1e-9**;
11. no WR/TE/RB/player mean changes;
12. sportsbook inputs = 0; same/future outcomes used as predictors/selectors = 0; production parameters changed = 0.

Otherwise disposition is `DEPLOYABLE_QB_STATE_SELECTOR_NOT_ELIGIBLE`.

No feature may be added back, no gate changed, and no threshold/alpha tuned after results.

## Deployment rule if eligible
Fit the exact frozen scaler + Ridge model on all eligible historical team-games through 2025 and persist its coefficients/scaler statistics as a versioned JSON artifact.

For 2026 Week 1:
- build the four strict-prior spot scores from completed PBP through 2025 plus the authoritative 2026 schedule/team-week map;
- inside promoted QB pricing, combine those four scores with M89/M90 `pred_attempts` and week=1;
- if model delta >0, use the exact C2-generated QB distribution shape while preserving the promoted M89/M90 point mean;
- if delta <=0, retain B0 QB distribution;
- WR/TE/RB outputs remain untouched.

A Phase-J pass authorizes a production-candidate dry run / Week-1 audit, not an automatic merge to main.
