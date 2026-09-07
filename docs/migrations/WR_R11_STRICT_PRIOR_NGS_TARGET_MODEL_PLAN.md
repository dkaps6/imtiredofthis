# WR-R11 Strict-Prior NGS Target Model — Frozen Plan

## Status
Frozen before results. Research candidate only. Production unchanged.

## Lineage
- WR production anchor: M38 target hierarchy (`1.40 / 1.14 / 0.91 / 0.78`).
- Exact multiseason B0 player baseline: Joint Pass/Receiving Conservation V1 run `34081764151`, artifact `10004223287`; B0 preserves the promoted M38 WR hierarchy.
- WR-R5 established that individual WR receiving-yard error is target/opportunity dominated.
- WR-R6/R8 showed simple recent target residual and snap/depth signals were not sufficient.
- WR-R10 run `34123975300`, artifact `10019327559`, disposition `STRICT_PRIOR_NGS_FEATURES_ELIGIBLE`: pooled prior1 coverage 86.33%, prior3 coverage 74.65%, all eight NGS fields clear the frozen availability gates, zero same/future observations used.
- Sportsbook inputs: 0.

## Football hypothesis
Strictly-prior tracking history may improve individual target entitlement beyond M38 because intended-air-yard share, aDOT, cushion, and separation describe how a receiver is being deployed and defended, not merely how many targets he happened to receive last week. If that information is predictive, it should improve future target forecasts and then improve receptions/receiving yards when the baseline catch/yard-per-target mechanics are preserved.

## Frozen OOS design
Target seasons scored: **2022-2025**.
Expanding-season walk-forward training:
- predict 2022 using 2021 only;
- predict 2023 using 2021-2022;
- predict 2024 using 2021-2023;
- predict 2025 using 2021-2024.

No target-season row may train its own prediction. No within-season future outcomes are used.

## Frozen feature set
Only strict-prior WR-R10 features:
- `avg_separation_prior1`, `avg_separation_prior3_mean`
- `avg_cushion_prior1`, `avg_cushion_prior3_mean`
- `avg_intended_air_yards_prior1`, `avg_intended_air_yards_prior3_mean`
- `percent_share_of_intended_air_yards_prior1`, `percent_share_of_intended_air_yards_prior3_mean`
- `prior_obs_count`
- current pregame `b0_expected_targets`

Catch percentage and YAC-family NGS fields are deliberately excluded from this target-opportunity candidate. They remain reserved for a separately frozen efficiency experiment if warranted.

Rows require at least 3 strictly-prior NGS observations. Rows without that history remain exactly B0 and are not used to claim candidate improvement.

## Frozen model
Predict the residual:
`actual_targets - b0_expected_targets`

Model: sklearn pipeline
1. median imputation fit on training data only;
2. StandardScaler fit on training data only;
3. Ridge regression, `alpha = 20.0`, fixed before results.

No hyperparameter search, feature selection, interaction search, player fixed effects, or post-result coefficient changes.

Candidate expected targets:
`clip(b0_expected_targets + predicted_target_residual, 0, 25)`.
The 25-target ceiling is a fixed football-domain sanity bound, not learned from results.

## Propagation to receiving markets
This experiment changes opportunity only.
For rows with positive B0 expected targets:
`target_ratio = candidate_targets / b0_expected_targets`.
Then:
- candidate receptions = `b0_receptions * target_ratio`
- candidate receiving yards = `b0_rec_yards * target_ratio`

Thus the candidate preserves B0 catch-per-target and receiving-yards-per-target mechanics and tests only whether NGS improves target entitlement.
Rows with zero B0 expected targets remain B0 for receptions/yards.

## Frozen scorecards
Pooled, by season, 2024-2025 combined, and highest-B0-target quartile:
- target MAE/RMSE/bias/correlation;
- reception MAE/RMSE/bias/correlation;
- receiving-yard MAE/RMSE/bias/correlation;
- median/p75/p90 absolute error;
- receiving-yard 20+/30+/40+ miss rates.

Also report correction distribution and percentage of eligible rows moved up/down.

## Frozen pass gates
All must pass on the NGS-eligible OOS cohort:
1. pooled target MAE improves by >= **0.05 targets**;
2. pooled receiving-yard MAE improves by >= **0.25 yards**;
3. target MAE improves in >= **3 of 4** scored seasons;
4. 2024-2025 target MAE improves;
5. 2024-2025 receiving-yard MAE improves;
6. no season target MAE regresses by > **0.10 targets**;
7. no season receiving-yard MAE regresses by > **0.75 yards**;
8. pooled target p90 absolute error does not worsen;
9. pooled receiving-yard p90 absolute error does not worsen;
10. pooled receiving-yard 30+ miss rate does not worsen by > **0.5 percentage points**;
11. pooled receiving-yard 40+ miss rate does not worsen by > **0.5 percentage points**;
12. highest-B0-target quartile target MAE does not worsen by > **0.05 targets**;
13. highest-B0-target quartile receiving-yard MAE does not worsen by > **0.25 yards**;
14. rows in the scored seasons that are NGS-ineligible remain exactly B0;
15. no leakage and sportsbook_inputs_used == false.

## Disposition
- Pass: `WR_NGS_TARGET_MODEL_SUPPORTED`, authorizing a separate full-stack integration test only.
- Fail: `WR_NGS_TARGET_MODEL_FAIL`. Do not search nearby Ridge alphas, target caps, prior windows, or coverage thresholds to rescue it. A future NGS efficiency experiment must stand on its own football hypothesis.

## Production
No direct production promotion is authorized by this plan.
