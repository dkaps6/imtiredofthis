# RB R27D — Strict-Prior YACOE Residual V1 Frozen Plan

Status: `FROZEN BEFORE IMPLEMENTATION / MODEL FIT / CANDIDATE EXECUTION / RESULTS`

## 1. Parent authority and scientific boundary

Protected production-code authority:
`bb76ba9eabb08e2f0875a9af49301c3877f4141f`

Immutable parent evidence:
- R27 first valid receiving-yard decomposition: Run `34423546037`, Artifact `10132290573`, digest `sha256:cd5c9e26efa44c47ef3374e4deefe9038dc504c155799b48548ad64746e10e45`.
- R27B V2 first valid context candidate: Run `34428917229`, Artifact `10134023092`, digest `sha256:1df1a14c9900b890fe86e04269849dc90d76fc03a9215ad8612c1820dee5341f`, disposition `R27B_V2_NOVEL_EFFICIENCY_CONTEXT_MIXED_OR_FAIL_NO_INTEGRATION`.
- R27C2 physical target-quality forensic: Run `34431286455`, Artifact `10134581843`, digest `sha256:6e3aa7ec2149f6f0b65adb59b4cc36c4f88b24a142234bc368131de8e4de60dd`.
- R27D0 xYAC/YACOE source audit: Run `34431705294`, Artifact `10134726607`, digest `sha256:b1a2a44d7ed0303300a792bf617ac6c5095aa7a0229186001504ebc2cad44bdf`.
- R27D0B 2019 source extension: Run `34431927369`, Artifact `10134799189`, digest `sha256:13853603029adaf820b3228916c519acfdf98794c70462fbbcebe5f93f97d6cd`.
- R27D0C first valid xYAC mechanism split: Run `34432497854`, Artifact `10134994377`, digest `sha256:2c3d6043dc25298b8806e82abe7e4a97e9c2c066408fc38e215eae0bdbc1c882`, result record commit `76e662c92a49743092d7e512a6a1b2e87a74032c`.

R26 receiving opportunity/receptions remain fixed. R22 remains the receiving-yard distribution/tail authority. Sportsbook inputs are prohibited upstream.

## 2. Why R27D exists

R27 showed that exact R26 opportunity improves vacancy-active receiving-yard MAE overall, but lead-back translation worsens when the same production YPT is used. R27B V2 added raw strict-prior target-shape/YAC/team/opponent context and produced useful but insufficient signal; it did not repair RB1 or 2023 enough to qualify.

R27C2 then showed that the 2023 vacancy-RB1 problem was downstream of the catch: catch rate, target depth, screen rate and explosive frequency were essentially stable, while YAC/YPR/YPT compressed.

R27D0C separated that compression with nflverse expected-YAC (`xyac_mean_yardage`):
- 2023 vacancy RB1 actual YAC/reception was about `0.7781` yards lower than non-2023;
- expected YAC/reception was actually about `0.2731` yards higher;
- YAC-over-expected (YACOE) was about `1.0512` yards/reception lower.

Therefore the next genuinely new question is not another raw historical YAC/YPR/YPT model and not a broad expected-YAC target-design correction. It is whether **relative post-catch execution versus xYAC expectation has strict-prior predictive persistence/context** that can improve lead-back receiving-yard means.

The R27D0C tail reference also showed that catastrophic upside was strongly positive YACOE, supporting continued separation between repeatable mean correction and R22 stochastic right-tail authority.

## 3. Frozen scientific question

> With exact R26 vacancy opportunity/receptions fixed, can a strict-prior model of RB YAC-over-expected state improve receiving-yard point means specifically for vacancy-active incumbent RB1s, relative to the exact R27/B1 production-YPT translation, while preserving RB2+, pooled vacancy, aggregate, Week1 and tail safety?

This is a new prospective historical test. The RB1 application scope is frozen before R27D results because R23, R27, R27B V2, R27C and R27C2 repeatedly localized the unresolved mean-translation failure to lead backs. No result-dependent router may be created after execution.

## 4. Evaluation population and chronology

Evaluation seasons: REG `2020–2025`, all six outer folds retained.

Outer fold for test season `S`:
- any model-fitting target must come only from seasons `< S`;
- every predictor for every training and test player-game must be constructed only from PBP rows before that player-game;
- 2019 is legal training/context history for the 2020 fold, as established by R27D0B;
- within a season, target-game PBP may never enter its own features.

Training population for the YACOE state model:
- all historical RB/FB/HB/TB player-games in legal outer-training seasons with at least one completed catch carrying non-null `xyac_mean_yardage` and `yards_after_catch`;
- target-game player-game YACOE is a training label only;
- rows without an xYAC-observed reception are not YACOE training targets.

Evaluation application scope:
`VACANCY_ACTIVE == 1 AND vacancy_incumbent == 1 AND role == "RB1"`.

Every row outside that exact scope remains the exact B1/R27 mean. In particular vacancy RB2+ is forced to exact B1 parity; stable/no-vacancy rows are unchanged.

## 5. Frozen xYAC/YACOE definitions

For each completed RB catch with non-null xYAC:
- `expected_yac = xyac_mean_yardage`
- `yacoe = yards_after_catch - xyac_mean_yardage`

For each historical player-game, aggregate only the common xYAC-observed catch set.

Define the strict-as-of league RB YACOE population level from all legal prior completed RB catches.

Define centered target label:
`relative_game_yacoe = game_yacoe_per_reception - league_rb_yacoe_prior_as_of_game`.

This centering prevents R27D from broadly recentering receiving-yard means merely because nflverse xYAC is not perfectly zero-centered in a historical period. R27D tests heterogeneity around the legal prior league level.

## 6. Frozen predictor family

Only the following primary predictors are eligible. Raw historical YAC, YPR, YPT, catch rate, receiving yards, sportsbook data, target-game outcomes and R19/R22 tail probabilities are forbidden as predictors.

### Player execution/context state
1. `player_relative_yacoe_prior`
2. `player_expected_yac_prior_relative_to_league`

### Team/offense RB execution/context state
3. `team_rb_relative_yacoe_prior`
4. `team_rb_expected_yac_prior_relative_to_league`

### Opponent defense-vs-RB execution/context state
5. `opp_rb_relative_yacoe_allowed_prior`
6. `opp_rb_expected_yac_allowed_prior_relative_to_league`

### Frozen structural controls
7. `week1`
8. `prior_xyac_reception_support_log1p`

No V2 raw-YAC, screen-rate, air-yard, explosive-rate or generic historical efficiency feature may be imported into the primary R27D candidate. No feature ablation may replace the primary candidate after results.

## 7. Frozen shrinkage / prior construction

All priors are strict-as-of the target player-game.

YACOE and expected-YAC signals are reception-weighted and shrunk to the corresponding legal prior population mean:
- player: K = `12` xYAC-observed receptions;
- team RB receiving environment: K = `30` xYAC-observed receptions;
- opponent defense-vs-RB environment: K = `30` xYAC-observed receptions.

For no-history states, use the corresponding legal prior population mean, making the centered relative signal `0` rather than imputing a future-season value.

`prior_xyac_reception_support_log1p = log1p(player prior xYAC-observed receptions)`.

No shrinkage K may be tuned from R27D test results.

## 8. Frozen estimator

Estimator:
- `StandardScaler`
- weighted `Ridge(alpha=100.0)`

Alpha `100.0` is fixed before results as the conservative fallback already used in the prior V2 design; there is no R27D hyperparameter search or model zoo.

Training target:
`relative_game_yacoe`.

Sample weight:
`clip(game_xyac_observed_receptions, 1, 8)`.

The model predicts:
`pred_relative_yacoe` in yards per reception.

Candidate correction cap:
`clip(pred_relative_yacoe, -1.5, +1.5)` yards per reception.

The ±1.5 cap is frozen before results and may not be retuned after seeing R27D performance.

## 9. Baselines and exact candidate arithmetic

### B0 — original production opportunity translation
`B0_rec_yards = baseline_targets × production_ypt`

### B1 — exact R27/R26 opportunity translation
`B1_rec_yards = R26_candidate_targets × production_ypt`

B1 is the primary incremental comparator.

Define production implied YPR:
`production_implied_ypr = production_ypt / production_catch_rate`
for rows with positive production catch rate.

The exact reception bridge must satisfy, within `1e-10` yards:
`R26_candidate_receptions × production_implied_ypr == B1_rec_yards`.

### C1 — R27D YACOE residual candidate
Only for the frozen vacancy-incumbent-RB1 application scope:

`C1_ypr = max(production_implied_ypr + clip(pred_relative_yacoe, -1.5, +1.5), 0)`

`C1_rec_yards = R26_candidate_receptions × C1_ypr`

Outside the scope:
`C1_rec_yards = B1_rec_yards` exactly.

R26 targets/receptions are never changed by R27D. R22 is not invoked or modified during point-mean evaluation.

## 10. Frozen reporting cohorts

Report B0/B1/C1 for:
- ALL_RB
- VACANCY_ACTIVE
- VACANCY_INCUMBENT
- VACANCY_RB1_INCUMBENT — primary
- VACANCY_RB2PLUS_INCUMBENT — exact-parity safety cohort
- each test season 2020–2025 within VACANCY_RB1_INCUMBENT
- 2023_VACANCY_RB1_INCUMBENT — pre-identified stress cohort
- WEEK1

Report n, MAE, RMSE, signed bias, median AE, p90 AE and rate of absolute error >=30 yards where defined.

Also report predicted correction distribution and feature support, but no post-result thresholding/router search is authorized.

## 11. Frozen integrity / structural gates

All must PASS before scientific interpretation:

1. sportsbook inputs used == 0.
2. target-game/future PBP rows used in features == 0.
3. exact R27B V2/R27 parent prediction authority verified by pinned artifact ID/digest or exact staged authority.
4. R26 candidate targets are unchanged from B1 parent.
5. R26 candidate receptions are unchanged from parent.
6. B0 and B1 receiving-yard means reproduce parent within `1e-10`.
7. reception/YPR bridge to B1 max gap <= `1e-10`.
8. R27D application scope is exactly `vacancy_active & vacancy_incumbent & role==RB1`.
9. every row outside application scope has C1 == B1 within `1e-10`.
10. vacancy RB2+ C1 == B1 within `1e-10`.
11. no raw historical YAC/YPR/YPT/catch-rate/sportsbook/tail feature enters primary estimator.
12. all xYAC/YACOE predictors are strict-as-of each row.
13. outer test season S model fit uses only seasons < S.
14. all six outer folds 2020–2025 are present.
15. production files unchanged.
16. R22 files/assets unchanged.
17. R26 production files/mechanism unchanged.
18. correction cap never exceeds ±1.5 YPR.

If any integrity gate fails, terminal disposition is mechanical/integrity failure with **no scientific decision**.

## 12. Frozen scientific gates

Primary comparator is B1 unless stated otherwise.

19. `VACANCY_RB1_INCUMBENT` MAE improves by at least `1.00%` vs B1.
20. `VACANCY_RB1_INCUMBENT` MAE is non-worse vs B0.
21. `2023_VACANCY_RB1_INCUMBENT` MAE improves by at least `2.00%` vs B1.
22. `2023_VACANCY_RB1_INCUMBENT` MAE is non-worse vs B0.
23. `VACANCY_ACTIVE` MAE is non-worse vs B1.
24. ALL_RB MAE is non-worse vs B1 and no more than `+0.10%` worse than B0.
25. VACANCY_RB1_INCUMBENT RMSE is non-worse vs B1.
26. VACANCY_RB1_INCUMBENT p90 AE is non-worse vs B1.
27. VACANCY_RB1_INCUMBENT >=30-yard miss rate is non-worse vs B1.
28. absolute VACANCY_RB1_INCUMBENT bias worsens by no more than `0.25` yards vs B1.
29. at least `4 of 6` season-specific vacancy-RB1 MAEs improve vs B1.
30. no season-specific vacancy-RB1 MAE worsens by more than `2.00%` vs B1.
31. WEEK1 all-RB MAE worsens by no more than `0.50%` vs B1.

RB2+ is protected structurally through exact parity rather than a performance tolerance.

## 13. Frozen dispositions

If all integrity and scientific gates pass:
`R27D_STRICT_PRIOR_YACOE_RESIDUAL_SUPPORT_READY_FOR_SEPARATE_INTEGRATION_DESIGN`

If all integrity gates pass but any scientific gate fails:
`R27D_STRICT_PRIOR_YACOE_RESIDUAL_MIXED_OR_FAIL_NO_INTEGRATION`

If any integrity gate fails:
`R27D_MECHANICAL_OR_INTEGRITY_FAILURE_NO_SCIENTIFIC_DECISION`

A PASS does **not** mutate production. It only authorizes a separately frozen integration/recentering study that must preserve R22 distribution authority and prove compatibility with the one-authoritative-projection stack.

## 14. Anti-overfitting / anti-reinvention rule

After the first valid R27D result, do not:
- retune K values, alpha, ±1.5 cap or application scope on the same test sample;
- drop 2023 or another losing season;
- switch to a winning post-hoc feature subset;
- import raw historical YPR/YPT/YAC persistence from R23/R24/V2;
- cherry-pick individual players/cohorts into production;
- modify R22 to make a point-mean candidate appear safer.

Any future follow-up must be motivated by preserved evidence and newly frozen before execution.
