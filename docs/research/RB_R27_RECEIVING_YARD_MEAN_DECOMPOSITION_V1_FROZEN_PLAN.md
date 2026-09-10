# RB R27 — Receiving-Yard Mean Decomposition V1

Status: **FROZEN BEFORE IMPLEMENTATION / CANDIDATE EXECUTION**  
Date: 2026-09-09  
Production parent: `main@bb76ba9eabb08e2f0875a9af49301c3877f4141f`  
Handoff parent: `d63ba0216d43e4763954e0be7351ece06f8fa6b4`  
Research branch: `research-rb-r27-receiving-yard-mean-decomposition-v1`

## 1. Scientific question

R26 has now been promoted as the Week-1 RB receptions/opportunity refinement. R22 remains the promoted RB receiving-yard tail/distribution layer but is explicitly **mean-preserving**.

The next question in the user's original hierarchy is therefore:

> **Does the exact historical R26 opportunity redistribution improve player-level RB receiving-yard point means when the existing pregame receiving-efficiency estimate is held fixed? If not, where does the remaining receiving-yard error live, and what efficiency work is actually justified next?**

This V1 is intentionally a **decomposition / translation test**, not a kitchen-sink efficiency model.

## 2. Why this is scientifically new

R23/R24 already showed that a broader receiving-entitlement improvement could improve targets/receptions, bias and RMSE while still failing receiving-yard MAE, especially for RB1 and p90 error. R24 specifically tested improved R23 entitlement with existing production efficiency and did not qualify receiving-yard mean promotion.

R26 is materially different from R23/R24:

- it is vacancy-gated rather than globally applied;
- it preserves the existing RB-room receiving-opportunity pool exactly;
- it uses the R8/R9 strict-prior player receiving-identity residual;
- it changes only rooms with leakage-safe vacated competition;
- it leaves stable rooms at baseline exactly;
- it was subsequently qualified and promoted for receptions.

Therefore R27 must test the **exact R26 mechanism**, not infer its receiving-yard effect from R23/R24.

## 3. Frozen historical R26 parent mechanism

Pin the historical mechanism to:

- branch: `research-rb-r26-role-transition-entitlement-v1`
- frozen plan blob SHA: `e8a2633454dcfb471bda57fa43e40fbb33bd7697`
- evaluator blob SHA: `17170baa072bcc3157dee03c7a0de75297060e71`

Exact R26 mechanism:

1. Build target-week RB/FB room from canonical historical roster state.
2. `VACANCY_ACTIVE = room_exits_n >= 1`.
3. Stable/no-vacancy rooms remain production baseline exactly.
4. Vacancy rooms preserve exact production RB-room target mass.
5. Attach strict-prior R8/R9 receiving identity.
6. Train R8 Ridge residual model on prior-season training data only.
7. Estimate R9 reliability from rolling-origin OOF training predictions only, clipped `[0,1]`.
8. Compute `log(production_within_rb_share + EPS) + reliability * raw_R8_residual`.
9. Softmax only inside the current RB/FB room and rescale to exact production RB-room target mass.
10. Non-RB entitlement remains unchanged.

No R26 formula, vacancy threshold, reliability rule, feature family, training window, clip, or cohort may be altered in R27 V1.

## 4. Historical evaluation seasons / folds

Evaluate REG seasons **2020-2025** exactly, with each test season using the immediately prior season for R26 model fitting as in the frozen R26 evaluator.

Historical outcomes are used **only as labels after pregame predictions are frozen for each fold**. They may never enter features or same-game calculations.

Week range follows the frozen R26 contract:

- 2020: Weeks 1-17
- 2021-2025: Weeks 1-18

No season may be removed after results are inspected, including known difficult 2020 and 2023 slices.

## 5. Population

Primary football population: all canonical historical **RB/FB** rows produced by the exact R26 fold construction.

Predeclared cohorts:

- `ALL`
- `VACANCY_ACTIVE`
- `VACANCY_INCUMBENT`
- `VACANCY_RB1_INCUMBENT`
- `VACANCY_RB2PLUS_INCUMBENT`
- `VACANCY_NEW_VETERAN`
- `VACANCY_NO_PRIOR_NFL`
- `WEEK1`
- `WEEKS2PLUS`
- each season individually 2020-2025

No cohort may be added as a rescue cohort after seeing results. Additional descriptive buckets may be emitted only if clearly labeled exploratory and cannot change disposition.

## 6. Frozen point-mean definitions

For each player-game, use the same pregame production quantities already available in the R26 historical frame:

- `baseline_targets = team_targets * baseline_entitlement_tgt_share`
- `candidate_targets = team_targets * R26_candidate_entitlement_tgt_share`
- `production_ypt = max(rules_ypt, fallback bayes_ypt, 0)` using only target-game pregame context
- `production_catch_rate = clip(rules_catch_rate / bayes_receptions_per_target fallback, 0.35, 0.95)` using the exact existing production fallback hierarchy

### A. Production mean baseline

`baseline_rec_yards = baseline_targets * production_ypt`

### B. R26-opportunity-only mean candidate

`r26_opportunity_rec_yards = candidate_targets * production_ypt`

This is the **primary R27 V1 candidate**.

### C. Reception-path equivalence audit

To make the user's receptions-first interpretation explicit without inventing a new model:

- `baseline_receptions = baseline_targets * production_catch_rate`
- `r26_receptions = candidate_targets * production_catch_rate`
- where catch rate > 0, define `implied_production_ypr = production_ypt / production_catch_rate`
- `r26_reception_bridge_rec_yards = r26_receptions * implied_production_ypr`

This must equal `r26_opportunity_rec_yards` to numerical tolerance. It is an **identity audit**, not a separate candidate.

No actual-game YPR/YPT/catch rate may be used in either baseline or candidate.

## 7. Actual labels

Use canonical historical actual receiving-yard outcomes from the same player-game log authority used by the existing backtest framework (`cp.build_actual_rows(..., market='rec_yards')`).

Actual targets/receptions may be used for **post-prediction decomposition diagnostics only**, never to construct a candidate prediction.

## 8. Frozen metrics

For baseline and R26-opportunity-only candidate, compute:

- N
- receiving-yard MAE (**primary accuracy metric**)
- RMSE
- signed bias
- median absolute error
- p75 absolute error
- p90 absolute error
- Pearson correlation
- Spearman correlation
- `>=30 yard` absolute-error miss rate / large-error rate, using the same definition as prior RB receiving work where available

Also report:

- target MAE
- reception MAE

These are mechanism-chain diagnostics and do not substitute for receiving-yard MAE.

For every cohort and season report absolute values plus:

- candidate minus baseline
- percent MAE change, with negative = improvement.

## 9. Opportunity-versus-efficiency decomposition

R27 V1 must explicitly quantify how much error remains after R26 changes opportunity while efficiency is fixed.

Post-prediction diagnostic columns may include:

- target error
- reception error
- production YPT
- actual YPT (diagnostic only)
- YPT error (diagnostic only)
- actual receiving yards
- baseline receiving-yard error
- candidate receiving-yard error
- whether R26 moved the player toward or away from actual receiving yards
- magnitude of R26 target/reception delta

The purpose is to determine whether the next justified study is:

- no new efficiency work because R26 opportunity alone sufficiently improves mean;
- a targeted efficiency residual model because opportunity improves but residual YPT/YPR error remains systematic;
- or a narrower router/guard because R26 receiving-yard translation is beneficial only in predeclared substructure.

Exploratory diagnostics cannot retrospectively change this V1 candidate or gates.

## 10. Frozen structural / leakage gates

All must pass:

1. sportsbook football inputs upstream = `0`.
2. target/future outcomes used in features/candidate construction = `0`.
3. exact R26 strict-prior fitting/state contract holds.
4. exact R26 vacancy gate is unchanged (`room_exits_n >= 1`).
5. max RB-room target-mass conservation gap `< 1e-10`.
6. max non-RB entitlement delta `< 1e-12`.
7. no protected production file is changed by this research run.
8. R22 assets/logic are not invoked as an upstream mean correction and remain unchanged.
9. stable/non-vacancy R26 opportunity equals baseline exactly.
10. reception-path receiving-yard identity max absolute gap `< 1e-10`.
11. no actual-game efficiency statistic enters baseline or candidate prediction.
12. all 2020-2025 REG folds complete with no post-result season exclusion.

## 11. Frozen scientific support gates for R26 opportunity → receiving-yard mean

This V1 may support a later mean integration only if **all** of the following pass:

13. pooled `VACANCY_ACTIVE` receiving-yard MAE improves (`candidate < baseline`).
14. pooled `VACANCY_INCUMBENT` receiving-yard MAE improves.
15. pooled `VACANCY_ACTIVE` receiving-yard RMSE is non-worse.
16. absolute pooled `VACANCY_ACTIVE` receiving-yard bias improves or is non-worse.
17. pooled `VACANCY_ACTIVE` p90 absolute error may not worsen by more than **2%**.
18. at least **4 of 6** seasons improve `VACANCY_ACTIVE` receiving-yard MAE.
19. no season may worsen `VACANCY_ACTIVE` receiving-yard MAE by more than **3%**.
20. neither `VACANCY_RB1_INCUMBENT` nor `VACANCY_RB2PLUS_INCUMBENT` receiving-yard MAE may worsen by more than **1.5%** pooled.
21. at least one of RB1/RB2+ vacancy-incumbent cohorts must improve receiving-yard MAE.
22. ALL-RB pooled receiving-yard MAE may not worsen by more than **0.25%**.
23. ALL-RB pooled receiving-yard RMSE may not worsen by more than **0.25%**.
24. Week-1 pooled receiving-yard MAE may not worsen by more than **0.50%**.
25. pooled `VACANCY_ACTIVE` target MAE must improve or remain effectively unchanged within **0.10%**.
26. pooled `VACANCY_ACTIVE` reception MAE must improve or remain effectively unchanged within **0.10%**.

These gates intentionally preserve known difficult years rather than tuning around them.

## 12. Material-improvement threshold

For a direct claim that R26 opportunity alone materially improves receiving-yard mean, require in addition:

27. pooled `VACANCY_ACTIVE` receiving-yard MAE improves by at least **0.50%**.

If Gates 1-26 pass but Gate 27 does not, classify the result as **safe directional support but insufficient standalone mean gain**; do not promote a receiving-yard mean change from V1 alone.

## 13. Dispositions / authority ceiling

### `R27_R26_OPPORTUNITY_REC_YARD_MEAN_SUPPORT_READY_FOR_INTEGRATION_DESIGN`

All Gates 1-27 pass.

Meaning: exact R26 opportunity redistribution, with production efficiency held fixed, materially and robustly improves historical receiving-yard point means. This authorizes a **separate frozen production-integration / R22-recentering qualification**, not direct production mutation from this run.

### `R27_R26_OPPORTUNITY_SAFE_BUT_EFFICIENCY_WORK_REQUIRED`

Gates 1-26 pass but Gate 27 fails, or receiving-yard MAE gain is too small for standalone promotion while structure/safety hold.

Meaning: R26 opportunity is safe but does not by itself solve receiving-yard mean. R27B may be frozen to model **strict-prior receiving-efficiency residuals on top of R26 opportunity**.

### `R27_R26_OPPORTUNITY_TRANSLATION_MIXED_OR_FAIL`

Any structural gate fails, or one/more scientific safety/support gates 13-26 fail.

Meaning: do not automatically propagate R26 reception improvements into receiving-yard means. Preserve R26 for receptions, preserve R22 mean/tails, and use the error decomposition to design a narrower efficiency/translation study without tuning this V1 after results.

## 14. R27B authorization if needed

If V1 does not directly qualify a mean change but structural integrity is intact, R27B may test **new efficiency information only**, with R26 opportunity fixed upstream.

Candidate families may include, only if strict-prior historical coverage is verified before fitting:

- shrunk player YPT / YPR persistence;
- recent-versus-long-term receiving efficiency;
- YAC / YACOE;
- aDOT / route type / screen-versus-downfield usage;
- explosive receiving / long-catch propensity;
- RB role / archetype interactions;
- opponent RB receiving efficiency allowed, LB coverage/tackling/YAC environment;
- QB checkdown tendency if independent of already-modeled opportunity;
- pressure/checkdown context only if it adds non-duplicative signal.

R27B must have its own frozen plan before any candidate results are inspected. No raw low-sample YPR multiplier may be deployed without shrinkage/reliability control.

## 15. R22 boundary

R22 is **not modified in R27 V1**.

If and only if a receiving-yard mean candidate later qualifies, a separate integration study must prove that R22 can be recentered around the new mean while preserving its qualified tail behavior, non-RB exactness, receptions authority, and `rush_rec_yards = rush_yards + rec_yards` identity.

## 16. Production boundary

This study makes **zero production changes**.

Current production remains:

- P3 rushing
- R26 receptions
- existing RB receiving-yard mean
- R22 receiving-yard tails/distribution around that mean
- existing QB/WR/TE authorities.

## 17. Expected outputs

The implementation must emit at minimum:

- `r27_player_predictions.csv`
- `r27_metrics_by_cohort.csv`
- `r27_metrics_by_season.csv`
- `r27_opportunity_efficiency_decomposition.csv`
- `r27_structural_audit.csv`
- `r27_gate_matrix.csv`
- `r27_disposition.json`
- exact input/code/provenance hash manifest.

## 18. No-retuning rule

After the first scientific execution:

- do not change vacancy threshold;
- do not change R26 R8/R9 formula/reliability;
- do not drop 2020/2023 or any failing season;
- do not change role definitions;
- do not change MAE/p90/safety thresholds;
- do not introduce an efficiency model into V1;
- do not use sportsbook lines to choose a candidate;
- do not relabel a scientific FAIL as mechanical.

Only documented value-neutral mechanical repairs may be rerun under this V1 contract.
