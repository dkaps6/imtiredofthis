# RB R27B V2 — NOVEL RECEIVING-EFFICIENCY CONTEXT — FROZEN PLAN

**Status:** FROZEN BEFORE IMPLEMENTATION / MODEL FIT / CANDIDATE EXECUTION / RESULTS

## 0. Scientific question

R27 proved that exact R26 vacancy-gated opportunity redistribution contains real receiving-yard signal when production YPT is held fixed, but it was not stable enough to promote because vacancy RB1, p90 and 2023 guardrails failed.

Production already contains empirical-Bayes player YPT and generic football matchup efficiency, and R23 already tested a strict-prior shrunk-YPR persistence candidate. Those avenues are not novel and are excluded from this study.

R27B V2 asks one narrower question:

> **With exact R26 opportunity fixed, can strict-prior target-shape, YAC, QB/team checkdown and RB-specific opponent context explain residual receiving efficiency that is not already represented by production YPT, enough to repair the R27 RB1/p90/2023 weaknesses without sacrificing aggregate accuracy?**

This is a football-context efficiency study. It is not an opportunity study, a generic YPT/YPR persistence study, a tail-model retune, or a sportsbook-assisted model.

---

## 1. Immutable authorities

### Production

Protected production authority remains:

`bb76ba9eabb08e2f0875a9af49301c3877f4141f`

R27B V2 may not change production code/assets, R22, R26, R26Q, P3, QB/WR/TE authorities, or sportsbook routing.

### Exact R27 parent

- run: `34423546037`
- job: `102703879430`
- execution head: `b7cfe5b2c450765208672fc6e2b017d0cb0c0dde`
- artifact: `10132290573`
- artifact name: `rb-r27-receiving-yard-mean-decomposition-v1`
- digest: `sha256:cd5c9e26efa44c47ef3374e4deefe9038dc504c155799b48548ad64746e10e45`
- disposition: `R27_R26_OPPORTUNITY_TRANSLATION_MIXED_OR_FAIL`
- gates: `24/27 PASS`
- result-record branch parent commit: `886c8432ff811882e006d84a61385862e3be7839`

Material R27 evidence defining this study:
- VACANCY_ACTIVE receiving-yard MAE: `-1.0967854669869492%` vs original baseline
- VACANCY_ACTIVE RMSE: `-1.3337252245802733%`
- target MAE: `-2.0869346310231074%`
- reception MAE: `-1.8026686460321795%`
- ALL-RB MAE: `-0.23517744112927508%`
- WEEK1 MAE: `-3.3081859534149105%`
- failed p90 gate: `+2.077705464816737%`
- failed 2023 vacancy MAE gate: `+4.915812026979438%`
- failed vacancy RB1 MAE gate: `+2.820339661055238%`
- vacancy RB2+ MAE: `-3.3577795492306994%`

No threshold may be changed to make those historical R27 results qualify.

---

## 2. Anti-reinvention authority

The following are controlling evidence and must be respected:

- `scripts/modeling/bayesian_v2.py` already supplies empirical-Bayes `bayes_ypt` from player/position history.
- `scripts/modeling/simulation_rules.py` already turns that football state into `rules_ypt` with existing matchup/pass-efficiency logic.
- R23 already tested a frozen strict-prior 6-game/16-game shrunk-YPR efficiency component and failed receiving-yard robustness, especially RB1/p90.
- R24 removed that failed new efficiency component and paired improved opportunity with unchanged production efficiency.
- `scripts/backtest/decompose_receiving_error.py` already diagnosed YPT as a receiving-error component.
- R19/R22 already own mean-neutral receiving-yard tail/distribution behavior.

Therefore the primary V2 candidate may NOT use career/recent/trailing YPT, career/recent/trailing YPR, generic catch-rate persistence, or a new generic empirical-Bayes YPT/YPR estimate as its claimed incremental signal.

The exact novelty boundary is recorded in:

`docs/research/RB_R27B_V2_NOVELTY_BOUNDARY_AUDIT.md`

R27B V1 remains separately preserved as superseded before execution on branch `research-rb-r27b-receiving-efficiency-v1`; it has no workflow run or scientific result.

---

## 3. Pre-plan source audit authority

Before this plan was frozen, a source/schema-only audit verified the proposed novel football inputs across 2019–2025 without fitting a model or scoring prediction error.

- source-audit run: `34427150810`
- source-audit job: `102714703407`
- source-audit head: `fd347b8f743c3829049a8052fcd0bc5d1ba72222`
- source-audit artifact: `10133096990`
- source-audit digest: `sha256:a9905a02243af2120cd71e85bcd460d47c5ccc73af87779adde4dd5c20a8263b`
- source-audit script SHA256: `c9a2921d3d75c02048a46de1ed0e59b08c5120f1e4d94b458cf5f1ca8f10886c`
- schema pass: all 2019–2025
- RB target rows present: all 2019–2025
- receiver position resolution: approximately 99.97%–99.99%
- RB air-yards non-null: approximately 99.63%–99.77%
- RB YAC non-null on completed receptions: 100% each season
- RB yards non-null: 100% each season
- team-week checkdown denominator finite: 100% each season
- opponent RB context derivable: yes each season
- sportsbook inputs used: 0
- model fit performed: false
- candidate projection created: false
- prediction error scored: false

These source results justify feature availability only. They are not evidence that any feature is predictive.

---

## 4. Exact opportunity scope

R26/R27 opportunity is immutable.

For every outer-test player-game:
- `candidate_targets` must equal the exact R26/R27 candidate targets;
- `candidate_receptions` must equal the exact R26/R27 candidate receptions;
- `VACANCY_ACTIVE = room_exits_n >= 1` remains exact;
- R8/R9 identity, R9 reliability, vacancy logic, room conservation and non-RB entitlement remain unchanged;
- no 2020, 2023, RB1, RB2+, Week1 or player-specific opportunity router is allowed.

### Application scope

The V2 efficiency correction is applied **only to R26 vacancy-active RB/FB player-games**. This is predeclared because V2 is explicitly the downstream translation study for the mechanism whose opportunity changes only vacancy rooms.

For stable/no-vacancy player-games, V2 receiving-yard mean must remain exactly the current production-efficiency path:

`candidate_ypt = production_ypt`

This is not a post-result router; it is the exact mechanism scope frozen before V2 results.

---

## 5. Baselines and candidate

Carry three point means.

### B0 — original production opportunity + production efficiency

`B0_rec_yards = baseline_targets × production_ypt`

### B1 — exact R27 / R26 opportunity + production efficiency

`B1_rec_yards = candidate_targets × production_ypt`

B1 is the primary incremental comparator.

### C1 — exact R26 opportunity + novel contextual efficiency residual

For vacancy-active rows only:

`context_residual_hat = Ridge(novel_context_features)`

`C1_ypt = max(production_ypt + clip(context_residual_hat, -1.5, +1.5), 0)`

`C1_rec_yards = candidate_targets × C1_ypt`

For stable/no-vacancy rows:

`C1_ypt = production_ypt`

`C1_rec_yards = B1_rec_yards = B0_rec_yards`

The fixed ±1.5 YPT correction cap is frozen before results to keep the new context layer subordinate to the established production efficiency state and prevent sparse target-shape context from manufacturing extreme means.

### Reception bridge audit

Using the unchanged production catch-rate path:

`implied_C1_ypr = C1_ypt / production_catch_rate`

`bridge_rec_yards = candidate_receptions × implied_C1_ypr`

The bridge must equal `C1_rec_yards` within `<1e-10` under the exact R26 reception identity.

---

## 6. Learning target and training chronology

The model predicts only residual efficiency beyond the production YPT state.

For an eligible historical training player-game with at least one actual target:

`actual_ypt = actual_rec_yards / actual_targets`

`target = actual_ypt - production_ypt_as_of_that_game`

Training labels are permitted only after all features for that historical row have been materialized strictly as-of its pregame cutoff.

### Outer folds

Test REG seasons:
- 2020
- 2021
- 2022
- 2023
- 2024
- 2025

For outer test season `S`, model fitting/selection may use only completed REG observations from seasons `< S`.

Within every training observation, novel PBP features are lagged strictly before that observation's game. No target-game PBP contributes to its own features.

All six outer seasons are retained regardless of outcome. 2020 and 2023 may not be excluded, downweighted after results, or routed differently.

### Training population

Use all eligible historical RB/FB player-games with valid pregame feature construction and actual targets >=1. The model learns general RB receiving-context residual relationships, but its candidate correction is applied only in the predeclared vacancy-active outer-test scope.

Training sample weight:

`clip(actual_targets, 1, 8)`

This weight uses only completed historical training labels, never outer-test outcomes.

---

## 7. Frozen primary feature family — NOVEL CONTEXT ONLY

The primary Ridge feature vector is limited to the following strict-prior football features.

### Player target-shape / YAC style

1. `player_air_yards_per_target_prior`
2. `player_yac_per_reception_prior`
3. `player_screen_target_rate_prior` where screen/behind-LOS means `air_yards <= 0`
4. `player_explosive20_target_rate_prior` where target-play yards gained >=20

### Team/QB RB receiving environment

5. `team_rb_targets_per_official_pass_attempt_prior`
6. `team_rb_air_yards_per_target_prior`
7. `team_rb_yac_per_reception_prior`
8. `team_rb_screen_target_rate_prior`

### Opponent RB-specific receiving environment

9. `opp_rb_air_yards_allowed_per_target_prior`
10. `opp_rb_yac_allowed_per_reception_prior`
11. `opp_rb_catch_rate_allowed_prior`
12. `opp_rb_explosive20_allowed_per_target_prior`
13. `opp_rb_screen_target_rate_faced_prior`

### Structural controls

14. `role_is_rb1`
15. `role_is_rb2plus`
16. `vacancy_incumbent`
17. `vacancy_new_veteran`
18. `vacancy_no_prior_nfl`
19. `week1`

`production_ypt` is the offset/baseline being corrected and is **not** an eligible primary predictor. Generic player YPT/YPR/catch-rate history is forbidden as a V2 primary feature.

No sportsbook variable, market line, price, consensus projection, target-game outcome, R19 tail probability, R22 residual state, or postgame diagnostic may enter C1.

---

## 8. Frozen as-of aggregation / shrinkage

PBP-derived rate features must not use noisy raw tiny samples directly.

For each rate-like feature, use all strict-prior available observations with empirical shrinkage toward the relevant football population prior computed from observations strictly before that cutoff:

`shrunk = (n * raw + K * prior) / (n + K)`

Frozen pseudo-counts:
- air yards / target: `K=20 targets`
- screen rate: `K=20 targets`
- explosive20 rate: `K=20 targets`
- YAC / reception: `K=12 receptions`
- catch rate allowed: `K=20 targets`
- team RB targets / official pass attempt: `K=40 team pass attempts`

Player features shrink toward RB/FB population priors. Team features shrink toward league team-RB priors. Opponent features shrink toward league defense-vs-RB priors.

No trailing-window player YPT/YPR feature is allowed. The new information is target shape/context, not restated raw efficiency history.

Missing values after legal history lookup are filled with the outer-training-fold median, with explicit missing indicators. Outer-test rows never affect imputation.

---

## 9. Frozen estimator / selection

Primary estimator:

`StandardScaler + weighted Ridge`

Frozen alpha grid:

`[1.0, 10.0, 100.0, 1000.0]`

For outer test seasons with at least two legal inner validation seasons, choose alpha using rolling-origin inner season validation only:
- inner validation season `V` is predicted using seasons `<V` only;
- objective = target-weighted YPT residual MAE;
- ties choose the larger alpha.

For outer test 2020, where only 2019 is legal history, use frozen fallback `alpha=100.0`.

No tree/boosting/neural candidate, feature selection by outer-test performance, cohort-specific fitted model, or hand override is allowed in V2.

### Diagnostic ablations

The evaluator may report, for explanation only:
- player-shape-only prediction;
- team/QB-environment-only prediction;
- opponent-context-only prediction.

These ablations are **not alternate promotion candidates**, may not replace C1 after results, and may not be used to choose a subset. Any future subset candidate requires a new frozen study.

---

## 10. Frozen cohorts / reporting

Report B0, B1 and C1 for:
- ALL RB/FB
- VACANCY_ACTIVE
- VACANCY_INCUMBENT
- VACANCY_RB1_INCUMBENT
- VACANCY_RB2PLUS_INCUMBENT
- VACANCY_NEW_VETERAN
- VACANCY_NO_PRIOR_NFL
- WEEK1
- WEEKS2PLUS
- each season 2020–2025

Also report C1 correction magnitude and novel-source evidence volume by role/history bucket. These diagnostics may not become post-result routers.

Primary metric: receiving-yard MAE.

Also report:
- RMSE
- signed bias / absolute bias
- median AE
- p75 AE
- p90 AE
- >=30-yard AE miss rate
- Pearson / Spearman
- YPT MAE on actual-target>0 rows
- mean/median absolute correction
- correction-cap hit rate
- missing rate for each novel feature
- B1 target/reception parity versus exact R27 parent

---

## 11. Frozen structural/integrity gates 1–15

All mandatory.

1. sportsbook inputs upstream = 0
2. target-game/future outcomes used in features = 0
3. exact R26/R27 parent opportunity identity verified
4. B1 candidate targets equal exact R27 parent within `<1e-12`
5. C1 candidate targets equal B1 within `<1e-12`
6. C1 candidate receptions equal B1 within `<1e-12`
7. vacancy definition remains exact `room_exits_n >= 1`
8. stable/no-vacancy C1 YPT and receiving-yard mean equal production/B1 exactly within `<1e-12`
9. RB-room opportunity mass conservation `<1e-10`
10. non-RB entitlement unchanged `<1e-12`
11. R22 untouched and not used as mean input
12. production files/assets unchanged from protected authority
13. all novel PBP/context features use strictly prior observations only
14. scaler/imputer/Ridge/alpha selection use legal outer-training rows only
15. all six 2020–2025 REG folds complete and reception bridge max gap `<1e-10`

Integrity failures are not scientific evidence for or against the candidate.

---

## 12. Frozen scientific gates 16–31

Percent change for error metrics is `(C1 - comparator) / comparator * 100`; negative is improvement.

16. pooled VACANCY_ACTIVE receiving-yard MAE improves at least **0.50% vs B1**
17. pooled VACANCY_ACTIVE receiving-yard MAE improves at least **1.25% vs B0**
18. pooled VACANCY_INCUMBENT receiving-yard MAE improves at least **0.25% vs B1**
19. VACANCY_RB1_INCUMBENT receiving-yard MAE improves at least **1.50% vs B1**
20. VACANCY_RB1_INCUMBENT receiving-yard MAE is **non-worse vs B0**
21. VACANCY_RB2PLUS_INCUMBENT receiving-yard MAE worsens no more than **0.75% vs B1**
22. pooled VACANCY_ACTIVE RMSE is **non-worse vs B1**
23. pooled VACANCY_ACTIVE p90 absolute error is **non-worse vs B1**
24. pooled VACANCY_ACTIVE >=30-yard miss rate is **non-worse vs B1**
25. pooled VACANCY_ACTIVE absolute bias worsens no more than **0.25 yards vs B1**
26. 2023 VACANCY_ACTIVE receiving-yard MAE improves at least **2.00% vs B1**
27. 2023 VACANCY_ACTIVE receiving-yard MAE is **non-worse vs B0**
28. at least **4 of 6** seasons improve VACANCY_ACTIVE MAE vs B1
29. no season worsens VACANCY_ACTIVE MAE more than **2.00% vs B1**
30. ALL-RB receiving-yard MAE is **non-worse vs B1** and no worse than **+0.10% vs B0**
31. WEEK1 receiving-yard MAE worsens no more than **0.50% vs B1**

The RB1 and 2023 gates are intentionally direct repairs of the exact R27 weaknesses. A candidate that improves pooled MAE while leaving those failures intact does not qualify.

---

## 13. Frozen dispositions

If all Gates 1–31 pass:

`R27B_V2_NOVEL_EFFICIENCY_CONTEXT_SUPPORT_READY_FOR_INTEGRATION_DESIGN`

If integrity Gates 1–15 pass but any scientific gate fails:

`R27B_V2_NOVEL_EFFICIENCY_CONTEXT_MIXED_OR_FAIL_NO_INTEGRATION`

If any integrity gate fails:

`R27B_V2_MECHANICAL_OR_INTEGRITY_FAILURE_NO_SCIENTIFIC_DECISION`

A scientific PASS does not modify production. It authorizes only a separately frozen integration/recentering qualification beneath unchanged R22.

---

## 14. First-result lock

Before any candidate execution:
1. implement the feature builder/evaluator/finalizer;
2. create an implementation lock pinning exact plan/code/parent hashes;
3. create a dedicated research workflow;
4. verify protected production boundary;
5. run once and preserve the first valid result exactly.

After the first valid result, no candidate feature, pseudo-count, alpha grid, correction cap, cohort, threshold, comparator, season, application scope or gate may be changed. Any new hypothesis requires a new frozen migration.

R26 and R22 remain untouched throughout V2.