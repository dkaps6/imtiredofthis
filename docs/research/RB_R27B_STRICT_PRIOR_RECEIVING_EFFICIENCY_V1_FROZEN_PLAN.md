# RB R27B Strict-Prior Receiving Efficiency V1 — FROZEN PLAN

**Status:** FROZEN BEFORE IMPLEMENTATION / CANDIDATE EXECUTION / RESULTS

## 0. Purpose

R27 V1 established that exact R26 vacancy-gated opportunity redistribution has real receiving-yard support when production YPT is held fixed, but the translation is not sufficiently stable for mean integration. The first valid R27 result passed 24/27 gates and materially improved pooled vacancy receiving-yard MAE, targets, receptions, RMSE and Week 1, while failing the p90 guardrail, the 2023 season guardrail, and the vacancy-RB1 guardrail.

R27B V1 asks the next isolated question:

> **With exact R26 opportunity held fixed, can genuinely strict-prior receiving-efficiency information improve RB receiving-yard point means beyond the existing production YPT without sacrificing role, season, tail, or all-RB stability?**

This is an efficiency study. It is not another opportunity/entitlement study and it may not alter R26.

---

## 1. Immutable parent authority

### Production authority

Production remains:

`bb76ba9eabb08e2f0875a9af49301c3877f4141f`

R27B may not change production code, production assets, R22, R26 production behavior, sportsbook routing, or the sealed R26Q prospective record.

### Exact R27 scientific parent

- R27 first valid run: `34423546037`
- R27 job: `102703879430`
- R27 execution head: `b7cfe5b2c450765208672fc6e2b017d0cb0c0dde`
- R27 artifact: `10132290573`
- R27 artifact name: `rb-r27-receiving-yard-mean-decomposition-v1`
- R27 digest: `sha256:cd5c9e26efa44c47ef3374e4deefe9038dc504c155799b48548ad64746e10e45`
- R27 disposition: `R27_R26_OPPORTUNITY_TRANSLATION_MIXED_OR_FAIL`
- R27 gates: 24/27 PASS
- exact result record parent commit on this branch: `886c8432ff811882e006d84a61385862e3be7839`

R27 material result to preserve:
- VACANCY_ACTIVE receiving-yard MAE: `-1.0967854669869492%`
- VACANCY_ACTIVE RMSE: `-1.3337252245802733%`
- target MAE: `-2.0869346310231074%`
- reception MAE: `-1.8026686460321795%`
- ALL-RB MAE: `-0.23517744112927508%`
- WEEK1 MAE: `-3.3081859534149105%`
- failed Gate17 p90: `+2.077705464816737%`
- failed Gate19 2023 vacancy MAE: `+4.915812026979438%`
- failed Gate20 vacancy RB1 MAE: `+2.820339661055238%`
- vacancy RB2+ MAE improved `-3.3577795492306994%`

These numbers are evidence defining the R27B question. They are not thresholds to be tuned against.

---

## 2. Non-negotiable R26 opportunity lock

For every test player-game, R27B must consume the exact R26/R27 opportunity outputs without modification:

- `candidate_targets` = exact historical R26 candidate targets
- `candidate_receptions` = exact historical R26 candidate receptions
- vacancy definition remains `VACANCY_ACTIVE = room_exits_n >= 1`
- stable/no-vacancy target opportunity remains exact R26/R27 parent behavior
- RB-room target-mass conservation remains exact
- non-RB entitlement remains unchanged
- R8/R9 receiving identity mechanism remains unchanged
- R9 reliability formula remains unchanged
- no new router may be introduced for 2020, 2023, RB1, RB2+, Week 1, or any other observed result cohort

R27B is prohibited from changing candidate targets or candidate receptions to make receiving-yard results improve.

---

## 3. Frozen outer evaluation folds

Evaluate REG seasons **2020–2025** with strict walk-forward chronology.

Outer test seasons:
- 2020
- 2021
- 2022
- 2023
- 2024
- 2025

All six seasons remain in the final result regardless of performance. 2020 and 2023 may not be removed, downweighted after results, or routed to a different candidate.

Efficiency training pool for test season `S`:

> all eligible completed REG player-games from seasons `2019..S-1`, with every feature for each training observation built as-of that observation's pregame cutoff.

For a target week inside a season, same-season lagged features may use completed earlier weeks only. No target-week or future-week football outcomes may enter a feature.

---

## 4. Frozen baselines and primary candidate

Three means are carried for every player-game.

### B0 — original production-path baseline

`b0_rec_yards = baseline_targets × production_ypt`

This is the R27 production baseline.

### B1 — fixed-R26-opportunity baseline

`b1_rec_yards = candidate_targets × production_ypt`

This is the exact R27 V1 candidate and is the **primary incremental comparator** for R27B efficiency.

### C1 — R27B strict-prior efficiency candidate

`r27b_candidate_ypt = max(production_ypt + clipped_efficiency_residual_hat, 0)`

`c1_rec_yards = candidate_targets × r27b_candidate_ypt`

where:

`clipped_efficiency_residual_hat = clip(efficiency_residual_hat, -2.0, +2.0)`

The fixed ±2.0 YPT correction cap is frozen before results to prevent a fitted efficiency layer from manufacturing extreme means through sparse/noisy player history.

The candidate changes **only YPT efficiency**. R26 candidate targets and receptions remain bit-for-bit/numerically identical to the parent outputs.

### Diagnostic reception bridge

For diagnostics only:

`r27b_implied_ypr = r27b_candidate_ypt / production_catch_rate`

`r27b_bridge_rec_yards = candidate_receptions × r27b_implied_ypr`

The bridge must equal `c1_rec_yards` within `<1e-10`, subject only to floating-point tolerance. This confirms that the new layer is an efficiency correction on the same frozen opportunity chain rather than a hidden reception change.

---

## 5. Frozen learning target

For eligible historical training player-games with at least one actual target:

`actual_ypt = actual_rec_yards / actual_targets`

`efficiency_residual_target = actual_ypt - production_ypt_as_of_that_game`

The fitted model predicts only this residual.

Training sample weight:

`sample_weight = clip(actual_targets, 1, 8)`

This is allowed only on completed historical training rows. Test-season actual targets, receptions, receiving yards, YPT or YPR may never be used in candidate construction, sample weighting, feature imputation, model fitting, model selection, or routing.

---

## 6. Frozen model family

Primary estimator:

**standardized weighted Ridge regression**.

No tree ensemble, boosting model, neural model, post-result feature selection, cohort-specific fit, or hand-tuned player override is authorized in R27B V1.

### Preprocessing

For each outer fold:
- numeric missing values: training-fold median only;
- explicit missing indicators retained for efficiency-history/PBP features where absence is meaningful;
- numeric standardization: training-fold mean/std only;
- categorical role flags: frozen binary indicators listed below;
- constant/zero-variance columns may be removed mechanically based only on the training fold;
- test data may never affect imputation/scaling.

### Ridge alpha

Candidate alpha grid is frozen:

`[0.1, 1.0, 10.0, 100.0, 1000.0]`

When at least two historical seasons are available inside an outer training pool, choose alpha using **inner rolling-origin season validation only**. For each inner validation season `V`, fit only on seasons `<V`, score on `V`, and minimize target-weighted YPT MAE. Ties choose the **larger alpha**.

For outer test 2020, where the frozen training pool contains only 2019 and therefore no valid inner rolling-origin season split exists, use the predeclared fallback:

`alpha = 100.0`

After alpha is selected, refit on the full legal outer training pool and predict the untouched outer test season.

No outer-test receiving-yard metric may participate in alpha selection.

---

## 7. Frozen feature family

Every feature must be available strictly pregame and reproducible from the historical nflverse/nflreadpy football sources already used by the repository or from deterministic R26/R27 parent outputs. Sportsbook data is forbidden.

### A. Existing production-efficiency state

1. `production_ypt`
2. `production_catch_rate`
3. `implied_production_ypr = production_ypt / production_catch_rate`

### B. Strict-prior player receiving persistence

4. empirical-Bayes/shrunk career-to-date YPT
5. empirical-Bayes/shrunk current-season-to-date YPT
6. empirical-Bayes/shrunk trailing-4-game YPT
7. empirical-Bayes/shrunk trailing-8-game YPT
8. empirical-Bayes/shrunk career-to-date catch rate
9. empirical-Bayes/shrunk career-to-date YPR
10. strict-prior targets per game
11. strict-prior receptions per game
12. strict-prior cumulative target count / evidence volume

All persistence features must use only games completed before the prediction row.

### C. Strict-prior player receiving shape from PBP

13. average air yards per target
14. yards after catch per reception
15. screen/behind-LOS target rate (`air_yards <= 0`)
16. explosive receiving target rate (`receiving target play yards >= 20`)

These features are lagged/as-of only. If PBP player identity cannot be resolved pregame, use the training-only imputation path plus missing indicator; do not use target-game PBP to repair it.

### D. Strict-prior team/QB receiving environment

17. offense RB target share among team targets
18. offense RB receiving yards per target
19. offense RB yards after catch per reception
20. QB/team RB checkdown proxy = RB targets / team official pass attempts

These are rolling/as-of features calculated from completed games only.

### E. Strict-prior opponent RB receiving environment

21. opponent RB receiving yards allowed per target
22. opponent RB YAC allowed per reception
23. opponent RB explosive-20 receiving rate allowed per target
24. opponent RB catch rate allowed

Opponent values are lagged before the target game; the target game itself may not contribute.

### F. Frozen structural/role context

25. `role_is_rb1`
26. `role_is_rb2plus`
27. `vacancy_active`
28. `vacancy_incumbent`
29. `vacancy_new_veteran`
30. `vacancy_no_prior_nfl`
31. `week1`

No feature may be added after the first candidate result is seen. A feature may be removed before first execution only if a documented source/schema audit proves it is mechanically unavailable for the historical window; such a removal must be frozen in a repair note before candidate execution and may not be based on predictive performance.

---

## 8. Shrinkage rules for strict-prior persistence features

To avoid noisy raw YPT/YPR from tiny samples, player rate features must be empirically shrunk using only the legal training/as-of population.

For a player rate with opportunity count `n` and raw rate `r`:

`shrunk_rate = (n * r + K * prior_rate) / (n + K)`

Frozen pseudo-counts:
- YPT: `K = 20 targets`
- catch rate: `K = 20 targets`
- YPR: `K = 12 receptions`
- PBP air-yards-per-target: `K = 20 targets`
- YAC-per-reception: `K = 12 receptions`
- explosive rate: `K = 20 targets`
- screen rate: `K = 20 targets`

`prior_rate` is the position-family RB/FB rate computed from football observations strictly before the prediction cutoff. If no legal position-family history exists, fall back to the corresponding training-pool rate; never use test outcomes.

Trailing-4 and trailing-8 features use the same pseudo-count rule after limiting the raw observation window.

---

## 9. Frozen cohorts

Report B0, B1 and C1 for:

- ALL
- VACANCY_ACTIVE
- VACANCY_INCUMBENT
- VACANCY_RB1_INCUMBENT
- VACANCY_RB2PLUS_INCUMBENT
- VACANCY_NEW_VETERAN
- VACANCY_NO_PRIOR_NFL
- WEEK1
- WEEKS2PLUS
- each season 2020–2025

Also report evidence-volume diagnostics for:
- prior targets = 0
- prior targets 1–19
- prior targets 20–49
- prior targets 50+

Evidence-volume cohorts are diagnostic only and may not be used as post-result routers.

---

## 10. Frozen metrics

Primary metric: receiving-yard MAE.

Also report:
- RMSE
- bias
- absolute bias
- median absolute error
- p75 absolute error
- p90 absolute error
- Pearson
- Spearman
- >=30-yard absolute-error miss rate
- target MAE (B0/B1 chain diagnostic; C1 does not alter targets)
- reception MAE (B0/B1 chain diagnostic; C1 does not alter receptions)
- candidate YPT MAE on rows with actual targets >0
- mean/median absolute YPT correction
- correction-cap hit rate
- missing/imputation rate by feature

All comparisons must include both C1 vs B1 (incremental efficiency value) and C1 vs B0 (whole-path value vs original production-path baseline).

---

## 11. Frozen structural/leakage gates 1–14

All are mandatory.

1. sportsbook inputs upstream = 0
2. future/test outcomes used in features = 0
3. exact R26 opportunity parent identity/hashes verified
4. C1 `candidate_targets` exactly equal B1/R27 parent candidate targets within `<1e-12`
5. C1 `candidate_receptions` exactly equal B1/R27 parent candidate receptions within `<1e-12`
6. R26 vacancy definition and stable-room behavior unchanged
7. RB-room opportunity mass conservation remains `<1e-10`
8. non-RB entitlement unchanged within `<1e-12`
9. production files/assets unchanged from protected authority
10. R22 untouched and not used as an upstream mean correction
11. every training/as-of feature cutoff is strictly before its prediction game; target-game PBP/stat rows in features = 0
12. imputer/scaler/Ridge fit only on legal outer-training rows; outer-test rows used in fit/selection = 0
13. reception bridge identity maximum absolute gap `<1e-10`
14. all six 2020–2025 REG outer folds complete, finite, and retained

A failure in Gates 1–14 is structural/mechanical/integrity failure, not scientific evidence for or against the candidate.

---

## 12. Frozen scientific gates 15–30

Percent changes use `(C1_metric - comparator_metric) / comparator_metric * 100`; negative is improvement for error metrics.

15. pooled VACANCY_ACTIVE receiving-yard MAE improves **at least 0.50% vs B1**
16. pooled VACANCY_ACTIVE receiving-yard MAE improves **at least 1.00% vs B0**
17. pooled VACANCY_INCUMBENT receiving-yard MAE improves **at least 0.25% vs B1**
18. VACANCY_RB1_INCUMBENT receiving-yard MAE improves **at least 1.00% vs B1**
19. VACANCY_RB1_INCUMBENT receiving-yard MAE is **non-worse vs B0** (<= `+0.00%`)
20. VACANCY_RB2PLUS_INCUMBENT receiving-yard MAE worsens **no more than 1.00% vs B1**
21. ALL-RB receiving-yard MAE improves **at least 0.25% vs B1**
22. ALL-RB receiving-yard MAE improves **at least 0.25% vs B0**
23. ALL-RB RMSE worsens **no more than 0.10% vs B0**
24. pooled VACANCY_ACTIVE RMSE is **non-worse vs B1**
25. pooled VACANCY_ACTIVE absolute bias is **non-worse vs B1**
26. pooled VACANCY_ACTIVE p90 absolute error worsens **no more than 0.25% vs B1**
27. pooled VACANCY_ACTIVE >=30-yard miss rate is **non-worse vs B1**
28. at least **4 of 6 seasons** improve VACANCY_ACTIVE receiving-yard MAE vs B1
29. **no season** worsens VACANCY_ACTIVE receiving-yard MAE by more than **2.50% vs B1**
30. WEEK1 receiving-yard MAE worsens **no more than 0.50% vs B1**

The RB1 gates are intentionally explicit because R27 V1 improved RB1 targets/receptions but worsened RB1 receiving-yard MAE. R27B is not considered successful if it wins only by helping RB2+ while leaving the identified RB1 translation problem unresolved.

---

## 13. Frozen dispositions

### Full scientific qualification

If Gates 1–30 all pass:

`R27B_STRICT_PRIOR_RECEIVING_EFFICIENCY_SUPPORT_READY_FOR_INTEGRATION_DESIGN`

This authorizes only a separately frozen integration/recentering design. It does **not** mutate production.

### Signal present but unstable

If structural Gates 1–14 pass and Gates 15, 18 and 21 pass, but one or more other scientific safety gates fail:

`R27B_RECEIVING_EFFICIENCY_SIGNAL_PRESENT_BUT_NOT_STABLE`

Preserve as evidence. Do not lower gates or promote.

### No incremental support

If structural Gates 1–14 pass but any of Gates 15, 18 or 21 fail:

`R27B_NO_INCREMENTAL_RECEIVING_EFFICIENCY_SUPPORT`

Preserve and close V1 without retuning.

### Structural/mechanical failure

If any Gate 1–14 fails before a scientifically valid candidate result exists:

`R27B_MECHANICAL_OR_INTEGRITY_FAILURE_NO_SCIENTIFIC_DECISION`

Only the minimum documented value-neutral repair is authorized. Candidate formulas, features, pseudo-counts, caps, folds and scientific thresholds remain frozen.

---

## 14. Diagnostic-only persistence ablation

A secondary **diagnostic-only** fit may be emitted using only feature groups A+B+F (production state + player persistence + role context). It must use the identical target, preprocessing, alpha-selection logic and outer folds.

Name:

`R27B_PERSISTENCE_ONLY_DIAGNOSTIC`

This ablation has **no promotion authority**. Its purpose is to reveal whether any C1 improvement comes mostly from player persistence or from the richer PBP/team/opponent context. A better diagnostic ablation result may motivate a future separately frozen study but may not replace C1 after results.

---

## 15. Prohibited actions after first execution

After the first scientifically valid R27B candidate result, do not:

- change the ±2.0 YPT correction cap;
- change Ridge family or alpha grid;
- add/remove features because of predictive results;
- alter pseudo-counts;
- alter 2020 or 2023 handling;
- create an RB1/RB2+ router;
- alter R26 target/reception opportunity;
- alter cohorts or thresholds;
- remove ugly seasons/subgroups;
- use sportsbook lines as model inputs;
- use target-game actual efficiency in features;
- touch R22 or production;
- promote a diagnostic ablation because it happens to outperform C1.

Any follow-up scientific candidate must receive a new study label and a new frozen plan.

---

## 16. Implementation order

1. verify exact R27/R26 parent hashes and protected production boundary;
2. build/reuse exact 2019–2025 REG historical football bundles with canonical REG schedule scope;
3. build a strict-as-of RB efficiency feature table and source-availability audit;
4. reproduce exact R26 opportunity folds for 2020–2025;
5. construct legal R27B training rows separately from untouched outer-test rows;
6. fit inner rolling-origin Ridge alpha using training history only;
7. create B0/B1/C1 predictions before joining target-season outcomes for scoring;
8. run all structural/leakage audits;
9. create an implementation lock **before first candidate execution/results**;
10. execute all outer folds;
11. apply all 30 frozen gates;
12. preserve exact disposition, metrics, predictions, feature audit, model coefficients/alpha per fold, hashes and artifact digest;
13. update `CURRENT_NFL_RESEARCH_HANDOFF.md` on `main` at the material checkpoint;
14. only after the result is recorded decide whether a separate integration study, follow-up efficiency study, or closure of the RB receiving-mean lane is scientifically justified.

---

## 17. One-sentence authority

**R27B V1 tests one thing only: whether a frozen, strict-prior, regularized receiving-efficiency residual model can improve receiving-yard means on top of exact fixed R26 opportunity across 2020–2025 without sacrificing the RB1 cohort, tails, seasons, or all-RB stability; R26, R22, production and sportsbook separation remain immutable.**
