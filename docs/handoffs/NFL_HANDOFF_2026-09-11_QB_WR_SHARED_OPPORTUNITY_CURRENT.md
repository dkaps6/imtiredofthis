# NFL HANDOFF — 2026-09-11 — QB/WR SHARED OPPORTUNITY CURRENT STOP

**Repository:** `dkaps6/imtiredofthis`  
**GitHub is canonical. Chat memory is secondary.**  
**Main head before this handoff write:** `e7c952a293722082566123d8a1bf3620a592559c` (`Create master NFL cross-chat continuity record`)  
**Current operational production authority:** `3079d8ab0512c5a1304662609e3e880d6846292f`  
**Protected scientific/model authority:** `bb76ba9eabb08e2f0875a9af49301c3877f4141f`

---

## 1. USER / PROJECT OPERATING CONTRACT — PRESERVE

- Predict real football first; sportsbook is downstream benchmark/opponent only.
- One authoritative production projection per player/market.
- Freeze hypotheses, cohorts, gates, routing and source boundaries before results.
- Preserve first valid scientific results, scientific failures, mechanical failures and repair lineage.
- Mechanical failures are not scientific failures; only minimum parity/plumbing repair is allowed without changing frozen science.
- Never rescue a failed hypothesis by post-hoc tuning, threshold shopping, window changes, routers or cherry-picked subgroups.
- Never use target-game outcomes/PBP as pregame features.
- Do not silently mutate production.
- Keep failed/null branches as anti-reinvention evidence rather than deleting them.
- NE-SEA grading remains parked by explicit user instruction.

Canonical architecture remains:

`GAME STATE -> TEAM OPPORTUNITY -> POSITION/ROOM POOL -> PLAYER ENTITLEMENT -> PLAYER EFFICIENCY -> JOINT MONTE CARLO -> PLAYER PROJECTIONS/DISTRIBUTIONS -> FAIR PROBABILITIES -> SPORTSBOOK COMPARISON`

---

## 2. CURRENT PROMOTED PRODUCTION STACK — UNCHANGED

No overnight/recent QB opportunity diagnostic changed production.

- QB mean: M89/M90 / `QB_PASS_SYNTHESIS_V1`
- QB distribution: `C2_QB_MEAN_NEUTRAL_DISTRIBUTION_V1`
- WR: M38 WR1 hierarchy + `WR_R15_PRODUCTION_MODEL_V1` WR2+
- TE: `TE_R5P_PRODUCTION_MODEL_V1`
- RB rushing: `RB_P3_SYNTHESIS_V1`
- RB receptions: R26
- RB receiving-yard mean: existing production YPT/mean path
- RB receiving-yard distribution/tails: R22 using frozen R19 assets, mean-preserving
- Current player availability/current roles: promoted availability-first plumbing
- Sportsbook: downstream only
- Master workbook: `outputs/NFL_BETTING_MODEL_MASTER.xlsx`

Do not treat any diagnostic/source-audit branch below as production authority.

---

## 3. WHY THE CURRENT QB/WR LANE EXISTS

Earlier shared-residual work established that QB attempt miss and receiver opportunity miss are materially coupled rather than independent systems.

The opportunity-chain work then localized the dominant QB volume error upstream to team pass opportunity/dropback volume. The play/rate decomposition further showed that the fixed production pass-opportunity-rate assumption is the larger physical bottleneck relative to total plays.

Important established result from `PLAY/RATE DECOMPOSITION V1`:
- current M89 projected pass-opportunity rate is exactly `0.57` in all 884 historical rows;
- pooled mean-abs Shapley contribution:
  - PASS_OPPORTUNITY_RATE `5.1714506751`
  - TOTAL_OFFENSIVE_PLAYS `4.3065421286`
- perfect-rate oracle improved team-D MAE much more than perfect-play-count oracle;
- 2025 WR target residual correlation was stronger with rate than plays.

The fixed 57% level was re-swept after corrected attempt semantics in a prospectively frozen anchor study. Anchor `0.59` looked directionally better but failed the frozen pass-rate MAE advancement gate, so it was **not promoted** and cannot be rescued post hoc.

---

## 4. CURRENT SHARED QB/RECEIVER ERROR LOCALIZATION

The recent work progressively decomposed the pass-rate miss rather than feature-mining.

### A. Down/distance decomposition

The dominant miss was **within-state pass propensity**, not merely how often the offense reached particular states. This means the model is more wrong about what the offense chooses to do inside a known down/distance situation than about state occupancy itself.

The same within-state term retained meaningful shared receiver correlation.

### B. State shared attribution

Branch:
- `research-qb-pass-rate-state-shared-attribution-v1`

Preserved result commit:
- `13d9ad428deaada4d92fe303a55948375dc1a491`

Frozen disposition:
- `FIRST_DOWN_SHARED_PRIMARY_DIAGNOSTIC`

Key result:
- first down carried ~44.4% of grouped within-state absolute mass;
- 2025 WR target residual Spearman ~`0.420`;
- pooled WR reception residual Spearman ~`0.313`;
- removing first down materially weakened the shared receiver linkage.

Meaning: the shared QB/receiver opportunity problem is centered on **first-down pass-origin play choice**, not merely generic passing volume.

### C. First-down relative choice economics — FAILED / CLOSED

Branch:
- `research-qb-first-down-choice-economics-d1`

Preserved result commit:
- `ac9e7706d02d11dbcb3cfe9cef02f2bc03d95058`

The source audit was extremely clean, but the untouched 2023 Weeks 10-18 predictive screen failed decisively:
- baseline first-down rate MAE `0.097668` -> candidate `0.097693` (worse);
- residual Spearman `-0.030`;
- only 4/9 weeks improved;
- paired-bootstrap support ~30.3%;
- RMSE/bias/p90 worsened.

Conclusion:
- simple offense+opponent first-down pass-vs-run efficiency/success economics do **not** explain the surviving first-down choice miss;
- 2024-2025 remained sealed;
- do not retune or rescue this family.

### D. First-down field-position decomposition — FIELD POSITION NOT THE ANSWER

Branch:
- `research-qb-first-down-field-position-decomp-v1`

Frozen plan:
- `d66b0a5c87273bd29dc4fa03df97f687135d6674`

Authoritative run:
- Run `34545401969`
- Job `103096807810`
- Tested head `1527ed2c2935175b222b9a428b2c2113a6eceb58`
- Artifact `10178825973`
- Digest `sha256:734b1603b69f45a39f33e6d4770690bcc81b42f8a3b3dae33b74dce6c2cbeb37`

Preserved result commit:
- `55d71b28d038d102c54b25a132c0cca79040b561`

Disposition:
- `FIRST_DOWN_WITHIN_FIELD_POSITION_PROPENSITY_PRIMARY_DIAGNOSTIC`

Key metrics:
- WITHIN_ZONE_PASS_PROPENSITY pooled mean abs `0.04474731937078716`
- 2024 `0.04415409734554893`
- 2025 `0.04534593432352756`
- 2025 WR-target Spearman `0.416549531388451`
- pooled WR-reception Spearman `0.31835559399817626`
- FIELD_POSITION_OCCUPANCY pooled mean abs only `0.006228897430577961`
- occupancy WR correlations ~zero
- zone reference-level component negligible

Conclusion:
- where first downs occur on the field does not explain the shared miss;
- the miss persists inside the same first-down field-position zone;
- do not add generic field position as a QB pass-rate correction.

### E. First-down score-state decomposition — SCORE STATE ALSO NOT THE MAIN ANSWER

Branch:
- `research-qb-first-down-score-state-decomp-v1`

Frozen plan:
- `34ba9fba2b73257a3be28163e2e0803a11ad7645`

Initial execution:
- Run `34546404935`
- scientifically uninterpretable due a mechanical universe mismatch: child evaluator excluded kneels while immutable parent retained them.

Mechanical-only parity repair:
- `735a248fa76f0a11004350f2580c6515dfd354ca`
- canonical execution head `c874f5a5743f069714965d37ff794845629f1986`

Canonical run:
- Run `34548863668`
- Job `103107307133`
- Artifact `10180050013`
- Artifact name `qb-first-down-score-state-decomp-v1`
- Digest `sha256:969b3a3a6c435c08034d5f63dc88ca44991aa99e27b0eca54698cba174beaef9`

Preserved result commit:
- `d819296f24f459170747040951177ae113704fbb`

Disposition:
- `FIRST_DOWN_WITHIN_SCORE_STATE_PROPENSITY_PRIMARY_DIAGNOSTIC`

Integrity:
- exact parent rows 884 = 444 (2024) + 440 (2025)
- exact shared receiver cohorts 440 WR-target + 884 pooled WR-reception
- score-state coverage 100%
- every reconciliation identity passed to floating-point tolerance
- zero sportsbook, zero model fitting, zero production change

Key metrics:
- WITHIN_SCORE_STATE_PASS_PROPENSITY pooled mean abs `0.03574055185869413`
- 2024 `0.03555504299231492`
- 2025 `0.03592774716931315`
- 2025 WR-target Spearman `0.3388118552453079`
- pooled WR-reception Spearman `0.3225090455074344`
- SCORE_STATE_OCCUPANCY pooled mean abs `0.02354084633667748` (secondary, not primary)
- SCORE_STATE_REFERENCE_LEVEL pooled mean abs `0.002200406937696577` (negligible)

Conclusion:
- the surviving shared first-down pass/run-choice miss persists **after down, field-position zone and realized score state are held constant**;
- this is strong evidence for week-specific first-down play-choice/game-plan uncertainty inside otherwise comparable football states;
- this result is diagnostic only and does not authorize a correction.

Anti-reinvention rule after this result:
- do not reopen generic M64/M65 score-state families;
- do not keep slicing postgame PBP simply because more states exist;
- do not rescue first-down choice economics;
- do not repackage M67/M68 opening-script/playcaller/intent history;
- do not use sportsbook/game markets as teacher.

---

## 5. PUBLIC PREGAME INTENT SOURCE AUDIT — CURRENT ACTIVE BRANCH, BUT MANUAL CRAWL IS NOW PAUSED

Reason this lane was opened:
- after down/distance, field position, score state and simple pass-vs-run economics failed to explain the remaining first-down miss, the next legitimate information family had to contain **genuinely new target-game pregame intent**, not another transform of prior PBP.

Branch:
- `research-qb-first-down-public-intent-source-v1`

Frozen source-audit plan commit:
- `4466c88a584a22f6cfebda41fd946a3cdc5bc220`

Frozen audit universe:
- 2023, 2024, 2025 regular seasons
- sampled Weeks 2, 5, 8, 11, 14, 17
- every scheduled team in those sampled weeks
- no outcome-driven dropping

Source hierarchy:
1. official team / coach / OC transcript or official article transcript;
2. official team preview/article containing attributable target-week offensive-plan statements;
3. attributable local beat reporting only when official material is insufficient.

Frozen source-only restrictions:
- no football outcomes;
- no parent residuals;
- no QB/WR targets/results;
- no sportsbook lines/spreads/totals;
- no predictive score/model;
- no production changes.

Frozen qualification gates include:
- >=70% pooled eligible team-week coverage;
- >=60% each season;
- >=50% each sampled week pooled across seasons;
- >=90% timestamp safety;
- >=80% stable official/local attributable source provenance;
- at least 24/32 franchises with >=50% eligible coverage.

Workflow/scaffold work preserved on branch:
- validator commit `a220f6fde8d6062af37967b6c14d709e3a0cc7cf`
- sample manifest `17266c112c0e4b5761946c017b18efe8e0cf8a3b`
- workflow `e76b3abda54478c45e12526567dd335ab47e56de`
- collection protocol `8ae3839c81e1c2850d32b6dbf9e3888ae6e29a93`
- historical schedule parsing repair `cf538603e55be30b6749dc27ec5ab9bdb5a74e58`
- ordered collection state `15fb0518d0d88a4967290c5f32200c1025344705`
- validated scaffold `2de863657121e257bc0efb18f601b8e24d558669`
- current branch head `61f3931394c383498ebd58f6dbdb923e6feb68ba`

Collection status at current stop:
- manual deterministic collection reached 2023 Week 2 through Detroit;
- 11 team-weeks collected: ARI, ATL, BAL, BUF, CAR, CHI, CIN, CLE, DAL, DEN, DET;
- all 11 currently have timestamp-safe eligible pregame intent evidence;
- most are official team sources; Atlanta used the allowed local-attributable fallback;
- examples include explicit pass emphasis (CIN/CLE/DAL), personnel/game-plan information (BAL/DEN/DET), etc.

**CRITICAL USER DECISION / NEW OPERATING BOUNDARY:**

The user correctly identified that manually crawling ~500+ historical team-weeks is operationally unacceptable and could take an enormous amount of time. Manual team-by-team/game-by-game source archaeology is now **PAUSED**. Do not continue the brute-force manual crawl.

The 11 collected rows are preserved as a small feasibility/gold sample, not as evidence that the full source family has qualified.

The V1 frozen qualification has **not** been passed or failed yet; it is simply incomplete and operationally non-scalable in its manual form.

---

## 6. NEXT AUTHORIZED PUBLIC-INTENT STEP — AUTOMATION-FIRST, HARD TIME CAP

User chose to continue the idea only if it can be made operationally efficient.

Next step should be a **separately frozen automation-first V1B retrieval/source-validation design**. Do not silently rewrite V1 after seeing its early 11/11 coverage.

Recommended design boundary:
- generate exact team-week universe automatically from schedule;
- fixed query/source hierarchy only;
- official sources first, predeclared local fallback second;
- retrieve only top few timestamp-safe pre-kickoff candidates;
- automatically extract publication time/date, speaker, opponent/target-week relevance, source class, semantic tag(s), locator and <=25-word evidence;
- humans review only ambiguous rows;
- use a small manually reviewed gold sample to measure retrieval precision/recall and timestamp correctness before scaling;
- stop early if automation quality is poor.

**Time budget / stopping rule requested by user:**
- do not sacrifice days/weeks to this hypothesis;
- target ~6-10 hours total serious research effort, one working day maximum;
- roughly 2-4h automation/retrieval build, 2-3h representative validation, 2-3h one frozen predictive screen only if source qualification is strong;
- if scalable retrieval + meaningful predictive promise are not demonstrated inside that budget, preserve findings and close/move on.

Potential use if source family eventually qualifies:
- likely sparse-event/context feature, not a mandatory feature requiring text for every team-game;
- baseline historical model remains for all games;
- strong explicit pregame intent can become additive context only after a separately frozen predictive study passes.

Do not proceed directly from source availability to production.

---

## 7. HISTORICAL VEGAS CERTIFICATION — IMPORTANT SEPARATE LANE, NOT CURRENT ACTIVE TASK

Do not incorrectly claim that the current full model has already been broadly validated against historical player-prop lines.

What happened:
- M60 built historical Vegas benchmark machinery but paid The Odds API history was blocked by quota economics;
- M60B found a large free Action Network-derived archive but canonical projection-game IDs did not reconcile, yielding zero trustworthy matches, so grading was correctly skipped;
- therefore broad historical market certification of the **current** production stack remains unfinished.

Benchmark machinery already exists for:
- model MAE vs actual;
- Vegas line MAE vs actual;
- model closer than Vegas;
- directional side win rate;
- priced bets / units / ROI;
- performance by disagreement-size bucket.

Potential future no/low-cost market-data route:
1. try to repair existing Action archive natural-key mapping;
2. audit ParlayAPI historical closing-line/bulk CSV coverage as a separate source candidate;
3. permanently archive our own live 2026 pregame boards going forward.

This is high-value but should remain separate from the current shared-opportunity source audit so sportsbook data never teaches upstream football projections.

---

## 8. PARKED / DO NOT ACCIDENTALLY REOPEN

- NE-SEA grading: explicitly parked by user.
- Generic QB mean feature hunt after M89/M90: frozen absent architecture-specific new information.
- M64/M65 generic pace/score-state opportunity work: closed absent materially new state-transition information.
- M67 generic offensive-intent/history family: already tested.
- M68 verified playcaller/opening-script/leverage family: already tested; partial signal, no actionable model.
- M81/FTN tactical formation/personnel/motion/screen/RPO family: closed/source-limited/failed point-prediction use.
- M87/M88 pass-funnel + short/intermediate interaction: directional but failed untouched confirmation; do not resurrect.
- directional personnel source family: failed coverage gates; do not loosen.
- designed-run D1 correction: failed frozen confirmation gates.
- schedule/rest D1 correction: failed downstream yardage/confirmation behavior.
- M89 opportunity reparameterization A1: no support.
- 0.59 fixed pass-rate anchor: level/bias evidence but failed frozen pass-rate MAE advancement gate; not production.
- first-down relative choice economics: failed clean untouched-2023 screen; closed.
- field-position occupancy: decisively not primary.
- score-state occupancy/reference-level as generic correction: not primary; do not repackage.

---

## 9. EXACT CURRENT STOP / NEXT CHAT RESUME INSTRUCTIONS

When a new chat/session starts:

1. Read `NFL_MASTER_CONTINUITY_RECORD.md` for project-wide history/philosophy.
2. Read this file: `docs/handoffs/NFL_HANDOFF_2026-09-11_QB_WR_SHARED_OPPORTUNITY_CURRENT.md`.
3. Read `CURRENT_NFL_RESEARCH_HANDOFF.md` and verify it still points here or to a newer handoff.
4. Verify live GitHub `main` and active branch heads before making any claim.
5. Verify `research-qb-first-down-public-intent-source-v1` is still at/after `61f3931394c383498ebd58f6dbdb923e6feb68ba` and whether any parallel work advanced.
6. Do **not** continue the manual public-source crawl.
7. If continuing the public-intent idea, first freeze an automation-first V1B retrieval/validation protocol prospectively; use current 11 rows only as source-quality/gold examples, not as target-labeled training evidence.
8. Keep the public-intent lane under the one-working-day / ~6-10h hard time budget.
9. If automation/source quality fails, close it and move to the next scalable shared-opportunity mechanism or the separate historical-market-certification lane.
10. No production change unless a separately frozen predictive/integration candidate passes all required gates.

**Current scientific description of the unresolved shared layer:**

`TEAM PASS OPPORTUNITY -> PASS-OPPORTUNITY RATE -> WITHIN-STATE PASS PROPENSITY -> FIRST-DOWN PLAY SELECTION -> SURVIVES FIELD POSITION -> SURVIVES SCORE STATE -> NOT EXPLAINED BY SIMPLE PASS-vs-RUN RECENT ECONOMICS -> LIKELY WEEK-SPECIFIC PREGAME GAME-PLAN / PLAY-CALL INTENT`

This remaining mechanism is still materially shared with receiver opportunity error, so a successful scalable predictor could improve QB attempts and WR/TE/RB receiving opportunity coherently rather than as independent patches.
