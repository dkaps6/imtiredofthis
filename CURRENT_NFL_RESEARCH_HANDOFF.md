# CURRENT NFL RESEARCH HANDOFF — READ FIRST

**Repository:** `dkaps6/imtiredofthis`  
**Purpose:** canonical cross-chat / scheduled-task continuity ledger for the NFL pregame projection research program.  
**Current local date:** 2026-09-09 (America/Indiana/Indianapolis).  
**Protected production-code authority:** `bb76ba9eabb08e2f0875a9af49301c3877f4141f`  
**Active research branch:** `research-rb-r27d-yacoe-residual-v1`  
**Exact current stop point:** R27D strict-prior YAC-over-expected residual V1 plan is frozen before implementation/model fit/candidate execution/results at commit `69edc12a16c691e3838eadcd75559b85dbba7865`, plan blob `d0c2b0ff2de154e52fa21fb9ce19b739039633f3`.  
**Next action:** implement R27D exactly from that frozen plan, static-audit it, create an implementation lock pinning all code/parent hashes, then and only then execute the first 2020–2025 walk-forward candidate. Preserve the first valid result exactly.

> **Future ChatGPT sessions / scheduled runs:** read this file first, then `AGENTS.md`, then the exact active frozen plan. GitHub is canonical; chat memory is secondary. Verify live branches, commits, workflow runs, jobs, artifacts, digests and dispositions before acting. Never infer current state from an old chat when GitHub disagrees.

## Historical handoff preservation

The prior full canonical ledger through the R27B V2 **plan-freeze** checkpoint is permanently preserved in Git history at:
- handoff commit `69e8de76bd1b508849d679fa22abd51aefa68a54`
- handoff blob `2f0ee91c1d5296c04114605afd5d4c4067a25a72`

That snapshot contains the deeper pre-R27B migration/R26 paper trail. This refreshed file supersedes its stale top-level stop point; it does **not** erase or invalidate the historical record.

---

# 1. Controlling objective

Build an elite pregame NFL prediction model that predicts **actual football outcomes** more accurately than competing forecasts/markets over valid samples.

Primary causal target:

> **Actual football outcome ← our football prediction → market prediction**

The sportsbook is a strong external benchmark and downstream pricing layer, not the teacher of the football model.

Desired architecture:

> **GAME / TEAM OPPORTUNITY → POSITION / ROOM POOL → INDIVIDUAL ENTITLEMENT → PLAYER + MATCHUP EFFICIENCY → JOINT DISTRIBUTION → GAME-STATE FEEDBACK → SCORING / FINAL SCORE**

Do not optimize merely to mimic Vegas. If Vegas repeatedly wins a valid cohort, investigate what football signal it may be capturing.

---

# 2. Non-negotiable research rules

1. Historical science must be strict-prior / walk-forward / leakage-safe.
2. Freeze the scientific question, population, candidate mechanics, metrics, thresholds, gates and authority ceiling **before results**.
3. Preserve the first valid scientific result exactly whether PASS, mixed or fail.
4. Preserve mechanical/plumbing failures separately; repair them only with the minimum documented value-neutral change.
5. Do not lower gates, change cohorts, drop losing seasons or retune a candidate after seeing results.
6. Sportsbook inputs remain downstream unless a separately frozen market-assisted experiment explicitly authorizes otherwise.
7. Do not silently change a frozen mechanism, architecture or research direction. Communicate scientific concerns first.
8. Production changes require separate qualification/integration/promotion evidence. A research PASS alone is not permission to mutate production.
9. One authoritative production projection per player/market. Old values may remain only as audit fields.
10. R26 receiving opportunity/receptions and R22 RB receiving-yard tail authority are protected during the current mean-research lane unless a separately frozen integration study explicitly changes that boundary.
11. Update this handoff on `main` at every material checkpoint with exact branch/commit/run/job/artifact/digest/disposition and next action.

---

# 3. Current protected production stack

Protected production-code authority:
`bb76ba9eabb08e2f0875a9af49301c3877f4141f`

Current stack:
- QB passing-yard mean: **M89/M90**
- QB distribution: **mean-neutral C2**
- WR: **M38 WR1 + WR-R15 WR2+**
- TE: **TE-R5P**
- RB rushing: **P3**
- RB receptions: **R26 production refinement**
- RB receiving-yard mean: existing production mean/YPT path; R26 opportunity has **not** yet been promoted to recenter the RB receiving-yard mean
- RB receiving-yard distribution/tails: **R22**, using pinned R19 assets and preserving the upstream mean
- sportsbook: downstream only

## R26 production qualification

- qualification run `34417740186`
- job `102686263562`
- artifact `10129819192`
- digest `sha256:f8fbd74187e811fc765c6d236c4c005c19be6d007d1bec33800790e56eabdcde`
- disposition `RB_R26_WEEK1_RECEPTIONS_PRODUCTION_INTEGRATION_PASS_READY_FOR_PROMOTION`
- 35/35 PASS

Post-promotion Full Slate:
- run `34418491952`
- job `102688556296`
- artifact `10130055472`
- digest `sha256:d84bbc9fb976e3862b19d458a929888c1ab5e0f7cec49e1f4705a961a236e15f`
- head `bb76ba9eabb08e2f0875a9af49301c3877f4141f`
- SUCCESS; live odds disabled, so this verifies football-stack/repository wiring rather than a fresh sportsbook board.

R26 vacancy mechanism remains:
- `VACANCY_ACTIVE = room_exits_n >= 1`
- stable/no-vacancy remains baseline exactly
- exact RB/FB room target pool is conserved
- non-RB entitlement is preserved exactly
- strict-prior R8/R9 receiving identity/reliability drives within-room redistribution
- softmax only inside the RB/FB room

## R22 / R19 tail authority

R22 integration:
- run `34298516960`
- artifact `10084118525`
- digest `sha256:2391bd9914e9d0029c63529829496fb9b462d008bdcc3f8c37bb3e1079580bd1`
- disposition `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_INTEGRATION_PASS`
- mean-neutral by design

R19 serialized tail authority:
- run `34288244770`
- artifact `10080377483`
- digest `sha256:11432b9d7b7f2367935a862b63c30df9f40955e479806ab67d90921b63a907c7`
- model SHA `9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba`
- residual-pools SHA `c69a268a5a1683e846bcb5f59fe55bcae20d70c679792449ea77e6548b37a362`

R19/R22 own stochastic receiving-yard right-tail shape; they are **not** the point-mean efficiency model.

---

# 4. Prospective R26 evidence that must remain immutable

R26Q Week1 pregame seal — do not recompute:
- run `34400524030`
- job `102630996205`
- artifact `10123251043`
- digest `sha256:dd3ec0e8e3831ab7f2255c2e5abf343cda8a7943d33a1d4863e52372d6f858a1`
- disposition `R26Q_2026_WEEK1_RECEPTIONS_PROSPECTIVE_SEAL_PASS_READY_FOR_OBSERVATION`
- NPZ SHA `7933bd7629d0e7108fe181e7a5474f8c8f6e8512b6a1b35081d3a8d71af8fe06`

R26R market observation:
- run `34401814588`
- job `102635265504`
- artifact `10124274040`
- digest `sha256:b4d3e573909803e892d57858c9b0c4bdac089bcf078c1d6f6e53366e4622303e`
- disposition `R26R_2026_WEEK1_PROSPECTIVE_OBSERVATION_SNAPSHOT_PASS_MARKET_CAPTURED`

R26S postgame evaluator is the frozen scorecard for R26Q, not a predictive model:
- plan commit `63722c61ca1b408b8ae77eba383e5fa8daecf9f2`
- evaluator commit `e42099dbf11cdd510eb64e43a0910444c75ec7f8`
- lock commit `3a706fa52f91f6584f6fd6594563239dd6ea3b53`
- canonical pregame dry run `34411889262`
- job `102667966181`
- artifact `10127562840`
- digest `sha256:e9ecd949816eafd813a50e60e8af28ce387e668848483ffcda3dfadf7614919e`
- disposition `R26S_2026_WEEK1_POSTGAME_PROSPECTIVE_EVALUATION_INCOMPLETE_NO_DECISION`

When Week1 outcome authority is available, rerun the **exact locked R26S evaluator unchanged**.

---

# 5. Anti-reinvention boundary for RB receiving-yard mean

Already established before the current R27D lane:

- Production already has empirical-Bayes player YPT (`bayes_ypt`) plus production matchup/team pass-efficiency adjustment (`rules_ypt`).
- R23 already tested a new shrunk historical YPR construction and did not qualify.
- R24 already tested an opportunity × existing-production-efficiency decomposition and did not qualify.
- R27 tested the exact later-qualified R26 opportunity mechanism × unchanged production YPT.
- R27B V2 tested novel raw target-shape/YAC/team/opponent context and did not qualify integration.
- R19/R22 own tail/distribution science, not mean recentering.
- Generic statements that “YPT/YPR/YAC matters” are not new science.

Do **not** restart generic rolling YPT/YPR/YAC models.

---

# 6. R27 — exact R26 opportunity → receiving-yard mean decomposition

Frozen question:
Does exact historical R26 opportunity redistribution improve RB receiving-yard means when the existing production efficiency estimate is held fixed?

Frozen plan:
- commit `5333d7e1cc33dcb567d03c924c774afb6877e932`
- plan blob `4ad4801dbe600238dd7f1090df0a6596ff8a6a46`

First valid scientific evidence:
- run `34423546037`
- job `102703879430`
- head `b7cfe5b2c450765208672fc6e2b017d0cb0c0dde`
- artifact `10132290573`
- digest `sha256:cd5c9e26efa44c47ef3374e4deefe9038dc504c155799b48548ad64746e10e45`
- disposition **`R27_R26_OPPORTUNITY_TRANSLATION_MIXED_OR_FAIL`**
- 24/27 gates PASS

Key evidence:
- ALL-RB rec-yard MAE change `-0.235%`
- Week1 rec-yard MAE change `-3.308%`
- vacancy-active n=1761: MAE `11.2830 → 11.1593` (`-1.097%`)
- vacancy-active RMSE `-1.334%`
- vacancy-active p90 worsened `+2.078%`
- vacancy target MAE improved `-2.087%`
- vacancy reception MAE improved `-1.803%`
- vacancy RB1 target MAE improved `-2.284%`
- vacancy RB1 reception MAE improved `-2.387%`
- vacancy RB1 receiving-yard MAE worsened `14.3059 → 14.7094` (`+2.820%`)
- vacancy RB2+ receiving-yard MAE improved `-3.358%`
- 2023 vacancy-active rec-yard MAE worsened `+4.916%`
- 4/6 seasons improved

Interpretation: R26 opportunity is real; the unresolved bottleneck is downstream efficiency/translation, especially lead backs, 2023 and tail safety.

---

# 7. R27B V1 supersession and V2 context experiment

R27B V1 was formally superseded **before execution** because it risked reinventing existing production/R23 generic historical efficiency science:
- supersession commit `ca5537321893eb5fecba890b3ee2aeb320dede1f`
- no V1 workflow, artifact or scientific result exists.

## R27B V2 source audit

- run `34427150810`
- job `102714703407`
- artifact `10133096990`
- digest `sha256:a9905a02243af2120cd71e85bcd460d47c5ccc73af87779adde4dd5c20a8263b`
- source/schema only; no model/candidate/performance score
- excellent 2019–2025 PBP coverage for novel RB target-shape/YAC/team/opponent context

## R27B V2 frozen plan

- commit `bbaa0e2bbf32b182ca768555fa17548f8493be18`
- plan blob `88c1377acc2e5389df092ec74ab876eaed468cdd`
- implementation lock `2c86520c84fbdd5a18aad3f4b88373e5f8c17051`

V2 kept R26 opportunity fixed and predicted a residual to production YPT using strict-prior player target shape/YAC, team/QB RB environment, opponent RB-specific environment and frozen role/vacancy controls. Stable rows remained exact production/B1; R22 stayed untouched.

## First valid R27B V2 result

- run `34428917229`
- job `102720004328`
- artifact **`10134023092`**
- digest `sha256:1df1a14c9900b890fe86e04269849dc90d76fc03a9215ad8612c1820dee5341f`
- disposition **`R27B_V2_NOVEL_EFFICIENCY_CONTEXT_MIXED_OR_FAIL_NO_INTEGRATION`**
- 24/31 gates PASS
- all 15 integrity gates PASS; seven scientific gates failed

Failed scientific gates: pooled vacancy improvement threshold, RB1 improvement threshold, RB1 non-worse vs B0, pooled vacancy RMSE, 30+ miss rate, 2023 improvement threshold, and 2023 non-worse vs B0.

Useful retained signal — evidence, **not cherry-picked production**:
- vacancy MAE `11.1593 → 11.1267` (`-0.292%` vs B1)
- vs B0 `11.2830 → 11.1267` (`-1.385%`)
- p90 `25.3747 → 24.9769` (~`-1.57%`), repairing R27 p90 weakness
- RB2+ improved ~`0.338%`
- 4/6 seasons improved
- ALL-RB and Week1 stayed stable/slightly better
- RB1 only improved ~`0.292%` vs B1 and remained ~`2.52%` worse than B0
- 2023 worsened again

Conclusion: V2 contained real context signal but did not qualify. Preserve the signal as evidence; do not route winning post-hoc subsets into production.

---

# 8. R27C / R27C2 — RB1 and 2023 forensics

## R27C diagnostic

Branch `research-rb-r27c-rb1-2023-forensic-v1`.
- canonical run `34430754033`
- job `102725570088`
- artifact `10134387002`
- digest `sha256:0c915a1c413d66c550990a1cc8bd45e0dc20a18e110fe1b96414a79eb7b598e9`
- result record commit `b2907345c50dceffdfb5c074ce9d556ff6d764c6`
- diagnostic only, exact V2 artifact consumed, no candidate/model

Clue: 2023 RB1 target count became slightly more accurate while receiving-yard translation worsened; tail misses showed the opposite high-YAC phenomenon.

## R27C2 realized target-quality forensic

Branch `research-rb-r27c2-realized-target-quality-forensic-v1`.
- first valid run `34431286455`
- job `102727137722`
- artifact `10134581843`
- digest `sha256:6e3aa7ec2149f6f0b65adb59b4cc36c4f88b24a142234bc368131de8e4de60dd`
- result-record commit `2cfca82d919290ac18ed01559994d2e0239b0798`
- disposition `R27C2_FORENSIC_COMPLETE_PHYSICAL_TARGET_QUALITY_HYPOTHESIS_IDENTIFIED`
- integrity PASS; 8429 parent rows; ~99.63% overall targeted PBP join; 100% 2023 vacancy-RB1 targeted join

2023 vacancy RB1 vs non-2023 vacancy RB1:
- catch rate `0.78345 vs 0.78724` — essentially unchanged
- air yards/target `0.0887 vs 0.1065` — essentially unchanged
- screen rate `0.5806 vs 0.5623` — similar
- explosive20 target rate `0.05046 vs 0.05097` — nearly identical
- YAC/reception `7.3395 vs 8.0667` — down `0.7272`
- YPR `6.4254 vs 7.4206` — down `0.9952`
- YPT `5.2236 vs 5.9001` — down `0.6765`

Production arithmetic, 2023 RB1:
- actual catch rate `0.78345`, production `0.77220` — actual slightly better
- actual YPR `6.4254`, production implied YPR `7.5774` — residual about `-1.188`
- actual YPT `5.2236`, production YPT `5.8393` — residual about `-0.616`
- non-2023 actual YPT `5.9001`, production `5.9029` — essentially perfect

Conclusion: problem is post-catch/YPR compression, not missed catches or basic target-shape frequency.

Tail reference remained a separate high-YAC/explosive phenomenon; do not solve it by broadly raising point means.

---

# 9. R27D0 / R27D0B — xYAC/YACOE source audit

## R27D0 source audit

Branch `research-rb-r27d0-yac-quality-source-audit`.
- result commit `8663b132c9910d07679d7b5e03ee8a8c9542f332`
- run `34431705294`
- job `102728396342`
- artifact `10134726607`
- digest `sha256:b1a2a44d7ed0303300a792bf617ac6c5095aa7a0229186001504ebc2cad44bdf`
- disposition `R27D0_YAC_QUALITY_SOURCES_SUPPORT_SEPARATELY_FROZEN_PREDICTIVE_STUDY`

Source finding:
- nflverse PBP exposes `xyac_mean_yardage`, `xyac_median_yardage`, `xyac_success`, `xyac_fd`.
- completed-RB-catch `xyac_mean_yardage` coverage is ~99% in every 2020–2025 season.
- targeted vacancy RB1 strict-prior PBP YACOE history is dense: >=1 prior game ~95.5%; >=3 ~91.2%; 2023 >=3 ~95.8%.
- down, yards-to-go, shotgun, no-huddle, pass location/length, score differential and air yards are ~99.6–100% available.

**NGS warning:** the Next Gen Stats receiving schema contains YACOE/expected-YAC/separation/cushion fields but the historical feed resolves to **zero RB rows** across 2020–2025. NGS is rejected for this RB-specific lane. Do not impute WR/TE tracking to RBs or claim NGS RB support from schema alone.

The genuinely new source is PBP expected YAC / YAC-over-expected, not another raw YAC average.

## R27D0B 2019 source extension

- run `34431927369`
- job `102729055139`
- artifact `10134799189`
- digest `sha256:13853603029adaf820b3228916c519acfdf98794c70462fbbcebe5f93f97d6cd`
- result commit `4e19b549dad2141ba3d55b77c563947342426461`

2019:
- RB target rows `3548`
- completed RB catches `2709`
- xYAC coverage `99.2617%`
- situational fields ~99.77–100%

Therefore the 2020 outer fold can have legal pre-2020 xYAC history; no arbitrary source-absence fallback is needed.

---

# 10. R27D0C — xYAC physical mechanism split

Branch `research-rb-r27d0c-xyac-mechanism-split`.

Frozen plan:
- commit `21db47af29f121e27c137c6e21773dd557094395`
- blob `6a01722bd0291805a44d6be37dec1e228c3adbd0`

Original implementation lock:
- `99f3347abc6af3ac300b4a78ac1237454678a6e1`

### Preserved Run1 mechanical failure

- run `34432353557`
- job `102730315801`
- plan hash / parent artifact / production boundary all PASS
- failure: actual YAC averaged all completed catches while expected YAC/YACOE necessarily used only xYAC-observed catches; ~99% rather than 100% xYAC coverage caused a `0.0509294626941692` yard/reception algebra gap
- no artifact, no canonical scientific/diagnostic conclusion

Frozen repair record:
`docs/research/RB_R27D0C_RUN1_XYAC_OBSERVATION_SET_MECHANICAL_REPAIR.md`
- repair-record commit `c87f3dcae924f8a9656f91533b81edccf86cd921`
- repaired script blob `8428aa5cfe675efcf3c366a48e80d705ffdf5aea`
- repaired workflow blob `23fcc7018ab560782556c7fd316308ec90154bbe`
- repair lock / first-valid head `6ee566b67abd87f0f33b37111aeb83011419b43d`

### First valid R27D0C result

- run `34432497854`
- job `102730752292`
- artifact `10134994377`
- artifact name `rb-r27d0c-xyac-mechanism-split`
- digest `sha256:2c3d6043dc25298b8806e82abe7e4a97e9c2c066408fc38e215eae0bdbc1c882`
- result record commit `76e662c92a49743092d7e512a6a1b2e87a74032c`
- status `R27D0C_XYAC_MECHANISM_SPLIT_COMPLETE`
- integrity PASS

2023 vacancy RB1:
- actual YAC/reception on xYAC-observed catches `7.33951`
- expected YAC/reception `7.92507`
- YACOE/reception `-0.58556`

Non-2023 vacancy RB1:
- actual YAC/reception `8.11761`
- expected YAC/reception `7.65199`
- YACOE/reception `+0.46562`

2023 minus non-2023:
- actual YAC difference `-0.77810`
- expected-YAC difference `+0.27308`
- YACOE difference `-1.05118`

**Key physical conclusion:** 2023 catches were **not** lower-value by xYAC target/play context; expected YAC was actually slightly higher. The observed compression occurred because realized post-catch execution came in dramatically below contextual expectation. In signed terms the lower YACOE more than explains the negative actual-YAC gap while expected-YAC context offsets part of it.

Tail reference:
- four RB1 rows V2 moved from <30 AE to >=30 AE: actual YAC/reception `22.5`, expected YAC `6.943`, YACOE `+15.557`
- RB1 rows >=30 AE under both B1/C1: actual YAC `11.894`, expected YAC `7.898`, YACOE `+3.996`

This reinforces architecture: repeatable mean correction should study pregame-identifiable YACOE state; stochastic extreme positive post-catch outcomes remain conceptually owned by R22.

---

# 11. ACTIVE: R27D strict-prior YACOE residual V1

Branch:
`research-rb-r27d-yacoe-residual-v1`

Frozen plan:
- commit **`69edc12a16c691e3838eadcd75559b85dbba7865`**
- plan blob **`d0c2b0ff2de154e52fa21fb9ce19b739039633f3`**
- file `docs/research/RB_R27D_STRICT_PRIOR_YACOE_RESIDUAL_V1_FROZEN_PLAN.md`
- status `FROZEN BEFORE IMPLEMENTATION / MODEL FIT / CANDIDATE EXECUTION / RESULTS`

Scientific question:
Can strict-prior relative YAC-over-expected state improve receiving-yard point means specifically for **vacancy-active incumbent RB1s**, with exact R26 opportunity/receptions fixed, without sacrificing pooled vacancy, aggregate, Week1 or tail safety?

Frozen application scope:
`VACANCY_ACTIVE == 1 AND vacancy_incumbent == 1 AND role == "RB1"`.

Every other row remains exact R27/B1. Vacancy RB2+ is forced to exact B1 parity by construction. This scope is frozen before R27D results and is justified by repeated R23/R27/V2/C/C2 localization of the unresolved translation error to lead backs; it is not a post-result R27D router.

Frozen primary predictor family — xYAC/YACOE only:
1. player relative YACOE prior
2. player expected-YAC prior relative to league
3. team-RB relative YACOE prior
4. team-RB expected-YAC prior relative to league
5. opponent RB relative YACOE allowed prior
6. opponent expected-YAC allowed prior relative to league
7. Week1
8. log prior player xYAC-reception support

Forbidden primary predictors:
- raw historical YAC/YPR/YPT
- generic catch-rate persistence
- V2 raw air-yard/screen/explosive/YAC feature family
- sportsbook
- target-game/future PBP as features
- R19/R22 tail probabilities

Strict-prior shrinkage:
- player K=12 xYAC receptions
- team K=30
- opponent K=30
- no-history centered relative signals = 0 using legal prior population level

Estimator:
- StandardScaler + weighted Ridge
- fixed `alpha=100.0`; no model zoo or hyperparameter search
- training label = player-game YACOE centered against legal prior league RB YACOE
- weight `clip(xYAC observed receptions,1,8)`
- correction capped at ±1.5 yards/reception

Candidate arithmetic:
- B0 = baseline targets × production YPT
- B1 = exact R26 candidate targets × production YPT
- production implied YPR = production YPT / production catch rate
- exact bridge: R26 candidate receptions × production implied YPR == B1 within `1e-10`
- C1 scoped RB1 YPR = production implied YPR + clipped predicted relative YACOE
- C1 yards = exact R26 candidate receptions × C1 YPR
- outside frozen RB1 scope C1 == B1 exactly

Frozen plan contains 18 structural/integrity gates and 13 scientific gates (31 total). Important science hurdles include:
- primary RB1 MAE improve >=1.00% vs B1
- primary RB1 non-worse vs B0
- 2023 RB1 improve >=2.00% vs B1 and non-worse vs B0
- pooled vacancy MAE non-worse vs B1
- ALL-RB non-worse vs B1 and <=+0.10% vs B0
- RB1 RMSE/p90/30+ miss safety
- bias safety
- >=4/6 seasons improve, no season >2% worse
- Week1 <=0.50% worse

Frozen PASS disposition:
`R27D_STRICT_PRIOR_YACOE_RESIDUAL_SUPPORT_READY_FOR_SEPARATE_INTEGRATION_DESIGN`

Frozen mixed/fail disposition:
`R27D_STRICT_PRIOR_YACOE_RESIDUAL_MIXED_OR_FAIL_NO_INTEGRATION`

Integrity failure:
`R27D_MECHANICAL_OR_INTEGRITY_FAILURE_NO_SCIENTIFIC_DECISION`

**A PASS still does not mutate production.** It only authorizes a separately frozen integration/recentering study compatible with R22 and the one-authoritative-projection stack.

## Exact next action

1. Stay on `research-rb-r27d-yacoe-residual-v1`.
2. Read exact frozen plan at commit `69edc12a16c691e3838eadcd75559b85dbba7865`, blob `d0c2b0ff2de154e52fa21fb9ce19b739039633f3`.
3. Implement strict-as-of PBP xYAC/YACOE history materialization for 2019–2025.
4. Consume exact R26/R27/V2 parent authority as specified; do not recreate a different opportunity candidate.
5. Fit only the frozen 8-feature relative-YACOE model with fixed Ridge alpha 100 and frozen shrinkage/cap.
6. Add all 31 gates exactly.
7. Static/anti-reinvention/protected-production audit implementation.
8. Create implementation lock pinning plan/code/workflow/parent blobs **before any candidate execution**.
9. Run first 2020–2025 walk-forward scientific execution.
10. Preserve first valid result exactly and record it in docs/research.
11. If mechanical failure occurs, preserve it and allow only minimum documented value-neutral repair.
12. If PASS, do not directly promote; write/freeze a separate integration design and prove R22/full-stack compatibility first.
13. If mixed/fail, do not retune on the same sample; use the evidence to decide whether another genuinely new hypothesis exists or whether RB receiving mean research has reached a defensible stopping point.
14. Update this handoff on `main` at the next material checkpoint.

---

# 12. Overnight / scheduled continuation instructions

An hourly ChatGPT task has been configured for **9 hourly runs, midnight through 8:00 AM America/Indiana/Indianapolis**.

Each run is instructed to:
- open this repo and read this handoff first;
- verify exact live GitHub state rather than rely on chat memory;
- do meaningful research/implementation work, not merely report status;
- obey all freeze, leakage, artifact and production-safety rules;
- continue the current RB receiving lane to a defensible scientific stopping point;
- follow a separately frozen integration/promotion path if and only if pre-frozen criteria genuinely authorize it;
- after RB receiving work is complete, automatically advance to the next documented roadmap item;
- update this handoff on `main` at every material checkpoint so later runs/chats inherit exact lineage.

Do not wait for the user between routine research checkpoints. Stop only when a scientific decision genuinely requires user preference, a safety/authorization boundary prevents an action, or the documented roadmap reaches a point with no defensible next experiment.

---

# 13. Roadmap after RB receiving-yard mean lane

Preserve this order unless new evidence gives a strong reason to discuss a change with the user first:

1. **Finish RB receiving efficiency / receiving-yard mean** — ACTIVE R27D.
2. **Current roster / late-week role handling** where needed as a separate operational lane; do not contaminate historical science.
3. **Grade sealed R26Q via the exact locked R26S evaluator** when authoritative Week1 outcomes are available.
4. **QB opportunity/efficiency:** attempts, dropbacks, pass rate, YPA, sacks, scrambles; build on existing M89/M90 rather than restarting old failed migrations.
5. **Selective WR/TE work:** only unresolved opportunity/efficiency/distribution mechanisms; preserve M38/WR-R15 and TE-R5P winners.
6. **Shared QB ↔ receiver conservation:** align team pass attempts, targets, completions, receiving yards and player entitlements coherently.
7. **Unified game simulation:** plays → pass/rush → player opportunity → outcomes → yards/explosives/TDs → game-state feedback → possessions/scoring.
8. **Anytime TD** modeling under coherent opportunity/game environment.
9. **Game ML / spread / total** from the football simulation rather than sportsbook imitation.
10. **Final operational package / prospective grading:** one authoritative slate, live input checks, distributions/fair probabilities, market comparison downstream, prospective scorecards.

Explosive receiver/QB interaction idea remains worth revisiting in the QB↔receiver distribution/conservation lane: player-specific explosive propensity vs coverage/DB environment may help explain QB high-end passing outcomes. Do not mix that parked hypothesis into current RB point-mean work; R22 already owns RB receiving-yard tail shape.

---

# 14. Promotion and production safety

The user has authorized autonomous continuation and trusts the research process, including promotion when genuinely qualified. That authorization does **not** waive scientific controls.

For any component:
1. Research result must satisfy its pre-frozen gates.
2. If the research plan says PASS only authorizes integration design, write/freeze that integration design first.
3. Verify exact parent/model hashes and preserve predecessor authority.
4. Prove no unintended cross-component changes.
5. Run full-stack / Full Slate verification where relevant.
6. Record exact production commit, run, job, artifact, digest and disposition.
7. Only then treat the new component as authoritative production.
8. Never promote a mixed/fail result or a cherry-picked post-result cohort.

GitHub remains the source of truth.
