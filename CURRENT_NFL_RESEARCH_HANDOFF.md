# CURRENT NFL RESEARCH HANDOFF — READ FIRST

**Repository:** `dkaps6/imtiredofthis`  
**Purpose:** canonical cross-chat continuity ledger for the NFL pregame projection research program.  
**Last updated:** 2026-09-09 after R26 receptions production promotion, post-promotion Full Slate/Repo CI verification, and the R27 receiving-yard mean decomposition plan freeze.  
**Current production head:** `bb76ba9eabb08e2f0875a9af49301c3877f4141f`  
**Active research branch:** `research-rb-r27-receiving-yard-mean-decomposition-v1`  
**Exact stop point:** R27 V1 frozen plan exists at commit `5333d7e1cc33dcb567d03c924c774afb6877e932`; **no R27 evaluator/implementation lock/workflow run/results exist yet**.

> **Future ChatGPT sessions / agents:** read this file first, then `AGENTS.md`, then the exact R27 frozen plan named below. Preserve failed studies and mechanical failures, distinguish science from plumbing, freeze plans before results, keep sportsbook data downstream only, and never silently change production.

---

## 1. Controlling objective and architecture

Project individual NFL player outcomes as accurately as possible **pregame**, player by player: yards, receptions, carries, touchdowns and downstream fair probabilities.

Core architecture:

> **GAME / TEAM OPPORTUNITY → POSITION / ROOM POOL → INDIVIDUAL PLAYER ENTITLEMENT → PLAYER + MATCHUP EFFICIENCY → JOINT MONTE CARLO DISTRIBUTION**

Sportsbook information is downstream benchmarking/pricing only. It does **not** alter football projections upstream.

Operating loop:

> **historical research/backtest → prove a mechanism helps pregame projection → integrate it into the live model → emit one authoritative football projection → compare that projection to Vegas → evaluate after the game**

The user does not want two competing model numbers requiring manual selection at bet time.

---

## 2. Non-negotiable methodology

- Historical science is strict walk-forward / leakage-safe.
- Freeze hypothesis, population, candidate definitions, metrics, thresholds, gates and authority ceiling **before** implementation/results.
- Preserve scientific failures exactly; never lower gates after seeing results.
- Mechanical failures may receive only the minimum documented value-neutral repair.
- No sportsbook football inputs upstream of football projections.
- Do not refit a protected model when a frozen study says to consume serialized authority.
- Production changes require a separate explicit qualification/promotion step.
- A promoted research component becomes part of the single authoritative output; predecessor values may remain only as audit fields.
- Production code must remain untouched during R27 V1 research.
- R22 must remain untouched during R27 mean discovery.

---

## 3. Original research hierarchy

Preserve this order unless the user explicitly changes it:

1. **RB receiving efficiency / receiving-yard mean**
2. **QB attempts / dropbacks / pass rate / YPA / sacks / scrambles**
3. **Selective WR/TE efficiency/distribution**
4. **Shared QB ↔ receiver conservation**
5. **Unified coherent game simulation**
6. **ATD**
7. **Game ML / spread / total**
8. **Week-1 final output package**

Desired end-state simulation:

> **plays → pass/rush decision → player opportunity → outcomes → yards/explosives/TDs → game-state feedback → possessions/scoring → final score**

**Current position:** item #1. R26 solved/improved RB receptions/opportunity allocation and is production-active. The current lane is now RB receiving-yard **mean/efficiency**, with R26 fixed upstream and R22 held as the already-qualified mean-neutral tail layer until a better mean is proven.

Parked unless explicitly reopened:
- 2026 ESPN/nflverse hierarchy audit
- RotoBaller WR/CB parser work
- old failed/mixed QB/WR/TE salvage audit
- acute same-week RB injury / role-inheritance transform.

---

## 4. Current production authority

### Production head

`main@bb76ba9eabb08e2f0875a9af49301c3877f4141f`

Current stack:
- QB passing-yard mean: **M89/M90**
- QB distribution: **mean-neutral C2**
- WR: **M38 WR1 + WR-R15 WR2+**
- TE: **TE-R5P**
- RB rushing: **RB-P3**
- RB receptions: **R26 production refinement**
- RB receiving-yard distribution/tails: **R22**, using pinned R19 assets and preserving receiving-yard mean
- sportsbook: downstream only.

### R26 production qualification authority

- branch: `production-rb-r26-receptions-v1`
- qualification head: `343372586bd4979c34487761d0af49b5986f68e8`
- run: `34417740186`
- job: `102686263562`
- artifact: `10129819192`
- digest: `sha256:f8fbd74187e811fc765c6d236c4c005c19be6d007d1bec33800790e56eabdcde`
- disposition: `RB_R26_WEEK1_RECEPTIONS_PRODUCTION_INTEGRATION_PASS_READY_FOR_PROMOTION`
- gates: **35/35 PASS**

R26 changes the RB/FB receptions MC component inside its qualified scope. The existing ML/state/ensemble consumes that component and emits **one final authoritative receptions `model_proj`**. P3 rushing, R22 receiving yards, QB, WR, TE, and non-RB arrays remain unchanged by the R26 receptions promotion.

Initial promotion commit: `1f0ea0d697f9a5b66cba42facc424a0185a147ed`.
A later static repo audit failure was mechanical only: the old audit expected an exact V4 import string in the public V3 compatibility wrapper. Compilation passed and R26 science was unchanged. Repair note:
`docs/production/RB_R26_PROMOTION_STATIC_AUDIT_COMPAT_MECHANICAL_REPAIR_V1.md`.
Final repaired production head is `bb76ba9eabb08e2f0875a9af49301c3877f4141f`.

### Post-promotion Full Slate verification

- workflow: `Full Slate`
- run: `34418491952`
- job: `102688556296`
- head: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`
- conclusion: **SUCCESS**
- artifact: `10130055472`
- artifact name: `run_34418491952`
- digest: `sha256:d84bbc9fb976e3862b19d458a929888c1ab5e0f7cec49e1f4705a961a236e15f`

Live odds were disabled/not requested in this verification, so this proves football-stack and repository wiring, not a fresh live-odds board.

### Post-promotion Repo CI

- run `34418492012`
- job `102688556448`
- head `bb76ba9eabb08e2f0875a9af49301c3877f4141f`
- conclusion **SUCCESS**
- compile/static audit/unit tests PASS.

---

## 5. Protected pre-R26/R22 lineage

Protected pre-R26 Full Slate authority:
- commit `f8417f55b04ce0e19baf260e9d532765034c47f1`
- run `34317211395`
- artifact `10090547415`
- digest `sha256:7eab77e41c5879d4f54d87497eee0d1186010784cfc4ef966178930e16fb7c3b`
- 468 players / 32 teams / 16 games.

R22 integration authority:
- run `34298516960`
- artifact `10084118525`
- digest `sha256:2391bd9914e9d0029c63529829496fb9b462d008bdcc3f8c37bb3e1079580bd1`
- head `4d0690f9827466d0792be42752bcc6b6d8a03f96`
- disposition `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_INTEGRATION_PASS`
- R22 changes receiving-yard distribution/tails while preserving receiving-yard means and receptions.

R19 serialized tail authority:
- run `34288244770`
- artifact `10080377483`
- digest `sha256:11432b9d7b7f2367935a862b63c30df9f40955e479806ab67d90921b63a907c7`
- inner model SHA `9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba`
- residual pools SHA `c69a268a5a1683e846bcb5f59fe55bcae20d70c679792449ea77e6548b37a362`.

---

## 6. R26 science/prospective lineage — preserve failures and support

R26 parent:
- run `34356222339`
- artifact `10106271075`
- digest `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`
- disposition `RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW`
- 19/20; sole blocker: 2023 all-season vacancy-incumbent receptions MAE worsened ~4.54% vs <=2% cap.

R26E:
- run `34368268224`
- artifact `10110785184`
- digest `sha256:2b64fe25a1024136f2bb2cdde42bc74de290b6b9df63b4bb6cd675f6095b8bb7`
- disposition `WEEK1_COMPONENT_NOT_QUALIFIED_NO_SHADOW`
- 19/20; 2020 W1 worsened ~8.9668%; 2021-2025 all improved.

R26J:
- run `34374987828`
- artifact `10113466373`
- digest `sha256:7d886f228487e052ee1617cc5f42b974cd9da6a9d448f3a86b1de82cb4fc4f46`
- disposition `2020_SOURCE_REGIME_DISTINCT_FOR_MECHANISM_FOLLOWUP`.

R26K:
- run `34376961740`
- artifact `10114261724`
- digest `sha256:74a3d9ac58fca360f6d2d23e19b28e1254f38e49e3be23878ee1b37cebe22c88`
- disposition `2020_SPECIFIC_MECHANISM_NO_REPLICATED_ROUTER`.
- Do not exclude 2020 or invent a router.

R26L:
- run `34389455694`
- artifact `10119058769`
- digest `sha256:3351dfb5bbf6b571174a94ddf0a03179d70786edbb91d316e4fdf2c3cf005c46`
- disposition `2026_SOURCE_REGIME_MODERN_LIKE_FOR_PROSPECTIVE_QUALIFICATION`.

R26M:
- run `34390505549`
- artifact `10119429741`
- digest `sha256:1306a3a2e58a0b129ac7e9fe34ad6407d87c491494e8fc27dd0284ba96996b76`
- disposition `2026_WEEK1_UNMODIFIED_R26_SHADOW_CANDIDATE_DESIGN_QUALIFIED`.

R26N structural candidate:
- run `34396075045`
- job `102616001356`
- artifact `10121598376`
- digest `sha256:887929203053cb62904aaaeda9d995c9645163814da181972799f08fe4465c62`
- disposition `R26N_2026_WEEK1_STRUCTURAL_CANDIDATE_PASS_READY_FOR_SHADOW_INTEGRATION_DESIGN`
- 28/28 PASS
- 107 RB/FB; 31 vacancy teams; 104 changed; 3 CIN controls.

R26O canonical MC integration:
- run `34399750746`
- job `102628405629`
- artifact `10123070453`
- digest `sha256:27307ad84c232935aed3b25c0e0c9bff4da70cd2915376f41dfa3f85c25f14d0`
- disposition `R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_PASS_READY_FOR_PROSPECTIVE_SEAL`
- 38/38 PASS
- 104 intended RB/FB reception arrays changed; 3 CIN exact; all non-reception arrays exact; R22 mean delta `3.552713678800501e-15`; 25k draws seed42.

Preserved R26O 37/38 mechanical run:
- run `34398759284`
- artifact `10122672501`
- digest `sha256:6d72de1eb8902fd8d956feb01b4d1a3cb5847aa58417ad4f416f3e1102e96e82`
- Gate15 evidence wiring only.

R26P repair authority:
- run `34399525657`
- artifact `10122862934`
- digest `sha256:2e596f3a3bcbef156983664d332ded94ef41dae1402d1a96ab0b61d185c68d47`
- disposition `R26P_GATE15_EVIDENCE_WIRING_DEFECT_CONFIRMED_MECHANICAL_RERUN_AUTHORIZED`
- 17/17.

R26Q immutable pregame seal:
- run `34400524030`
- job `102630996205`
- artifact `10123251043`
- digest `sha256:dd3ec0e8e3831ab7f2255c2e5abf343cda8a7943d33a1d4863e52372d6f858a1`
- disposition `R26Q_2026_WEEK1_RECEPTIONS_PROSPECTIVE_SEAL_PASS_READY_FOR_OBSERVATION`
- 28/28 PASS
- 107 exact arrays; 104 changed; 3 CIN; 25k seed42
- NPZ SHA `7933bd7629d0e7108fe181e7a5474f8c8f6e8512b6a1b35081d3a8d71af8fe06`
- **Do not recompute the sealed R26Q record.**

R26R pregame market observation:
- run `34401814588`
- job `102635265504`
- artifact `10124274040`
- digest `sha256:b4d3e573909803e892d57858c9b0c4bdac089bcf078c1d6f6e53366e4622303e`
- disposition `R26R_2026_WEEK1_PROSPECTIVE_OBSERVATION_SNAPSHOT_PASS_MARKET_CAPTURED`
- 30/30 PASS
- 320 all-position reception lines; 70 RB/FB book lines; 35 matched sealed players; sportsbook downstream only.

---

## 7. R26S locked postgame evaluator

R26S is the postgame scorecard for sealed R26Q, **not the predictive model**.

Frozen authority:
- plan commit `63722c61ca1b408b8ae77eba383e5fa8daecf9f2`
- evaluator commit `e42099dbf11cdd510eb64e43a0910444c75ec7f8`
- lock commit `3a706fa52f91f6584f6fd6594563239dd6ea3b53`.

Canonical repaired pregame dry-run:
- run `34411889262`
- job `102667966181`
- artifact `10127562840`
- digest `sha256:e9ecd949816eafd813a50e60e8af28ce387e668848483ffcda3dfadf7614919e`
- disposition `R26S_2026_WEEK1_POSTGAME_PROSPECTIVE_EVALUATION_INCOMPLETE_NO_DECISION`
- primary evaluable rows 0 because Week1 outcomes do not exist yet.

Preserved first mechanical run:
- run `34405689393`
- artifact `10125223220`
- digest `sha256:6130a64872739436a0626ee45eb249a06edb3281a8f750755c8b4483af6aa9bd`
- metadata-contract failure only; scientific gates not evaluated.

**After Week1:** rerun the exact locked R26S evaluator unchanged when authoritative weekly stats and snaps are available.

---

## 8. Week1 RB operational readiness before R26 promotion

Final pre-promotion readiness:
- run `34412854521`
- job `102671015758`
- artifact `10127920603`
- digest `sha256:b619429022a5bfc50257ebbb55600c1ff1299e4b706997d278545f66a469dd5e`
- disposition `RB_WEEK1_2026_PREGAME_OPERATIONAL_READINESS_PASS_PRODUCTION_STACK_READY_R26_SIDECAR_READY`
- 35/35 PASS
- 107 RB/FB; 32 teams; exact P3/R26 universe alignment; 94 RB exact R22 alignment; outcomes0.

A later readiness run exposed one exact current-roster swap:
- sealed-only NE Corey Kiner, RB3
- current-only NE Lan Larison, RB3
- matched-player role changes 0.

Do not infer or invent a general role-inheritance transform from this one swap.

---

## 9. Current RB model tree

> **current roster / game context**
> → team / position opportunity
> → **RB-P3 rushing opportunity + yards**
> → **R26 RB receiving/receptions entitlement refinement**
> → **current receiving efficiency / mean machinery**
> → **R22 receiving-yard distribution/tails around the existing mean**
> → existing ML/state/ensemble
> → one final player projection
> → sportsbook comparison downstream.

Important limitation:

**R26 improved the opportunity/reception side, but R22 deliberately preserved the existing RB receiving-yard mean.** The current scientific question is whether R26's better opportunity itself improves receiving-yard mean and, if not enough, which strict-prior efficiency signals add incremental accuracy.

---

## 10. Prior receiving-yard evidence that constrains R27

R23/R24 already tested broader receiving-entitlement / production-efficiency ideas. In particular, **R24 did not qualify receiving-yard mean promotion**: better generic entitlement multiplied by existing production efficiency did not robustly improve overall RB receiving-yard MAE.

Do not retest R24 under a new label.

R27 is scientifically distinct because it tests the **exact R26 vacancy-gated mechanism** that was later qualified/promoted for receptions:
- vacancy-gated rather than globally applied;
- exact RB-room receiving-opportunity mass conservation;
- strict-prior R8/R9 player receiving-identity residual;
- only vacancy rooms change;
- stable rooms remain exact baseline.

---

## 11. ACTIVE R27 V1 — FROZEN, NOT YET IMPLEMENTED OR RUN

### Exact authority

Research branch:
`research-rb-r27-receiving-yard-mean-decomposition-v1`

Frozen plan:
`docs/research/RB_R27_RECEIVING_YARD_MEAN_DECOMPOSITION_V1_FROZEN_PLAN.md`

- plan commit: **`5333d7e1cc33dcb567d03c924c774afb6877e932`**
- plan blob SHA: **`4ad4801dbe600238dd7f1090df0a6596ff8a6a46`**
- production parent: `main@bb76ba9eabb08e2f0875a9af49301c3877f4141f`
- handoff parent: `d63ba0216d43e4763954e0be7351ece06f8fa6b4`
- status: **FROZEN BEFORE IMPLEMENTATION / CANDIDATE EXECUTION**
- **No R27 evaluator exists yet.**
- **No implementation lock exists yet.**
- **No R27 workflow run exists yet.**
- **No candidate results have been seen.**

### Scientific question

> Does the exact historical R26 opportunity redistribution improve player-level RB receiving-yard point means when the existing pregame receiving-efficiency estimate is held fixed? If not, where does the remaining receiving-yard error live, and what efficiency work is justified next?

### Exact historical R26 parent pinned in R27

- R26 branch `research-rb-r26-role-transition-entitlement-v1`
- R26 frozen plan blob SHA `e8a2633454dcfb471bda57fa43e40fbb33bd7697`
- R26 evaluator blob SHA `17170baa072bcc3157dee03c7a0de75297060e71`.

Exact mechanism may not be altered:
1. target-week RB/FB room;
2. `VACANCY_ACTIVE = room_exits_n >= 1`;
3. stable/no-vacancy rooms exact baseline;
4. vacancy rooms preserve exact production RB-room target mass;
5. strict-prior R8/R9 receiving identity;
6. R8 Ridge fit on immediately prior-season training data only;
7. R9 reliability from rolling-origin OOF training predictions only, clipped `[0,1]`;
8. score = `log(production_within_rb_share + EPS) + reliability * raw_R8_residual`;
9. softmax only inside current RB/FB room, rescaled to exact RB-room mass;
10. non-RB entitlement unchanged.

### Frozen historical folds

- test seasons 2020-2025 REG
- each test season uses immediately prior season for R26 fitting
- 2020 Weeks 1-17
- 2021-2025 Weeks 1-18
- 2020 and 2023 may not be removed after results.

Primary/predeclared cohorts:
- ALL
- VACANCY_ACTIVE
- VACANCY_INCUMBENT
- VACANCY_RB1_INCUMBENT
- VACANCY_RB2PLUS_INCUMBENT
- VACANCY_NEW_VETERAN
- VACANCY_NO_PRIOR_NFL
- WEEK1
- WEEKS2PLUS
- each season 2020-2025.

### Frozen receiving-yard mean formulas

For each player-game:
- `baseline_targets = team_targets * baseline_entitlement_tgt_share`
- `candidate_targets = team_targets * R26_candidate_entitlement_tgt_share`
- `production_ypt = max(rules_ypt, fallback bayes_ypt, 0)` using only pregame context
- `production_catch_rate = clip(rules_catch_rate / bayes_receptions_per_target fallback, 0.35, 0.95)`.

Baseline:
- `baseline_rec_yards = baseline_targets * production_ypt`

Primary R27 V1 candidate:
- **`r26_opportunity_rec_yards = candidate_targets * production_ypt`**

Reception-path identity audit:
- baseline receptions = baseline targets × same production catch rate
- R26 receptions = candidate targets × same production catch rate
- implied production YPR = production YPT / production catch rate
- bridge yards = R26 receptions × implied production YPR
- bridge yards must equal the primary candidate within `<1e-10`.

No actual-game YPT/YPR/catch rate may enter baseline or candidate construction.

### Frozen metrics

Primary: receiving-yard MAE.
Also: N, RMSE, bias, median AE, p75 AE, p90 AE, Pearson, Spearman, >=30-yard large-error miss rate, plus target MAE and reception MAE chain diagnostics.

### Frozen gates

Structural/leakage Gates 1-12 require:
- sportsbook upstream0
- outcomes/future features0
- exact R26 strict-prior contract/gate
- RB-room mass conservation `<1e-10`
- non-RB entitlement delta `<1e-12`
- production files unchanged
- R22 untouched/not used as mean correction
- stable rooms exact baseline
- reception bridge identity `<1e-10`
- no actual-game efficiency in candidate
- all 2020-2025 folds complete.

Scientific Gates 13-26 require, among other exact frozen thresholds:
- pooled VACANCY_ACTIVE and VACANCY_INCUMBENT receiving-yard MAE improve
- VACANCY_ACTIVE RMSE non-worse
- absolute bias non-worse
- p90 no worse than +2%
- at least 4/6 seasons improve VACANCY_ACTIVE MAE
- no season worsens >3%
- neither RB1 nor RB2+ vacancy-incumbent MAE worsens >1.5%; at least one improves
- ALL-RB MAE/RMSE no worse than +0.25%
- Week1 MAE no worse than +0.50%
- vacancy target and reception MAE improve or remain within +0.10%.

Gate 27 requires pooled VACANCY_ACTIVE receiving-yard MAE improvement of at least **0.50%** for direct material support.

### Frozen dispositions

All 1-27 pass:
`R27_R26_OPPORTUNITY_REC_YARD_MEAN_SUPPORT_READY_FOR_INTEGRATION_DESIGN`

Gates 1-26 pass but 27 does not:
`R27_R26_OPPORTUNITY_SAFE_BUT_EFFICIENCY_WORK_REQUIRED`

Any structural/scientific support gate fails:
`R27_R26_OPPORTUNITY_TRANSLATION_MIXED_OR_FAIL`

A PASS does **not** directly mutate production. It can only authorize the next separately frozen integration/recentering qualification.

### If V1 is insufficient

R27B may test only **new strict-prior efficiency information on top of fixed R26 opportunity**. Candidate families can include shrunk YPT/YPR persistence, recent-vs-long-term efficiency, YAC/YACOE, aDOT/route/screen usage, explosive receiving propensity, RB role/archetype, opponent RB receiving efficiency/LB tackling/YAC environment, and QB checkdown/pressure context only if independent and leakage-safe.

Do not blindly multiply noisy raw YPR by R26 receptions.

---

## 12. Exact next actions from this handoff

1. Stay on branch `research-rb-r27-receiving-yard-mean-decomposition-v1` for R27 research code.
2. Read the exact frozen plan at commit `5333d7e1cc33dcb567d03c924c774afb6877e932` before writing code.
3. Implement an R27 evaluator/finalizer that reproduces the exact frozen R26 historical mechanism and changes only the receiving-yard mean candidate to `R26 candidate targets × same production YPT`.
4. Add explicit structural audits, especially stable-room exactness, RB-room pool conservation, no actual efficiency leakage, and reception-path equivalence.
5. Create an **implementation lock before any candidate execution/results**.
6. Add a dedicated research workflow for the 2020-2025 folds.
7. Run it and preserve the first result exactly, whether PASS, mixed, scientific FAIL, or mechanical FAIL.
8. If a mechanical defect occurs, document and make only the minimum value-neutral repair; do not alter the frozen candidate or gates.
9. If R27 V1 does not qualify direct mean integration but structural integrity holds, design/freeze R27B efficiency research rather than tuning V1 after results.
10. Do not touch R22 or production during R27 V1.
11. After Week1, rerun locked R26S unchanged for prospective audit.
12. Update this handoff on `main` at the next material checkpoint with exact commit/run/job/artifact/digest/disposition lineage.

---

## 13. One-sentence current state

**R26 receptions is production-active at `main@bb76ba9eabb08e2f0875a9af49301c3877f4141f`; the active next study is R27 V1 on branch `research-rb-r27-receiving-yard-mean-decomposition-v1`, whose plan was frozen at `5333d7e1cc33dcb567d03c924c774afb6877e932` before implementation/results to test whether exact R26 opportunity redistribution improves RB receiving-yard mean using the same production YPT, with R22 and all production code untouched.**
