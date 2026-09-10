# CURRENT NFL RESEARCH HANDOFF — READ FIRST

**Repository:** `dkaps6/imtiredofthis`  
**Purpose:** canonical cross-chat continuity ledger for the NFL pregame projection research program.  
**Last updated:** 2026-09-09 after the first valid R27 receiving-yard decomposition result, R27B V1 pre-execution novelty stop, the 2019-2025 novel-efficiency source audit, and the R27B V2 plan freeze.  
**Protected production-code authority:** `bb76ba9eabb08e2f0875a9af49301c3877f4141f`  
**Active research branch:** `research-rb-r27b-v2-novel-efficiency-context`  
**Exact stop point:** R27B V2 novel receiving-efficiency-context plan is frozen before implementation/model fitting/candidate execution/results at commit `bbaa0e2bbf32b182ca768555fa17548f8493be18`, plan blob `88c1377acc2e5389df092ec74ab876eaed468cdd`. No R27B V2 candidate result exists yet.

> **Future ChatGPT sessions / agents:** read this file first, then `AGENTS.md`, then `docs/research/RB_R27B_V2_NOVEL_EFFICIENCY_CONTEXT_FROZEN_PLAN.md` at the exact commit above. GitHub is canonical. Preserve failed science and mechanical failures separately, freeze before results, keep sportsbook data downstream only, and never silently change production or a frozen research mechanism.

---

## 1. Controlling objective and architecture

Project NFL outcomes as accurately as possible **pregame**, player by player and game by game: attempts, carries, targets, receptions, yards, touchdowns and downstream fair probabilities.

Core architecture:

> **GAME / TEAM OPPORTUNITY → POSITION / ROOM POOL → INDIVIDUAL PLAYER ENTITLEMENT → PLAYER + MATCHUP EFFICIENCY → JOINT MONTE CARLO DISTRIBUTION**

Sportsbook information is an external benchmark/pricing layer, not the teacher of the football projection.

Operating loop:

> **historical research/backtest → prove a mechanism improves football prediction → integrate only qualified winners → emit one authoritative football projection → compare to reality and Vegas → grade prospectively**

The north star is to build a more accurate pregame representation of NFL football than the market. Matching Vegas is not the objective; predicting actual outcomes better than Vegas over valid samples is.

---

## 2. Non-negotiable methodology

- Historical science is strict walk-forward / leakage-safe.
- Freeze hypothesis, population, candidate definitions, metrics, thresholds, gates and authority ceiling **before** implementation/results.
- Preserve scientific failures exactly; never lower gates after seeing results.
- Mechanical/plumbing failures may receive only the minimum documented value-neutral repair.
- No sportsbook football inputs upstream of football projections unless a separately frozen market-assisted experiment explicitly says so.
- Do not refit a protected model when a study says to consume pinned/serialized authority.
- Production changes require a separate explicit qualification/promotion step.
- A promoted component must feed the single authoritative production output; predecessor values may remain only as audit fields.
- R22 receiving-yard tail authority remains untouched during RB mean research.
- R26 receiving opportunity/receptions remains fixed during R27B.
- The assistant may disagree with a research direction, but must communicate the concern before changing any frozen mechanism, threshold, architecture or research direction.

---

## 3. Program hierarchy

Preserve this order unless explicitly changed:

1. **RB receiving efficiency / receiving-yard mean** — ACTIVE
2. **QB attempts / dropbacks / pass rate / YPA / sacks / scrambles**
3. **Selective WR/TE efficiency/distribution**
4. **Shared QB ↔ receiver conservation**
5. **Unified coherent game simulation**
6. **Anytime TD**
7. **Game ML / spread / total**
8. **Final operational package / prospective grading**

Desired end-state simulation:

> **plays → pass/rush decision → player opportunity → outcomes → yards/explosives/TDs → game-state feedback → possessions/scoring → final score**

Current position: item #1. R26 solved/improved the RB receiving opportunity/receptions side and is production-active. R27 showed that translating that improved opportunity through the existing production YPT helps in aggregate but is not robust enough for mean promotion. R27B V2 now isolates genuinely novel football-context efficiency information.

Parked unless explicitly reopened:
- 2026 ESPN/nflverse hierarchy audit
- RotoBaller WR/CB parser work
- old failed/mixed QB/WR/TE salvage audits
- acute same-week RB injury / role-inheritance transform

---

## 4. Current production authority

Protected production-code authority:

`bb76ba9eabb08e2f0875a9af49301c3877f4141f`

Current stack:
- QB passing-yard mean: **M89/M90**
- QB distribution: **mean-neutral C2**
- WR: **M38 WR1 + WR-R15 WR2+**
- TE: **TE-R5P**
- RB rushing: **RB-P3**
- RB receptions: **R26 production refinement**
- RB receiving-yard distribution/tails: **R22**, using pinned R19 assets and preserving the upstream receiving-yard mean
- sportsbook: downstream only

### R26 production qualification

- branch: `production-rb-r26-receptions-v1`
- qualification head: `343372586bd4979c34487761d0af49b5986f68e8`
- run: `34417740186`
- job: `102686263562`
- artifact: `10129819192`
- digest: `sha256:f8fbd74187e811fc765c6d236c4c005c19be6d007d1bec33800790e56eabdcde`
- disposition: `RB_R26_WEEK1_RECEPTIONS_PRODUCTION_INTEGRATION_PASS_READY_FOR_PROMOTION`
- gates: **35/35 PASS**

R26 changes the RB/FB receptions MC component only inside its qualified scope. P3 rushing, R22 receiving-yard tail logic, QB, WR, TE and non-RB arrays remain unchanged.

Initial promotion commit: `1f0ea0d697f9a5b66cba42facc424a0185a147ed`.
Final repaired production-code authority after a static-audit compatibility-only repair: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`.

Repair note:
`docs/production/RB_R26_PROMOTION_STATIC_AUDIT_COMPAT_MECHANICAL_REPAIR_V1.md`

### Post-promotion verification

Full Slate:
- run `34418491952`
- job `102688556296`
- head `bb76ba9eabb08e2f0875a9af49301c3877f4141f`
- conclusion **SUCCESS**
- artifact `10130055472`
- digest `sha256:d84bbc9fb976e3862b19d458a929888c1ab5e0f7cec49e1f4705a961a236e15f`
- live odds disabled, so this verifies football-stack/repository wiring, not a fresh sportsbook board

Repo CI:
- run `34418492012`
- job `102688556448`
- conclusion **SUCCESS**
- compile/static audit/unit tests PASS

---

## 5. Protected R22/R19 authority

R22 integration:
- run `34298516960`
- artifact `10084118525`
- digest `sha256:2391bd9914e9d0029c63529829496fb9b462d008bdcc3f8c37bb3e1079580bd1`
- head `4d0690f9827466d0792be42752bcc6b6d8a03f96`
- disposition `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_INTEGRATION_PASS`
- mean-neutral: changes receiving-yard distribution/tails while preserving receiving-yard means and receptions

R19 serialized tail authority:
- run `34288244770`
- artifact `10080377483`
- digest `sha256:11432b9d7b7f2367935a862b63c30df9f40955e479806ab67d90921b63a907c7`
- inner model SHA `9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba`
- residual pools SHA `c69a268a5a1683e846bcb5f59fe55bcae20d70c679792449ea77e6548b37a362`

R19/R22 are tail/distribution authorities, not a replacement point-mean efficiency model. R19 uses existing opportunity/mean/identity state and does not establish a target-depth/YAC/checkdown/opponent-context receiving-yard mean model.

---

## 6. R26 science / prospective lineage

Preserve the full failed/supporting path.

R26 parent:
- run `34356222339`
- artifact `10106271075`
- digest `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`
- disposition `RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW`
- 19/20; 2023 vacancy-incumbent receptions MAE worsened ~4.54%

R26E:
- run `34368268224`
- artifact `10110785184`
- digest `sha256:2b64fe25a1024136f2bb2cdde42bc74de290b6b9df63b4bb6cd675f6095b8bb7`
- disposition `WEEK1_COMPONENT_NOT_QUALIFIED_NO_SHADOW`
- 19/20; 2020 W1 worsened ~8.9668%, 2021-2025 improved

R26J:
- run `34374987828`
- artifact `10113466373`
- digest `sha256:7d886f228487e052ee1617cc5f42b974cd9da6a9d448f3a86b1de82cb4fc4f46`
- disposition `2020_SOURCE_REGIME_DISTINCT_FOR_MECHANISM_FOLLOWUP`

R26K:
- run `34376961740`
- artifact `10114261724`
- digest `sha256:74a3d9ac58fca360f6d2d23e19b28e1254f38e49e3be23878ee1b37cebe22c88`
- disposition `2020_SPECIFIC_MECHANISM_NO_REPLICATED_ROUTER`
- do not exclude 2020 or invent a router

R26L:
- run `34389455694`
- artifact `10119058769`
- digest `sha256:3351dfb5bbf6b571174a94ddf0a03179d70786edbb91d316e4fdf2c3cf005c46`
- disposition `2026_SOURCE_REGIME_MODERN_LIKE_FOR_PROSPECTIVE_QUALIFICATION`

R26M:
- run `34390505549`
- artifact `10119429741`
- digest `sha256:1306a3a2e58a0b129ac7e9fe34ad6407d87c491494e8fc27dd0284ba96996b76`
- disposition `2026_WEEK1_UNMODIFIED_R26_SHADOW_CANDIDATE_DESIGN_QUALIFIED`

R26N:
- run `34396075045`
- job `102616001356`
- artifact `10121598376`
- digest `sha256:887929203053cb62904aaaeda9d995c9645163814da181972799f08fe4465c62`
- disposition `R26N_2026_WEEK1_STRUCTURAL_CANDIDATE_PASS_READY_FOR_SHADOW_INTEGRATION_DESIGN`
- 28/28 PASS
- 107 RB/FB; 31 vacancy teams; 104 changed; 3 CIN controls

R26O canonical MC integration:
- run `34399750746`
- job `102628405629`
- artifact `10123070453`
- digest `sha256:27307ad84c232935aed3b25c0e0c9bff4da70cd2915376f41dfa3f85c25f14d0`
- disposition `R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_PASS_READY_FOR_PROSPECTIVE_SEAL`
- 38/38 PASS
- 104 intended RB/FB reception arrays changed; 3 CIN exact; all non-reception arrays exact; R22 mean delta `3.552713678800501e-15`; 25k draws seed42

Preserved R26O 37/38 mechanical run:
- run `34398759284`
- artifact `10122672501`
- digest `sha256:6d72de1eb8902fd8d956feb01b4d1a3cb5847aa58417ad4f416f3e1102e96e82`
- Gate15 evidence wiring only

R26P repair authority:
- run `34399525657`
- artifact `10122862934`
- digest `sha256:2e596f3a3bcbef156983664d332ded94ef41dae1402d1a96ab0b61d185c68d47`
- disposition `R26P_GATE15_EVIDENCE_WIRING_DEFECT_CONFIRMED_MECHANICAL_RERUN_AUTHORIZED`
- 17/17

R26Q immutable pregame seal:
- run `34400524030`
- job `102630996205`
- artifact `10123251043`
- digest `sha256:dd3ec0e8e3831ab7f2255c2e5abf343cda8a7943d33a1d4863e52372d6f858a1`
- disposition `R26Q_2026_WEEK1_RECEPTIONS_PROSPECTIVE_SEAL_PASS_READY_FOR_OBSERVATION`
- 28/28 PASS
- 107 exact arrays; 104 changed; 3 CIN; 25k seed42
- NPZ SHA `7933bd7629d0e7108fe181e7a5474f8c8f6e8512b6a1b35081d3a8d71af8fe06`
- **do not recompute this sealed record**

R26R pregame market observation:
- run `34401814588`
- job `102635265504`
- artifact `10124274040`
- digest `sha256:b4d3e573909803e892d57858c9b0c4bdac089bcf078c1d6f6e53366e4622303e`
- disposition `R26R_2026_WEEK1_PROSPECTIVE_OBSERVATION_SNAPSHOT_PASS_MARKET_CAPTURED`
- 30/30 PASS
- 320 all-position reception lines; 70 RB/FB lines; 35 matched sealed players; sportsbook downstream only

---

## 7. R26S postgame evaluator

R26S is the scorecard for sealed R26Q, not the predictive model.

Frozen authority:
- plan commit `63722c61ca1b408b8ae77eba383e5fa8daecf9f2`
- evaluator commit `e42099dbf11cdd510eb64e43a0910444c75ec7f8`
- lock commit `3a706fa52f91f6584f6fd6594563239dd6ea3b53`

Canonical pregame dry-run:
- run `34411889262`
- job `102667966181`
- artifact `10127562840`
- digest `sha256:e9ecd949816eafd813a50e60e8af28ce387e668848483ffcda3dfadf7614919e`
- disposition `R26S_2026_WEEK1_POSTGAME_PROSPECTIVE_EVALUATION_INCOMPLETE_NO_DECISION`
- evaluable rows 0 because Week1 outcomes were not yet available

Preserved first mechanical run:
- run `34405689393`
- artifact `10125223220`
- digest `sha256:6130a64872739436a0626ee45eb249a06edb3281a8f750755c8b4483af6aa9bd`
- metadata-contract failure only

After Week1 outcomes/stat/snap authority is available, rerun the exact locked R26S evaluator unchanged.

---

## 8. RB operational readiness note

Final pre-promotion readiness:
- run `34412854521`
- job `102671015758`
- artifact `10127920603`
- digest `sha256:b619429022a5bfc50257ebbb55600c1ff1299e4b706997d278545f66a469dd5e`
- disposition `RB_WEEK1_2026_PREGAME_OPERATIONAL_READINESS_PASS_PRODUCTION_STACK_READY_R26_SIDECAR_READY`
- 35/35 PASS
- 107 RB/FB; 32 teams; exact P3/R26 universe alignment; 94 RB exact R22 alignment; outcomes0

A later readiness run exposed one current-roster swap:
- sealed-only NE Corey Kiner, RB3
- current-only NE Lan Larison, RB3
- matched-player role changes 0

This proves an operational current-roster refresh problem exists; it does **not** justify a general role-inheritance formula.

---

## 9. Current RB model tree

> current roster / game context  
> → team / position opportunity  
> → **RB-P3 rushing opportunity + yards**  
> → **R26 RB receiving/receptions entitlement refinement**  
> → **production Bayesian/rules receiving efficiency (`bayes_ypt` → `rules_ypt`)**  
> → **R22 receiving-yard distribution/tails around the upstream mean**  
> → existing ML/state/ensemble  
> → one final football projection  
> → sportsbook comparison downstream

Key limitation: R26 improves receiving opportunity/receptions. R22 is intentionally mean-neutral. The remaining research problem is whether genuinely incremental pregame efficiency/context information can improve the receiving-yard mean beneath unchanged R22.

---

## 10. Prior receiving-yard science that constrains current work

### Production efficiency already exists

`scripts/modeling/bayesian_v2.py` already builds `bayes_ypt` using position-family prior + prior/current player evidence. `scripts/modeling/simulation_rules.py` uses that football state and existing matchup/pass-efficiency logic to create `rules_ypt`.

Therefore generic player YPT persistence/shrinkage is not a novel R27B hypothesis.

### R23

R23 tested improved RB receiving opportunity/receptions plus a strict-prior shrunk-YPR efficiency component using frozen 6-game recent and 16-game stabilizing histories.

Authoritative run:
- run `34332613867`
- head `0b4e641ec9fa6ba7ea9464a140268dfdbb897d28`
- artifact `10096546836`
- digest `sha256:e3e3087f03d0a8d697225e61de435e9a1668ef74e35ca850e60ffe451e9c66b6`
- disposition `MIXED_OR_FAIL_NO_PROMOTION`

R23 pooled 2023-2025:
- target MAE improved `1.375079 → 1.352808`
- reception MAE improved `1.161790 → 1.151914`
- receiving-yard MAE worsened `10.683986 → 10.725492` (~+0.39%)
- p90 receiving-yard AE worsened ~3.35%
- RB1 receiving-yard MAE worsened ~2.68%
- RB2+ improved ~1.77%

R23 is a preserved scientific null for combined opportunity + generic shrunk-YPR efficiency. Do not recreate it.

### R24

R24 intentionally removed R23's failed new YPR component and paired the improved opportunity with unchanged production YPT. It did not qualify receiving-yard mean promotion. Do not rebrand R24.

### Earlier diagnostic decomposition

`scripts/backtest/decompose_receiving_error.py` already decomposed receiving error into opportunity, catch conversion and YPT error. Identifying YPT as an error source is therefore not itself a new study.

---

## 11. R27 V1 — COMPLETE FIRST VALID SCIENTIFIC RESULT

Research branch:
`research-rb-r27-receiving-yard-mean-decomposition-v1`

Frozen plan:
`docs/research/RB_R27_RECEIVING_YARD_MEAN_DECOMPOSITION_V1_FROZEN_PLAN.md`

### Frozen/implementation lineage

- plan commit `5333d7e1cc33dcb567d03c924c774afb6877e932`
- plan blob `4ad4801dbe600238dd7f1090df0a6596ff8a6a46`
- exact R25/R26 parent staging `62e1b285f930e4cc2c437d85710c4579c0941faa`
- evaluator implementation `934443e2c750b3d506fb9722043a54bd39f846e7`
- finalizer / 27-gate implementation `2ceb6b372bf04156cf61ffc23234b12c87b6352a`
- implementation lock `a20f172db9d635a01029e3af8c772e0556ee2666`
- workflow commit `d6de4665fb520240d09225957d55b26a6c4cdcfc`
- evaluator locked blob `a284592a0b2f948b8f12f973ca9eb7d9c08d6b23`
- finalizer locked blob `4f8bf99b25210e8c88a6c8c4511ff9d565d47652`

### Preserved mechanical repair lineage

- `7740d02be1f916accc3b8d8ace60583a21566d84` — freeze historical schedule mechanical repair
- `acaf8993d260871e8c409b21d5ec775eb2a53170` — add REG schedule staging repair
- `4fe26a31cd988fad84c0ac3d00426ad69fe86c71` — apply schedule scope repair
- `efebb966922ea6703236bd2ec60f1351fdc1a08c` — freeze player-log REG scope repair v2
- `e6e30d3da8bdbad461a5abeb04f50904b8753bcf` — fail-closed exact player-log blob guard
- `b7cfe5b2c450765208672fc6e2b017d0cb0c0dde` — stage exact R25 REG-scope player-log authority
- exact canonical player-log blob `b993bd46fd44c88785bb37344c756c51b5d39afa`

Mechanical runs preserved:
- run `34420935737` — failure before candidate; no scientific result
- run `34421267285` / job `102696999664` — schedule/player-log REG scope failure; no scientific result
- guard-only fail-close run `34423519513` — exact blob guard active before canonical file staging; no model/data execution

### First valid R27 scientific execution

- run `34423546037`
- job `102703879430`
- head `b7cfe5b2c450765208672fc6e2b017d0cb0c0dde`
- artifact `10132290573`
- artifact name `rb-r27-receiving-yard-mean-decomposition-v1`
- digest `sha256:cd5c9e26efa44c47ef3374e4deefe9038dc504c155799b48548ad64746e10e45`
- workflow conclusion **SUCCESS** at the structural level
- scientific disposition `R27_R26_OPPORTUNITY_TRANSLATION_MIXED_OR_FAIL`
- gates **24/27 PASS**

Material results vs original production-path baseline:
- VACANCY_ACTIVE receiving-yard MAE: **-1.0967854669869492%**
- VACANCY_ACTIVE RMSE: **-1.3337252245802733%**
- vacancy target MAE: **-2.0869346310231074%**
- vacancy reception MAE: **-1.8026686460321795%**
- ALL-RB receiving-yard MAE: **-0.23517744112927508%**
- WEEK1 receiving-yard MAE: **-3.3081859534149105%**
- VACANCY_RB2PLUS_INCUMBENT MAE: **-3.3577795492306994%**

Failed frozen gates:
- Gate17: VACANCY_ACTIVE p90 AE worsened **+2.077705464816737%** vs allowed +2.0%
- Gate19: 2023 VACANCY_ACTIVE MAE worsened **+4.915812026979438%** vs allowed +3.0%
- Gate20: VACANCY_RB1_INCUMBENT MAE worsened **+2.820339661055238%** vs allowed +1.5%

Season VACANCY_ACTIVE MAE changes:
- 2020 `-1.58%`
- 2021 `-2.35%`
- 2022 `-4.25%`
- 2023 `+4.92%`
- 2024 `-3.56%`
- 2025 `+2.87%`

Interpretation: exact R26 opportunity is supported and materially useful, but opportunity alone does not make receiving-yard mean robust enough to promote. The remaining weakness is downstream efficiency/context, especially RB1 and the upper-error tail. R27 is preserved as a scientific mixed/fail and may not be retuned.

Production remains unchanged.

---

## 12. R27B V1 — SUPERSEDED BEFORE EXECUTION

Branch:
`research-rb-r27b-receiving-efficiency-v1`

The initial V1 design was stopped after the user explicitly challenged novelty. Repository audit showed that part of V1 would overlap production empirical-Bayes YPT and R23's already-tested shrunk-YPR/persistence work.

V1 lineage:
- R27 result-record parent commit `886c8432ff811882e006d84a61385862e3be7839`
- V1 plan commit `c9c7906748065f25cf7cb0d5ce24259144465dc5`
- V1 feature-dataset builder commit `7ec3ccf53607e42a5bfdfbc1e2e6596f0c3d913c`
- V1 evaluator commit `a55e42ea7a7eaf7f318024ea0e4c1d7c4688f640`
- supersession record commit `ca5537321893eb5fecba890b3ee2aeb320dede1f`
- record: `docs/research/RB_R27B_V1_PREEXECUTION_NOVELTY_AUDIT_AND_SUPERSESSION.md`

Critical status:
- **no V1 workflow exists**
- **no V1 workflow run exists**
- **no V1 artifact exists**
- **no V1 candidate metric exists**
- **no scientific result exists**

V1 frozen files remain preserved as historical design evidence and are not authorized for execution.

Ruled out as the primary novelty for V2:
- career/current/trailing YPT
- career/current/trailing YPR
- generic catch-rate persistence
- generic empirical-Bayes YPT/YPR re-shrinkage

---

## 13. ACTIVE R27B V2 — NOVEL EFFICIENCY CONTEXT

Active branch:
`research-rb-r27b-v2-novel-efficiency-context`

Novelty boundary:
`docs/research/RB_R27B_V2_NOVELTY_BOUNDARY_AUDIT.md`

Novelty-boundary commit:
`4bdb9351ff9f40d956d88b915d70af86f3cf5a17`

### Pre-plan source audit — COMPLETE

Source-only audit script commit:
`25c337ba957e366d803c4f741d803dae82329cdb`

Workflow commit/head:
`fd347b8f743c3829049a8052fcd0bc5d1ba72222`

Workflow:
`RB R27B V2 Novel Efficiency Source Audit`

- run `34427150810`
- job `102714703407`
- conclusion **SUCCESS**
- artifact `10133096990`
- artifact name `rb-r27b-v2-source-audit`
- digest `sha256:a9905a02243af2120cd71e85bcd460d47c5ccc73af87779adde4dd5c20a8263b`
- audit script SHA256 `c9a2921d3d75c02048a46de1ed0e59b08c5120f1e4d94b458cf5f1ca8f10886c`

Audit constraints/results:
- 2019-2025 schema pass: all seasons
- model fit performed: false
- candidate projection created: false
- prediction error scored: false
- sportsbook inputs: 0
- receiver position resolution ~99.97%-99.99%
- RB air-yards non-null ~99.63%-99.77%
- RB YAC non-null on completed catches 100% each season
- RB yards non-null 100% each season
- team-week RB checkdown denominator finite 100% each season
- opponent RB context derivable every season

RB target rows audited by season:
- 2019: 3548
- 2020: 3326
- 2021: 3522
- 2022: 3366
- 2023: 3301
- 2024: 2974
- 2025: 2986

The source audit proves reconstructability only; it is not predictive evidence.

### Frozen V2 plan

Plan:
`docs/research/RB_R27B_V2_NOVEL_EFFICIENCY_CONTEXT_FROZEN_PLAN.md`

- plan commit `bbaa0e2bbf32b182ca768555fa17548f8493be18`
- plan blob `88c1377acc2e5389df092ec74ab876eaed468cdd`
- status **FROZEN BEFORE IMPLEMENTATION / MODEL FIT / CANDIDATE EXECUTION / RESULTS**

Scientific question:

> With exact R26 opportunity fixed, can strict-prior target-shape, YAC, QB/team checkdown and RB-specific opponent context explain residual receiving efficiency not already represented by production YPT, enough to repair R27's RB1/p90/2023 weaknesses without sacrificing aggregate accuracy?

### Exact V2 candidate scope

- R26 candidate targets/receptions remain immutable.
- V2 efficiency correction applies only to `VACANCY_ACTIVE` RB/FB rows, matching the exact R26 mechanism scope.
- stable/no-vacancy YPT and receiving-yard mean remain exact production/B1.
- production YPT is the baseline/offset, **not** a new predictive feature.
- generic historical YPT/YPR persistence features are forbidden.

Primary new football features, strict-prior only:
- player air yards / target
- player YAC / reception
- player screen/behind-LOS target rate
- player explosive-20 receiving target rate
- team/QB RB targets per official pass attempt
- team RB air yards / target
- team RB YAC / reception
- team RB screen rate
- opponent RB air yards allowed / target
- opponent RB YAC allowed / reception
- opponent RB catch rate allowed
- opponent RB explosive-20 allowed / target
- opponent RB screen rate faced
- structural role/vacancy flags

Estimator:
- standardized weighted Ridge
- alpha grid `[1.0, 10.0, 100.0, 1000.0]`
- rolling-origin inner-season selection only
- 2020 fallback alpha `100.0`
- residual correction clipped to `[-1.5, +1.5]` YPT

Outer folds:
- 2020-2025 REG
- all seasons retained
- no 2020/2023 special router

The frozen gates explicitly require repair of RB1 and 2023, not merely pooled improvement. Full details are in the frozen plan. A scientific PASS still cannot mutate production; it would authorize only a separate integration/recentering qualification under unchanged R22.

### Exact next action

Implement V2 feature materialization/evaluator/finalizer under the frozen plan, create an implementation lock before any candidate run, then add a dedicated workflow and preserve the first valid scientific result exactly.

Do not execute or reuse R27B V1 scripts as the V2 candidate.

---

## 14. Remaining-work register

### A. Finish RB receiving-yard mean lane

**A1 — R27 V1: COMPLETE / MIXED-FAIL PRESERVED**
- exact first valid run recorded above
- do not retune

**A2 — R27B V1: SUPERSEDED PRE-EXECUTION**
- preserve only as design lineage
- never infer a result from its scripts

**A3 — R27B V2: ACTIVE**
- implement exact frozen novel-context feature family
- lock implementation before candidate execution
- run 2020-2025 strict walk-forward
- preserve first scientific result exactly

**A4 — receiving-yard mean integration/recentering: CONDITIONAL**
- only if a mean candidate qualifies
- separately freeze how the new mean composes with mean-neutral R22
- do not directly mutate production from research result

### B. Current-roster / late-week role handling

- preserve sealed R26Q as scientific pregame authority
- production operations still need safe handling of late roster changes/inactives/new players
- the Corey Kiner → Lan Larison example is evidence of operational refresh need, not a validated inheritance formula
- separate frozen study required for automatic acute role transfer

### C. Post-Week1 prospective grading

- rerun exact locked R26S unchanged when authoritative Week1 outcomes/snaps are available
- do not recompute R26Q from postgame knowledge
- any later promoted RB mean should get its own prospective grading path

### D. QB opportunity/efficiency

After RB receiving-yard mean is closed or explicitly parked:
- attempts/dropbacks
- pass rate
- sacks/scramble conversion
- YPA/efficiency
- game-script/personnel effects where pregame-safe

Existing M89/M90/C2 remains authority until a separately qualified replacement exists.

### E. Selective WR/TE

- preserve WR M38 + WR-R15 and TE-R5P
- reopen only concrete unresolved error sources/new information
- do not recycle failed ideas under new labels

### F. Shared QB ↔ receiver conservation

Build one coherent passing environment so QB attempts/completions/yards and receiver targets/receptions/yards conserve together.

### G. Unified game simulation

Target causal simulation:
`plays → pass/rush → player opportunity → outcomes → yards/explosives/TDs → state feedback → possessions/scoring → final score`

### H. Anytime TD

Build after opportunity/efficiency/shared-game structure is trustworthy. Sportsbook ATD prices remain downstream.

### I. Game ML / spread / total

Build only when possession/scoring mechanics support coherent team/game distributions. Market lines remain benchmark/pricing references.

### J. Operational live package

Before final live use:
- fresh roster/depth/injury state
- fresh weather/context
- latest qualified football stack only
- full 32-team/16-game integrity
- live sportsbook capture downstream
- one authoritative model number per market
- model-vs-book comparison
- saved pregame artifacts/digests

### K. Continuity maintenance

At every major checkpoint update this file on `main` with:
- active branch / exact stop point
- plan/code/lock commits and blobs
- run/job/artifact/digest
- exact disposition/gate count
- mechanical vs scientific changes
- production-active vs research-only status
- next unresolved question

---

## 15. Future-chat / future-agent evidence reconstruction protocol

A future chat must be able to continue without conversational memory. GitHub is canonical.

### Read order

1. `CURRENT_NFL_RESEARCH_HANDOFF.md` from `main`
2. `AGENTS.md`
3. the active branch and exact frozen plan named at the top of this file
4. inspect active branch head vs exact stop point before writing anything

### Where truth lives

- production authority: production code/assets and production workflow lineage recorded here
- frozen research designs/results: `docs/research/` and research branches
- research executors: `scripts/backtest/` and matching research workflows
- production runtime: `scripts/modeling/`, `scripts/run_*`, `scripts/simulation_*`, `.github/workflows/full-slate.yml`
- serialized model authority: `data/models/` and `model/`
- run truth: GitHub Actions run → job → logs → artifact
- final scientific truth: artifact disposition/gate files, not a green workflow badge alone

### Completed-run protocol

For the exact branch/commit:
1. locate workflow run
2. record run ID/head/conclusion
3. fetch job ID/steps/logs
4. fetch artifact ID/name/digest
5. open artifact disposition/gates/metrics
6. classify failures as mechanical or scientific
7. preserve every first result exactly
8. never relabel or silently discard a failed run

### If handoff and repo disagree

Stop and reconcile against exact commits/runs/artifacts before continuing. Do not guess. Then update this handoff so the next session inherits one unambiguous state.