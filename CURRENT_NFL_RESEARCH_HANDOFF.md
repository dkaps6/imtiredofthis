# CURRENT NFL RESEARCH HANDOFF — READ FIRST

**Repository:** `dkaps6/imtiredofthis`  
**Purpose:** canonical cross-chat continuity ledger for the NFL pregame projection research program.  
**Last updated:** 2026-09-09 after R26 receptions production promotion, post-promotion Repo CI PASS, and post-promotion Full Slate PASS.  
**Current production head:** `bb76ba9eabb08e2f0875a9af49301c3877f4141f`  
**Current RB state:** RB-P3 rushing + R22 receiving-yard distribution/tails + **R26 receptions refinement are production-active in the Full Slate stack**. R26 is no longer research-only. R26S remains a locked postgame audit of the originally sealed prospective candidate.

> **Future ChatGPT sessions / agents:** read this file first, then `AGENTS.md`, then the exact active plan/result/repair files named below. Preserve failed studies and mechanical failures, distinguish science from plumbing, freeze plans before results, keep sportsbook data downstream only, and never silently change production.

---

## 1. User's controlling objective

Project individual NFL player outcomes as accurately as possible **pregame**, player by player: yards, receptions, carries, touchdowns and downstream fair probabilities.

Core architecture:

> **GAME / TEAM OPPORTUNITY → POSITION / ROOM POOL → INDIVIDUAL PLAYER ENTITLEMENT → PLAYER + MATCHUP EFFICIENCY → JOINT MONTE CARLO DISTRIBUTION**

Sportsbook information is downstream benchmarking/pricing only. It does **not** alter football projections upstream.

The operating loop is:

> **historical research/backtest → prove a mechanism helps pregame projection → integrate it into the live model → emit one authoritative football projection → compare that projection to Vegas → evaluate after the game**

The user does **not** want two competing model numbers that require a manual choice at bet time.

---

## 2. Original research hierarchy — preserve this order unless explicitly changed

The hierarchy the user originally established is:

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

Player props, ATD, moneyline, spread and total should ultimately be views of the same coherent simulated football game.

**Current position in hierarchy:** R26 solved/improved the RB receptions/opportunity allocation problem and is now production-active. The next scientific lane returns to **#1: RB receiving-yard mean / efficiency**, using promoted R26 receptions as the upstream opportunity authority.

Parked work that should not interrupt this hierarchy unless explicitly reopened:
- 2026 ESPN/nflverse hierarchy audit
- RotoBaller WR/CB parser work
- old failed/mixed QB/WR/TE experiment salvage audit
- acute same-week RB injury / role-inheritance transform (important later, but not ahead of the current hierarchy item)

---

## 3. Non-negotiable methodology

- Historical science is strict walk-forward / leakage-safe.
- Freeze hypothesis, population, candidate definitions, metrics, thresholds, gates and authority ceiling **before** implementation/results.
- Preserve scientific failures exactly; never lower gates after seeing results.
- Mechanical failures may receive only the minimum documented value-neutral repair.
- No sportsbook football inputs upstream of football projections.
- Do not refit a protected model when a frozen study says to consume its serialized authority.
- Production changes require a separate, explicit qualification/promotion step.
- A research component that is promoted becomes part of the **single authoritative output**, while predecessor values may remain only as audit fields.
- R26S is postgame evaluation only; it is not permission for the already-authorized R26 production integration.

---

## 4. Current production authority — Full Slate after R26 promotion

### Production head

- `main`: **`bb76ba9eabb08e2f0875a9af49301c3877f4141f`**
- public pricing compatibility entrypoint: `scripts/run_pricing_with_full_roster_universe_v3.py`
- final pricing authority now routes through **V5/R26**, while V4/R22 remains the protected parent and preserved R22 authority.

### Post-promotion Full Slate verification

- workflow: `Full Slate`
- run: **`34418491952`**
- job: **`102688556296`**
- head: **`bb76ba9eabb08e2f0875a9af49301c3877f4141f`**
- conclusion: **SUCCESS**
- artifact: **`10130055472`**
- artifact name: `run_34418491952`
- digest: **`sha256:d84bbc9fb976e3862b19d458a929888c1ab5e0f7cec49e1f4705a961a236e15f`**
- strict repository audits: PASS
- fresh Ourlads / team map / TeamForm / QB / weather / injuries / coverage / PlayerForm / Bayesian / ML / State / ensemble / P3 / QB C2 all built successfully.
- **Live odds were disabled/not requested in this verification run, so the live offer/pricing step was skipped.** This run proves the promoted football stack and repository wiring, not a fresh live-odds board.

### Post-promotion Repo CI

- workflow: `Repo CI`
- run: **`34418492012`**
- job: **`102688556448`**
- head: **`bb76ba9eabb08e2f0875a9af49301c3877f4141f`**
- conclusion: **SUCCESS**
- compile: PASS
- static repo audit: PASS
- unit tests: PASS

### Current model authorities

- QB passing-yard mean: **M89/M90**
- QB distribution: **mean-neutral C2**
- WR: **M38 WR1 anchor + WR-R15 WR2+ entitlement**
- TE: **TE-R5P**
- RB rushing: **RB-P3**
- RB receiving-yard distribution/tails: **R22**, using pinned R19 assets and preserving receiving-yard mean
- RB receptions: **R26 production refinement**, consumed inside the existing MC/ML/state ensemble to produce **one final authoritative receptions `model_proj`**

---

## 5. Protected pre-R26 production authority retained for lineage

Previous protected production / Full Slate authority:
- commit **`f8417f55b04ce0e19baf260e9d532765034c47f1`**
- Full Slate run `34317211395`
- artifact `10090547415`
- digest `sha256:7eab77e41c5879d4f54d87497eee0d1186010784cfc4ef966178930e16fb7c3b`
- 468 players / 32 teams / 16 games.

This remains the protected parent lineage for P3/R22/QB/WR/TE behavior. R26 promotion was qualified against that production chain rather than rebuilding unrelated components.

### R22 integration authority

- run `34298516960`
- artifact `10084118525`
- digest **`sha256:2391bd9914e9d0029c63529829496fb9b462d008bdcc3f8c37bb3e1079580bd1`**
- head `4d0690f9827466d0792be42752bcc6b6d8a03f96`
- disposition `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_INTEGRATION_PASS`
- R22 changes receiving-yard distribution/tails while preserving receiving-yard means and receptions.

### R19 serialized R9 authority used by R22

- run `34288244770`
- artifact `10080377483`
- digest **`sha256:11432b9d7b7f2367935a862b63c30df9f40955e479806ab67d90921b63a907c7`**
- inner model SHA **`9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba`**
- residual pools SHA **`c69a268a5a1683e846bcb5f59fe55bcae20d70c679792449ea77e6548b37a362`**

---

## 6. R26 historical science — preserve both support and failures

### R26 parent

- run `34356222339`
- artifact `10106271075`
- digest `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`
- disposition `RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW`
- **19/20** gates.
- sole blocker: all-season 2023 vacancy-incumbent reception MAE worsened about **4.54%** versus frozen <=2% cap.
- useful evidence preserved: vacancy signal, R9 identity, pooled targets/receptions, Week1 behavior, RB1 behavior, 5/6 seasons.

### R26E Week1 qualification

- run `34368268224`
- artifact `10110785184`
- digest `sha256:2b64fe25a1024136f2bb2cdde42bc74de290b6b9df63b4bb6cd675f6095b8bb7`
- disposition `WEEK1_COMPONENT_NOT_QUALIFIED_NO_SHADOW`
- **19/20**; 2020 Week1 worsened about **8.9668%**; **2021-2025 all improved**.

### R26J / R26K 2020 forensic follow-up

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

Conclusion: 2020 was distinct, but no robust historical router replicated. Do **not** invent a special 2020 exclusion rule.

### R26L 2026 transportability

- run `34389455694`
- artifact `10119058769`
- digest `sha256:3351dfb5bbf6b571174a94ddf0a03179d70786edbb91d316e4fdf2c3cf005c46`
- disposition `2026_SOURCE_REGIME_MODERN_LIKE_FOR_PROSPECTIVE_QUALIFICATION`
- 31 2026 Week1 vacancy rooms
- 6/7 features modern-closer
- modern distance `0.72555857`
- 2020 distance `1.22202458`
- ratio `0.593735`
- zero outcomes / sportsbook / same-week depth / R9 refit.

### R26M prospective synthesis

- run `34390505549`
- artifact `10119429741`
- digest `sha256:1306a3a2e58a0b129ac7e9fe34ad6407d87c491494e8fc27dd0284ba96996b76`
- disposition `2026_WEEK1_UNMODIFIED_R26_SHADOW_CANDIDATE_DESIGN_QUALIFIED`

---

## 7. R26N / O / Q / R prospective construction chain

### R26N — structural candidate

- run `34396075045`
- job `102616001356`
- artifact `10121598376`
- digest **`sha256:887929203053cb62904aaaeda9d995c9645163814da181972799f08fe4465c62`**
- disposition `R26N_2026_WEEK1_STRUCTURAL_CANDIDATE_PASS_READY_FOR_SHADOW_INTEGRATION_DESIGN`
- **28/28 PASS**
- 107 RB/FB, 31 vacancy teams, 104 changed, 3 CIN controls
- RB room entitlement conserved; non-RB exact; no outcomes/odds/same-week depth/R9 refit.

### R26O — receptions MC integration

- canonical run `34399750746`
- job `102628405629`
- artifact `10123070453`
- digest **`sha256:27307ad84c232935aed3b25c0e0c9bff4da70cd2915376f41dfa3f85c25f14d0`**
- disposition `R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_PASS_READY_FOR_PROSPECTIVE_SEAL`
- **38/38 PASS**
- 104 intended RB/FB reception arrays changed; 3 CIN exact; forbidden arrays changed 0
- all non-reception arrays exact
- R22 mean delta `3.552713678800501e-15`
- 25k draws, seed 42.

Preserved earlier R26O 37/38 mechanical fail:
- run `34398759284`
- artifact `10122672501`
- digest `sha256:6d72de1eb8902fd8d956feb01b4d1a3cb5847aa58417ad4f416f3e1102e96e82`
- sole failure was Gate15 evidence wiring.

R26P forensic repair authority:
- run `34399525657`
- artifact `10122862934`
- digest `sha256:2e596f3a3bcbef156983664d332ded94ef41dae1402d1a96ab0b61d185c68d47`
- disposition `R26P_GATE15_EVIDENCE_WIRING_DEFECT_CONFIRMED_MECHANICAL_RERUN_AUTHORIZED`
- 17/17.

### R26Q — immutable pregame seal

- run `34400524030`
- job `102630996205`
- artifact `10123251043`
- digest **`sha256:dd3ec0e8e3831ab7f2255c2e5abf343cda8a7943d33a1d4863e52372d6f858a1`**
- disposition `R26Q_2026_WEEK1_RECEPTIONS_PROSPECTIVE_SEAL_PASS_READY_FOR_OBSERVATION`
- **28/28 PASS**
- 107 exact arrays, 104 changed, 3 CIN, 25k draws, seed42
- NPZ SHA `7933bd7629d0e7108fe181e7a5474f8c8f6e8512b6a1b35081d3a8d71af8fe06`
- **Do not recompute this sealed candidate.** It remains the prospective scientific record even though the mechanism has now been promoted to production.

### R26R — pregame sportsbook observation

- run `34401814588`
- job `102635265504`
- artifact `10124274040`
- digest **`sha256:b4d3e573909803e892d57858c9b0c4bdac089bcf078c1d6f6e53366e4622303e`**
- disposition `R26R_2026_WEEK1_PROSPECTIVE_OBSERVATION_SNAPSHOT_PASS_MARKET_CAPTURED`
- **30/30 PASS**
- 320 all-position player-reception book lines
- 70 RB/FB book lines
- 35 matched sealed players
- sportsbook remained downstream; outcomes 0.

---

## 8. R26S — locked postgame evaluator

R26S is the **postgame scorecard for the sealed pregame R26Q candidate**, not the predictive model itself.

Frozen plan/evaluator/lock:
- plan `63722c61ca1b408b8ae77eba383e5fa8daecf9f2`
- evaluator `e42099dbf11cdd510eb64e43a0910444c75ec7f8`
- lock `3a706fa52f91f6584f6fd6594563239dd6ea3b53`

Canonical repaired pregame dry-run:
- run `34411889262`
- job `102667966181`
- artifact `10127562840`
- digest **`sha256:e9ecd949816eafd813a50e60e8af28ce387e668848483ffcda3dfadf7614919e`**
- disposition `R26S_2026_WEEK1_POSTGAME_PROSPECTIVE_EVALUATION_INCOMPLETE_NO_DECISION`
- primary evaluable rows 0 because Week1 games/outcomes do not yet exist.

Preserved first mechanical run:
- run `34405689393`
- artifact `10125223220`
- digest `sha256:6130a64872739436a0626ee45eb249a06edb3281a8f750755c8b4483af6aa9bd`
- mechanical metadata-contract failure only; scientific accuracy gates were not evaluated.

**After Week1:** rerun the exact locked R26S evaluator unchanged when authoritative weekly stats and snap counts are available. Promotion has already occurred by explicit user governance decision; R26S now audits that decision prospectively rather than granting permission for it.

---

## 9. Week1 RB operational-readiness authority before promotion

Final pre-promotion readiness:
- run `34412854521`
- job `102671015758`
- artifact `10127920603`
- digest **`sha256:b619429022a5bfc50257ebbb55600c1ff1299e4b706997d278545f66a469dd5e`**
- disposition `RB_WEEK1_2026_PREGAME_OPERATIONAL_READINESS_PASS_PRODUCTION_STACK_READY_R26_SIDECAR_READY`
- **35/35 PASS**
- 107 RB/FB, 32 teams, exact P3/R26 universe alignment, 94 RB exact R22 alignment, outcomes 0.

A later operational readiness run also exposed one current-roster swap:
- sealed-only NE Corey Kiner, RB3
- current-only NE Lan Larison, RB3
- no matched-player role changes.

Interpretation: roster freshness must remain explicit, but this RB3-for-RB3 drift did not authorize inventing role-inheritance logic.

---

## 10. R26 production promotion — CURRENT AUTHORITY

### Qualification branch

`production-rb-r26-receptions-v1`

Key production files:
- `scripts/modeling/rb_r26_receptions_production_adapter_v1.py`
- `scripts/run_pricing_with_full_roster_universe_v5_production.py`
- `scripts/validate_rb_r26_week1_production_integration_v1.py`
- `.github/workflows/production-rb-r26-receptions-v1.yml`
- `docs/production/RB_R26_WEEK1_RECEPTIONS_PRODUCTION_PROMOTION_CERTIFICATION.md`

### Frozen qualification

- qualification head: **`343372586bd4979c34487761d0af49b5986f68e8`**
- run: **`34417740186`**
- job: **`102686263562`**
- conclusion: **SUCCESS**
- artifact: **`10129819192`**
- artifact name: `rb-r26-week1-receptions-production-qualification-v1`
- digest: **`sha256:f8fbd74187e811fc765c6d236c4c005c19be6d007d1bec33800790e56eabdcde`**
- disposition: **`RB_R26_WEEK1_RECEPTIONS_PRODUCTION_INTEGRATION_PASS_READY_FOR_PROMOTION`**
- **35/35 PASS**

The qualification ran protected V4 control and V5/R26 candidate on the **exact same fixed Week1 football inputs**.

Scientific/production interpretation:
- R26 changes the RB/FB **receptions Monte Carlo component** for its qualified scope.
- It does **not** bypass the existing ML/state/ensemble framework.
- The existing ensemble consumes the R26-improved MC component and emits **one final authoritative receptions `model_proj`**.
- P3 rushing remains unchanged.
- R22 receiving-yard distributions remain unchanged.
- QB / WR / TE / non-RB distributions remain unchanged.
- sportsbook football inputs remain 0.
- Week1 outcomes used 0.

### Promotion to main

Initial promotion commit:
- `1f0ea0d697f9a5b66cba42facc424a0185a147ed`

A static audit then failed because the old audit required the public V3 compatibility wrapper to contain an exact V4 import string. Module compilation itself passed. This was preserved as a **mechanical audit-compatibility failure**, not an R26 scientific failure.

Mechanical repair note:
- `docs/production/RB_R26_PROMOTION_STATIC_AUDIT_COMPAT_MECHANICAL_REPAIR_V1.md`

Final repaired production head:
- **`bb76ba9eabb08e2f0875a9af49301c3877f4141f`**

The compatibility repair preserves V4 as the explicit protected parent while V5/R26 remains the final execution authority. R26 adapter/V5 science was not changed by this repair.

**Current rule:** Full Slate now has one RB receptions model output. The old baseline is predecessor/audit information, not an alternate number for manual bet selection.

---

## 11. Current RB model tree

Current production structure:

> **current roster / game context**
> → team / position opportunity
> → **RB-P3 rushing opportunity + yards**
> → **R26 RB receiving/receptions entitlement refinement**
> → **current receiving efficiency / mean machinery**
> → **R22 receiving-yard distribution/tails around the existing mean**
> → existing ML/state/ensemble
> → one final player projection
> → sportsbook comparison downstream.

Important limitation now exposed:

**R26 fixed/improved the opportunity/reception side, but R22 deliberately preserved the existing RB receiving-yard mean.** Therefore the next research question is whether receiving-yard mean improves simply from R26's better reception volume, and whether a new player/matchup efficiency layer adds additional out-of-sample accuracy beyond that.

---

## 12. ACTIVE NEXT STUDY — RB receiving-yard mean / efficiency

This is the next item in the user's original hierarchy.

### Core decomposition

The next frozen study should compare, historically and leakage-safely:

1. **Current production receiving-yard mean baseline**
2. **R26-opportunity-only candidate**: improved R26 reception/opportunity expectation with the existing receiving-efficiency assumption held fixed
3. **R26 + efficiency candidates**: add only pregame-available, strict-prior player/matchup efficiency information
4. determine incremental gain from opportunity versus true efficiency
5. only after a mean candidate qualifies, test re-centering/preserving R22's already-qualified tail/distribution behavior around the improved mean.

### Candidate efficiency families worth evaluating without presuming they work

- strict-prior yards per reception / yards per target with shrinkage
- recent vs longer-term receiving efficiency stability
- YAC / YAC-over-expected where historical sources permit leakage-safe coverage
- aDOT / route type / screen-vs-downfield role where available
- explosive receiving frequency / long-catch propensity
- receiving usage by RB role / depth / player archetype
- defensive RB receiving matchup / LB coverage / tackling / YAC allowance where historical data are available
- offensive line / pressure / checkdown environment only if it adds independent pregame signal
- QB checkdown tendency only if measured leakage-safely and without double-counting team opportunity

Do **not** blindly multiply a noisy raw YPR by R26 receptions. Efficiency must be shrunk/validated because low-volume RB samples are volatile.

### Required research boundaries

- R26 production opportunity is the upstream authority; do not revert to old baseline receptions as the candidate foundation.
- Preserve a baseline column for decomposition/audit only.
- R22 remains untouched during mean discovery; it should be adapted only after a mean improvement qualifies.
- No sportsbook inputs upstream.
- No 2026 Week1 outcomes.
- Freeze exact historical seasons, player population, features, formulas, thresholds and gates before looking at candidate results.
- Prefer strict walk-forward evaluation and player/team clustered uncertainty where appropriate.
- Separate mean accuracy from tail calibration.

### Desired end state

> **R26 opportunity/receptions → validated RB receiving efficiency → improved receiving-yard mean → R22 tail/distribution around that improved mean → one final authoritative RB receiving-yards output**

---

## 13. Other remaining RB issue — parked behind current hierarchy item

Acute same-week injury / role inheritance remains unsolved as a separately validated transform.

Desired future tree:

> player identity → historical ability/usage → current role → team opportunity pool → injury/availability change → player-specific role inheritance → matchup/game environment → workload/efficiency → distribution.

Do not assume RB2 simply becomes RB1. Do not silently copy one player's R26 distribution to another player. This remains valuable future work, but **the immediate active scientific lane is receiving-yard mean/efficiency**.

---

## 14. Broader lane reminders

### QB
M89/M90 mean authority remains protected; C2 is production-qualified distribution work. The next hierarchy item after RB receiving-yard mean is the deeper QB attempts/dropback/pass-rate/YPA/sacks/scrambles decomposition.

### WR
M38 remains WR1 anchor; WR-R15 handles WR2+ entitlement. Failed broad additive target-boost ideas remain closed unless new independent evidence appears.

### TE
TE-R5P remains production. Keep opportunity allocation distinct from player efficiency.

### RB
- P3 rushing: production-active
- R26 receptions: **production-active**
- R22 receiving-yard tails: production-active, mean-preserving
- receiving-yard mean/efficiency: **next active research problem**
- R26S: locked postgame prospective audit after Week1
- acute injury/role inheritance: parked future study

---

## 15. Exact next actions

1. Treat `bb76ba9eabb08e2f0875a9af49301c3877f4141f` as current production head.
2. Treat R26 qualification run `34417740186` / artifact `10129819192` / digest `sha256:f8fbd74187e811fc765c6d236c4c005c19be6d007d1bec33800790e56eabdcde` as the production-promotion qualification authority.
3. Treat Full Slate run `34418491952` / artifact `10130055472` / digest `sha256:d84bbc9fb976e3862b19d458a929888c1ab5e0f7cec49e1f4705a961a236e15f` as the post-promotion football-stack verification authority; remember its live-pricing step was skipped because live odds were not requested.
4. Start a **new research branch** for RB receiving-yard mean decomposition.
5. Freeze that study **before** implementation/results.
6. First isolate the gain from R26 opportunity alone with efficiency held fixed.
7. Then evaluate pregame efficiency features for incremental gain.
8. Only after mean qualification, test R22 re-centering/integration without damaging its tail behavior.
9. After Week1, rerun locked R26S unchanged for prospective auditing.
10. Continue the original hierarchy after the RB receiving-yard mean lane closes.

---

## 16. One-sentence current state

**R26 receptions has completed historical research, prospective construction, 35/35 production qualification and successful promotion into Full Slate at `bb76ba9eabb08e2f0875a9af49301c3877f4141f`, giving RBs one authoritative receptions output; the next active scientific problem is the user's original hierarchy item #1—RB receiving-yard mean/efficiency—using R26 as the fixed upstream opportunity authority and preserving R22 as the already-qualified mean-neutral tail layer until a better receiving-yard mean is proven.**
