# NFL MASTER CONTINUITY RECORD — ALL-CHAT / ALL-HISTORY HANDOFF

**Created:** 2026-09-10 15:23 ET  
**Repository:** `dkaps6/imtiredofthis`  
**Purpose:** one durable cross-chat continuity record so a new session can recover the project’s scientific philosophy, production architecture, research history, stopping rules, exact current operational state, and next authorized work without reconstructing dozens of prior chats.

> **Authority rule:** GitHub committed plans/result records/workflow runs/artifacts are canonical for exact lineage. Chat history supplies intent, interpretation, priorities and anti-reinvention context. If a chat summary and live GitHub conflict, prefer the newer committed GitHub evidence.

---

## 1. USER / PROJECT OPERATING CONTRACT

The project is not a generic betting-model exercise. The user’s North Star is to build the most accurate possible **football model of individual NFL game outcomes** and then use sportsbook markets only as a downstream opponent/benchmark.

Permanent user directives:

- Predict real football first: QB passing, RB rushing/receiving, WR/TE receiving, then derive sportsbook edges.
- Sportsbook information is downstream only unless a separately frozen study explicitly labels itself market-assisted.
- Vegas is not allowed to teach the upstream football projection which direction to move.
- One authoritative production projection per player/market; no silent competing projections.
- Preserve exact frozen plans, first valid scientific results, failed scientific results, mechanical failures, repair records, artifacts, digests and dispositions.
- Mechanical/plumbing/data failures are not scientific failures. A minimum frozen mechanical repair may rerun the experiment without changing scientific gates or model meaning.
- Never rescue a scientific failure post-result by tuning thresholds, alphas, windows, subgroups or routers unless a new study is frozen prospectively.
- Never silently mutate production.
- Tell the user when an idea is weak, redundant, already tested, or unsupported.
- Live-status questions require checking GitHub first.
- Major cross-chat checkpoints belong in `CURRENT_NFL_RESEARCH_HANDOFF.md` on `main`.

Canonical architecture direction:

`GAME STATE -> TEAM OPPORTUNITY -> POSITION/ROOM POOL -> PLAYER ENTITLEMENT -> PLAYER EFFICIENCY -> JOINT MONTE CARLO -> PLAYER PROJECTIONS/DISTRIBUTIONS -> FAIR PROBABILITIES -> SPORTSBOOK COMPARISON`

The project deliberately moved away from treating final player yardage as a mostly independent positional output. The finite football available in a game must be modeled and conserved before it is allocated to individual players.

---

## 2. CANONICAL PRODUCTION ENTRYPOINT AND CURRENT AUTHORITIES

Only canonical live production orchestration:

`.github/workflows/full-slate.yml`

Legacy `engine/engine.py` is retired and must not become an alternate production pipeline.

### Current repository authorities

Latest documentation-only main head when this master record was written:
- `ab5f8acf1c3155ba42d5ddd1c4d3ad9535a18a6e` — points the canonical handoff to the Sep 10 current stop.

Current operational production authority before documentation-only handoff commits:
- `3079d8ab0512c5a1304662609e3e880d6846292f`
- PR `#515` added the automatic downstream master betting workbook and did not change model science.

Protected scientific/model authority remains:
- `bb76ba9eabb08e2f0875a9af49301c3877f4141f`

Important distinction: operational plumbing/reporting can advance while the protected scientific authority remains frozen.

### Current promoted football stack

- **QB mean:** M89/M90 / `QB_PASS_SYNTHESIS_V1`
- **QB distribution:** `C2_QB_MEAN_NEUTRAL_DISTRIBUTION_V1`
- **WR:** M38 WR1 hierarchy anchor + `WR_R15_PRODUCTION_MODEL_V1` for WR2+
- **TE:** `TE_R5P_PRODUCTION_MODEL_V1`
- **RB rushing:** `RB_P3_SYNTHESIS_V1` for the qualified 2026 Week-1 route
- **RB receptions:** R26 production refinement
- **RB receiving-yard mean:** existing production YPT/mean path
- **RB receiving-yard distribution/tails:** R22 using frozen R19 assets; exactly mean-preserving
- **Current player availability/current roles:** promoted availability-first production plumbing
- **Sportsbook:** downstream only, after football eligibility/projection generation

Failed, mixed, forensic, diagnostic or source-audit branches are not production merely because their scripts exist.

---

## 3. HISTORICAL PRODUCTION ARC — HIGH-LEVEL

The project evolved through several eras:

1. **Early production/model migration era:** stabilize provider inputs, player/team identity, schedule/opponent semantics, Monte Carlo, ensemble and football rules.
2. **QB migration era:** increasingly isolate opportunity, pressure, rushing competition, receiver ecosystem, catastrophic passing misses and synthesis residuals; culminated in M89/M90.
3. **RB rushing era:** M91-M96 and STACK-series work decomposed team rush volume, backfield concentration, workload regimes, role transitions, efficiency and tail risk; culminated in P3 Week-1 production authority.
4. **Receiving architecture era:** cross-position conservation, WR hierarchy/entitlement, TE pool/participation, RB receiving opportunity/receptions and tail separation.
5. **Production-hardening era:** sportsbook-independent football universe, current-role/availability certification, T-75 fail-close logic, exact Full Slate integration, strict audits and automatic human-facing workbook publication.
6. **Current era:** Week 1 has started; production is operationally live. RB receiving-yard mean research is closed/no-integration after R27D. The next model-development lane is QB opportunity/efficiency anti-reinvention/error decomposition.

---

## 4. QB RESEARCH HISTORY — DECISIVE LINEAGE

### Foundational QB mechanisms retained from earlier migrations

Several earlier migrations established durable football mechanics later carried into production:

- **M21:** opportunity framework / pass-vs-rush structure; historical project shorthand references the 57/43 opportunity split.
- **M23:** gentle pressure adjustment; pressure belongs as a modest football efficiency/opportunity factor, not an overpowering correction.
- **M30:** top-five rushing opportunity pool / rushing competition handling.
- **M38:** receiver hierarchy became a major shared offensive mechanism; although M38 is a WR production model, it materially improved the QB/receiver ecosystem by sharpening finite target entitlement rather than treating receivers symmetrically.

These are historical foundations, not invitations to reopen generic versions of the same ideas.

### M69-M89 forensic / frontier work

The later QB migration family investigated game script, extreme efficiency, opportunity recoverability, low-chaos catastrophic misses, regime replication and data integrity. Important branch families preserved in GitHub include:

- `backtest-migration-69-qb-gamescript-forensic-atlas`
- `backtest-migration-70-qb-extreme-ypa-mechanism-autopsy`
- `backtest-migration-71-qb-efficiency-uncertainty-audit`
- `backtest-migration-72-qb-matchup-persistence-explosive-weapons`
- `backtest-migration-73-qb-opportunity-recoverability`
- `backtest-migration-80-qb-research-frontier-batch-audit`
- `backtest-migration-83-defensive-adaptive-gameplan-audit`
- `backtest-migration-84-top-weapon-escape-hatch-audit`
- `backtest-migration-85-true-blocker-rusher-source-audit`
- `backtest-migration-86-qb-error-floor-recoverability-audit`
- `backtest-migration-87-low-chaos-catastrophic-pregame-atlas`
- `backtest-migration-88-2023-regime-replication`
- `backtest-migration-89-qb-data-integrity-casebook-synthesis`
- `backtest-migration-90-qb-synthesis-confirmation-promotion`

The point of this era was not endless feature accumulation. It progressively established which catastrophic QB misses were structurally recoverable pregame and which were true tail/explosive events.

### M89/M90 — current QB mean authority

M89 built the football-only residual synthesis. M90 was confirmation/promotion only and explicitly prohibited rediscovering the M89 casebook as new evidence.

Canonical records in `docs/migrations/PROJECT_CONTINUITY_LATEST.md`:
- M89 run `33331073376`
- M90 prospective confirmation run `33333730480`

M90 2023 confirmation:
- MAE `60.632751 -> 56.559869`
- RMSE `75.634635 -> 69.629921`
- bias `-27.248932 -> -6.606555`
- correlation `.172628 -> .243467`
- 100+ yard misses `81 -> 64`

Production synthesis contract preserved from M89/M90 includes Ridge alpha `20.0` and the frozen residual cap used by the confirmation work. Broad generic QB mean hunting was formally frozen after M90 unless a specific architecture diagnostic identifies a failed layer or genuinely new independent information becomes available.

### QB C2 distribution authority

Cross-position conservation work proved that a mean-neutral QB distribution refinement could improve calibration while preserving M89/M90 point means.

On the exact 884 M89/M90 QB-game cohort:
- QB mean MAE unchanged `55.0601 -> 55.0601`
- CRPS `40.3866 -> 39.0938`
- bootstrap probability of CRPS improvement `1.000`
- 80% interval coverage `58.71% -> 71.27%`
- 90% interval coverage `69.34% -> 82.47%`
- conservation identity gap `0.0`

This became the QB-only `C2_QB_MEAN_NEUTRAL_DISTRIBUTION_V1` production selector.

Important non-promotion boundary:
- the **full shared QB/receiver C2 integration** is not the same thing as the QB-only production selector;
- C1 group target-mass calibration and C3 joint combination failed player-level protection gates and must not be recycled as if they were winners.

### QB internal-reliability diagnostic

`research-qb-pd3-internal-disagreement-reliability`
- Run `34122984048`
- Job `101745071358`
- Artifact `10018942911`
- digest `sha256:cde910939dba0631a2a68e9acfbcafcdb90af1d1b2128627442073854bf3f849`
- n `884`
- disposition `NO_ACTIONABLE_QB_INTERNAL_RELIABILITY_STATE`

Meaning: large synthesis moves, cap hits and component disagreement did not provide a stable historical reason to distrust M89/M90. Week-1 QB concerns therefore must be decomposed through real football layers—team pass opportunity, attempts/dropbacks, YPA/efficiency, sacks/scrambles, current-team history mapping, opponent, receiver ecosystem and game environment—not by generic confidence heuristics.

---

## 5. CROSS-POSITION / PASS-RECEIVING ARCHITECTURE

### Shared pass-volume evidence

The project found strong coupling between QB attempt error and receiver opportunity error:
- 2025 QB-attempt residual vs aggregate WR opportunity residual Pearson `0.6973`
- Spearman `0.6706`
- same direction `73.2%`
- independent 2024-2025 WR reception-error confirmation Pearson `0.5239`

This is one of the clearest architectural findings in the entire project: QB and receivers cannot be modeled as independent opportunity systems.

### Joint Pass/Receiving Conservation V1

Canonical scientific run:
- Run `34081764151`
- Job `101618243530`
- SHA `4f620f1ea24a10cdf840d52d42a8930e5991f1ee`
- Artifact `10004223287`
- digest `sha256:753aa191e6c80a059918553d8567499c9cdbce82b12a26b4d5b19b21225764ac`
- disposition `CONSERVATION_ONLY_SUPPORTED`

The conservation mechanism was valuable, but broad group-mass and joint-combination candidates did not automatically improve individual player projections. The architecture lesson survived even where a candidate did not.

### Receiving attempt-semantics family — closed

Canonical repaired C4 run:
- Run `34077637287`
- Job `101606772786`
- SHA `d5f0ba8af56caa28f8665fbea15d74885e09a65c`
- Artifact `10002832205`
- disposition `ATTEMPT_SEMANTICS_CANDIDATE_FAIL`

Results:
- team targets `6.3014 -> 6.1874`
- player target MAE `1.7747 -> 1.8015` worse
- macro receiving-yard MAE `16.9594 -> 16.9990` worse
- `0/6` seasons improved receiving-yard MAE

Do not retry this generic attempt-semantics family.

---

## 6. WR RESEARCH HISTORY

### M38 — foundational production winner

M38 target-share multipliers:
- WR1 `1.40`
- WR2 `1.14`
- WR3 `.91`
- WR4+ `.78`

Exact M38 parent SHA:
- `b98518d97b3038f471aee9ae3201009b2c70bb29`

Historical confirmation:
- 6/6 seasons improved across 2020-2025
- pooled n `12396`
- receiving-yard MAE `24.215219 -> 23.247553`

M38 established the core WR hierarchy prior.

### WR NGS/source research

WR-R10 strict-prior NGS source eligibility:
- Run `34123975300`
- disposition `STRICT_PRIOR_NGS_FEATURES_ELIGIBLE`
- strict-prior eligible rows `10427` (2021-2025)
- prior-1 coverage `86.33%`
- prior-3 coverage `74.65%`
- zero same/future observations used

WR-R11 NGS residual target model — scientific fail:
- Run `34124533822`
- Job `101749972264`
- Head `d2f251dc267db854e6747734d6cbe692d56f93ae`
- Artifact `10019545653`
- digest `sha256:388973f0fc67c6a2f0e4868a7c874d310ff7711cd7b0a6fb38c74128dcd43370`
- disposition `WR_NGS_TARGET_MODEL_FAIL`

Although target RMSE/p90 improved, receiving-yard MAE worsened and the correction was positive almost universally. It behaved like a generic underprojection correction rather than differentiated player entitlement. Do not rescue WR-R11 by retuning.

### WR-R15 — production specialist

Canonical PASS research:
- Run `34238301577`
- Artifact `10061328722`
- 4,193 OOS WR player-games
- target MAE `2.1285 -> 2.0439`
- receiving-yard MAE `22.8566 -> 22.5264`
- p90 AE `51.56 -> 50.31`
- 8/8 phases non-worse
- leakage/conservation gates passed

Final refit:
- Run `34240496725`
- 6,305 rows, 2022-2025
- 15 frozen features
- Ridge alpha `20`
- sportsbook inputs `0`

Production behavior:
- one M38 WR1 anchor per team is preserved;
- WR-R15 redistributes only within WR2+;
- WR-room/team entitlement mass is conserved.

Do not allow Ourlads lane labels to masquerade as a unique team-wide WR1/WR2 hierarchy. LWR/RWR/SWR depth charts can produce duplicate depth labels. The model hierarchy remains a separately derived football concept.

---

## 7. TE RESEARCH HISTORY

### TE error decomposition

Dedicated TE research established that target opportunity is the dominant receiving-yard error mechanism.

TE-R2 pool-vs-individual diagnostic:
- Run `34126026512`
- Job `101754774701`
- Artifact `10020131404`
- digest `sha256:54309eda0886cd0b327333ea89bde12a2fc506d141e2b72ddc3b1d1d778b874d`
- rows `6371`, team-games `3197`
- disposition `TE_TARGET_POOL_FIRST`

Absolute target-error mass:
- team TE pool `60.405%`
- individual allocation `39.595%`

Highest-error quartile:
- team TE pool `72.073%`
- individual allocation `27.927%`

Architecture lesson: team pass state -> finite TE pool -> individual TE entitlement -> catch/yard efficiency.

### TE-R3 — scientific fail

- Run `34126813280`
- Artifact `10020423842`
- digest `sha256:99699c7e4745ab8cf678b527fc57325c5f402a5c7be930ff9cb90dba96e63ff8`
- disposition `TE_TARGET_POOL_CONTEXT_MODEL_FAIL`

Team TE-pool MAE and player target MAE improved, but receiving-yard MAE worsened. Correction was positive in `99.48%` of games. Do not reuse as a universal TE boost.

### TE-R4 source eligibility — passed

- Run `34127474412`
- Head `ffe4e1101193b3502a6b069d2b77347049719b1d`
- Artifact `10020700686`
- digest `sha256:56de7efe302cdf9c329c3a0386d798789d15e6a495c865535291173b6840c163`
- prior-1 participation coverage `97.91%`
- prior-1 same-team `96.59%`
- prior-3 `95.92%`
- zero same/future observations used

### TE-R5 / TE-R5P — winner and production path

TE-R5 scientific winner:
- Run `34132127351`
- 3,214 OOS player-games, 2023-2025
- receiving-yard MAE `16.267971 -> 15.858042`
- RMSE `23.498617 -> 21.620042`
- p90 AE `37.227233 -> 33.284738`
- 40+ miss rate `8.7430% -> 6.5028%`
- all three seasons improved MAE

Production authority is **TE-R5P**, not the later scientific-only R3+R5 combination. TE-R5P preserves the finite TE pool and redistributes individual entitlement within it.

---

## 8. RB RUSHING RESEARCH HISTORY

The RB rushing program was extensive. Major branch families preserved in GitHub include M91-M96, M95A-M95T, M96A-M96E, STACK1-STACK7, ND/PD diagnostics, role-remap studies and availability source audits.

### Durable lessons

- Simple depth-rank remapping is bad workload authority. Run `34063904515`, Artifact `9998334300` worsened carry MAE `3.483 -> 4.106` and rush-yard MAE `20.424 -> 22.837`. Depth chart is context, not direct carry assignment.
- Central opportunity/carry modeling and workload concentration were more useful than detached late tail patches.
- M95-family work repeatedly explored workload regimes, vacancy/transition, feed tendency and carry ceiling. The project ultimately stopped detached retrospective carry-tail invention.
- M95T formal stop: Run `33455690862`, Artifact `9781352939`. Do not create another detached generic tail overlay candidate.
- Several residual-calibration candidates improved central MAE but hurt p90; this recurring tradeoff is a stopping signal, not an invitation to tune until a pass appears.

Historical project shorthand from prior chat audits:
- M94C became a central opportunity/carry model reference.
- M95F focused workload-tail/risk.
- M95I focused vacancy/transition.
- M95K failed sealed 2023 confirmation.
- M95R/T and M96D/E were rejected by stability/materiality gates.

### P3 production rushing authority

Current RB rushing production authority:
- `RB_P3_SYNTHESIS_V1`
- qualified Week-1 route uses the exact STACK3 Week-1 override behavior.

Important scope boundary:
- Week-1 P3 promotion does **not** automatically certify Weeks 2-18 enriched allocation.
- historical W2-18 availability/injury timestamp provenance remained unresolved at the time of qualification.
- future-week behavior must fail closed rather than silently route through an unqualified implementation.

P3 remains protected production science unless a separately frozen replacement beats it.

---

## 9. RB RECEIVING RESEARCH — R23 THROUGH R27D

This is one of the most important recent histories because it separates **opportunity/receptions**, **mean efficiency**, and **tail distribution** rather than conflating them.

### Foundational decomposition

Historical baseline RB receiving-yard MAE was about `10.42` yards.

Oracle actual RB targets reduced MAE to about `7.29` yards, proving target allocation was a major bottleneck. In early catastrophic-error analysis, 84 misses of 30+ receiving yards were all underprojections; actual RB targets averaged ~`5.71` vs model ~`2.16`. Replacing projected targets with actual targets eliminated 53/84 30+ misses and 25/32 50+ misses.

But catastrophic cases also had extreme efficiency:
- actual YPT ~`13.09`
- model YPT ~`6.17`

Strict-prior RB receiving identity was highly relevant. Top ~20% by RB-room receiving identity generated disproportionate error/tail risk.

R16 tail-identification evidence showed pregame predictability increased with severity:
- AUC ~`.673` for 30+
- `.766` for 50+
- `.802` for 60+

This motivated the permanent architectural separation: **do not raise means simply to chase explosive right-tail games**.

### R22 — production tail authority

R22 reshapes RB receiving-yard distributions/tails using pinned R19 assets while preserving the receiving-yard mean exactly. It owns stochastic 30-49 / 50+ right-tail redistribution. It does not recenter receiving-yard means.

Therefore later receiving-mean studies are not allowed to steal tail authority simply because a realized game exploded.

### R23 — generic shrunk YPR path failed

R23 tested:
`E[receiving yards] = E[receptions] * shrunk YPR`
using prior 6/16 games and empirical-Bayes role prior.

Run:
- `34332613867`
- Artifact `10096546836`
- digest `sha256:e3e3087f03d0a8d697225e61de435e9a1668ef74e35ca850e60ffe451e9c66b6`
- mixed/fail

Targets/receptions improved, but receiving-yard quality worsened:
- overall rec-yard MAE ~`+0.39%` worse
- p90 ~`+3.35%` worse
- RB1 ~`+2.68%` worse
- RB2+ ~`-1.77%` better

Permanent anti-reinvention rule: generic prior YPR/YPT/catch-rate shrinkage is not a novel future avenue.

### R24 — production YPT held fixed; no qualification

R24 removed the failed R23 YPR formulation and translated improved opportunity with the existing production YPT. It did not qualify. This further established that receiving opportunity can improve without automatically improving receiving-yard means.

### R26 — vacancy-gated receiving opportunity/receptions

R26 evolved into the production receiving-opportunity/receptions refinement.

Core production mechanics:
- `VACANCY_ACTIVE = room_exits_n >= 1`
- stable/no-vacancy exact baseline
- vacancy preserves exact RB/FB room target pool
- uses strict-prior R8/R9 receiving-identity residual
- frozen R8 Ridge/R9 reliability
- softmax only within RB/FB room
- non-RB entitlement preserved exactly

Canonical production qualification:
- Run `34417740186`
- Job `102686263562`
- Artifact `10129819192`
- digest `sha256:f8fbd74187e811fc765c6d236c4c005c19be6d007d1bec33800790e56eabdcde`
- disposition `RB_R26_WEEK1_RECEPTIONS_PRODUCTION_INTEGRATION_PASS_READY_FOR_PROMOTION`
- `35/35 PASS`

Post-promotion Full Slate:
- Run `34418491952`
- Job `102688556296`
- Artifact `10130055472`
- digest `sha256:d84bbc9fb976e3862b19d458a929888c1ab5e0f7cec49e1f4705a961a236e15f`

### R26Q prospective seal / R26S evaluator

R26Q immutable Week-1 seal:
- Run `34400524030`
- Job `102630996205`
- Artifact `10123251043`
- digest `sha256:dd3ec0e8e3831ab7f2255c2e5abf343cda8a7943d33a1d4863e52372d6f858a1`
- disposition `R26Q_2026_WEEK1_RECEPTIONS_PROSPECTIVE_SEAL_PASS_READY_FOR_OBSERVATION`
- NPZ SHA `7933bd7629d0e7108fe181e7a5474f8c8f6e8512b6a1b35081d3a8d71af8fe06`

R26S frozen evaluator:
- plan `63722c61ca1b408b8ae77eba383e5fa8daecf9f2`
- evaluator `e42099dbf11cdd510eb64e43a0910444c75ec7f8`
- lock `3a706fa52f91f6584f6fd6594563239dd6ea3b53`
- dry run `34411889262`
- Job `102667966181`
- Artifact `10127562840`
- digest `sha256:e9ecd949816eafd813a50e60e8af28ce387e668848483ffcda3dfadf7614919e`

Do not recompute the seal. Rerun the exact frozen evaluator only when its full authoritative Week-1 outcome scope is available.

### R27 — exact R26 opportunity translated to receiving-yard mean

Question: does exact historical R26 opportunity redistribution improve player-level RB receiving-yard point means when production YPT is held fixed?

First valid result:
- Run `34423546037`
- Job `102703879430`
- Artifact `10132290573`
- digest `sha256:cd5c9e26efa44c47ef3374e4deefe9038dc504c155799b48548ad64746e10e45`
- disposition `R27_R26_OPPORTUNITY_TRANSLATION_MIXED_OR_FAIL`
- `24/27` gates PASS

Vacancy-active:
- MAE `11.283018 -> 11.159268` = `-1.0968%`
- RMSE `16.5021 -> 16.2820` = `-1.3337%`
- bias improved `-2.8996 -> -1.9276`
- p90 worsened ~`+2.08%`

RB1 vacancy incumbent:
- target MAE improved ~`-2.28%`
- receptions improved ~`-2.39%`
- receiving-yard MAE worsened `+2.82%`

RB2+ receiving-yard MAE improved `-3.36%`.

Interpretation: R26 opportunity is real, especially RB2+, but fixed production YPT overtranslates some RB1 opportunity. That is an efficiency/context bottleneck, not evidence for generic new YPT shrinkage.

### R27B V1 — superseded before science

Generic historical YPT/YPR/catch-rate candidate was recognized as reinvention of R23/R24 and superseded before model/candidate/result. No scientific conclusion.

### R27B V2 — novel efficiency context, mixed/fail

First valid execution:
- Run `34428917229`
- Job `102720004328`
- Head `2c86520c84fbdd5a18aad3f4b88373e5f8c17051`
- Artifact `10134023092`
- digest `sha256:1df1a14c9900b890fe86e04269849dc90d76fc03a9215ad8612c1820dee5341f`
- disposition `R27B_V2_NOVEL_EFFICIENCY_CONTEXT_MIXED_OR_FAIL_NO_INTEGRATION`
- `24/31 PASS`; all 15 integrity gates PASS

Vacancy active:
- B1 MAE `11.1593` -> C1 `11.1267`, only `-0.292%`
- p90 improved ~`-1.57%`

RB1:
- C1 improved only slightly vs B1 but remained worse than B0; RB1 wall persisted.

RB2+:
- modest improvement vs B1.

2023 was especially bad: C1 worsened further vs both B1 and B0.

Valid sub-signals were preserved as evidence, but no post-hoc cherry-pick/router was authorized.

### R27C — RB1/2023 forensic

Canonical:
- Run `34430754033`
- Job `102725570088`
- Artifact `10134387002`
- digest `sha256:0c915a1c413d66c550990a1cc8bd45e0dc20a18e110fe1b96414a79eb7b598e9`

Finding: two different failure modes existed.
1. 2023 RB1: R26 improved target count slightly but realized YPT fell below production YPT; V2 nudged YPT upward, exactly the wrong direction.
2. Tail regressions: some newly created 30+ misses were high realized-YPT explosive RB1 games, the opposite phenomenon.

This ruled out a single universal mean correction.

### R27C2 — physical realized target-quality forensic

First valid canonical:
- Run `34431286455`
- Job `102727137722`
- Artifact `10134581843`
- digest `sha256:6e3aa7ec2149f6f0b65adb59b4cc36c4f88b24a142234bc368131de8e4de60dd`
- result commit `2cfca82d919290ac18ed01559994d2e0239b0798`
- disposition `R27C2_FORENSIC_COMPLETE_PHYSICAL_TARGET_QUALITY_HYPOTHESIS_IDENTIFIED`

Join integrity:
- 8,429 parent rows preserved
- targeted PBP join coverage ~`.9963`
- minimum primary cohort >`.9898`
- 2023 vacancy RB1 join coverage `1.000`

Primary physical finding: **2023 vacancy-RB1 receiving-yard failure was YAC/YPR compression, not catch-rate, target-depth, screen-rate or explosive-rate collapse.**

2023 vs non-2023 vacancy RB1:
- catch rate `.78345` vs `.78724` ~same
- air yards/target `.0887` vs `.1065` ~same
- screen rate `.5806` vs `.5623` no collapse
- explosive20 target rate `.05046` vs `.05097` identical
- YAC/reception `7.3395` vs `8.0667`
- YPR `6.4254` vs `7.4206`
- YPT `5.2236` vs `5.9001`
- max receiving gain `10.68` vs `13.20`

Production arithmetic showed actual catch rate was slightly better than production while implied YPR was too high. The miss was post-catch value, not catching.

Tail finding remained separate: new 30+ miss rows were genuine YAC/explosive outcomes and therefore conceptually belong to R22 rather than broad mean inflation.

### R27D — strict-prior xYAC/YACOE residual test; lane closed

Latest valid mean-study result:
- branch `research-rb-r27d-yacoe-residual-v1`
- first valid head `641b25419c4f5ff3c234d1c000222fb4909ef940`
- Run `34436178615`
- Job `102741600329`
- Artifact `10136250846`
- digest `sha256:b975511dc54e961c7745e9d1422ac48be4f867c65d3733adfaeabe93af9951e6`
- result-record commit `a20e757a1b85806ae5e244cc42d29cfbc7ed2742`
- disposition `R27D_STRICT_PRIOR_YACOE_RESIDUAL_MIXED_OR_FAIL_NO_INTEGRATION`
- integrity `18/18 PASS`
- scientific gates `4/13 PASS`

Vacancy RB1:
- B0 MAE `14.305919`
- B1 `14.709395`
- C1 `14.710366`

2023 vacancy RB1:
- B0 `12.349471`
- B1 `13.992139`
- C1 `14.055947`

Only `3/6` seasons improved C1 vs B1.

Conclusion: strict-prior relative xYAC/YACOE persistence did not provide a reliable vacancy-RB1 receiving-yard mean correction. **RB receiving-yard mean lane is now closed/no-integration.** Future reopening requires genuinely new pregame football information, not another transform of generic prior YAC/YPR/YPT/team/opponent averages.

Preserved invalid/mechanical R27D runs:
- `34435661834` / Job `102740069405`: duplicate deterministic Week1 column collision; no science.
- `34435897671` / Job `102740771822` / Artifact `10136169669`: null-outcome scoring/integrity failure; emitted result rejected as invalid.

---

## 10. CURRENT-PLAYER AVAILABILITY / CURRENT-ROLE PRODUCTION

This became urgent immediately before/after Week 1 started because stale roster hierarchy could invalidate otherwise good football projections.

### Frozen behavior

- Resolve availability before opportunity.
- `ACTIVE_ROLES_CSV=data/roles_current_production_eligible_v1.csv` is current eligible role authority.
- Definitive unavailable players cannot survive into eligible roles, PlayerForm or simulation arrays.
- Official NFL inactive authority applies inside the frozen T-75 window.
- More than 75 minutes before kickoff: official inactives are not yet required.
- Within 75 minutes: incomplete official inactive evidence fails closed.
- At/after kickoff: game is `KICKED_OFF_LOCKED` and withheld from current betting/output eligibility.
- Sportsbook acquisition occurs only after football eligibility and cannot resurrect a kicked-off game or unavailable player.
- Sportsbook inputs used to define availability/opportunity = `0`.

Canonical T-75 timing validation:
- Run `34437715931`
- Job `102746163583`

First mechanically valid no-odds candidate:
- Run `34447900206`
- Job `102776660124`
- Artifact `10140425929`
- digest `sha256:6b31ae40d648780673b7953b57323d509b302ea7a82fe99801b024ede2603f37`

Final 35/35 certification:
- substantive Run `34461561636`
- Job `102820358570`
- Artifact `10145975346`
- `34/34 PASS` substantive gates
- gate-35 finalizer Run `34463888613`
- final Artifact `10146675272`
- digest `sha256:dcc0fec6f07c62542e7115a49c1c45e9a185d18fd3baa2e423e0bd3dd08d0286`
- disposition `CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_PASS_READY_FOR_PROMOTION`

### Production verification lineage

Run1 mechanical audit-compatibility failure:
- Parent `34470549704`
- Job `102849180219`
- Artifact `10149440197`
- child Full Slate `34470613780`
- old strict audit required a direct literal call to `run_player_form_v2_loader.py`; certified current-role wrapper delegated to the protected loader correctly.
- classification: mechanical/no decision.

Run2 mechanical verifier failure:
- Parent `34477957076`
- Job `102873279264`
- head `944426bae1a6d9292fded5dcf41f8951e7357467`
- Artifact `10152500791`
- child Full Slate `34478009290` succeeded
- verifier accepted obsolete R22 disposition names while child emitted the canonical current R22 PASS disposition.
- classification: mechanical/no decision.

Run3 first successful locked production-branch verification:
- final head `f55b14b99844d5d4de9899db91bc9a7abdc15bf3`
- Parent Run `34497510664`
- Job `102939616416`
- Artifact `10160648036`
- digest `sha256:e1fbd76f5f5189383acea770ff3b669d22d40313c0042cb07c02fe85b92b41ed`
- child Full Slate `34497578081`
- Artifact `10160594516`
- digest `sha256:656a2ede23db107002abf75dcb568ecf26afcabc987f3b0b1bbdeba1a568e7db`
- disposition `CURRENT_PLAYER_AVAILABILITY_PRODUCTION_BRANCH_VERIFY_PASS_READY_FOR_MAIN_PROMOTION`

Promotion:
- PR `#513`
- merged as `f813f85ed814cc7c231a459e2301170171b8ed10`

First clean-main no-odds proof:
- Run `34498365769`
- Job `102942531505`
- Artifact `10160866044`
- digest `sha256:89f36a3d1f100e4aefeb2167873485f271faf40b86d0ac2415b05da5f7b2e94b`
- SUCCESS

Post-promotion CI had one stale P3 workflow-label test only; runtime was already correct. Frozen one-line contract repair was merged through PR `#514` as `3e49e0e3a155c0fe74b134542eb359d6444cd8c8`, after Repo CI Run `34500788405` passed.

Availability/current-role production is **COMPLETE / PROMOTED / CLEAN-MAIN VERIFIED**.

---

## 11. AUTOMATIC MASTER BETTING WORKBOOK

Purpose: every Full Slate run should publish one human-facing downstream workbook without contaminating football projections.

Production files:
- `scripts/build_master_betting_workbook_v1.py`
- `scripts/master_betting_workbook_core_v2.py`
- `.github/workflows/full-slate.yml`

Output:
- `outputs/NFL_BETTING_MODEL_MASTER.xlsx`

PR `#515`:
- reporting/integration only
- merged as `3079d8ab0512c5a1304662609e3e880d6846292f`

First production proof:
- Full Slate Run `34505435434`
- Job `102966347569`
- Artifact `10163711046`
- digest `sha256:e272ea4c1398bae9c9fa617566cfd17565fbd19efeacff01c08c7f193b16214b`
- artifact physically verified to contain `outputs/NFL_BETTING_MODEL_MASTER.xlsx`

No-live-odds behavior:
- workbook still publishes
- football/availability remain current
- no sportsbook rows are invented
- betting decisions are suppressed

With live odds:
- lines/offers enter only after football projections/eligibility are established
- workbook can display projection, Vegas line, odds, model probability, no-vig market probability, line/probability edge, EV and PLAY/LEAN/PASS/BLOCKED status
- display status is downstream and is not a model input

Anytime TD remains blocked/research-only until separately scientifically certified.

---

## 12. WEEK 1 LIVE OPERATING STATE

Week 1 has started. New England vs Seattle has already kicked off and is expected to be `KICKED_OFF_LOCKED` in a current Full Slate run. Its absence from current live prop output is expected and is not an Odds API failure.

### If Full Slate is run right now

It uses the latest **promoted** production stack, including:
- availability/current-role logic
- M89/M90 QB mean
- QB-only C2 distribution
- M38 + WR-R15
- TE-R5P
- P3
- R26
- R22
- automatic master workbook

It does **not** include failed/research-only R27/R27B/R27C/R27D ideas or other unpromoted research.

---

## 13. NE-SEA CURRENT-STACK PREGAME COUNTERFACTUAL — CURRENT OPERATIONAL STOP

This is diagnostic only and must not be confused with production status.

Branch:
- `audit-ne-sea-pregame-current-stack-v1`

Frozen plan:
- `docs/audits/NE_SEA_PREGAME_CURRENT_STACK_COUNTERFACTUAL_V1_FROZEN_PLAN.md`

Historical sportsbook source:
- Run `34152868136`
- Artifact `10030344451`
- digest `sha256:c19bd303a0eb7ca58a3484117e28b5e5144459b74459bd1032970873cae6d035`
- odds are not refetched

Run1:
- `34508864138`
- Job `102977690436`
- Artifact `10164979620`
- stale diagnostic clock `23:00Z` was 160 minutes after authoritative kickoff
- timing system correctly returned `KICKED_OFF_LOCKED`
- classification mechanical timestamp error / no projection decision

A documentation-only push triggered duplicate stale-clock failure Run `34509347795`; same classification.

Corrected rerun:
- Run `34509425408`
- Job `102979570511`
- head `3beb00ee518b8076873a5f54dffed51ee2009e5e`
- Artifact `10165290158`
- digest `sha256:41c0e81b489f67d4b57bb98a249865e019fa644c04b4cf891b51734203e068c5`
- corrected pregame clock `19:00Z`, exactly 80 minutes before kickoff

Corrected pregame state passed:
- Rhamondre Stevenson = RB1 / eligible
- TreVeyon Henderson = definitive unavailable / excluded
- Sam Darnold = QB1 / eligible
- A.J. Brown = eligible
- sportsbook inputs to availability = 0

Strict-prior football rebuild also passed, including 32-team PlayerForm and no target-game PBP leakage.

Valid P3 outputs produced before later diagnostic plumbing failure:
- Rhamondre Stevenson rushing mean `30.790770`
- Jadarian Price rushing mean `18.159696`

Historical NE-SEA offers were staged successfully after football generation:
- 162 side rows
- 22 players
- teams NE/SEA only
- markets included ATD, pass TD, pass yards, receiving yards, receptions, rush+rec and rush yards
- odds refetched = false

Exact current blocker:
- `run_metrics_context.py` produced 103 metrics rows
- `scripts/metrics_ready.py` then failed because `data/opponent_map_from_props.csv` had not been staged by the special diagnostic
- complete cross-market pricing did not run

Classification: **diagnostic plumbing/staging failure, not model science**.

Exact next diagnostic move:
1. identify the normal Full Slate command that builds `data/opponent_map_from_props.csv` from already-staged props;
2. freeze the smallest Run3 diagnostic-only repair;
3. hash/lineage lock it;
4. rerun from the same protected parent and same `19:00Z` state;
5. compare only after complete diagnostic succeeds;
6. never use actual NE-SEA target-game PBP/results as pregame features.

---

## 14. DO-NOT-REINVENT / STOPPING RULES

These are permanent unless genuinely new evidence justifies a separately frozen study:

- Do not reopen generic QB mean feature hunting after M90.
- Do not treat large Vegas disagreement as permission to pull the model toward the market.
- Do not retry generic receiving attempt-semantics C4.
- Do not retry WR-R11 formulation by tuning; NGS source may be useful in a different architecture, but WR-R11 itself failed.
- Do not retry TE-R3 as a universal TE boost.
- Do not use simple depth-rank remapping as RB workload authority.
- Do not create another detached M95-style RB rushing tail overlay; M95T stopped that family.
- Do not promote residual corrections that improve central MAE while repeatedly worsening p90 without a new causal mechanism.
- Do not reinvent RB receiving efficiency through generic historical YPT/YPR/catch-rate shrinkage; R23/R24 already addressed that family.
- Do not cherry-pick R27B V2 subgroups after seeing the same evaluation sample.
- Do not merge the distinct 2023 YAC-compression problem with explosive tail misses; they are opposite phenomena.
- Do not reopen RB receiving-yard mean work after R27D unless genuinely new strict-prior football information becomes available.
- Do not let R22 tail logic recenter the mean.
- Do not modify R26Q prospective seal.
- Do not treat branch existence as production certification.
- Do not allow sportsbook presence/absence to define football roster, role, targets, carries, receptions or passing opportunity.
- Do not permit kicked-off games to reenter current betting output.

---

## 15. NEXT ACTIVE MODEL-DEVELOPMENT PHASE

After the narrow NE-SEA diagnostic is closed, the authorized next development lane is:

# **QB OPPORTUNITY / EFFICIENCY ANTI-REINVENTION AUDIT + ERROR DECOMPOSITION**

This is not “restart QB research from scratch.” It is a targeted audit of what M89/M90 still gets wrong and which layer owns the remaining error.

Before freezing any candidate:

1. Audit all prior QB migrations/result records and current production code.
2. Inventory what was already tested for:
   - attempts
   - dropbacks
   - pass rate
   - official-attempt conversion semantics
   - sacks/pressure
   - scrambles/designed QB runs
   - YPA/efficiency
   - opponent context
   - game environment
   - explosive passing
   - receiver ecosystem / top-weapon interaction
   - distribution/tails
3. Decompose current production error into **opportunity vs efficiency vs tail** using strict-prior football information.
4. Explicitly identify prior work that would make a proposed feature family redundant.
5. Source/schema-audit genuinely new football information before model design.
6. Freeze question, mechanism, data boundary, walk-forward protocol, baseline/candidate and pass/fail gates before first scientific execution.
7. Sportsbook remains downstream benchmark only.

Conceptual QB roadmap:
- attempts / dropbacks / pass-rate opportunity
- YPA / efficiency
- sacks and scrambles
- explosive passing / receiver interaction only where independently football-supported
- distribution/tail calibration after mean/opportunity decomposition

Do not change M89/M90 merely because Week 1 is underway.

---

## 16. LATER ROADMAP AFTER QB OPPORTUNITY/EFFICIENCY

The longer architecture roadmap remains:

- selective WR/TE efficiency/distribution work only where diagnostics show a specific remaining layer;
- shared QB↔receiver conservation/one-pass-state integration, without reviving failed C1/C3 formulations;
- unified game simulation where pass/rush opportunity is jointly coherent;
- dedicated Anytime-TD model/calibration before ATD betting recommendations are authoritative;
- game-level ML/spread/total research only after player-stat architecture is stable;
- prospective Week-1 grading, including exact R26S evaluation when its complete outcome scope is available.

---

## 17. RESUME PROTOCOL FOR ANY NEW CHAT / SESSION

A new session should do this in order:

1. Read `NFL_MASTER_CONTINUITY_RECORD.md` — this file.
2. Read `CURRENT_NFL_RESEARCH_HANDOFF.md`.
3. Read the latest dated handoff referenced there; currently `docs/handoffs/NFL_HANDOFF_2026-09-10_1516_CURRENT_STOP.md`.
4. Verify live `main` before acting; documentation may have advanced.
5. If the NE-SEA diagnostic is still open, verify Run `34509425408` lineage and finish only the missing opponent-map staging seam under a frozen minimum repair.
6. Preserve the diagnostic result separately from production.
7. Move immediately into the QB opportunity/efficiency audit after that narrow diagnostic.
8. Before any new research, search repo plans/results/branches to prove the idea has not already been tested.
9. After every substantive milestone, update the canonical handoff on `main` with exact branch/commit/run/job/artifact/digest/disposition.

### Current stop in one sentence

**Production is live and availability-aware with the full promoted QB/WR/TE/RB stack plus automatic workbook; RB receiving-yard mean research is closed after R27D; the only open operational side-task is the NE-SEA diagnostic opponent-map staging repair, after which the next science lane is QB opportunity/efficiency decomposition.**

---

## 18. FINAL CONTINUITY PRINCIPLE

The project’s progress comes from **not confusing signal with promotion**. Many studies contained useful football evidence while still failing the frozen candidate gates. Preserve those findings, but never turn them into production by hindsight. Production changes only when an independently frozen, leakage-safe, football-coherent candidate earns it.
