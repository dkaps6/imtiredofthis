# CURRENT NFL RESEARCH HANDOFF — READ FIRST

**Repository:** `dkaps6/imtiredofthis`  
**Purpose:** canonical cross-chat / scheduled-task continuity ledger for the NFL pregame projection research program.  
**Protected production-code authority:** `bb76ba9eabb08e2f0875a9af49301c3877f4141f`  
**Current production stack:** QB M89/M90 + mean-neutral C2 distribution; WR M38 WR1 + WR-R15 WR2+; TE TE-R5P; RB rushing P3; RB receptions R26; RB receiving-yard tails R22; sportsbook downstream only.  
**Completed research lane:** RB receiving-yard mean R27→R27B→R27C/C2→R27D has reached a defensible scientific stopping point with no new mean integration authorized.  
**Current next roadmap item:** current roster / late-week role handling as a separate operational lane. Do not contaminate historical science.

> Future ChatGPT sessions / scheduled runs: read this file first, then `AGENTS.md`, then inspect the exact live branch/run/artifact named below. GitHub is canonical; chat memory is secondary. Verify branches, commits, Actions runs, jobs, artifacts, digests and result records before acting.

## Historical ledger preservation

The immediately prior detailed canonical handoff is permanently preserved at:
- handoff commit `84c9ffa6ce3617757bcd6b41705d7e55fa8403e0`
- handoff blob `622f4857e75e9ef8939dc80755adbe837c017164`

That snapshot contains the full pre-R27D history and earlier migration/R26/R22/R27/R27B/R27C paper trail. Earlier deep-history checkpoint remains at handoff commit `69e8de76bd1b508849d679fa22abd51aefa68a54`, blob `2f0ee91c1d5296c04114605afd5d4c4067a25a72`. Do not discard those historical records.

---

# 1. Non-negotiable operating rules

1. Historical science is strict-prior / walk-forward / leakage-safe.
2. Freeze question, population, features/mechanics, metrics, thresholds and gates before results.
3. Preserve the first valid scientific result exactly whether PASS, mixed or fail.
4. Preserve mechanical/integrity failures separately and repair only the minimum value-neutral defect.
5. Never lower gates, change cohorts, drop losing seasons or post-hoc route a failed candidate.
6. Sportsbook data stays downstream unless a separately frozen market-assisted study explicitly authorizes it.
7. A research PASS does not directly authorize production. Follow any separately frozen integration/promotion requirement.
8. One authoritative production projection per player/market.
9. Protect R26 receiving opportunity/receptions and R22 RB receiving-yard tail authority unless a separately frozen study explicitly changes them.
10. After each material checkpoint, update this file on `main` with exact lineage and next action.

---

# 2. Protected production / prospective authorities

Protected production-code authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`.

R26 production qualification:
- run `34417740186`
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

R22 receiving-yard tail integration:
- run `34298516960`
- artifact `10084118525`
- digest `sha256:2391bd9914e9d0029c63529829496fb9b462d008bdcc3f8c37bb3e1079580bd1`
- disposition `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_INTEGRATION_PASS`

R19 serialized tail authority:
- run `34288244770`
- artifact `10080377483`
- digest `sha256:11432b9d7b7f2367935a862b63c30df9f40955e479806ab67d90921b63a907c7`
- model SHA `9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba`
- residual pools SHA `c69a268a5a1683e846bcb5f59fe55bcae20d70c679792449ea77e6548b37a362`

R26Q sealed 2026 Week1 pregame authority — never recompute:
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

R26S exact postgame evaluator:
- plan commit `63722c61ca1b408b8ae77eba383e5fa8daecf9f2`
- evaluator commit `e42099dbf11cdd510eb64e43a0910444c75ec7f8`
- lock commit `3a706fa52f91f6584f6fd6594563239dd6ea3b53`
- pregame dry run `34411889262`, job `102667966181`, artifact `10127562840`
- digest `sha256:e9ecd949816eafd813a50e60e8af28ce387e668848483ffcda3dfadf7614919e`
- rerun unchanged only when authoritative Week1 outcomes are available.

---

# 3. RB receiving-yard mean research conclusion

The following families have now been tested without qualifying a new RB receiving-yard mean integration:
- R23: new shrunk historical YPR construction — did not qualify.
- R24: opportunity × existing production efficiency decomposition — did not qualify.
- R27: exact later-qualified R26 opportunity × unchanged production YPT — mixed/fail.
- R27B V2: novel raw strict-prior target-shape/YAC/team/opponent context — mixed/fail.
- R27C/C2: forensic localization showed lead-back failure is downstream post-catch/YPR compression, not catch rate/basic target shape.
- R27D0/D0B/D0C: xYAC/YACOE source audit and physical split showed 2023 catches had slightly *higher* expected YAC but much worse YAC over expected.
- R27D: strict-prior player/offense/opponent relative YACOE residual prediction — first valid result mixed/fail, no integration.

Do not restart generic rolling YPT/YPR/YAC, raw target-shape, or xYAC/YACOE persistence transforms from this same evaluated sample. Future RB receiving-mean work requires genuinely new pregame information/mechanism.

## R27 core result

- run `34423546037`
- job `102703879430`
- artifact `10132290573`
- digest `sha256:cd5c9e26efa44c47ef3374e4deefe9038dc504c155799b48548ad64746e10e45`
- disposition `R27_R26_OPPORTUNITY_TRANSLATION_MIXED_OR_FAIL`
- 24/27 gates PASS
- vacancy target/reception accuracy improved, RB2+ yards improved, but vacancy RB1 receiving-yard MAE worsened and 2023 worsened.

## R27B V2 result

- run `34428917229`
- job `102720004328`
- artifact `10134023092`
- digest `sha256:1df1a14c9900b890fe86e04269849dc90d76fc03a9215ad8612c1820dee5341f`
- disposition `R27B_V2_NOVEL_EFFICIENCY_CONTEXT_MIXED_OR_FAIL_NO_INTEGRATION`
- 24/31 gates PASS
- modest pooled vacancy/p90/RB2+ signal but RB1 and 2023 remained insufficient.

## R27C2 physical result

- run `34431286455`
- job `102727137722`
- artifact `10134581843`
- digest `sha256:6e3aa7ec2149f6f0b65adb59b4cc36c4f88b24a142234bc368131de8e4de60dd`
- result commit `2cfca82d919290ac18ed01559994d2e0239b0798`
- 2023 vacancy RB1 catch rate, air yards/target, screen rate and explosive frequency were roughly stable, but YAC/reception, YPR and YPT compressed.

## R27D0 source audit / R27D0B extension

R27D0:
- run `34431705294`, job `102728396342`, artifact `10134726607`
- digest `sha256:b1a2a44d7ed0303300a792bf617ac6c5095aa7a0229186001504ebc2cad44bdf`
- result commit `8663b132c9910d07679d7b5e03ee8a8c9542f332`
- nflverse PBP xYAC ~99% complete for RB catches; NGS historical receiving feed resolves to zero RB rows and is rejected for this lane.

R27D0B:
- run `34431927369`, job `102729055139`, artifact `10134799189`
- digest `sha256:13853603029adaf820b3228916c519acfdf98794c70462fbbcebe5f93f97d6cd`
- result commit `4e19b549dad2141ba3d55b77c563947342426461`
- 2019 xYAC source support established for legal 2020 training history.

## R27D0C mechanism split

- first valid run `34432497854`
- job `102730752292`
- artifact `10134994377`
- digest `sha256:2c3d6043dc25298b8806e82abe7e4a97e9c2c066408fc38e215eae0bdbc1c882`
- result record commit `76e662c92a49743092d7e512a6a1b2e87a74032c`
- 2023 vs non-2023 vacancy RB1: actual YAC/reception `-0.7781`; expected YAC/reception `+0.2731`; YACOE/reception `-1.0512`.
- tail-reference catastrophic games were strongly positive YACOE, reinforcing that R22 owns stochastic right-tail shape.

---

# 4. R27D strict-prior YACOE residual V1 — CLOSED, NO INTEGRATION

Research branch: `research-rb-r27d-yacoe-residual-v1`.

Frozen plan:
- commit `69edc12a16c691e3838eadcd75559b85dbba7865`
- blob `d0c2b0ff2de154e52fa21fb9ce19b739039633f3`

Frozen model remained exactly:
- 8 strict-prior xYAC/YACOE features only
- player K=12; team K=30; opponent K=30
- StandardScaler + weighted Ridge(alpha=100)
- correction cap ±1.5 yards/reception
- application only to vacancy-active incumbent RB1
- all other rows exact B1 parity
- R26 opportunity/receptions fixed
- R22 untouched
- sportsbook 0

### Preserved Run1 mechanical failure

- run `34435661834`
- job `102740069405`
- head `1757d7549c0fc8c44ad207f55971fdd8c3eab754`
- duplicate deterministic Week1 column collision
- no model fit / no scientific result
- repair record `docs/research/RB_R27D_RUN1_WEEK1_COLUMN_COLLISION_MECHANICAL_REPAIR.md`

### Preserved Run2 integrity/scoring failure

- run `34435897671`
- job `102740771822`
- head `775d9bae7f264ae343209a75cd4947bc88994da9`
- artifact `10136169669`
- digest `sha256:d00eab566113e0c43502bf75f07fe477295dbef6a7ce20389e23ea941ec6d845`
- null outcome rows propagated NaN into the scientific scorecard; emitted mixed/fail string explicitly rejected as invalid
- repair record `docs/research/RB_R27D_RUN2_OUTCOME_NULL_SCORING_MECHANICAL_REPAIR.md`

### First valid scientific result

- first valid head `641b25419c4f5ff3c234d1c000222fb4909ef940`
- run **`34436178615`**
- job **`102741600329`**
- artifact **`10136250846`**
- artifact name `rb-r27d-strict-prior-yacoe-residual-v1`
- digest **`sha256:b975511dc54e961c7745e9d1422ac48be4f867c65d3733adfaeabe93af9951e6`**
- result-record commit **`a20e757a1b85806ae5e244cc42d29cfbc7ed2742`**
- disposition **`R27D_STRICT_PRIOR_YACOE_RESIDUAL_MIXED_OR_FAIL_NO_INTEGRATION`**
- integrity 18/18 PASS
- total scorecard 22/31 PASS
- scientific gates 4/13 PASS

Observed scoring universe: 8,429 RB rows. Vacancy RB1 observed n=503.

Primary RB1:
- B0 MAE `14.305919`
- B1 MAE `14.709395`
- C1 MAE `14.710366`
- C1 vs B1 `+0.006600%` worse — failed >=1% improvement
- C1 remains `+0.404446` yards worse than B0
- B1 RMSE `19.863416` vs C1 `19.900574` — worse
- B1 p90 AE `29.064913` vs C1 `29.555785` — worse
- 30+ miss rate `0.095427 -> 0.093439` — slight improvement
- absolute-bias worsening `0.086634` — within tolerance

2023 vacancy RB1 observed n=74:
- B0 MAE `12.349471`
- B1 MAE `13.992139`
- C1 MAE `14.055947`
- C1 vs B1 `+0.456024%` worse
- C1 remains `+1.706475` yards worse than B0

Season RB1 C1 vs B1:
- 2020 `+1.568506%` worse
- 2021 `-0.712228%` better
- 2022 `-1.712705%` better
- 2023 `+0.456024%` worse
- 2024 `+0.645117%` worse
- 2025 `-0.318705%` better
- only 3/6 improve; no season >2% worse

Aggregate:
- vacancy-active B1 MAE `11.159268`, C1 `11.159545` — microscopically worse, exact non-worse gate fails
- all-RB B1 MAE `10.967611`, C1 `10.967669` — microscopically worse, exact non-worse gate fails
- Week1 B1 `10.830435`, C1 `10.815477` (`-0.138113%`, better)

Scientific conclusion: relative xYAC/YACOE persistence is not a reliable lead-back mean correction. Local wins may not be routed post hoc. No R27D integration design is authorized.

RB receiving-yard mean research is therefore paused at a defensible frontier. Reopen only for genuinely new pregame information/mechanism, not transformations of the exhausted historical efficiency/context families.

---

# 5. Current next action — roster / late-week role handling

Move to a separate operational lane for **current roster / late-week role handling**. This is not historical model science and must not be used to rewrite historical outcomes or leak late information backward.

Required first actions:
1. audit the current Ourlads roster/depth-role path and all current-season provenance/freshness fields;
2. identify where late-week inactive/injury/depth-chart changes can leave stale player role/entitlement in the live Full Slate;
3. inventory existing fallback behavior and whether critical current-role failures fail closed or silently fall back;
4. freeze an operational correctness plan before mutating production behavior;
5. keep historical backtests untouched unless a separately timestamped historical source contract exists;
6. after any operational fix, run production-readiness and Full Slate verification and preserve exact lineage.

Do not mutate the production stack merely because an audit finds an issue. Separate diagnosis, frozen fix plan, implementation lock and verification.

---

# 6. Remaining roadmap after roster/late-week role handling

1. Current roster / late-week role handling — **ACTIVE NEXT**.
2. Grade sealed R26Q using the exact locked R26S evaluator once authoritative Week1 outcomes are available.
3. QB opportunity/efficiency: attempts, dropbacks, pass rate, YPA, sacks, scrambles; build on M89/M90, do not restart failed generic feature hunts.
4. Selective WR/TE unresolved opportunity/efficiency/distribution mechanisms; preserve M38/WR-R15 and TE-R5P winners.
5. Shared QB ↔ receiver conservation across pass attempts, targets, completions, receiving yards and player entitlements.
6. Unified game simulation: plays → pass/rush → player opportunity → outcomes → yards/explosives/TDs → game-state feedback → possessions/scoring.
7. Anytime TD modeling under coherent opportunity/game environment.
8. Game ML / spread / total from football simulation rather than sportsbook imitation.
9. Final operational package / prospective grading: one authoritative slate, live input checks, distributions/fair probabilities, market comparison downstream, prospective scorecards.

Parked hypothesis for later QB↔receiver distribution/conservation lane: player-specific explosive propensity × coverage/DB environment may explain QB high-end passing outcomes. Do not mix it into RB point-mean work; R22 owns RB receiving-yard tail shape.

---

# 7. Promotion safety

For any component:
- research must satisfy pre-frozen gates;
- if PASS only authorizes integration design, freeze that design separately;
- verify exact parents/model hashes and predecessor authority;
- prove no unintended cross-component change;
- run Full Slate verification where relevant;
- record production commit/run/job/artifact/digest/disposition;
- never promote mixed/fail results or cherry-picked post-result cohorts.

GitHub remains the source of truth.
