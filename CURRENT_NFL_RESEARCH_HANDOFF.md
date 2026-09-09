# CURRENT NFL RESEARCH HANDOFF — READ FIRST

**Repository:** `dkaps6/imtiredofthis`  
**Purpose:** canonical cross-chat continuity ledger for the active NFL player-projection research program.  
**Last updated:** 2026-09-09 after R26R final pregame observation, R26S pregame evaluator lock/dry-run, and the 2026 Week-1 RB operational-readiness 35/35 PASS.  
**Protected production-code / Full-Slate authority:** `f8417f55b04ce0e19baf260e9d532765034c47f1`.  
**Current RB state:** protected Week-1 production is operationally ready for RB-P3 rushing + R22 receiving yards + baseline receptions; the fully sealed R26 receptions candidate is available side-by-side as a research-only pregame sidecar. R26 is **not production-active** yet.

> **Future ChatGPT sessions / agents:** read this file first, then `AGENTS.md`, then the exact active plan/result/repair files named below. Preserve scientific failures, distinguish mechanical failures from science, never let sportsbook data flow upstream, and do not promote research into production without a separately authorized promotion step.

---

## 1. User's controlling objective

Project individual NFL player outcomes as accurately as possible pregame: yards, receptions, carries, and downstream fair probabilities.

Current architecture:

> **GAME / TEAM OPPORTUNITY → POSITION/ROOM POOL → INDIVIDUAL PLAYER ENTITLEMENT → PLAYER+MATCHUP EFFICIENCY → JOINT MONTE CARLO DISTRIBUTION**

Sportsbook data is downstream benchmarking/pricing only. It is never an upstream correction to football projections.

For the current RB receiving lane, the question is **who inside a changing backfield should own a finite RB receiving-opportunity pool**, not whether every RB should receive a broad positive adjustment.

---

## 2. Non-negotiable methodology

- Canonical production authority is `.github/workflows/full-slate.yml` at protected head `f8417f55b04ce0e19baf260e9d532765034c47f1`.
- Historical science is strict walk-forward / leakage-safe.
- Freeze hypothesis, candidate, population, metrics, thresholds, gates and authority ceiling before results.
- Mechanical failures may receive only the minimum documented value-neutral repair; they are not scientific failures.
- Scientific failures remain preserved; do not lower gates or tune around them after seeing results.
- No sportsbook football inputs upstream of projections.
- Current R26 prospective work uses **zero 2026 Week-1 outcomes**.
- Research PASS does not silently change production.
- Same-week/current depth may be used for market identity or separate future research, but it did not change the sealed R26Q candidate.

---

## 3. Protected current production stack

### Current Full Slate population authority

- production head: `f8417f55b04ce0e19baf260e9d532765034c47f1`
- Full Slate run: `34317211395`
- artifact: `10090547415`
- artifact name: `run_34317211395`
- digest: `sha256:7eab77e41c5879d4f54d87497eee0d1186010784cfc4ef966178930e16fb7c3b`
- player rows: `468`
- teams: `32`
- games: `16`

### Protected model authorities

- QB passing-yard mean: **M89/M90**
- QB distribution: **mean-neutral C2**
- WR: **M38 WR1 anchor + WR-R15 WR2+ entitlement**
- TE: **TE-R5P**
- RB rushing: **RB-P3**
- RB receiving-yard distribution/tails: **R22**, using pinned R19 assets
- RB receptions in production: **current protected baseline**; R26 is not yet production-active.

### R22 production integration authority

- run `34298516960`
- artifact `10084118525`
- artifact name `rb-r22-week1-production-integration-v2`
- digest `sha256:2391bd9914e9d0029c63529829496fb9b462d008bdcc3f8c37bb3e1079580bd1`
- adapter disposition `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_ADAPTER_PASS`
- integration disposition `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_INTEGRATION_PASS`
- pricing disposition `RB_R22_WEEK1_RECEIVING_TAIL_PRICING_LINEAGE_PASS`
- adapted RB keys: `94`; FB rows: `13` exact/unadapted
- max receiving-yard mean delta `5.329070518200751e-15`
- receptions exact, FB exact, non-RB exact, RB other markets exact, rush+receiving identity preserved
- current/future outcomes used `0`; sportsbook inputs to adapter `0`; production mean parameters changed `0`.

### R19 serialized R9 authority

- run `34288244770`
- artifact `10080377483`
- digest `sha256:11432b9d7b7f2367935a862b63c30df9f40955e479806ab67d90921b63a907c7`
- inner model SHA `9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba`
- residual pools SHA `c69a268a5a1683e846bcb5f59fe55bcae20d70c679792449ea77e6548b37a362`
- no R9 refit in the R26 prospective chain.

---

## 4. R26 historical foundation — preserve support and failures

### R26 parent

- run `34356222339`
- artifact `10106271075`
- digest `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`
- disposition `RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW`
- 19/20 gates; all-season 2023 vacancy-incumbent reception MAE worsened about 4.54% versus the frozen <=2% cap.

Useful/supporting components remain preserved: vacancy signal, R9 player identity, pooled targets/receptions, Week-1 behavior, RB1 behavior, and 5/6 seasons.

### R26E Week-1 qualification

- run `34368268224`
- artifact `10110785184`
- digest `sha256:2b64fe25a1024136f2bb2cdde42bc74de290b6b9df63b4bb6cd675f6095b8bb7`
- disposition `WEEK1_COMPONENT_NOT_QUALIFIED_NO_SHADOW`
- 19/20; sole blocker was 2020 Week 1 worsening about `8.9668%`; 2021-2025 improved.

Do not erase or retroactively exclude 2020.

### R26J / R26K 2020 follow-up

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

Conclusion: 2020 was distinct, but no historical router/guard replicated. Do not invent one.

### R26L — 2026 transportability

- branch `research-rb-r26l-2026-week1-regime-transportability-v1`
- run `34389455694`
- artifact `10119058769`
- digest `sha256:3351dfb5bbf6b571174a94ddf0a03179d70786edbb91d316e4fdf2c3cf005c46`
- disposition `2026_SOURCE_REGIME_MODERN_LIKE_FOR_PROSPECTIVE_QUALIFICATION`
- 31 vacancy rooms / exact non-vacancy CIN
- 6/7 features modern-closer; modern distance `0.72555857`, 2020 distance `1.22202458`, ratio about `0.593735`
- zero outcomes / sportsbook / same-week depth / R9 refit.

### R26M — prospective synthesis

- run `34390505549`
- artifact `10119429741`
- digest `sha256:1306a3a2e58a0b129ac7e9fe34ad6407d87c491494e8fc27dd0284ba96996b76`
- disposition `2026_WEEK1_UNMODIFIED_R26_SHADOW_CANDIDATE_DESIGN_QUALIFIED`

R26M authorized only prospective construction of the **unmodified** R26 candidate.

---

## 5. R26N — exact 2026 structural candidate — PASS 28/28

Branch: `research-rb-r26n-2026-week1-unmodified-r26-structural-candidate-v1`

- frozen plan `919285fd0e2042461a5a89472f6832c01da857b4`
- builder `5299ce54575ffcfe33ad203db0ee00285181291f`
- successful head `3b7a00e282cb925bd7a33175bc0b7d08d1467b2f`
- run `34396075045`
- job `102616001356`
- artifact `10121598376`
- digest `sha256:887929203053cb62904aaaeda9d995c9645163814da181972799f08fe4465c62`
- disposition `R26N_2026_WEEK1_STRUCTURAL_CANDIDATE_PASS_READY_FOR_SHADOW_INTEGRATION_DESIGN`
- **28/28 PASS**.

State:
- 468 players / 32 teams / 16 games
- 107 RB/FB rows
- 31 vacancy teams; CIN exact non-vacancy control
- 104 RB/FB rows changed
- fixed RB-room pool conserved exactly
- non-RB entitlement exact
- strict-prior R9 only; no refit
- zero outcomes / sportsbook / same-week depth
- R22 receiving yards and all other protected model outputs untouched.

R26N mechanical repairs were identity-key / pandas dtype compatibility only; original scientific builder stayed byte-identical.

---

## 6. R26O / R26P — receptions-only MC integration — FINAL PASS 38/38

Branch: `research-rb-r26o-2026-week1-receptions-shadow-integration-compatibility-v1`

Frozen R26O:
- plan `96e4a6977458b55ec00cb29fbc9de05c56c89b88`
- evaluator `6a171cb32e77c7d6691ceb75da9861a5d6dd7bc4`
- 25,000 draws / seed 42 / 38 gates / mean compatibility tolerance `0.05 receptions`.

Preserved earlier 37/38 execution:
- run `34398759284`
- artifact `10122672501`
- digest `sha256:6d72de1eb8902fd8d956feb01b4d1a3cb5847aa58417ad4f416f3e1102e96e82`
- disposition `R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_FAIL_NO_SHADOW`
- sole Gate-15 failure was null R22 audit evidence.

R26P forensic:
- run `34399525657`
- artifact `10122862934`
- digest `sha256:2e596f3a3bcbef156983664d332ded94ef41dae1402d1a96ab0b61d185c68d47`
- disposition `R26P_GATE15_EVIDENCE_WIRING_DEFECT_CONFIRMED_MECHANICAL_RERUN_AUTHORIZED`
- 17/17 forensic checks.

Final corrected R26O:
- head `e7014a6e365cbb776e48085dcef12dfece744ca4`
- run `34399750746`
- job `102628405629`
- artifact `10123070453`
- digest `sha256:27307ad84c232935aed3b25c0e0c9bff4da70cd2915376f41dfa3f85c25f14d0`
- disposition `R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_PASS_READY_FOR_PROSPECTIVE_SEAL`
- **38/38 PASS**.

Exact state:
- 104 intended RB/FB reception arrays changed
- forbidden changed arrays `0`
- 3 CIN controls exact baseline
- all non-reception arrays exact
- RB/FB receiving-yard and rush+receiving arrays exact R22 baseline
- RB rushing / QB passing / non-RB arrays exact
- R22 Gate-15 max mean delta `3.552713678800501e-15`
- max analytical/MC compatibility gaps all under frozen `0.05`
- zero outcomes / sportsbook / same-week depth / R9 refit / production changes.

---

## 7. R26Q — immutable Week-1 prospective seal — PASS 28/28

Branch: `research-rb-r26q-2026-week1-prospective-seal-v1`

- plan `03b0cdf6367ca17d085d55aa1633f7f705f642ee`
- sealer `fb6610be550cb4bdefec640ae83d202f2adb2c4a`
- lock `8e01b46b5c0f7c944fbe04ea1d3cc8885c094ff4`
- head `68661da94f03cab2f96182d47636cf55e088b5de`
- run `34400524030`
- job `102630996205`
- artifact `10123251043`
- digest `sha256:dd3ec0e8e3831ab7f2255c2e5abf343cda8a7943d33a1d4863e52372d6f858a1`
- disposition `R26Q_2026_WEEK1_RECEPTIONS_PROSPECTIVE_SEAL_PASS_READY_FOR_OBSERVATION`
- **28/28 PASS**.

Seal:
- 107 RB/FB arrays
- 104 changed / 3 CIN controls
- 25,000 draws / seed 42
- every array hash exact
- 17 parent evidence files byte-exact
- no football regeneration, outcomes, sportsbook upstream, same-week depth, R9 refit, production change or promotion.

**R26Q is the immutable pregame R26 football candidate. Do not recompute it.**

---

## 8. R26R — final prospective market observation — PASS 30/30

Branch: `research-rb-r26r-2026-week1-prospective-observation-snapshot-v1`

- plan `5ebdc456dfb58fee3cf9c2087401d332895908e9`
- evaluator `81aa92e2c74c46ea7191ba36444498f11219c4de`
- lock `5259db7d561752469f2fcaf9f513f747d1b7d56a`
- head `469aa40c90c738e12a82ee32ccca70c9cdbbc29f`
- run `34401814588`
- job `102635265504`
- artifact `10124274040`
- digest `sha256:b4d3e573909803e892d57858c9b0c4bdac089bcf078c1d6f6e53366e4622303e`
- disposition `R26R_2026_WEEK1_PROSPECTIVE_OBSERVATION_SNAPSHOT_PASS_MARKET_CAPTURED`
- **30/30 PASS**.

Captured:
- 320 all-position player-reception book lines
- 70 RB/FB reception book lines
- 35 matched sealed RB/FB players.

Boundary:
- Week-1 outcomes used `0`
- sportsbook football inputs `0`
- current roster used only for market identity, not candidate
- football values regenerated `false`
- tuning / production promotion / live-shadow activation `false`.

Market evidence remains downstream benchmarking only.

---

## 9. R26S — frozen postgame prospective evaluator — READY, CURRENTLY INCOMPLETE BY DESIGN

Branch: `research-rb-r26s-2026-week1-postgame-prospective-evaluation-v1`

Frozen contract:
- plan commit `63722c61ca1b408b8ae77eba383e5fa8daecf9f2`
- evaluator commit `e42099dbf11cdd510eb64e43a0910444c75ec7f8`
- implementation lock `3a706fa52f91f6584f6fd6594563239dd6ea3b53`

R26S is a **postgame evaluator only**. It is intentionally built and locked before games so the rules cannot move after outcomes.

Frozen science includes:
- official Week-1 receptions + snap participation
- primary changed-player MAE
- 10,000 team-cluster bootstrap, seed 42
- lower CI floor `-0.05 receptions/player`
- within-room share MAE
- large-mover / frozen pregame-role cohort guards
- DNP/zero-snap protection
- sportsbook benchmark downstream only
- at least 50 evaluable changed rows + all 16 games required, otherwise INCOMPLETE.

### Preserved first mechanical execution

- run `34405689393`
- job `102647981880`
- head `51806c896fdd79f1d1f9a56f7e3bfa3b0ce4b065`
- artifact `10125223220`
- digest `sha256:6130a64872739436a0626ee45eb249a06edb3281a8f750755c8b4483af6aa9bd`
- emitted FAIL before science because evaluator expected R26Q gate-count keys not stored in canonical disposition JSON
- Week-1 weekly data unavailable; snap-count support only through 2025; primary evaluable rows `0`
- all scientific accuracy gates were explicitly not evaluated.

Frozen mechanical repair:
- note `b78d4e1671e30eee392ed07a219d409aa00f8b41`
- compatibility wrapper `adb4b07fa3d7f0585c477d217a58095ac0b9ae10`
- staged compatibility copy only adds `gate_count:28` / `gate_pass_count:28`; all football files remain exact.

### Authoritative current R26S pregame state

- run `34411889262`
- job `102667966181`
- head `af539bd1926845f438d41156e76b44382acb2078`
- artifact `10127562840`
- digest `sha256:e9ecd949816eafd813a50e60e8af28ce387e668848483ffcda3dfadf7614919e`
- disposition `R26S_2026_WEEK1_POSTGAME_PROSPECTIVE_EVALUATION_INCOMPLETE_NO_DECISION`
- 22 frozen gates; source/scientific outcome gates remain unscored because Week 1 has not happened
- scientific evaluation ready `false`
- primary evaluable rows `0`
- production/sportsbook/model changes `0`.

**This is the correct pregame R26S state, not a scientific failure. Rerun the exact locked evaluator unchanged after authoritative Week-1 weekly stats and snap counts exist.**

---

## 10. 2026 Week-1 RB pregame operational-readiness authority — PASS 35/35

Branch: `audit-rb-week1-2026-operational-readiness-v1`

Purpose: certify what can actually run **before kickoff** across RB rushing, receiving yards and receptions while exposing R26 side-by-side without promoting it.

Frozen lineage:
- plan `41449230c36dbc008e2efc9b8e1ffe1bd28bb3f0`
- evaluator `db351aa5c1a0a0cc7fcd657e5c222490a4535366`
- implementation lock `947135e16ee51ab3da5b4a7a3caf5b6414e0bd52`.

### Preserved first run — mechanical identity-format miss only

- run `34412565779`
- head `66e50f9345ecab8dbf58920526ddfba404738ca8`
- artifact `10127822769`
- digest `sha256:fd847cfb6ff1dc9e8a1d0afc76bc98661b8b5ea02c42ecceeb59bc0ea48bb013`
- disposition `RB_WEEK1_2026_PREGAME_OPERATIONAL_READINESS_FAIL_NOT_READY`
- **34/35 gates passed**
- sole failure: exact raw role-key join for three provider-punctuation keys:
  - CHI D'Andre Swift `d'andreswift` vs `dandreswift`
  - MIA De'Von Achane `de'vonachane` vs `devonachane`
  - WAS Jacory Croskey-Merritt `jacorycroskey-merritt` vs `jacorycroskeymerritt`.

Every football-value/readiness gate already passed in that run.

Frozen value-neutral repair:
- repair note `b825c6479fb69a66c1e8bc71cb717dd4d950b8d2`
- staging wrapper `cd695b72ce51b60bebcd0b03ab4dec2273703239`
- exactly three `player_clean_key` cells changed in staged R26R role evidence only
- all role/depth values unchanged
- 20 non-role R26R files byte-exact
- no fuzzy matching
- football values / R26 arrays unchanged.

### Final authoritative readiness PASS

- run **`34412854521`**
- job **`102671015758`**
- head **`ee6221bb484efb993ba6aa8149085daef0c04db1`**
- artifact **`10127920603`**
- artifact name `rb-week1-2026-pregame-operational-readiness-v1`
- digest **`sha256:b619429022a5bfc50257ebbb55600c1ff1299e4b706997d278545f66a469dd5e`**
- disposition **`RB_WEEK1_2026_PREGAME_OPERATIONAL_READINESS_PASS_PRODUCTION_STACK_READY_R26_SIDECAR_READY`**
- **35/35 gates passed**.

Certified Week-1 RB universe:
- 107 RB/FB players
- 32 teams
- 94 RB / 13 FB
- all current roles resolved (`0` unresolved)
- exact 107-key equality: protected P3 production ↔ R26Q sealed manifest
- exact 94-key equality: protected production RBs ↔ R22 adapted RBs
- all P3 `stack_att` / `stack_yards` finite and nonnegative
- P3 route `WEEK1_STACK_OVERRIDE`, version `RB_P3_SYNTHESIS_V1`, 25,000 iterations, sportsbook football inputs `0`
- R22 adapter/integration/pricing all PASS; receiving-yard mean parity preserved
- R26Q 28/28, all 107 25k arrays/hash seals exact
- R26R 30/30, 35 players with captured reception market context
- Week-1 outcomes used `0`
- football values regenerated `false`
- production parameters changed `false`
- production promotion performed `false`
- R26 live-production activation performed `false`.

The final artifact contains `rb_week1_player_readiness_manifest.csv`, a 107-row player board with:
- current role/model role
- protected production rush-attempt mean
- protected production rush-yard mean + implied YPC
- current production receptions mean
- sealed R26 candidate receptions mean + delta
- R26 p10/p25/p50/p75/p90
- vacancy flag
- R22 receiving-yard mean/status for RBs; explicit FB exact/unadapted status
- R22 tail-state evidence
- R26R market coverage / line median where available
- component readiness flags.

**Operational interpretation:** the Week-1 RB production stack is ready to run now. R26 is also fully materialized and aligned for Week 1 as a research sidecar, but is not silently replacing production receptions before the postgame prospective test.

---

## 11. Player-specific identity / acute injury role-transfer note

The user's “player tree” concept is correct.

Each RB already has his own strict-prior player identity/history rather than being treated as a copy of the starter: prior/recent targets and receptions, target share, RB-room share, same-team usage, experience/sample size and related R9 identity signals.

However, sealed R26 deliberately used `same_week_depth_used=false`. Therefore a sudden same-week promotion (for example, Bijan Robinson ruled out and RB2 starting) does **not** yet have a separately validated dynamic role-inheritance transform.

Future research question:

> Given a sudden role promotion, how much opportunity should this specific replacement inherit based on his own identity/profile, rather than assuming he becomes the injured starter?

Preserve that as a separately frozen future study. Do not backfill it into the already-sealed Week-1 candidate.

---

## 12. Exact next actions

### Before Week-1 kickoff

1. Treat final readiness run `34412854521` / artifact `10127920603` as the RB pregame operational-readiness authority.
2. Production RB execution remains:
   - RB-P3 carries/rushing yards
   - R22 receiving-yard distributions
   - current protected baseline receptions.
3. R26Q/R26R may be inspected side-by-side through the 107-row readiness manifest; R26 must remain research-sidecar only unless the governance requirement is explicitly changed.
4. If current rosters/injuries change materially before kickoff, do **not** rewrite the sealed R26 candidate. Handle ordinary protected production live-input behavior normally; preserve acute RB role-transfer as separate research.

### After Week-1 games

5. Rerun the exact locked R26S evaluator only when authoritative Week-1 weekly stats + snap counts are complete.
6. Preserve PASS / FAIL / INCOMPLETE exactly as frozen.
7. If R26S passes, it authorizes only a separately frozen production-promotion review (R26T or later), not automatic promotion.
8. If R26S scientifically fails, preserve the result and do not tune/reopen the sealed Week-1 candidate.

### Later RB research

9. Separately freeze and study acute injury / role-inheritance behavior by player identity.
10. After the R26 receptions chain is resolved, continue other RB lanes only without disturbing protected P3/R22 authorities.

---

## 13. Broader lane reminders

### QB
M89/M90 mean authority remains protected; C2 distribution remains production-qualified QB-only mean-neutral state work.

### WR
M38 remains WR1 anchor; WR-R15 handles WR2+ entitlement. Broad additive target boosts that failed remain closed.

### TE
TE-R5P remains production. Keep opportunity allocation separate from efficiency.

### RB
- RB-P3 rushing production is Week-1 operationally ready.
- R22 receiving-yard production is Week-1 operationally ready.
- current baseline receptions remain production-active.
- R26Q is the immutable receptions candidate; R26R is its pregame market observation; R26S is the locked future postgame outcome test.
- final Week-1 readiness authority is run `34412854521`, artifact `10127920603`, digest `sha256:b619429022a5bfc50257ebbb55600c1ff1299e4b706997d278545f66a469dd5e`, 35/35 PASS.

---

## 14. One-sentence current state

**The 2026 Week-1 RB stack is operationally certified 35/35 across protected RB-P3 rushing, R22 receiving yards, baseline receptions and the exact 107-player identity universe; the new R26 receptions allocation has progressed through historical research, 2026 transportability, structural construction, Monte Carlo integration, immutable sealing and live pregame observation and is now available as a fully aligned research sidecar, while the separately frozen R26S postgame test waits—correctly—for Week-1 games to actually occur before any production-promotion decision is made.**
