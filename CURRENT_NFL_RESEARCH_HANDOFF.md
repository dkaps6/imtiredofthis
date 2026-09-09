# CURRENT NFL RESEARCH HANDOFF — READ FIRST

**Repository:** `dkaps6/imtiredofthis`  
**Purpose:** canonical cross-chat continuity ledger for the active NFL player-projection research program.  
**Last updated:** 2026-09-09 after R26Q prospective seal PASS; R26R prospective market observation is the live frontier.  
**Protected production-code / Full-Slate authority:** `f8417f55b04ce0e19baf260e9d532765034c47f1`.  
**Current research frontier:** exact R26O receptions shadow is immutably sealed by R26Q; R26R is observing current Week-1 reception markets downstream without changing the sealed football candidate.

> **Future ChatGPT sessions / agents:** read this file first, then `AGENTS.md`, then the exact active R26 plan/result files named below. Do not alter production, do not reinterpret preserved failures, and do not let sportsbook data flow upstream. Update this file on `main` after each substantive frozen plan, result, mechanical repair, promotion, failure, or change in authorized next step.

---

## 1. User's controlling objective

Project individual NFL player outcomes as accurately as possible pregame: yards, receptions, carries, and downstream fair probabilities.

Current architecture:

> **GAME / TEAM OPPORTUNITY → POSITION/ROOM POOL → INDIVIDUAL PLAYER ENTITLEMENT → PLAYER+MATCHUP EFFICIENCY → JOINT MONTE CARLO DISTRIBUTION**

Sportsbook data is downstream benchmarking/pricing only. It is never an upstream correction to football projections.

For the current RB receiving lane, the research question is specifically **who inside a changing backfield should own a finite RB receiving-opportunity pool**, not whether every RB should receive a broad positive target/reception adjustment.

---

## 2. Non-negotiable methodology

- Canonical production authority is `.github/workflows/full-slate.yml` at protected head `f8417f55b04ce0e19baf260e9d532765034c47f1`.
- Historical science is strict walk-forward / leakage-safe.
- Freeze hypothesis, candidate, population, metrics, thresholds, gates and authority ceiling before results.
- Mechanical failures may receive only the minimum documented plumbing repair and exact rerun; they are not scientific failures.
- Scientific failures remain preserved; do not lower gates or tune around them after seeing results.
- No sportsbook football inputs upstream of projections.
- Current R26 prospective work uses **zero 2026 Week-1 outcomes**.
- Research PASS does not silently change production.
- Current roster/depth may be used for sportsbook identity/scoping in R26R, but it may not modify the sealed R26Q football candidate.

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

R26N/R26O/R26Q/R26R have **not** altered RB receiving-yard means, R22 distributions, QB arrays, rushing arrays, WR/TE arrays, or production parameters.

### R22 production integration authority

- run `34298516960`
- artifact `10084118525`
- artifact name `rb-r22-week1-production-integration-v2`
- digest `sha256:2391bd9914e9d0029c63529829496fb9b462d008bdcc3f8c37bb3e1079580bd1`
- disposition `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_INTEGRATION_PASS`

R22 adapts RB `rec_yards` / `rush_rec_yards`; receptions remain unadapted in production.

### R19 serialized R9 authority

- run `34288244770`
- artifact `10080377483`
- digest `sha256:11432b9d7b7f2367935a862b63c30df9f40955e479806ab67d90921b63a907c7`
- inner model SHA `9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba`
- residual pools SHA `c69a268a5a1683e846bcb5f59fe55bcae20d70c679792449ea77e6548b37a362`
- no R9 refit in the R26 prospective chain.

---

## 4. R26 historical foundation — preserve both support and failures

R26 tested vacancy-gated R9 redistribution of a fixed RB/FB receiving-opportunity pool.

### R26 parent

- run `34356222339`
- artifact `10106271075`
- digest `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`
- disposition `RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW`
- 19/20 gates; the full candidate failed because all-season 2023 vacancy-incumbent reception MAE worsened about 4.54% versus the frozen <=2% cap.

Useful/supporting components remain preserved: vacancy signal, R9 player identity, pooled targets/receptions, Week-1 behavior, RB1 behavior, and 5/6 seasons.

### R26E Week-1 qualification

- run `34368268224`
- artifact `10110785184`
- digest `sha256:2b64fe25a1024136f2bb2cdde42bc74de290b6b9df63b4bb6cd675f6095b8bb7`
- disposition `WEEK1_COMPONENT_NOT_QUALIFIED_NO_SHADOW`
- 19/20 gates
- sole blocker: 2020 Week 1 worsened about `8.9668%`; 2021-2025 improved.

**Do not erase or exclude 2020 retroactively.**

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

---

## 5. R26L — 2026 source-regime transportability — PASS

Branch: `research-rb-r26l-2026-week1-regime-transportability-v1`

- run `34389455694`
- artifact `10119058769`
- digest `sha256:3351dfb5bbf6b571174a94ddf0a03179d70786edbb91d316e4fdf2c3cf005c46`
- disposition `2026_SOURCE_REGIME_MODERN_LIKE_FOR_PROSPECTIVE_QUALIFICATION`

Frozen evidence:
- 31 2026 Week-1 vacancy rooms; exact non-vacancy team `CIN`
- 6/7 source-regime features modern-closer
- normalized modern distance `0.72555857`
- 2020 distance `1.22202458`
- ratio about `0.593735`
- zero features beyond 2020 in anomalous direction
- zero outcomes / sportsbook inputs / same-week depth / R9 refit.

---

## 6. R26M — prospective synthesis — PASS

Branch: `research-rb-r26m-2026-week1-prospective-qualification-synthesis-v1`

- run `34390505549`
- artifact `10119429741`
- digest `sha256:1306a3a2e58a0b129ac7e9fe34ad6407d87c491494e8fc27dd0284ba96996b76`
- disposition `2026_WEEK1_UNMODIFIED_R26_SHADOW_CANDIDATE_DESIGN_QUALIFIED`

R26M preserved the historical 2020 failure and authorized only prospective construction of the **unmodified** R26 Week-1 candidate.

---

## 7. R26N — exact 2026 structural candidate — PASS 28/28

Branch: `research-rb-r26n-2026-week1-unmodified-r26-structural-candidate-v1`

Read:
- `docs/research/RB_R26N_2026_WEEK1_UNMODIFIED_R26_STRUCTURAL_CANDIDATE_V1_FROZEN_PLAN.md`
- `docs/research/RB_R26N_2026_WEEK1_UNMODIFIED_R26_STRUCTURAL_CANDIDATE_V1_RESULT.md`
- both R26N mechanical repair notes.

Canonical:
- frozen plan commit `919285fd0e2042461a5a89472f6832c01da857b4`
- frozen builder commit `5299ce54575ffcfe33ad203db0ee00285181291f`
- successful head `3b7a00e282cb925bd7a33175bc0b7d08d1467b2f`
- run `34396075045`
- job `102616001356`
- artifact `10121598376`
- digest `sha256:887929203053cb62904aaaeda9d995c9645163814da181972799f08fe4465c62`
- disposition `R26N_2026_WEEK1_STRUCTURAL_CANDIDATE_PASS_READY_FOR_SHADOW_INTEGRATION_DESIGN`
- **28/28 frozen gates passed**.

Structural state:
- 468 players / 32 teams / 16 games
- 107 RB/FB rows
- 31 vacancy teams
- CIN exact non-vacancy baseline
- 104 RB/FB rows changed
- fixed RB-room pool conserved exactly
- non-RB entitlement exact
- R9 training season 2025, reliability 1.0, no refit
- strict-prior history only
- zero outcomes / sportsbook inputs / same-week depth
- R22 and receiving-yard means/distributions untouched.

R26N mechanical repairs were identity-key / pandas join-dtype compatibility only, frozen before implementation. The original scientific builder remained unchanged.

---

## 8. R26O — receptions-only MC shadow integration — FINAL PASS 38/38

Branch: `research-rb-r26o-2026-week1-receptions-shadow-integration-compatibility-v1`

Read first:
- `docs/research/RB_R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_COMPATIBILITY_V1_FROZEN_PLAN.md`
- `docs/research/RB_R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_COMPATIBILITY_V1_RESULT.md`
- R26O mechanical repair notes
- R26P Gate-15 forensic plan/result if auditing the preserved earlier 37/38 run.

Frozen R26O:
- plan `96e4a6977458b55ec00cb29fbc9de05c56c89b88`
- evaluator `6a171cb32e77c7d6691ceb75da9861a5d6dd7bc4`
- 38 gates
- 25,000 Monte Carlo draws
- seed `42`
- frozen mean compatibility tolerance `0.05 receptions`.

### Preserved earlier R26O 37/38 run

- head `550d3d532e9f27c34b90ecd027f3811292674862`
- run `34398759284`
- job `102625073624`
- artifact `10122672501`
- digest `sha256:6d72de1eb8902fd8d956feb01b4d1a3cb5847aa58417ad4f416f3e1102e96e82`
- disposition `R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_FAIL_NO_SHADOW`
- 37/38 gates; only Gate 15 recorded null R22 mean-parity evidence.

R26P later proved the issue was evidence wiring, not a changed scientific gate:
- R26P run `34399525657`
- artifact `10122862934`
- digest `sha256:2e596f3a3bcbef156983664d332ded94ef41dae1402d1a96ab0b61d185c68d47`
- disposition `R26P_GATE15_EVIDENCE_WIRING_DEFECT_CONFIRMED_MECHANICAL_RERUN_AUTHORIZED`
- 17/17 forensic gates.

The earlier R26O FAIL remains preserved; it was not retroactively relabeled.

### Final corrected canonical R26O

R26P-authorized correction lineage:
- Gate-15 repair note `c40de155c3f19785f8398cbcb9aaa2ab2445c2f4`
- audit-payload wrapper `85c3f2ca6bc73c302a3a33f3b6e75f4df42ae78a`
- final launch/head `e7014a6e365cbb776e48085dcef12dfece744ca4`

Canonical execution:
- run `34399750746`
- job `102628405629`
- artifact `10123070453`
- artifact name `rb-r26o-2026-week1-receptions-shadow-integration-compatibility-v1`
- digest `sha256:27307ad84c232935aed3b25c0e0c9bff4da70cd2915376f41dfa3f85c25f14d0`
- disposition `R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_PASS_READY_FOR_PROSPECTIVE_SEAL`
- **38/38 frozen gates passed**.

Exact compatibility evidence:
- 104 intended RB/FB reception arrays changed
- forbidden changed arrays `0`
- 3 CIN RB/FB receptions exact baseline
- all non-RB/FB receptions exact baseline
- all non-reception arrays exact baseline
- all RB/FB rec-yard arrays exact R22 baseline
- all RB/FB rush+rec arrays exact R22 baseline
- all RB rushing arrays exact baseline
- all QB pass-yard arrays exact baseline
- full 2,892 simulation-key universe exact
- deterministic replay exact
- Gate 15 R22 mean delta `3.552713678800501e-15`
- max baseline mean gap vs sealed R26N `0.017563893432533284`
- max candidate mean gap `0.018766081922851008`
- max candidate-minus-baseline delta gap `0.023037323394851983`
- all below frozen `0.05` tolerance
- zero outcomes / sportsbook football inputs / same-week depth / R9 refit / production changes.

Representative 25k research-shadow reception shifts:
- Omarion Hampton `2.12164 -> 3.23328`
- Jonathan Taylor `1.57412 -> 2.56996`
- Jaylen Warren `1.54380 -> 2.52768`
- Woody Marks `1.35000 -> 2.18748`
- D'Andre Swift `1.67208 -> 2.38604`
- Bijan Robinson `2.89060 -> 3.42608`
- Bam Knight `1.11328 -> 1.62536`
- Brian Robinson `0.68560 -> 0.19776`
- Jeremiyah Love `1.15676 -> 0.81148`.

These remain **research-shadow receptions only**. Production remains unchanged.

---

## 9. R26Q — immutable 2026 Week-1 prospective seal — PASS 28/28

Branch: `research-rb-r26q-2026-week1-prospective-seal-v1`

Read:
- `docs/research/RB_R26Q_2026_WEEK1_PROSPECTIVE_SEAL_V1_FROZEN_PLAN.md`
- `docs/research/RB_R26Q_2026_WEEK1_PROSPECTIVE_SEAL_V1_IMPLEMENTATION_LOCK.md`
- `scripts/backtest/seal_rb_r26q_2026_week1_receptions_prospective_v1.py`.

Frozen lineage:
- plan commit `03b0cdf6367ca17d085d55aa1633f7f705f642ee`
- sealer commit `fb6610be550cb4bdefec640ae83d202f2adb2c4a`
- implementation lock `8e01b46b5c0f7c944fbe04ea1d3cc8885c094ff4`
- launch/head `68661da94f03cab2f96182d47636cf55e088b5de`.

Canonical execution:
- run **`34400524030`**
- job **`102630996205`**
- artifact **`10123251043`**
- artifact name `rb-r26q-2026-week1-receptions-prospective-seal-v1`
- digest **`sha256:dd3ec0e8e3831ab7f2255c2e5abf343cda8a7943d33a1d4863e52372d6f858a1`**
- disposition **`R26Q_2026_WEEK1_RECEPTIONS_PROSPECTIVE_SEAL_PASS_READY_FOR_OBSERVATION`**
- **28/28 frozen gates passed**.

Seal evidence:
- exact R26O run/artifact/digest/head pinned
- 107 RB/FB reception arrays sealed
- 104 changed vacancy-active arrays
- 3 unchanged CIN arrays
- 25,000 draws / seed 42
- all per-array SHA256 values exact
- 17 R26O evidence files copied byte-exact
- no football values regenerated
- zero Week-1 outcomes used
- zero sportsbook football inputs
- no same-week depth
- no R9 refit
- no production parameter changes
- no live shadow production activation
- no production promotion.

R26Q is the immutable pregame football candidate authority for all subsequent observation/postgame evaluation. **Do not recompute it.**

---

## 10. Player-specific identity / injury-role-transfer note

The user's “player tree” concept is directionally correct and important.

The RB receiving identity layer already gives each player his own strict-prior historical branch rather than treating an RB2 as a copy of an injured RB1. Existing player-specific inputs include prior/last-8/previous-season targets and receptions, target share, RB-room share, high-target rates, same-team prior usage, experience/sample-size indicators, and related R9 identity features.

However, the sealed R26N/R26O/R26Q candidate deliberately reports `same_week_depth_used=false`. Therefore do **not** claim that a sudden same-week injury promotion (example: established RB1 ruled out and RB2 inherits the start) already has a separately validated dynamic role-transfer magnitude.

Scientific interpretation:

> The replacement player has his own identity tree; what remains to be separately audited is the **role-inheritance transform** — how much a sudden promotion should expand that specific player's workload given his own historical/athletic/role profile, rather than assuming he becomes the injured starter.

This is a future separately frozen study. It must not be smuggled into the already-sealed Week-1 candidate.

---

## 11. R26R — CURRENT LIVE FRONTIER: prospective sportsbook observation

Branch: `research-rb-r26r-2026-week1-prospective-observation-snapshot-v1`

Purpose: capture a genuinely prospective Week-1 sportsbook observation beside the immutable R26Q candidate. Sportsbook information remains downstream and cannot alter any football value.

Read:
- `docs/research/RB_R26R_2026_WEEK1_PROSPECTIVE_OBSERVATION_SNAPSHOT_V1_FROZEN_PLAN.md`
- `docs/research/RB_R26R_2026_WEEK1_PROSPECTIVE_OBSERVATION_SNAPSHOT_V1_IMPLEMENTATION_LOCK.md`
- `scripts/backtest/observe_rb_r26r_2026_week1_receptions_market_snapshot_v1.py`
- `.github/workflows/research-rb-r26r-2026-week1-prospective-observation-snapshot-v1.yml`.

Frozen lineage:
- R26R plan commit **`5ebdc456dfb58fee3cf9c2087401d332895908e9`**
- evaluator commit **`81aa92e2c74c46ea7191ba36444498f11219c4de`**
- implementation lock **`5259db7d561752469f2fcaf9f513f747d1b7d56a`**
- launch/head **`469aa40c90c738e12a82ee32ccca70c9cdbbc29f`**
- frozen gates: **30**.

Pinned immutable parent:
- R26Q run `34400524030`
- artifact `10123251043`
- digest `sha256:dd3ec0e8e3831ab7f2255c2e5abf343cda8a7943d33a1d4863e52372d6f858a1`
- head `68661da94f03cab2f96182d47636cf55e088b5de`.

R26R frozen behavior:
1. independently verify R26Q digest/head/PASS/28 gates;
2. verify all sealed R26O files and 107 arrays/hashes;
3. build current Week-1 schedule and current Ourlads roster only for market scoping/player identity;
4. call the protected production live-odds boundary;
5. materialize exact sportsbook book/player/market/line offers with no fabricated consensus line;
6. filter to `player_receptions`;
7. deterministically match active RB/FB book offers to sealed players;
8. preserve exact book lines and over/under American prices;
9. compare sealed baseline and sealed candidate reception means/distributions to each exact book line;
10. calculate candidate `P(< line)`, `P(= line)`, `P(> line)` and no-vig market over probability when exact paired prices permit it;
11. forbid sportsbook feedback into the football candidate.

Frozen dispositions:
- `R26R_2026_WEEK1_PROSPECTIVE_OBSERVATION_SNAPSHOT_PASS_MARKET_CAPTURED`
- `R26R_2026_WEEK1_PROSPECTIVE_OBSERVATION_SNAPSHOT_NO_RECEPTIONS_MARKET_YET`
- `R26R_2026_WEEK1_PROSPECTIVE_OBSERVATION_SNAPSHOT_FAIL_NO_OBSERVATION`.

`NO_RECEPTIONS_MARKET_YET` is a legitimate prospective observation state, not a scientific failure. It permits a later market snapshot only against the exact same sealed R26Q parent.

### Live run at this handoff update

- run **`34401814588`**
- job **`102635265504`**
- head **`469aa40c90c738e12a82ee32ccca70c9cdbbc29f`**
- status at update: **in progress**
- completed successfully before live provider call:
  - checkout/setup/dependencies
  - frozen R26R contract verification
  - protected production market-utility exactness check
  - immutable R26Q parent verification
  - current 32-team Week-1 schedule + 32-team Ourlads roster identity build.
- current active step at update: **Capture hardened live sportsbook snapshot**.

Do **not** call this run a failure merely because the live provider step is slow. No R26R scientific/frozen gate failure has been emitted at this point.

---

## 12. Exact next actions

1. Check run `34401814588`, job `102635265504` until it reaches a legitimate final state.
2. If R26R emits `PASS_MARKET_CAPTURED`, record exact artifact ID/digest, 30-gate result, number of all-position reception book lines, RB/FB reception book lines, matched rows/players, and representative market gaps. Do not use those gaps to alter projections.
3. If R26R emits `NO_RECEPTIONS_MARKET_YET`, preserve it as a legitimate prospective snapshot and repeat later only against the exact same R26Q artifact/digest/head.
4. If execution fails mechanically before a frozen disposition, freeze a repair note **before** implementation and preserve all 30 gates / exact R26Q parent.
5. If a frozen gate truly fails, preserve `FAIL_NO_OBSERVATION`; do not tune around it.
6. Once Week-1 games are final and canonical outcomes are available, design the separately frozen postgame evaluation of production baseline vs immutable R26Q shadow on the predeclared population.
7. Preserve the same-week RB injury/role-transfer question for a future separately frozen study; do not modify the Week-1 sealed candidate.
8. Production remains unchanged unless a later separately frozen promotion study authorizes otherwise.

---

## 13. Broader lane reminders

### QB
M89/M90 mean authority remains protected. Continue football-only attempts/YPA/path integrity/shared-pass-state work when returning to QB.

### WR
M38 remains the entitlement anchor; WR-R11 additive NGS target correction failed. Future work should remain finite-pool allocation, not broad positive target boosts.

### TE
TE pool-first evidence is strong; TE-R5P remains production. Keep opportunity and efficiency layers separate.

### RB
- static depth/role-order remaps remain closed as direct opportunity authority;
- failed retrospective tail overlays remain failed;
- R22 receiving-yard authority remains protected;
- R26Q is now the immutable receptions-shadow pregame authority;
- R26R is only a prospective downstream market observation, not a model update.

---

## 14. One-sentence current state

**The unmodified R26 vacancy/R9 mechanism has been materialized into 2026 Week-1 RB receiving entitlements (R26N 28/28), safely converted into receptions-only Monte Carlo shadow distributions while leaving R22 and every non-reception array exact (canonical R26O 38/38), immutably sealed before outcomes (R26Q 28/28), and the active R26R study is now observing live Week-1 reception markets downstream without permitting sportsbook data, current same-week role information, or any other source to alter that sealed football candidate.**
