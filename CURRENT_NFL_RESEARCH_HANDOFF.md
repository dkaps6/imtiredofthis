# CURRENT NFL RESEARCH HANDOFF — READ FIRST

**Repository:** `dkaps6/imtiredofthis`  
**Purpose:** canonical cross-chat continuity ledger for the active NFL player-projection research program.  
**Last updated:** 2026-09-09 after corrected R26O 38/38 PASS.  
**Protected production-code / Full-Slate authority:** `f8417f55b04ce0e19baf260e9d532765034c47f1`.  
**Current research frontier:** R26O receptions-only shadow integration has passed; next authorized step is a separately frozen **pre-outcome prospective seal**.

> **Future ChatGPT sessions / agents:** read this file first, then `AGENTS.md`, then the exact active R26 result/plan files named below. Do not alter production or reinterpret prior failures. Update this file on `main` after each substantive frozen plan, result, repair, promotion, failure, or change in authorized next step.

---

## 1. User's controlling objective

Project individual NFL player outcomes as accurately as possible pregame: yards, receptions, carries, and downstream fair probabilities. Football projections must remain sportsbook-independent; sportsbook data is downstream pricing/benchmarking only.

Current architecture:

> **GAME / TEAM OPPORTUNITY → POSITION/ROOM POOL → INDIVIDUAL PLAYER ENTITLEMENT → PLAYER+MATCHUP EFFICIENCY → JOINT MONTE CARLO DISTRIBUTION**

The current RB receiving lane is specifically solving **who inside a changing backfield should own a finite RB receiving-opportunity pool**, rather than applying broad positive target/reception corrections.

---

## 2. Non-negotiable methodology

- Canonical production authority is `.github/workflows/full-slate.yml`.
- Historical science is strict walk-forward / leakage-safe.
- Freeze hypotheses, candidate, population, metrics, thresholds and gates before results.
- Mechanical failures may receive only the minimum documented plumbing repair and exact rerun; they are not scientific failures.
- Scientific failures remain preserved; do not lower gates or retune around them.
- No sportsbook football inputs upstream of projections.
- Current R26 prospective work uses **zero 2026 outcomes**.
- Production promotion always requires separately frozen authority; research PASS does not silently change `main`.

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

R26N/R26O have **not** altered RB receiving-yard means, R22 distributions, QB arrays, rushing arrays, or production parameters.

---

# 4. RB R26 historical foundation — preserve the failed science

R26 tested vacancy-gated R9 redistribution of a fixed RB/FB receiving-opportunity pool.

### R26 parent
- run `34356222339`
- artifact `10106271075`
- digest `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`
- disposition `RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW`
- 19/20-style broad support, but historical temporal safety failed because 2020 behaved differently.

### R26E Week-1 qualification
- run `34368268224`
- artifact `10110785184`
- digest `sha256:2b64fe25a1024136f2bb2cdde42bc74de290b6b9df63b4bb6cd675f6095b8bb7`
- disposition `WEEK1_COMPONENT_NOT_QUALIFIED_NO_SHADOW`
- 19/20 gates
- only blocker: 2020 Week-1 worsened about 8.97%; 2021-2025 improved.

**Do not erase or exclude 2020 retroactively.**

### R26J 2020 source comparability
- run `34374987828`
- artifact `10113466373`
- digest `sha256:7d886f228487e052ee1617cc5f42b974cd9da6a9d448f3a86b1de82cb4fc4f46`
- disposition `2020_SOURCE_REGIME_DISTINCT_FOR_MECHANISM_FOLLOWUP`

### R26K mechanism atlas
- run `34376961740`
- artifact `10114261724`
- digest `sha256:74a3d9ac58fca360f6d2d23e19b28e1254f38e49e3be23878ee1b37cebe22c88`
- disposition `2020_SPECIFIC_MECHANISM_NO_REPLICATED_ROUTER`

Conclusion: 2020 was distinct, but no historical router/guard replicated. Do not invent one.

---

# 5. R26L — 2026 source-regime transportability — PASS

Branch: `research-rb-r26l-2026-week1-regime-transportability-v1`

- run `34389455694`
- artifact `10119058769`
- digest `sha256:3351dfb5bbf6b571174a94ddf0a03179d70786edbb91d316e4fdf2c3cf005c46`
- disposition `2026_SOURCE_REGIME_MODERN_LIKE_FOR_PROSPECTIVE_QUALIFICATION`

Frozen evidence:
- 6/7 source-regime features closer to 2021-2025 than anomalous 2020
- normalized modern/2020 distance ratio about `0.594`
- zero features beyond 2020 in the anomalous direction
- 31 2026 Week-1 vacancy teams
- sole non-vacancy team: `CIN`
- zero outcomes / sportsbook inputs / same-week depth / R9 refit

---

# 6. R26M — prospective synthesis — PASS

Branch: `research-rb-r26m-2026-week1-prospective-qualification-synthesis-v1`

- run `34390505549`
- artifact `10119429741`
- digest `sha256:1306a3a2e58a0b129ac7e9fe34ad6407d87c491494e8fc27dd0284ba96996b76`
- disposition `2026_WEEK1_UNMODIFIED_R26_SHADOW_CANDIDATE_DESIGN_QUALIFIED`

R26M preserved the historical 2020 failure and authorized only prospective construction of the **unmodified** R26 Week-1 candidate.

---

# 7. R26N — exact 2026 structural candidate — PASS 28/28

Branch: `research-rb-r26n-2026-week1-unmodified-r26-structural-candidate-v1`

Read:
- `docs/research/RB_R26N_2026_WEEK1_UNMODIFIED_R26_STRUCTURAL_CANDIDATE_V1_FROZEN_PLAN.md`
- `docs/research/RB_R26N_2026_WEEK1_UNMODIFIED_R26_STRUCTURAL_CANDIDATE_V1_RESULT.md`
- both R26N mechanical repair notes

Canonical:
- frozen plan commit `919285fd0e2042461a5a89472f6832c01da857b4`
- frozen builder commit `5299ce54575ffcfe33ad203db0ee00285181291f`
- successful head `3b7a00e282cb925bd7a33175bc0b7d08d1467b2f`
- run `34396075045`
- job `102616001356`
- artifact `10121598376`
- digest `sha256:887929203053cb62904aaaeda9d995c9645163814da181972799f08fe4465c62`
- disposition `R26N_2026_WEEK1_STRUCTURAL_CANDIDATE_PASS_READY_FOR_SHADOW_INTEGRATION_DESIGN`
- **28/28 frozen gates passed**

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
- R22 and receiving-yard means/distributions untouched

R19 serialized R8/R9 authority:
- run `34288244770`
- artifact `10080377483`
- digest `sha256:11432b9d7b7f2367935a862b63c30df9f40955e479806ab67d90921b63a907c7`
- inner model SHA `9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba`

---

# 8. R26O — receptions-only MC shadow integration — FINAL PASS 38/38

**This is the current decisive scientific checkpoint.**

Branch: `research-rb-r26o-2026-week1-receptions-shadow-integration-compatibility-v1`

Read first:
- `docs/research/RB_R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_COMPATIBILITY_V1_FROZEN_PLAN.md`
- `docs/research/RB_R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_COMPATIBILITY_V1_RESULT.md`
- R26O mechanical repair notes
- `docs/research/RB_R26P_R26O_GATE15_CONTRACT_FORENSIC_V1_RESULT.md` on the R26P branch if auditing Gate 15

Frozen R26O:
- plan `96e4a6977458b55ec00cb29fbc9de05c56c89b88`
- evaluator `6a171cb32e77c7d6691ceb75da9861a5d6dd7bc4`
- 38 gates
- 25,000 Monte Carlo draws
- seed 42
- frozen MC compatibility tolerance `0.05 receptions`

## Preserved first scientific R26O execution — FAIL 37/38

- head `550d3d532e9f27c34b90ecd027f3811292674862`
- run `34398759284`
- job `102625073624`
- artifact `10122672501`
- digest `sha256:6d72de1eb8902fd8d956feb01b4d1a3cb5847aa58417ad4f416f3e1102e96e82`
- disposition `R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_FAIL_NO_SHADOW`
- 37/38 gates
- sole failed Gate 15 recorded `mean_parity=null`, `max_mean_delta=null`

This failure remains preserved and was **not** simply relabeled.

## R26P Gate-15 governance forensic — PASS 17/17

Branch: `research-rb-r26p-r26o-gate15-contract-forensic-v1`

- frozen plan `51eac07260752809bf9426e38a1b5c46abafe771`
- evaluator `8c787208a70615bd4c5f596a3457f7fa08f5447a`
- run `34399525657`
- job `102627643995`
- artifact `10122862934`
- digest `sha256:2e596f3a3bcbef156983664d332ded94ef41dae1402d1a96ab0b61d185c68d47`
- disposition `R26P_GATE15_EVIDENCE_WIRING_DEFECT_CONFIRMED_MECHANICAL_RERUN_AUTHORIZED`
- **17/17 forensic tests passed**

R26P proved the frozen R26O evaluator read R22's trace DataFrame where it intended to read R22's audit payload. Canonical protected R22 independently had:
- `gates.mean_parity = true`
- `max_mean_delta = 5.329070518200751e-15`
- `gates.receptions_exact = true`

R26P authorized only a mechanical evidence-wiring correction and exact rerun; it did not reinterpret the old FAIL.

## Final corrected canonical R26O — PASS 38/38

R26P-authorized correction lineage:
- Gate-15 repair note `c40de155c3f19785f8398cbcb9aaa2ab2445c2f4`
- audit-payload wrapper `85c3f2ca6bc73c302a3a33f3b6e75f4df42ae78a`
- final launch/head `e7014a6e365cbb776e48085dcef12dfece744ca4`
- canonical result doc commit `8cac09adcd1f413d351b011123fc221bee816b1d`

Final canonical execution:
- run `34399750746`
- job `102628405629`
- artifact `10123070453`
- artifact name `rb-r26o-2026-week1-receptions-shadow-integration-compatibility-v1`
- digest `sha256:27307ad84c232935aed3b25c0e0c9bff4da70cd2915376f41dfa3f85c25f14d0`
- disposition `R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_PASS_READY_FOR_PROSPECTIVE_SEAL`
- **38/38 frozen gates passed**

Final compatibility evidence:
- 104 intended RB/FB reception arrays changed
- forbidden changed arrays `0`
- CIN RB/FB receptions exact baseline
- non-RB/FB receptions exact baseline
- all non-reception arrays exact baseline
- all RB/FB rec-yard arrays exact R22 baseline
- all RB/FB rush+rec arrays exact R22 baseline
- all RB rushing arrays exact baseline
- all QB pass-yard arrays exact baseline
- full 2,892 simulation-key universe exact
- deterministic replay exact
- corrected Gate 15: `mean_parity=true`, `max_mean_delta=3.552713678800501e-15`
- max baseline MC reception-mean gap vs sealed R26N `0.017563893432533284`
- max candidate MC gap `0.018766081922851008`
- max candidate-minus-baseline delta gap `0.023037323394851983`
- all below frozen `0.05` tolerance
- 2026 outcomes `0`
- sportsbook football inputs `0`
- same-week depth `false`
- R9 refit `false`
- R22 changed by splice `false`
- receiving-yard means changed `false`
- production parameters changed `false`

Representative 25k research-shadow reception shifts:
- Omarion Hampton `2.12164 -> 3.23328`
- Jonathan Taylor `1.57412 -> 2.56996`
- Jaylen Warren `1.54380 -> 2.52768`
- Woody Marks `1.35000 -> 2.18748`
- D'Andre Swift `1.67208 -> 2.38604`
- Bijan Robinson `2.89060 -> 3.42608`
- Bam Knight `1.11328 -> 1.62536`
- Brian Robinson `0.68560 -> 0.19776`
- Jeremiyah Love `1.15676 -> 0.81148`

These are **research-shadow receptions only**. Production remains unchanged.

---

# 9. EXACT CURRENT AUTHORIZED NEXT STEP

R26O PASS authorizes only a separately frozen **pre-outcome prospective seal**.

The next study should be R26Q-style and must be frozen **before any 2026 Week-1 outcome is consumed**.

It should:

1. pin corrected R26O run `34399750746`, artifact `10123070453`, digest `sha256:27307ad84c232935aed3b25c0e0c9bff4da70cd2915376f41dfa3f85c25f14d0`;
2. copy/hash-seal the exact R26O disposition, 38-gate matrix, RB receptions manifest, and Monte Carlo draw arrays without recomputation;
3. freeze the future Week-1 scoring population and metrics now, before outcomes;
4. score production baseline vs sealed R26O only after target games are final and canonical outcomes become available;
5. use paired player-level receptions errors, with at minimum MAE, RMSE, bias, median/p75/p90 absolute error and role/tier stability diagnostics;
6. make the primary paired comparison on the 104 changed vacancy-active RB/FB rows, while preserving suitable secondary all-RB/room-level diagnostics;
7. leave production, R22, receiving-yard means/distributions, QB, rushing, and all other markets unchanged;
8. authorize no production promotion merely from sealing.

Do **not** recompute R26O during the seal. The point is to lock the exact pregame candidate before outcomes.

---

## 10. Broader lane reminders

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
- the active frontier is now **prospective validation of the sealed R26O receptions-only shadow**.

---

## 11. What a new chat must do first

1. Read this file and `AGENTS.md`.
2. Verify production head/artifact above.
3. Read canonical R26N and R26O result documents.
4. If auditing Gate 15, read the R26P frozen plan/result and preserve the earlier R26O 37/38 FAIL.
5. Verify corrected R26O run `34399750746`, artifact `10123070453`, digest `sha256:27307ad84c232935aed3b25c0e0c9bff4da70cd2915376f41dfa3f85c25f14d0`.
6. Check whether an R26Q prospective-seal branch already exists before creating one.
7. Continue only with the separately frozen pre-outcome seal / future scorecard.
8. Do not change production.

---

## 12. One-sentence current state

**The unmodified R26 vacancy/R9 mechanism has been materialized into 2026 Week-1 RB receiving entitlements (R26N 28/28), safely converted into receptions-only Monte Carlo shadow distributions on the exact current 468-player Full Slate while leaving R22 and every non-reception array exact (corrected R26O 38/38), and the only authorized next step is to seal those exact pregame arrays and freeze their prospective Week-1 production-vs-shadow scorecard before outcomes.**
