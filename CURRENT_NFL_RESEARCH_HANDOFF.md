# CURRENT NFL RESEARCH HANDOFF — READ FIRST

**Repository:** `dkaps6/imtiredofthis`  
**Purpose:** canonical cross-chat continuity ledger for the active NFL player-projection research program.  
**Last updated:** 2026-09-09 ~15:40 ET.  
**Protected production-code / Full-Slate authority used by the current RB study:** `f8417f55b04ce0e19baf260e9d532765034c47f1`.  
**Current research frontier branch:** `research-rb-r26n-2026-week1-unmodified-r26-structural-candidate-v1`.

> **Future ChatGPT sessions / agents: read this file first, then `AGENTS.md`, then the exact active branch plan/result documents named below before making any model or workflow change. Update this file on `main` after every substantive result, mechanical repair, frozen plan, promotion, failure, or change in authorized next step.**

---

## 1. User's north star

The user's controlling objective is simple:

> If a QB, RB, WR, or TE is projected for **X yards / X receptions / X carries**, make X as close as possible to the actual game outcome, player by player, pregame.

The ultimate market objective is to make the football-only player distributions and fair probabilities outperform sportsbook player-prop pricing. Sportsbook data is **downstream only**. Large model-vs-market discrepancies are audit triggers, never automatic corrections.

For the current RB receiving/receptions lane, the emphasis is real football opportunity and allocation accuracy—not merely directional market agreement.

---

## 2. Non-negotiable methodology

- Canonical production authority is `.github/workflows/full-slate.yml`; read `AGENTS.md` before touching production.
- Historical tests are strict walk-forward / leakage-safe.
- Sportsbook inputs are prohibited upstream of football projections.
- Freeze hypothesis, candidate, cohort, thresholds, gates, and outcome definitions **before** results.
- Mechanical/data/integrity failures may receive only the minimum plumbing repair and exact rerun; they are not scientific failures.
- Mechanical repair intent should be frozen/documented before implementation when practical.
- Scientific failures are preserved. Do not lower gates, tune nearby thresholds, window-hunt, or rescue a near miss.
- Failed experiments remain evidence and must not be silently reopened.
- Full-stack/production promotion always requires a separately frozen integration confirmation.
- Current RB R26-series authority is explicitly prospective and does **not** use 2026 game outcomes.

---

## 3. Current project architecture

The active architecture remains:

> **GAME / TEAM OPPORTUNITY → POSITION/ROOM POOL → INDIVIDUAL PLAYER ENTITLEMENT → PLAYER+MATCHUP EFFICIENCY → JOINT MONTE CARLO DISTRIBUTION**

Do not treat a player's final yardage as an independent primary object when a finite team/room opportunity pool can be modeled and conserved first.

Strong repeated evidence supporting this architecture includes:
- QB/receiver shared pass-volume residual coupling;
- WR target/opportunity error dominance and the durable M38 hierarchy;
- TE team-pool-first evidence;
- RB carry/yard coupling and failures of static depth remaps;
- C2 pass/receiving conservation improving QB distribution quality without moving the promoted QB mean.

---

## 4. Current production authorities that must remain protected

### Production code / current Full Slate state

Protected production-code reference used throughout the current R26N research:
`f8417f55b04ce0e19baf260e9d532765034c47f1`

Current Full Slate artifact used as the exact 2026 Week-1 population authority:
- run `34317211395`
- artifact `10090547415`
- artifact name `run_34317211395`
- digest `sha256:7eab77e41c5879d4f54d87497eee0d1186010784cfc4ef966178930e16fb7c3b`
- head `f8417f55b04ce0e19baf260e9d532765034c47f1`
- current player rows `468`
- teams `32`
- games `16`

### QB

`QB_PASS_SYNTHESIS_V1 / M89-M90` remains the passing-yard mean authority. Do not replace it casually.

### WR

M38 target-share hierarchy remains the canonical WR entitlement prior/winner.

### TE

TE-R5P is the promoted TE entitlement specialist in the current stack.

### WR specialist

WR-R15 is the promoted conserved WR specialist in the current stack.

### RB rushing

`RB_P3_SYNTHESIS_V1`, 2026 Week-1 route `WEEK1_STACK_OVERRIDE`, remains the qualified RB rushing-yard mean authority.

### RB receiving-yard distribution

R22 / R19 receiving-tail production authority remains protected. The current R26N study did **not** alter, regenerate, or refit receiving-yard means/distributions.

---

# 5. CURRENT RB RECEIVING / RECEPTIONS FRONTIER — R26 SERIES

This is the highest-priority current handoff state.

## R26 historical mechanism

R26 tested a vacancy-gated R9 redistribution mechanism inside the RB/FB receiving opportunity room.

Historical R26 parent:
- run `34356222339`
- artifact `10106271075`
- digest `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`
- disposition `RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW`

The mechanism broadly helped but failed a frozen temporal safety gate because 2020 behaved differently.

## R26E Week-1 qualification

Latest valid R26E authority:
- run `34368268224`
- artifact `10110785184`
- digest `sha256:2b64fe25a1024136f2bb2cdde42bc74de290b6b9df63b4bb6cd675f6095b8bb7`
- disposition `WEEK1_COMPONENT_NOT_QUALIFIED_NO_SHADOW`
- gate pattern `19/20`
- only failed gate: `14_no_w1_season_worsens_more_than_5pct`
- 2020 Week-1 relative MAE change: `+0.08966783207056839`
- 2021–2025 each improved

Do **not** erase the 2020 failure.

## R26J — 2020 source comparability

- run `34374987828`
- artifact `10113466373`
- digest `sha256:7d886f228487e052ee1617cc5f42b974cd9da6a9d448f3a86b1de82cb4fc4f46`
- disposition `2020_SOURCE_REGIME_DISTINCT_FOR_MECHANISM_FOLLOWUP`

2020 was source/regime-distinct across multiple dimensions, justifying follow-up—but not automatic exclusion.

## R26K — mechanism atlas

- run `34376961740`
- artifact `10114261724`
- digest `sha256:74a3d9ac58fca360f6d2d23e19b28e1254f38e49e3be23878ee1b37cebe22c88`
- disposition `2020_SPECIFIC_MECHANISM_NO_REPLICATED_ROUTER`

No replicated historical state was found that authorized a new 2020-style router/guard. Therefore:
- do not exclude 2020;
- do not invent a historical regime router;
- preserve the original R26 failed gate.

---

# 6. R26L — 2026 WEEK-1 REGIME TRANSPORTABILITY — COMPLETE

Branch:
`research-rb-r26l-2026-week1-regime-transportability-v1`

Canonical successful run:
- run `34389455694`
- artifact `10119058769`
- digest `sha256:3351dfb5bbf6b571174a94ddf0a03179d70786edbb91d316e4fdf2c3cf005c46`
- disposition `2026_SOURCE_REGIME_MODERN_LIKE_FOR_PROSPECTIVE_QUALIFICATION`

Key evidence:
- `6 of 7` frozen features closer to 2021–2025 than to 2020;
- zero features beyond 2020 in the anomalous direction;
- mean normalized distance to modern `0.7255585711888131`;
- mean normalized distance to 2020 `1.2220245842254218`;
- modern/2020 distance ratio approximately `0.594`;
- 2026 Week-1 vacancy teams `31`;
- exact non-vacancy team `CIN`.

Important interpretation:
“Modern-like” does not mean every 2026 metric lies inside every modern historical range. It means the frozen distance rule places the current state materially closer to the modern regime than anomalous 2020.

R26L used:
- 2026 outcomes `0`;
- sportsbook football inputs `0`;
- same-week depth `false`;
- no R9 refit;
- no production/R22/receiving-mean change.

---

# 7. R26M — PROSPECTIVE QUALIFICATION SYNTHESIS — COMPLETE

Branch:
`research-rb-r26m-2026-week1-prospective-qualification-synthesis-v1`

Frozen/implementation lineage:
- frozen-plan commit `4201985b23ba96b58e81a1b78a37bd91edc66cb4`
- evaluator commit `fbdad82591e038d89235f70d6b9f4e0c638c3504`
- implementation-lock commit `4d9a1f2c52431f580cb8e30486339d1ff7f73436`
- launch commit `d4756157331b0bd2281f5e6580816344bd84e645`

Canonical run:
- run `34390505549`
- artifact `10119429741`
- digest `sha256:1306a3a2e58a0b129ac7e9fe34ad6407d87c491494e8fc27dd0284ba96996b76`
- disposition `2026_WEEK1_UNMODIFIED_R26_SHADOW_CANDIDATE_DESIGN_QUALIFIED`
- canonical result commit `9ae4e6639865a4ca25f48f33a1d49e1318eefb78`

R26M preserved all parent failures/limits and authorized only design of an unmodified 2026 Week-1 R26 structural candidate.

It did **not** authorize:
- excluding 2020;
- a new historical router;
- a live shadow;
- production promotion;
- receiving-yard/R22 changes.

---

# 8. R26N — 2026 WEEK-1 UNMODIFIED-R26 STRUCTURAL CANDIDATE — COMPLETE / PASS

This is the newest substantive checkpoint.

Branch:
`research-rb-r26n-2026-week1-unmodified-r26-structural-candidate-v1`

Frozen plan:
`docs/research/RB_R26N_2026_WEEK1_UNMODIFIED_R26_STRUCTURAL_CANDIDATE_V1_FROZEN_PLAN.md`

Canonical result:
`docs/research/RB_R26N_2026_WEEK1_UNMODIFIED_R26_STRUCTURAL_CANDIDATE_V1_RESULT.md`

Original frozen implementation:
`scripts/backtest/build_rb_r26n_2026_week1_unmodified_r26_structural_candidate_v1.py`

### Frozen lineage

- plan commit `919285fd0e2042461a5a89472f6832c01da857b4`
- original builder commit `5299ce54575ffcfe33ad203db0ee00285181291f`
- implementation-lock commit `be1f2bc5219c378aeb081ceaf00bcb54b02a91a2`
- final successful launch head `3b7a00e282cb925bd7a33175bc0b7d08d1467b2f`
- canonical result commit `cefc91b5c7cee31d51208821c71835f4ea57af06`

### Canonical successful run / artifact

- run `34396075045`
- job `102616001356`
- artifact `10121598376`
- artifact name `rb-r26n-2026-week1-unmodified-r26-structural-candidate-v1`
- digest `sha256:887929203053cb62904aaaeda9d995c9645163814da181972799f08fe4465c62`

### Disposition

`R26N_2026_WEEK1_STRUCTURAL_CANDIDATE_PASS_READY_FOR_SHADOW_INTEGRATION_DESIGN`

**All 28 frozen structural gates passed.**

### Exact 2026 structural candidate state

- production population `468`
- teams `32`
- games `16`
- RB/FB rows `107`
- vacancy teams `31`
- non-vacancy teams `[CIN]`
- changed RB/FB rows `104`
- R9 training season `2025`
- R9 reliability `1.0`
- R9 refit `false`
- serialized R19/R9 inner SHA-256 `9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba`

### Structural conservation

- maximum RB/FB room-pool gap `0.0`
- maximum team-entitlement delta `1.1102230246251565e-16`
- maximum non-RB/FB entitlement delta `0.0`
- maximum non-vacancy RB/FB entitlement delta `0.0`
- CIN exact baseline
- player universe exact `468 -> 468`
- strict-prior max identity time key `202518`
- candidate entitlement finite/nonnegative

### Leakage / authority ceiling

- `2026_outcomes_used = 0`
- `sportsbook_football_inputs_used = 0`
- `same_week_depth_used = false`
- `r9_refit = false`
- `production_parameters_changed = false`
- `r22_changed = false`
- `receiving_yard_means_changed = false`
- `receiving_distribution_regenerated = false`
- `live_shadow_activation_authorized = false`
- `production_promotion_authorized = false`
- `shadow_integration_design_authorized = true`

### Successful-run hashes

- frozen plan SHA-256 `5033bbd7a8de619333f1a6c290d0eb574b830e3cff76dbb72fb6f062bebf4ff4`
- implementation lock SHA-256 `afa16ef8695088e83e0698a5656092a676d8c9de330220b7040e18f238a8f619`
- original frozen builder SHA-256 `ddec40e2373611971572d7f11fd5e966660e4412256736c0a84fbab68c313e6a`

---

## 9. R26N mechanical repair lineage — preserve this exactly

These were execution/plumbing failures, **not scientific failures**. The original R26N plan and original candidate builder remained byte-identical through both repairs.

### Repair 1 — missing compact identity key in model-context

First authoritative launch:
- run `34395291505`
- job `102613379938`
- head `3834b6765f60efc80566b7a28ecd081d6fe8fd00`

Failure occurred after all governance/parent checks and before any scientific disposition because:
- PlayerForm had compact `player_clean_key`;
- model-context lacked `player_clean_key` and fell back to display names;
- immutable sources nevertheless had the exact same 468 `(team, display player)` identities with zero duplicates.

Frozen repair note:
`docs/research/RB_R26N_RUN1_MECHANICAL_REPAIR_V1.md`
- commit `8268327683e9d826c05d2d72a4467e353f4fa9f1`
- successful-run note SHA `9508cd0dee8cea4f60e35eb284764829f3dcd7a267f328007b5186c37465019f`

Hash-tracked staging helper:
`scripts/backtest/stage_r26n_production_identity_key_repair_v1.py`
- commit `0b4c2df7c14b16bd0e945b425aac7f35e015fa3d`
- helper SHA `6fb5407de5ced156966437702634d51c95a9ec02787be308fd208dac0ea54079`

Successful audit proved:
- 468 source / 468 staged rows;
- exact display identity match;
- zero fuzzy matching;
- zero normalization heuristics;
- zero football-value changes;
- zero players added/removed;
- immutable parent untouched.

### Repair 2 — pandas identity join dtype mismatch

After repair 1 passed, run `34395790276` reached strict-prior history and failed before R9 scoring because current keys were pandas `string[python]` while historical keys were `object`.

Frozen second repair note:
`docs/research/RB_R26N_SECOND_MECHANICAL_REPAIR_V1.md`
- commit `dbfe863212dc5b2684ec1005d66102342bf87777`
- note SHA `a5fd7708c73c39324a1c46e0ec85aa24931d75b5971e59e9f3f8883191fe8f96`

Dtype-only wrapper:
`scripts/backtest/run_rb_r26n_with_identity_dtype_compat_v1.py`
- commit `9abdaa9a6262aa87ac601d7bc665209b94036b20`
- wrapper SHA `537807be0eddbf2ec7a21006792c4e992b79a24c08ad58613db8dfc524a7ecda`

The wrapper casts only identity join-key dtypes to plain object, verifies key values and all non-key values are unchanged, then delegates to the original protected identity function and original frozen builder.

---

# 10. CURRENT AUTHORIZED NEXT STEP

R26N **does not authorize a live shadow**.

R26N authorizes only a separately frozen downstream **shadow-integration design / compatibility study**.

The next legitimate study must answer:

> Can the R26N opportunity/reception overlay be integrated into the exact current 2026 Week-1 Full Slate/R22 stack while preserving every protected receiving-yard distribution/mean and all unrelated markets/production authorities unless a later, separately frozen scientific study explicitly authorizes changing them?

The next study should be design/integration validation first, not a production promotion.

Required protections for the next study should include at minimum:
- exact R26N artifact/digest parent;
- exact current production Full Slate artifact/digest parent;
- exact R22/R19 authority parents and model hashes;
- no 2026 outcomes;
- no sportsbook football inputs;
- no R9 refit;
- no same-week depth;
- exact non-RB/FB invariance;
- exact RB non-reception/non-target market invariance unless explicitly in scope;
- exact receiving-yard mean/distribution invariance if R26N is integrated only as an opportunity/reception shadow layer;
- exact production-code boundary clean;
- explicit authority ceiling: pass may authorize shadow activation design/confirmation only, not production promotion unless separately frozen.

Before creating it, check whether an R26O branch/plan already exists. Do not duplicate an existing study.

---

## 11. Broader position-lane reminders

### QB
- M89/M90 mean authority stays protected.
- Continue football-only attempts/YPA/path integrity and shared pass-state work when returning to QB.
- Do not shrink QBs merely because the synthesis correction is large; QB-PD3 found no stable internal-disagreement distrust state.

### WR
- M38 is the baseline entitlement prior.
- WR-R11 NGS additive target model failed; do not retry nearby tuning.
- Future WR work should allocate a finite WR/team opportunity pool rather than apply another broad positive target correction.

### TE
- TE target-pool-first evidence is strong.
- TE-R3 generic context model failed the full gates despite partial pool signal.
- Strict-prior participation source qualified.
- TE work should remain finite-pool + individual entitlement + separate efficiency.

### RB
- Static role-order/depth remaps are closed as direct opportunity authority.
- Retrospective tail-overlay families that failed remain closed.
- Current productive frontier is the R26 vacancy/R9 receiving-opportunity mechanism, now structurally materialized for 2026 Week 1 but not yet integrated as a live shadow.

---

## 12. What a new chat must do first

1. Read this file.
2. Read `AGENTS.md`.
3. Read:
   - `docs/research/RB_R26N_2026_WEEK1_UNMODIFIED_R26_STRUCTURAL_CANDIDATE_V1_FROZEN_PLAN.md`
   - `docs/research/RB_R26N_2026_WEEK1_UNMODIFIED_R26_STRUCTURAL_CANDIDATE_V1_IMPLEMENTATION_LOCK.md`
   - `docs/research/RB_R26N_2026_WEEK1_UNMODIFIED_R26_STRUCTURAL_CANDIDATE_V1_RESULT.md`
   - both R26N mechanical repair notes.
4. Verify canonical R26N run `34396075045` and artifact `10121598376` digest `sha256:887929203053cb62904aaaeda9d995c9645163814da181972799f08fe4465c62`.
5. Inspect repository branches/files for an existing R26O study before creating one.
6. Continue only with the authorized separately frozen shadow-integration design study.
7. Keep production authority protected; R26N itself gives no live-shadow or production-promotion permission.

---

## 13. One-sentence current state

**The unmodified R26 vacancy-gated R9 RB receiving-opportunity mechanism has passed all 28 frozen structural gates on the exact 468-player 2026 Week-1 production population, with 31 vacancy teams and CIN baseline-exact, while leaving R22/receiving-yard authority untouched; the only authorized next step is a separately frozen shadow-integration compatibility study.**
