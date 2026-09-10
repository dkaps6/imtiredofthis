# CURRENT NFL RESEARCH HANDOFF — READ FIRST

**Repository:** `dkaps6/imtiredofthis`  
**Protected production-code authority:** `bb76ba9eabb08e2f0875a9af49301c3877f4141f`  
**Production stack unchanged:** QB M89/M90 + mean-neutral C2; WR M38 WR1 + WR-R15 WR2+; TE TE-R5P; RB rushing P3; RB receptions R26; RB receiving-yard tails R22; sportsbook downstream only.  
**RB receiving-yard mean lane:** CLOSED at a defensible scientific frontier; no new mean integration authorized.  
**Active lane:** current roster / late-week role handling operational correctness.  
**Active implementation branch:** `ops-current-player-availability-v1`  
**Do not wire/promote until the frozen operational fix plan and all validation gates are satisfied.**

Future sessions / scheduled tasks: read this file, then `AGENTS.md`, then verify the exact branches/runs/artifacts below. GitHub is canonical; conversation memory is secondary.

## Historical handoff preservation

Detailed predecessor snapshots remain immutable in Git history:
- immediately prior handoff commit `99d0ae6f6e0c4d60458a919096ce5cec1dfe695e`, blob `d9ddcfe40811f1f2dbeed7dc259c1ccbc2625a7b`
- pre-R27D detailed handoff commit `84c9ffa6ce3617757bcd6b41705d7e55fa8403e0`, blob `622f4857e75e9ef8939dc80755adbe837c017164`
- deep-history checkpoint commit `69e8de76bd1b508849d679fa22abd51aefa68a54`, blob `2f0ee91c1d5296c04114605afd5d4c4067a25a72`

Do not discard those histories.

---

# 1. Non-negotiable rules

1. Historical science is strict-prior and leakage-safe.
2. Freeze questions/mechanics/gates before results.
3. Preserve first valid scientific results exactly.
4. Preserve mechanical/integrity failures separately; repair only value-neutral defects.
5. Never lower gates, drop losing seasons, retune after seeing results, or post-hoc route a failed candidate.
6. Sportsbook stays downstream.
7. Research PASS is not direct production authority; separately freeze integration/promotion when required.
8. Protect R26 receiving opportunity/receptions and R22 RB receiving-tail authority unless separately authorized.
9. Operational current-roster fixes must not rewrite historical science.
10. Update this handoff at every material checkpoint.

---

# 2. Key protected production authorities

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

R26Q sealed 2026 Week1 pregame authority — never recompute:
- run `34400524030`
- job `102630996205`
- artifact `10123251043`
- digest `sha256:dd3ec0e8e3831ab7f2255c2e5abf343cda8a7943d33a1d4863e52372d6f858a1`
- disposition `R26Q_2026_WEEK1_RECEPTIONS_PROSPECTIVE_SEAL_PASS_READY_FOR_OBSERVATION`

R26S exact postgame evaluator:
- plan commit `63722c61ca1b408b8ae77eba383e5fa8daecf9f2`
- evaluator commit `e42099dbf11cdd510eb64e43a0910444c75ec7f8`
- lock commit `3a706fa52f91f6584f6fd6594563239dd6ea3b53`
- rerun unchanged only when authoritative Week1 outcomes are available.

---

# 3. RB receiving-yard mean conclusion — CLOSED

Tested without qualifying a new receiving-yard mean integration:
- R23 shrunk historical YPR
- R24 opportunity × existing production efficiency decomposition
- R27 exact R26 opportunity × unchanged production YPT
- R27B V2 raw strict-prior player/team/opponent target-shape and YAC context
- R27C/C2 physical forensics
- R27D0/D0B/D0C xYAC/YACOE source/physical decomposition
- R27D strict-prior xYAC/YACOE residual mean correction

Future RB receiving-mean work requires genuinely new pregame information/mechanism. Do not restart transformations of historical YPR/YPT/YAC, basic target shape, or xYAC/YACOE persistence from the same evaluation sample.

## R27D first valid result

Research branch `research-rb-r27d-yacoe-residual-v1`.

Frozen plan:
- commit `69edc12a16c691e3838eadcd75559b85dbba7865`
- blob `d0c2b0ff2de154e52fa21fb9ce19b739039633f3`

Preserved Run1 mechanical failure:
- run `34435661834`, job `102740069405`, head `1757d7549c0fc8c44ad207f55971fdd8c3eab754`
- duplicate deterministic Week1 merge collision
- no model fit/scientific result
- record `docs/research/RB_R27D_RUN1_WEEK1_COLUMN_COLLISION_MECHANICAL_REPAIR.md`

Preserved Run2 integrity/scoring failure:
- run `34435897671`, job `102740771822`, head `775d9bae7f264ae343209a75cd4947bc88994da9`
- artifact `10136169669`
- digest `sha256:d00eab566113e0c43502bf75f07fe477295dbef6a7ce20389e23ea941ec6d845`
- null outcomes produced NaN scientific metrics; emitted label rejected as invalid
- record `docs/research/RB_R27D_RUN2_OUTCOME_NULL_SCORING_MECHANICAL_REPAIR.md`

First valid scientific execution:
- head `641b25419c4f5ff3c234d1c000222fb4909ef940`
- run `34436178615`
- job `102741600329`
- artifact `10136250846`
- digest `sha256:b975511dc54e961c7745e9d1422ac48be4f867c65d3733adfaeabe93af9951e6`
- result record commit `a20e757a1b85806ae5e244cc42d29cfbc7ed2742`
- disposition `R27D_STRICT_PRIOR_YACOE_RESIDUAL_MIXED_OR_FAIL_NO_INTEGRATION`
- integrity 18/18 PASS; total 22/31; scientific only 4/13 PASS

Primary vacancy RB1 observed n=503:
- B0 MAE `14.305919`
- B1 `14.709395`
- C1 `14.710366` (`+0.006600%` worse vs B1)
- RMSE and p90 worsened
- 30+ miss rate slightly improved

2023 vacancy RB1 n=74:
- B0 `12.349471`
- B1 `13.992139`
- C1 `14.055947` (`+0.456024%` worse vs B1)

Only 3/6 seasons improved. No integration, no retuning, no post-hoc routing.

---

# 4. Current roster / late-week role audit — GAP CONFIRMED

Audit branch: `audit-current-roster-late-week-role-v1`.

Frozen audit plan:
- `docs/operations/CURRENT_ROSTER_LATE_WEEK_ROLE_AUDIT_V1_FROZEN_PLAN.md`
- commit `848e5e44b1078c9d9dd7ab40ad0ccf16136f3754`

Audit result:
- `docs/operations/CURRENT_ROSTER_LATE_WEEK_ROLE_AUDIT_V1_RESULT.md`
- result commit `810c344a437d411707185317033acc7004f1c7db`
- disposition `CURRENT_ROSTER_LATE_WEEK_ROLE_GAP_CONFIRMED_FIX_PLAN_REQUIRED`

Confirmed architecture gap:
1. Ourlads parser detects `active/inactive`, but canonical `roles_ourlads.csv` defaults to include inactive rows and strips every status/injury column before writing.
2. `roles_ourlads` artifact contract requires no availability/as-of/freshness provenance.
3. Full Slate weekly injury builder uses nflverse/NFL.com injury reports, not official game-day inactive lists.
4. Generic simulation injury rules can retain 50% opportunity for OUT/IR/PUP/inactive players; definitive unavailable is not zero eligibility.
5. Promoted RB P3 defines its active RB universe directly from all Ourlads RB/HB/FB rows before later injury context joins; no definitive-unavailable filter or depth re-ranking occurs first.
6. Current Full Slate has no canonical role+availability reconciliation before PlayerForm/promoted opportunity construction.
7. Prior QB M78 already established a hardened official NFL game-day inactive source contract and live `nfl.com/inactives/` acquisition semantics; this was never promoted into shared current-player availability.

---

# 5. Frozen operational fix architecture

File: `docs/operations/CURRENT_ROSTER_LATE_WEEK_ROLE_FIX_V1_FROZEN_PLAN.md`
- commit `b2206e7ad693148623447bcf9a3ad6b594033500`

Core frozen semantics:
- availability precedence: validated complete official NFL inactive section > definitive weekly injury designation (`OUT/IR/PUP`) > Ourlads provider inactive > uncertain/unknown
- `QUESTIONABLE/DOUBTFUL` are not automatically unavailable
- preserve raw source facts/provenance separately
- build `data/current_player_availability.csv` + JSON status sidecar
- definitive unavailable => zero eligibility/opportunity and no active reconciled role
- deterministic depth re-ranking for eligible players
- removed opportunity must be conserved/reallocated through qualified component logic or a separately frozen deterministic availability redistribution layer
- sportsbook may not define availability/roles
- timing-aware official inactive coverage must be certified per game window; endpoint reachability alone is not evidence
- historical backtests unchanged
- 20 promotion/validation gates plus frozen fixture scenarios

---

# 6. Active implementation branch — NOT PRODUCTION WIRED

Branch: `ops-current-player-availability-v1`
Base: frozen fix-plan commit `b2206e7ad693148623447bcf9a3ad6b594033500`.

Implemented so far:

### A. Timestamped Ourlads depth/status sidecar
`scripts/providers/ourlads_depth_status_v1.py`
- commit `86e024be635691555df4a53434916942870f5e55`
- reuses production Ourlads parser but preserves `status`, `source_url`, `source_asof_utc`
- requires 32-team completeness
- legacy `roles_ourlads.csv` still untouched

### B. Current-player availability resolver
`scripts/build/build_current_player_availability_v1.py`
- commit `2ccd425c79e78a401495a77021f933ea2e1a4603`
- combines depth status + weekly injury + optional official inactive sections
- implements frozen authority precedence
- produces `definitive_unavailable`, `eligible_for_opportunity`, authority/reason and reconciled roles
- deterministic RB/QB/TE/FB re-ranking; WR raw alignment role preserved pending component-specific conservation integration
- sportsbook inputs 0
- production_wired false

### C. Official NFL game-day inactive adapter
`scripts/providers/nfl_official_inactives_v1.py`
- commit `824ad677f4899ea74bddc755615bda61849d3397`
- adapts already-hardened M78 source/parser semantics
- emits explicit section ledger rows
- only complete parseable sections can certify absence/listed status
- endpoint reachability is separately recorded from payload validity

### D. Frozen fixture tests
`tests/test_current_player_availability_v1.py`
- commit `c676164a71a43ea298da06c30319c37fefc42e45`
- 8 scenarios: RB1 OUT→RB2 promotion; QUESTIONABLE remains; Ourlads inactive persists with empty injury report; complete official inactive authority; complete-section absence evidence; incomplete section cannot clear inactive; QB1 inactive→QB2 promotion; sportsbook 0

Fixture workflow `.github/workflows/test-current-player-availability-v1.yml`.

Preserved fixture Run1 mechanical workflow failure:
- run `34436894543`
- job `102743747393`
- head `75a02a4f14d9dcce69970130c9cf01138a0cbb90`
- all 8 fixtures PASSED, but final parent-diff assertion failed because checkout was shallow and pinned parent object unavailable
- record `docs/operations/CURRENT_PLAYER_AVAILABILITY_V1_RUN1_SHALLOW_CHECKOUT_MECHANICAL_REPAIR.md`
- minimum repair: checkout `fetch-depth: 0`

First clean fixture run:
- run `34436970099`
- job `102743973821`
- head `d3fadbd82953a3b2b1ad4168dbb82741ca62d167`
- conclusion SUCCESS
- all 8 frozen fixture tests PASS
- protected Full Slate/RB P3/simulation/context/artifact-contract production files unchanged from parent

### E. Live source smoke
Workflow: `.github/workflows/smoke-current-player-availability-sources-v1.yml`
- creation/head `36815ec75b41941131f8866667031292d4d8f01d`
- current run `34437032282`
- job `102744156238`
- status at this handoff checkpoint: IN PROGRESS, currently building timestamped Ourlads depth/status after dependencies installed
- source smoke is not a production run and does not require official inactive payload before publication time

## Exact next action

1. Inspect run `34437032282` to completion; preserve any source/plumbing failure separately.
2. If source smoke succeeds, record row/team/inactive counts, official endpoint/payload state and artifact ID/digest.
3. Before production wiring, add timing-aware game-window availability certification and fixture tests for required/not-yet-required official sections.
4. Add an implementation lock pinning the exact resolver/providers/tests and the frozen operational plan.
5. Only then wire availability reconciliation into a dedicated candidate Full Slate branch, ensuring definitive unavailable players are removed/zeroed before PlayerForm/promoted opportunity paths and role re-ranking feeds those paths.
6. Run no-odds Full Slate + fixture/invariant gates. Live pricing verification only when appropriate; sportsbook cannot resolve availability.
7. Promote only if all frozen operational gates pass. Otherwise preserve result and do not mutate production authority.

---

# 7. Remaining roadmap after roster/late-week role handling

1. Current roster / late-week role handling — ACTIVE.
2. Grade sealed R26Q with exact locked R26S once authoritative Week1 outcomes exist.
3. QB opportunity/efficiency: attempts, dropbacks, pass rate, YPA, sacks, scrambles; build on M89/M90.
4. Selective unresolved WR/TE opportunity/efficiency/distribution; preserve M38/WR-R15 and TE-R5P.
5. Shared QB↔receiver conservation across attempts, targets, completions, receiving yards and entitlements.
6. Unified game simulation: plays → pass/rush → player opportunity → outcomes → yards/explosives/TDs → game-state feedback → scoring.
7. Anytime TD modeling.
8. Game ML/spread/total from football simulation rather than sportsbook imitation.
9. Final operational package/prospective grading.

GitHub remains the source of truth.
