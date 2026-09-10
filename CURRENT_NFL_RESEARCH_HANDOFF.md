# CURRENT NFL RESEARCH HANDOFF — READ FIRST

**Repository:** `dkaps6/imtiredofthis`  
**Protected production-code authority:** `bb76ba9eabb08e2f0875a9af49301c3877f4141f`  
**Production stack unchanged:** QB M89/M90 + mean-neutral C2; WR M38 WR1 + WR-R15 WR2+; TE TE-R5P; RB rushing P3; RB receptions R26; RB receiving-yard tails R22; sportsbook downstream only.  
**RB receiving-yard mean lane:** CLOSED at a defensible scientific frontier; no new mean integration authorized.  
**Active lane:** current roster / late-week role handling operational correctness.  
**Active implementation branch:** `ops-current-player-availability-v1`  
**Do not wire/promote until the frozen operational fix plan, canonical timing plan, implementation lock and all validation gates are satisfied.**

Future sessions / scheduled tasks: read this file, then `AGENTS.md`, then verify the exact branches/runs/artifacts below. GitHub is canonical; conversation memory is secondary.

## Historical handoff preservation

Detailed predecessor snapshots remain immutable in Git history:
- immediately prior handoff blob `5ceec583744c25ced11c006640b7a13b75242820`
- predecessor blob `76a1b0d3af6315d033110db43920c89b6808a27c`
- earlier detailed handoff commit `99d0ae6f6e0c4d60458a919096ce5cec1dfe695e`, blob `d9ddcfe40811f1f2dbeed7dc259c1ccbc2625a7b`
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
11. Earlier frozen authority beats later conflicting implementation attempts; never silently replace a prior lock.

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

R27D strict-prior xYAC/YACOE residual correction was the final tested lane and did not qualify:
- branch `research-rb-r27d-yacoe-residual-v1`
- frozen plan commit `69edc12a16c691e3838eadcd75559b85dbba7865`, blob `d0c2b0ff2de154e52fa21fb9ce19b739039633f3`
- first valid run `34436178615`
- job `102741600329`
- artifact `10136250846`
- digest `sha256:b975511dc54e961c7745e9d1422ac48be4f867c65d3733adfaeabe93af9951e6`
- result record commit `a20e757a1b85806ae5e244cc42d29cfbc7ed2742`
- disposition `R27D_STRICT_PRIOR_YACOE_RESIDUAL_MIXED_OR_FAIL_NO_INTEGRATION`
- integrity 18/18 PASS; total 22/31; scientific only 4/13 PASS
- no integration, retuning or post-hoc routing

Future RB receiving-mean work requires genuinely new pregame information/mechanism. Do not restart historical YPR/YPT/YAC, basic target-shape, or xYAC/YACOE persistence transformations on the same sample.

---

# 4. Current roster / late-week role gap — CONFIRMED

Audit result commit `810c344a437d411707185317033acc7004f1c7db` confirmed:
1. Ourlads provider inactive status was discarded from canonical roles.
2. Weekly injury data is not official game-day inactive authority.
3. Generic simulation injury logic can retain positive opportunity for definitively unavailable players.
4. Promoted RB P3 builds its current RB universe before definitive-unavailable reconciliation.
5. Full Slate lacks a canonical role+availability authority before PlayerForm/promoted opportunity construction.
6. M78 already established hardened NFL official-inactive source semantics but they were never promoted to shared current-player operations.

Frozen operational fix plan:
- `docs/operations/CURRENT_ROSTER_LATE_WEEK_ROLE_FIX_V1_FROZEN_PLAN.md`
- commit `b2206e7ad693148623447bcf9a3ad6b594033500`

Core semantics:
- validated complete official NFL inactive section > definitive weekly injury (`OUT/IR/PUP`) > Ourlads provider inactive > uncertain/unknown
- QUESTIONABLE/DOUBTFUL remain eligible/uncertain
- definitive unavailable => zero eligibility/opportunity and no active reconciled role
- deterministic depth re-ranking after unavailable removal
- source provenance retained separately
- timing-aware official inactive coverage is game-window scoped and fail-closed only when required
- sportsbook cannot define availability/roles
- historical backtests remain unchanged

---

# 5. Implemented availability components on `ops-current-player-availability-v1`

### Timestamped Ourlads depth/status sidecar
`scripts/providers/ourlads_depth_status_v1.py`
- commit `86e024be635691555df4a53434916942870f5e55`
- preserves provider status, source URL and source timestamp
- requires 32 teams

### Current-player availability resolver
`scripts/build/build_current_player_availability_v1.py`
- commit `2ccd425c79e78a401495a77021f933ea2e1a4603`
- frozen precedence implemented
- definitive_unavailable / eligible_for_opportunity / authority / reason / reconciled roles
- deterministic RB/QB/TE/FB re-ranking; WR alignment preserved pending component-specific conservation integration
- sportsbook 0; production_wired false

### Official NFL game-day inactive adapter
`scripts/providers/nfl_official_inactives_v1.py`
- commit `824ad677f4899ea74bddc755615bda61849d3397`
- adapts hardened M78 parser/source
- complete parseable team section required for official evidence
- endpoint reachability separate from payload validity

### Base availability fixtures
`tests/test_current_player_availability_v1.py`
- 8 frozen semantic scenarios

Preserved Run1 workflow-only failure:
- run `34436894543`
- job `102743747393`
- all 8 semantic tests passed; only shallow-checkout protected-parent diff failed
- record `docs/operations/CURRENT_PLAYER_AVAILABILITY_V1_RUN1_SHALLOW_CHECKOUT_MECHANICAL_REPAIR.md`

First clean fixture run:
- run `34436970099`
- job `102743973821`
- head `d3fadbd82953a3b2b1ad4168dbb82741ca62d167`
- SUCCESS
- all 8 semantic fixtures PASS
- protected production files unchanged

---

# 6. Live source smoke — SUCCESS

Run `34437032282`, job `102744156238`, head `36815ec75b41941131f8866667031292d4d8f01d`.
Artifact `10136545256`, name `current-player-availability-source-smoke-v1`, digest `sha256:decb703afe4769befba790d8b1adceb0accb5a49eec4f5fd8f5d2adc6c7eb75a`.

Ourlads snapshot `2026-09-10T04:24:36Z`:
- complete true
- 32/32 teams
- 468 role/depth rows
- 0 team failures
- 0 provider-inactive rows at that exact scrape
- status/source_url/source_asof preserved

NFL official-inactives probe `2026-09-10T04:24:45Z`:
- HTTP 200
- endpoint_reachable true
- complete_team_sections 0
- listed_players 0
- payload_valid false
- sections []

This is a valid pre-publication state, not an error. It proves HTTP reachability is not being misused as inactive evidence.

---

# 7. Canonical timing authority — EARLIER FROZEN T-75 PLAN

Authoritative frozen plan already existed on the branch before a later T-90 continuation attempt:
- `docs/operations/CURRENT_PLAYER_AVAILABILITY_TIMING_V1_FROZEN_PLAN.md`
- validator `scripts/validate_current_player_availability_timing_v1.py`
- frozen constant `REQUIRE_MINUTES=75.0`

The T-75 threshold intentionally allows a fixed 15-minute publication/ingestion buffer after the league's T-90 inactive-list submission meeting. It was frozen before testing and must not be tuned from current source results.

Canonical game states:
- `NOT_YET_REQUIRED`: >75 minutes before kickoff; missing official sections do not block.
- `REQUIRED_AND_CERTIFIED`: <=75 minutes, pre-kickoff, both scheduled team sections complete with valid pre-kickoff timestamps.
- `REQUIRED_MISSING_FAIL_CLOSED`: <=75 minutes and one/both sections missing/incomplete or timestamp invalid/not pre-kickoff.
- `KICKED_OFF_LOCKED`: as-of at/after kickoff; no new pregame pricing.

Failure is scoped to the affected game/teams, not the entire slate. Endpoint reachability is never certification.

## Conflicting T-90 attempt — PRESERVED AND REMOVED FROM LIVE BRANCH

A later continuation created a duplicate T-90 certifier/lock before discovering the earlier frozen T-75 authority. The conflict was explicitly reconciled; the earlier frozen plan wins.

Preserved failed duplicate timing run:
- run `34437394807`
- job `102745226871`
- head `3219f584092946d9be81f8b30c2abd2b5f4b4019`
- protected-production boundary PASS
- duplicate fixture step FAIL
- record `docs/operations/CURRENT_PLAYER_AVAILABILITY_TIMING_V1_RUN1_FIXTURE_EXPECTATION_MECHANICAL_REPAIR.md`

The duplicate `scripts/build/certify_current_player_availability_timing_v1.py` and `CURRENT_PLAYER_AVAILABILITY_TIMING_CERTIFICATION_V1_LOCK.md` were deleted from the current branch after discovering the earlier frozen authority. History remains preserved. Do not restore those T-90 files.

## Canonical T-75 timing fixtures

`tests/test_current_player_availability_timing_v1.py` is now bound to `scripts/validate_current_player_availability_timing_v1.py` and tests the exact earlier frozen cases:
- T-120 no sections => NOT_YET_REQUIRED
- T-76 no sections => NOT_YET_REQUIRED
- exactly T-75 no sections => REQUIRED_MISSING_FAIL_CLOSED
- T-60 both complete => REQUIRED_AND_CERTIFIED
- T-60 only one complete => fail closed
- post-kickoff source snapshot => fail closed
- as-of at/after kickoff => KICKED_OFF_LOCKED
- staggered kickoff windows => early affected game fails closed while later game remains NOT_YET_REQUIRED

Canonical trigger commit:
- `6bbbcd9bb57fdfc344c777db8a615f6816d54b39`
- run `34437595637`
- job `102745806064`
- state at this handoff checkpoint: IN PROGRESS (dependency installation)

---

# 8. Exact next action

1. Inspect run `34437595637` through completion.
2. If it fails, preserve the exact mechanical/integrity failure and repair only value-neutral defects. Do not change T-75.
3. If it succeeds, record the first clean canonical timing-fixture result.
4. Add one complete implementation lock pinning the operational fix plan, T-75 plan, Ourlads-status provider, official-inactives provider, availability resolver, T-75 validator, both fixture suites and workflows.
5. Only then create a dedicated candidate Full Slate integration branch; do not wire directly on main.
6. Candidate Full Slate order must be: timestamped Ourlads status -> authoritative schedule/kickoffs -> weekly injuries -> official inactive acquisition -> T-75 game certification -> current-player availability reconciliation -> reconciled eligible roles -> PlayerForm/promoted QB/RB/WR/TE opportunity paths.
7. Explicitly prove unavailable players cannot receive positive RB P3/R26/QB/WR/TE mean/distribution and prove deterministic role promotion + team opportunity conservation.
8. Run no-odds Full Slate + all frozen operational/static-readiness gates. Live pricing only when appropriate; sportsbook cannot resolve availability.
9. Promote only if all frozen operational gates pass; otherwise preserve the result and leave protected production unchanged.

---

# 9. Remaining roadmap

1. Current roster / late-week role handling — ACTIVE.
2. Grade sealed R26Q with exact locked R26S once authoritative Week1 outcomes exist.
3. QB opportunity/efficiency: attempts/dropbacks/pass rate/YPA/sacks/scrambles; build on M89/M90.
4. Selective unresolved WR/TE opportunity/efficiency/distribution; preserve M38/WR-R15 and TE-R5P.
5. Shared QB↔receiver conservation.
6. Unified game simulation.
7. Anytime TD modeling.
8. Game ML/spread/total from football simulation rather than sportsbook imitation.
9. Final operational package/prospective grading.

GitHub remains the source of truth.