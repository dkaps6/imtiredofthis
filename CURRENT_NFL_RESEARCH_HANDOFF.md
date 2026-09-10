# CURRENT NFL RESEARCH HANDOFF — READ FIRST

**Repository:** `dkaps6/imtiredofthis`  
**Protected production-code authority:** `bb76ba9eabb08e2f0875a9af49301c3877f4141f`  
**Production stack unchanged:** QB M89/M90 + mean-neutral C2; WR M38 WR1 + WR-R15 WR2+; TE TE-R5P; RB rushing P3; RB receptions R26; RB receiving-yard tails R22; sportsbook downstream only.  
**RB receiving-yard mean lane:** CLOSED at a defensible scientific frontier; no new mean integration authorized.  
**Active lane:** current roster / late-week availability + role reconciliation.  
**Canonical validated core branch:** `ops-current-player-availability-t75-v1`  
**Candidate Full Slate branch:** `ops-current-player-availability-full-slate-v1`  
**Production is NOT yet wired or promoted.**

Future sessions / scheduled tasks: read this file, then `AGENTS.md`, then verify the exact branches/runs/commits below. GitHub is canonical; chat memory is secondary.

## Historical handoff preservation

The immediately prior detailed snapshot is preserved at:
- main handoff commit `261256ae4105cc3220668777a7c77f377252b961`
- handoff blob `63364a0d3e5ea2931a305094a5acd8a843a76fd8`

Earlier snapshots remain in Git history, including `99d0ae6f6e0c4d60458a919096ce5cec1dfe695e`, `84c9ffa6ce3617757bcd6b41705d7e55fa8403e0`, and deep-history checkpoint `69e8de76bd1b508849d679fa22abd51aefa68a54`. Do not discard them.

---

# 1. Non-negotiable rules

- freeze questions/mechanics/gates before results;
- preserve first valid results exactly;
- preserve mechanical/integrity failures separately and repair only value-neutral defects;
- no lowering gates, dropping losing seasons, post-hoc routing or retuning after results;
- sportsbook remains downstream and cannot define football eligibility/role;
- protect R26 receptions/opportunity and R22 RB receiving-tail authority unless separately authorized;
- operational availability work must not alter historical research;
- earlier frozen authority wins over later unacknowledged concurrent conflicts;
- update this handoff at every material checkpoint.

---

# 2. RB receiving-yard mean lane — CLOSED

R27D was the final tested mean-correction family and did not qualify.

First valid R27D result:
- branch `research-rb-r27d-yacoe-residual-v1`
- frozen plan `69edc12a16c691e3838eadcd75559b85dbba7865`, blob `d0c2b0ff2de154e52fa21fb9ce19b739039633f3`
- run `34436178615`
- job `102741600329`
- artifact `10136250846`
- digest `sha256:b975511dc54e961c7745e9d1422ac48be4f867c65d3733adfaeabe93af9951e6`
- result record commit `a20e757a1b85806ae5e244cc42d29cfbc7ed2742`
- disposition `R27D_STRICT_PRIOR_YACOE_RESIDUAL_MIXED_OR_FAIL_NO_INTEGRATION`
- integrity 18/18 PASS; total 22/31; scientific 4/13 PASS
- vacancy RB1 B1 MAE `14.709395` -> C1 `14.710366` (worse)
- 2023 RB1 B1 `13.992139` -> C1 `14.055947` (worse)
- only 3/6 seasons improved

No integration, retuning or post-result routing. Reopen RB receiving mean only for genuinely new pregame information/mechanism.

---

# 3. Current roster / late-week availability gap — CONFIRMED

Audit branch `audit-current-roster-late-week-role-v1`.

Frozen audit plan commit `848e5e44b1078c9d9dd7ab40ad0ccf16136f3754`.
Audit result commit `810c344a437d411707185317033acc7004f1c7db`, disposition `CURRENT_ROSTER_LATE_WEEK_ROLE_GAP_CONFIRMED_FIX_PLAN_REQUIRED`.

Confirmed:
- Ourlads parser detected inactive state but production roles stripped status and included inactive by default;
- weekly injury reports are not official game-day inactive authority;
- generic injury rules could retain positive opportunity for definitive unavailable players;
- RB P3 built its current RB universe before definitive-unavailable reconciliation;
- no canonical shared availability+role authority existed before PlayerForm/promoted opportunity;
- prior QB M78 already supplied hardened official NFL inactive source semantics.

Frozen operational fix plan:
- `docs/operations/CURRENT_ROSTER_LATE_WEEK_ROLE_FIX_V1_FROZEN_PLAN.md`
- commit `b2206e7ad693148623447bcf9a3ad6b594033500`

---

# 4. Validated current-player availability core

Core components:
- Ourlads status/provenance sidecar: `scripts/providers/ourlads_depth_status_v1.py`, blob `c115816ea8aa4ba7150a635c3115546d43f94b3c`
- availability resolver: `scripts/build/build_current_player_availability_v1.py`, blob `9a1b0a672db7854ff52764ad93e5fcd89f9cb0ea`
- official NFL inactive adapter: `scripts/providers/nfl_official_inactives_v1.py`, blob `0d67316b6d7b7b9aa9d3637a07da4cd2b171639e`
- canonical T-75 timing validator: `scripts/validate_current_player_availability_timing_v1.py`, blob `d67ae30ed6e837f62098671c499d05462fe9d837`
- semantic fixtures: `tests/test_current_player_availability_v1.py`, blob `c51ac79173b651395fb733d92ac8138e7422e822`
- timing fixtures: `tests/test_current_player_availability_timing_v1.py`, blob `249fe23bff606fa977a245097eeb5875908337c1`

Core implementation lock:
- `docs/operations/CURRENT_PLAYER_AVAILABILITY_V1_IMPLEMENTATION_LOCK.md`
- commit `596f084750a8c8e3c36ca09d737dc2edb808dc09`

Semantic fixture evidence:
- preserved workflow-only failure `34436894543`, job `102743747393`: all 8 tests passed; only shallow checkout parent-diff failed
- first clean semantic run `34436970099`, job `102743973821`, head `d3fadbd82953a3b2b1ad4168dbb82741ca62d167`: SUCCESS, 8/8 semantic fixtures PASS

Live source smoke:
- run `34437032282`
- job `102744156238`
- artifact `10136545256`
- digest `sha256:decb703afe4769befba790d8b1adceb0accb5a49eec4f5fd8f5d2adc6c7eb75a`
- Ourlads: 32/32 teams, 468 rows, timestamp/source provenance preserved
- NFL `/inactives/`: HTTP 200 but zero complete sections at pre-publication snapshot; correctly NOT treated as availability evidence

Locked availability semantics:
- complete validated official inactive section is strongest source;
- definitive weekly OUT/IR/PUP next;
- Ourlads inactive next;
- QUESTIONABLE/DOUBTFUL remain uncertain/eligible until definitive evidence;
- definitive unavailable => zero eligibility and no active reconciled role;
- sportsbook inputs to eligibility/role = 0.

---

# 5. Canonical official-inactive timing rule — T-75

Frozen timing plan:
- `docs/operations/CURRENT_PLAYER_AVAILABILITY_TIMING_V1_FROZEN_PLAN.md`
- commit `905f1bbe55d51676587d941295d357a1d2c31e9b`
- blob `e72bf0c5e32aa7b06c559a5836fd883028e9cc1a`

NFL inactive lists are delivered at the league's T-90 game-day administration point; V1 freezes a 15-minute publication/ingestion allowance and requires certification at T-75.

States:
- >75 min: `NOT_YET_REQUIRED`
- <=75 min pre-kickoff + both complete sections/pre-kickoff timestamps: `REQUIRED_AND_CERTIFIED`
- <=75 min + missing/incomplete/invalid source: `REQUIRED_MISSING_FAIL_CLOSED`
- at/after kickoff: `KICKED_OFF_LOCKED`

Concurrent T-90 work was detected before a valid timing result and preserved as superseded draft history. Conflict resolution commit `910e707159bfd98f6d62f0c493d0e0fb30ab1881` establishes first-frozen T-75 as canonical.

Isolated canonical timing branch: `ops-current-player-availability-t75-v1`.

Preserved isolated Run1 dependency failure:
- run `34437550771`
- job `102745673287`
- head `86c68b98b4ed6974b5d25c06e6ef99d6ed9aff17`
- frozen T-75/production-boundary checks PASS
- fixture command never ran because pytest was missing
- record `docs/operations/CURRENT_PLAYER_AVAILABILITY_TIMING_V1_RUN1_PYTEST_DEPENDENCY_MECHANICAL_REPAIR.md`

First valid timing result:
- run `34437715931`
- job `102746163583`
- head `092fee088f3402d6313bc02b2f8cc05d1f3f54f9`
- conclusion SUCCESS
- frozen T-75 contract PASS
- protected-production boundary PASS
- all 8 timing fixtures PASS

Do not change T-75 after this result.

---

# 6. Full Slate integration is now separately frozen

Frozen integration plan:
- `docs/operations/CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_V1_FROZEN_PLAN.md`
- commit `91ee6aa3ad3813c7d285f6f3163368205937eb09`
- 35 predeclared validation gates

Core principle: **availability is resolved before opportunity**. Definitively unavailable players are removed/re-ranked before PlayerForm/current opportunity. Existing qualified component allocators remain the football logic; no new generic injury percentage redistribution.

Frozen component seams:
- QB: unavailable QB cannot start; highest eligible depth QB becomes QB1; M89/M90/C2 parameters unchanged.
- RB: unavailable RB/FB removed before P3/R26 current universe; eligible backs re-ranked; P3/R26/R22 parameters unchanged.
- WR/TE: remove unavailable players before explicit target-entitlement universe; M38 establishes team entitlement, TE-R5P conserves within TE room, WR-R15 conserves within WR2+ room/eligible WR1 anchor; no legacy 50% retention for definitive unavailable.
- sportsbook matching cannot resurrect a player/game excluded by football availability.

Candidate integration branch:
- `ops-current-player-availability-full-slate-v1`
- based on frozen integration-plan commit `91ee6aa3ad3813c7d285f6f3163368205937eb09`

First candidate implementation file:
- `scripts/build/build_reconciled_active_roles_v1.py`
- commit `85001ca4a7e6d7831c2b2efa9bf0bf5c3bc0659e`
- builds `roles_ourlads_active_v1.csv` from availability state
- excludes definitive unavailable from active artifact while retaining them in audit availability artifact
- preserves WR alignment roles and uses locked ordinal role re-ranking for QB/RB/FB/TE
- validates unique active identity and gap-free ordinal role ranks
- sportsbook inputs 0

Important integration discovery:
- current `.github/workflows/full-slate.yml` builds raw Ourlads roles first, live-odds gate before injuries, then weather/injuries, then PlayerForm, then promoted RB P3.
- availability can therefore be reconciled after injury build and before PlayerForm/promoted opportunity without allowing sportsbook data to define eligibility.
- the only existing `ROLES_CSV` environment override found is in sportsbook fetching; core football builders do not yet expose a universal active-role override.
- DO NOT overwrite raw Ourlads source silently. Add one explicit shared current-role input seam for football builders and prove parity when no unavailable players exist.

---

# 7. Exact next action

1. On `ops-current-player-availability-full-slate-v1`, identify the minimal shared role-loader/input seam used by PlayerForm, full-roster universe construction and RB P3; add an explicit `ACTIVE_ROLES_CSV`/current-role resolver rather than overwriting raw Ourlads.
2. Freeze/lock the exact candidate integration implementation before its first Full Slate result.
3. First prove **parity with protected production when availability removes nobody**: same player universe/roles and unchanged promoted means/distributions to frozen tolerances.
4. Then run fixture-injected candidate Full Slate cases required by the 35-gate plan: RB1 OUT, QB1 inactive, WR/TE unavailable; prove unavailable zero, deterministic successor roles and existing opportunity-conservation invariants.
5. Run current real-source no-odds candidate Full Slate with timing certification. Do not require official inactive lists for games outside T-75; fail closed only affected games when inside T-75 and uncertified.
6. Add static production-readiness checks for availability source/timing/role wiring.
7. If and only if all 35 frozen gates PASS, record `CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_PASS_READY_FOR_PROMOTION`, promote exact implementation, and run exact post-promotion Full Slate verification. Otherwise preserve failure and leave production unchanged.
8. Keep `CURRENT_NFL_RESEARCH_HANDOFF.md` updated with branch/head/run/job/artifact/digest/disposition.

---

# 8. Remaining roadmap after availability lane

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
