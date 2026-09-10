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

Prior detailed snapshots remain in Git history, including:
- `a2a8f32d0e39672052cab94be297a36b03238fbc` — availability core + Full Slate freeze checkpoint
- `261256ae4105cc3220668777a7c77f377252b961` — canonical T-75 timing checkpoint
- `99d0ae6f6e0c4d60458a919096ce5cec1dfe695e` — R27D closure checkpoint
- `84c9ffa6ce3617757bcd6b41705d7e55fa8403e0` — R27D plan freeze
- `69e8de76bd1b508849d679fa22abd51aefa68a54` — deep R27B V2 checkpoint

Do not discard those snapshots or historical research records.

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
- frozen plan commit `69edc12a16c691e3838eadcd75559b85dbba7865`, blob `d0c2b0ff2de154e52fa21fb9ce19b739039633f3`
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

- frozen audit plan commit `848e5e44b1078c9d9dd7ab40ad0ccf16136f3754`
- audit result commit `810c344a437d411707185317033acc7004f1c7db`
- disposition `CURRENT_ROSTER_LATE_WEEK_ROLE_GAP_CONFIRMED_FIX_PLAN_REQUIRED`
- frozen operational fix plan commit `b2206e7ad693148623447bcf9a3ad6b594033500`

Confirmed defects included status loss from raw Ourlads roles, non-authoritative weekly injury handling for final game-day availability, positive opportunity surviving definitive unavailability, and no canonical shared availability+role authority before current opportunity.

---

# 4. Locked availability core and canonical T-75 timing

Locked core:
- Ourlads status/provenance sidecar `scripts/providers/ourlads_depth_status_v1.py`, blob `c115816ea8aa4ba7150a635c3115546d43f94b3c`
- availability resolver `scripts/build/build_current_player_availability_v1.py`, blob `9a1b0a672db7854ff52764ad93e5fcd89f9cb0ea`
- official NFL inactive adapter `scripts/providers/nfl_official_inactives_v1.py`, blob `0d67316b6d7b7b9aa9d3637a07da4cd2b171639e`
- T-75 validator `scripts/validate_current_player_availability_timing_v1.py`, blob `d67ae30ed6e837f62098671c499d05462fe9d837`
- reconciled active-role builder `scripts/build/build_reconciled_active_roles_v1.py`, blob `b2d05882cada048337f1f5f8b8db8ec7f9eef001`
- core implementation lock commit `596f084750a8c8e3c36ca09d737dc2edb808dc09`

Clean evidence:
- semantic run `34436970099`, job `102743973821`: 8/8 PASS
- live source smoke run `34437032282`, job `102744156238`, artifact `10136545256`, digest `sha256:decb703afe4769befba790d8b1adceb0accb5a49eec4f5fd8f5d2adc6c7eb75a`
- first valid T-75 timing run `34437715931`, job `102746163583`, head `092fee088f3402d6313bc02b2f8cc05d1f3f54f9`: SUCCESS, 8/8 timing fixtures PASS

Canonical timing semantics:
- >75 minutes: `NOT_YET_REQUIRED`, game remains eligible;
- <=75 minutes pre-kickoff with complete pre-kickoff official sections: `REQUIRED_AND_CERTIFIED`;
- <=75 minutes with missing/incomplete/invalid official sections: `REQUIRED_MISSING_FAIL_CLOSED` for that game only;
- at/after kickoff: `KICKED_OFF_LOCKED`.

Do not change T-75 after the first valid result.

---

# 5. Frozen Full Slate integration authority

Frozen integration plan:
- `docs/operations/CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_V1_FROZEN_PLAN.md`
- commit `91ee6aa3ad3813c7d285f6f3163368205937eb09`
- blob `54eb4629c48062fcaef3153918b2069238584d0a`
- 35 predeclared gates; all required before promotion.

Core rule: **availability is resolved before opportunity**. Definitively unavailable players and timing-withheld games are removed before current PlayerForm/opportunity. Existing M89/M90/C2, M38, TE-R5P, WR-R15, P3, R26 and R22 mechanics remain unchanged. Sportsbook data cannot define eligibility or resurrect a removed player.

---

# 6. Candidate Full Slate implementation — LOCKED, FIRST RUN IN FLIGHT

Candidate branch: `ops-current-player-availability-full-slate-v1`.

Pre-lock mechanical role-seam smoke:
- Run `34439533260`
- Job `102751487393`
- head `bb1c4f4634d832ca3db7b6fc0d98885a59cc8848`
- conclusion SUCCESS
- explicit current-role resolver fixtures PASS
- timing-withheld game filter fixtures PASS
- compile PASS
- no scientific model file changes from frozen integration-plan parent
- this is plumbing evidence only, not the first Full Slate result.

Candidate-only implementation added before lock:
- `scripts/utils/current_roles_v1.py`, blob `7540cc40e02546b4feb2cc503aa506aa613c44f3`
  - raw `data/roles_ourlads.csv` remains default/provider evidence;
  - explicit `ACTIVE_ROLES_CSV` activates reconciled roles;
  - explicit missing active artifact fails closed and never silently falls back to raw.
- `scripts/run_player_form_current_roles_v1.py`, blob `6252142859042dac7411ba23a482d8c0c128c601`
  - redirects only PlayerForm's current-role input; PlayerForm historical/blend mechanics unchanged.
- `scripts/run_rb_week1_current_roles_v1.py`, blob `5874128aaa31dca9ed0a401f4dfa9197ff4c282e`
  - redirects only RB P3's current Ourlads-role read; P3/R26/R22 math unchanged.
- `scripts/build/build_production_eligible_active_roles_v1.py`, blob `8be1e607f4d35359459aa2d888c4cfee12cd27c7`
  - removes whole games with `production_eligible==0` from active roles before opportunity;
  - preserves `NOT_YET_REQUIRED` games;
  - records withheld/eligible teams and state counts; sportsbook inputs 0.
- `scripts/run_current_player_availability_candidate_prep_v1.py`, blob `1b16c8052f079ba94423b21a755ce4107f60a311`
  - candidate pre-opportunity order: timestamped Ourlads -> official inactives -> T-75 certification -> availability -> active roles -> production-eligible roles.
- candidate workflow `.github/workflows/ops-current-player-availability-full-slate-v1.yml`, blob `1ea734ab5dd9e5f9846857a449d8c8c948774bf5`
  - no-odds first run;
  - rebuilds current football stack using active roles before PlayerForm/P3;
  - asserts unavailable identities are absent from PlayerForm;
  - protects scientific models and historical research;
  - uploads artifacts even on failure.

Implementation lock:
- `docs/operations/CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_V1_IMPLEMENTATION_LOCK.md`
- lock commit `9147bde6d08c05249415ec4cd364db482d812a35`
- status `LOCKED BEFORE FIRST CANDIDATE FULL_SLATE RESULT`

**First lock-triggered candidate Full Slate run:**
- Run `34439714153`
- Job `102752015236`
- head `9147bde6d08c05249415ec4cd364db482d812a35`
- status at this handoff checkpoint: `IN_PROGRESS`
- this is the immutable first candidate Full Slate lineage if it reaches a valid integration result.
- NO PASS/FAIL scientific or integration disposition has been declared yet.
- production remains unchanged.

If this run fails before substantive execution for a plumbing/dependency/provider reason, preserve it as mechanical failure and make only value-neutral repair. Do not change the T-75 rule, availability hierarchy, role semantics, scientific model parameters, or frozen 35 gates.

---

# 7. Exact next action

1. Inspect Run `34439714153`, Job `102752015236` first.
2. If mechanical failure: identify exact failed step, preserve a repair record, change only value-neutral plumbing, pin repaired blobs if needed, and rerun without altering scientific/integration semantics.
3. If the no-odds Full Slate completes: archive artifact ID/digest and current availability counts (unavailable/uncertain/unknown/withheld).
4. Complete the 35-gate integration evaluator, including fixture-injected RB1 OUT, QB1 inactive, and WR/TE unavailable demonstrations and existing entitlement-conservation checks.
5. Record the immutable first valid integration disposition. Promotion is allowed only for exact `CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_PASS_READY_FOR_PROMOTION` with all 35 gates PASS.
6. If PASS, promote the exact locked implementation and run exact post-promotion Full Slate verification. Otherwise leave protected production unchanged.
7. Update this handoff with exact run/job/artifact/digest/disposition at every material checkpoint.

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
