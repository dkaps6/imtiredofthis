# CURRENT NFL RESEARCH HANDOFF — READ FIRST

**Repository:** `dkaps6/imtiredofthis`  
**Protected production-code authority:** `bb76ba9eabb08e2f0875a9af49301c3877f4141f`  
**Production stack unchanged:** QB M89/M90 + mean-neutral C2; WR M38 WR1 + WR-R15 WR2+; TE TE-R5P; RB rushing P3; RB receptions R26; RB receiving-yard tails R22; sportsbook downstream only.  
**RB receiving-yard mean lane:** CLOSED at a defensible scientific frontier; no new mean integration authorized.  
**Active lane:** current roster / late-week availability + role reconciliation.  
**Canonical availability core:** `ops-current-player-availability-t75-v1`.  
**Candidate Full Slate branch:** `ops-current-player-availability-full-slate-v1`.  
**Production is NOT yet wired or promoted.**

GitHub is canonical; chat memory is secondary. Historical handoff snapshots remain in Git history, especially `5556841d4a2b49cc19b2c2e3780af9f72c76303a`, `a2a8f32d0e39672052cab94be297a36b03238fbc`, `261256ae4105cc3220668777a7c77f377252b961`, `99d0ae6f6e0c4d60458a919096ce5cec1dfe695e`, and `84c9ffa6ce3617757bcd6b41705d7e55fa8403e0`.

## Non-negotiable rules

- freeze questions/mechanics/gates before results;
- preserve first valid results exactly;
- preserve mechanical/integrity failures separately and repair only value-neutral defects;
- no lowering gates, dropping losing seasons, post-hoc routing or retuning;
- sportsbook remains downstream and cannot define football eligibility/role;
- protect R26 receptions/opportunity and R22 RB receiving-tail authority unless separately authorized;
- operational availability work must not alter historical research;
- earlier frozen authority wins over later unacknowledged conflicts;
- update this file at every material checkpoint.

## RB receiving-yard mean lane — CLOSED

Final tested mean family R27D did not qualify:
- branch `research-rb-r27d-yacoe-residual-v1`
- run `34436178615`, job `102741600329`
- artifact `10136250846`
- digest `sha256:b975511dc54e961c7745e9d1422ac48be4f867c65d3733adfaeabe93af9951e6`
- disposition `R27D_STRICT_PRIOR_YACOE_RESIDUAL_MIXED_OR_FAIL_NO_INTEGRATION`
- 22/31 total gates; 4/13 scientific gates
- vacancy RB1 B1 MAE `14.709395` -> C1 `14.710366`; 2023 RB1 `13.992139` -> `14.055947`; 3/6 seasons improved.

No integration/retuning/router. Reopen only for genuinely new pregame information.

## Availability core authority

Frozen Full Slate integration plan:
- commit `91ee6aa3ad3813c7d285f6f3163368205937eb09`
- blob `54eb4629c48062fcaef3153918b2069238584d0a`
- 35 predeclared gates, all required before promotion.

Canonical T-75 semantics are frozen by first valid result Run `34437715931`, Job `102746163583`:
- >75 min: `NOT_YET_REQUIRED` and eligible;
- <=75 min with complete official pre-kickoff sections: `REQUIRED_AND_CERTIFIED`;
- <=75 min missing/incomplete/invalid official sections: `REQUIRED_MISSING_FAIL_CLOSED` for that game only;
- at/after kickoff: `KICKED_OFF_LOCKED`.

Locked core blobs remain:
- Ourlads status sidecar `c115816ea8aa4ba7150a635c3115546d43f94b3c`
- availability resolver `9a1b0a672db7854ff52764ad93e5fcd89f9cb0ea`
- official NFL inactives adapter `0d67316b6d7b7b9aa9d3637a07da4cd2b171639e`
- T-75 validator `d67ae30ed6e837f62098671c499d05462fe9d837`
- reconciled active roles `b2d05882cada048337f1f5f8b8db8ec7f9eef001`

Current-role seams remain unchanged:
- `scripts/utils/current_roles_v1.py` `7540cc40e02546b4feb2cc503aa506aa613c44f3`
- `scripts/run_player_form_current_roles_v1.py` `6252142859042dac7411ba23a482d8c0c128c601`
- `scripts/run_rb_week1_current_roles_v1.py` `5874128aaa31dca9ed0a401f4dfa9197ff4c282e`
- `scripts/build/build_production_eligible_active_roles_v1.py` `8be1e607f4d35359459aa2d888c4cfee12cd27c7`
- `scripts/run_current_player_availability_candidate_prep_v1.py` `1b16c8052f079ba94423b21a755ce4107f60a311`

## First locked Full Slate candidate — PRESERVED MECHANICAL FAILURE

Original candidate lock head: `9147bde6d08c05249415ec4cd364db482d812a35`.

Run `34439714153`, Job `102752015236`:
- conclusion `failure`;
- artifact `10137497418`;
- digest `sha256:0f3f5195fce8bc6514804b411e0b312475a379cfeb722eb5f5280237f8ff50f2`;
- failed before availability/opportunity execution in `Build team context and promoted QB context`;
- decoded log proves exact failing command was `python scripts/run_team_form_context.py --season 2026 --box-backfill-prev`;
- exact exception: `RuntimeError: active-season PBP has no completed pre-target rows for season=2026 week=1`;
- `run_qb_promoted_context.py` never executed;
- therefore this is `MECHANICAL_FAILURE_BEFORE_AVAILABILITY_EXECUTION_NO_INTEGRATION_RESULT`, not a 35-gate FAIL.

Preserved/corrected failure record on candidate branch:
`docs/operations/CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_RUN1_QB_CONTEXT_MECHANICAL_FAILURE.md`.

## Value-neutral Week-1 TeamForm repair — RELOCKED

Root cause: the runtime TeamForm wrapper's documented Week-1 contract says preseason/Week1 should use PRIOR_SEASON because no completed current-season games can exist strictly before target Week 1. The active-season nflverse feed was non-empty, so the original loader treated it as current-season data before the later strict `week < 1` cutoff rejected it. The legacy TeamForm had written only a partial current-season table before aborting.

Repair:
- new Week-1-only wrapper `scripts/run_team_form_context_week1_prior_v1.py`
- blob `240f62d018fa419567663cae3c96647483305361`
- wrapper forces the already-declared 2025 PRIOR_SEASON PBP source for target Week 1 and refuses to run outside Week 1;
- no TeamForm formulas, scientific model parameters, T-75 semantics, availability hierarchy, role semantics, R22/R26 mechanics, sportsbook boundary, or frozen 35 gates changed.

Repaired candidate workflow:
- `.github/workflows/ops-current-player-availability-full-slate-v1.yml`
- blob `c432d6985323e1a5f6029464ca401dbd68393b78`
- only substantive execution change is calling the Week-1 declared-prior wrapper in the TeamForm step.

Repaired implementation lock:
- `docs/operations/CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_V1_IMPLEMENTATION_LOCK.md`
- relock head `ec01f7f8087679d6650cae98bf765258b073a261`
- status `RELOCKED AFTER VALUE_NEUTRAL_WEEK1_TEAMFORM_MECHANICAL_REPAIR`.

## Current live candidate execution

Run `34443710690`
Job `102763847787`
Head `ec01f7f8087679d6650cae98bf765258b073a261`
Workflow run number `2`
Status at this checkpoint: `IN_PROGRESS` (dependency-install stage when last inspected).

This is the first repaired candidate run after the preserved Week-1 TeamForm mechanical failure. It has not yet produced a valid integration disposition. Production remains unchanged.

## Exact next action

1. Inspect Run `34443710690`, Job `102763847787` first.
2. If another mechanical failure occurs, fetch decoded job logs, preserve exact failure lineage/artifact/digest, and make only a value-neutral repair; do not change the 35 gates or football semantics.
3. If the no-odds Full Slate reaches completion, archive artifact ID/digest and current availability counts.
4. Complete the frozen 35-gate integration evaluator, explicitly including RB1 OUT, QB1 inactive, WR/TE unavailable fixtures and entitlement-conservation checks.
5. Record the first valid integration disposition. Promotion is authorized only for exact `CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_PASS_READY_FOR_PROMOTION` with 35/35 PASS.
6. If 35/35 PASS, promote exact locked implementation and run post-promotion Full Slate verification. Any other valid disposition leaves protected production unchanged.
7. Update this handoff with every material run/job/artifact/digest/disposition.

## Remaining roadmap after availability

1. Current roster / late-week role handling — ACTIVE.
2. Grade sealed R26Q with exact locked R26S once authoritative Week1 outcomes exist.
3. QB opportunity/efficiency: attempts/dropbacks/pass rate/YPA/sacks/scrambles; build on M89/M90.
4. Selective unresolved WR/TE opportunity/efficiency/distribution; preserve M38/WR-R15 and TE-R5P.
5. Shared QB↔receiver conservation.
6. Unified game simulation.
7. Anytime TD modeling.
8. Game ML/spread/total from football simulation rather than sportsbook imitation.
9. Final operational package/prospective grading.
