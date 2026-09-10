# CURRENT NFL RESEARCH HANDOFF — READ FIRST

**Repository:** `dkaps6/imtiredofthis`  
**Protected production-code authority:** `bb76ba9eabb08e2f0875a9af49301c3877f4141f`  
**Production stack unchanged:** QB M89/M90 + mean-neutral C2; WR M38 WR1 + WR-R15 WR2+; TE TE-R5P; RB rushing P3; RB receptions R26; RB receiving-yard tails R22; sportsbook downstream only.  
**RB receiving-yard mean lane:** CLOSED at a defensible scientific frontier; no new mean integration authorized.  
**Active lane:** current roster / late-week availability + role reconciliation / Full Slate integration certification.  
**Canonical availability core:** `ops-current-player-availability-t75-v1`.  
**Candidate Full Slate branch:** `ops-current-player-availability-full-slate-v1`.  
**35-gate certification branch:** `ops-current-player-availability-35gate-cert-v1`.  
**Production is NOT yet wired or promoted.**

GitHub is canonical; chat memory is secondary. Preserve all prior handoff snapshots in Git history.

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

Final tested mean family R27D:
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
- 35 predeclared gates; all required before promotion.

Canonical T-75 semantics are frozen by first valid result Run `34437715931`, Job `102746163583`:
- >75 min: `NOT_YET_REQUIRED` and eligible;
- <=75 min with complete official pre-kickoff sections: `REQUIRED_AND_CERTIFIED`;
- <=75 min missing/incomplete/invalid official sections: `REQUIRED_MISSING_FAIL_CLOSED` for that game only;
- at/after kickoff: `KICKED_OFF_LOCKED`.

Locked core blobs:
- Ourlads status sidecar `c115816ea8aa4ba7150a635c3115546d43f94b3c`
- availability resolver `9a1b0a672db7854ff52764ad93e5fcd89f9cb0ea`
- official NFL inactives adapter `0d67316b6d7b7b9aa9d3637a07da4cd2b171639e`
- T-75 validator `d67ae30ed6e837f62098671c499d05462fe9d837`
- reconciled active roles `b2d05882cada048337f1f5f8b8db8ec7f9eef001`

Current-role/timing seams:
- `scripts/utils/current_roles_v1.py` `7540cc40e02546b4feb2cc503aa506aa613c44f3`
- `scripts/run_player_form_current_roles_v1.py` `0a82649801794c560ccfd0ef368b7a340ec6f38f`
- `scripts/run_rb_week1_current_roles_v1.py` `5874128aaa31dca9ed0a401f4dfa9197ff4c282e`
- `scripts/build/build_production_eligible_active_roles_v1.py` `8be1e607f4d35359459aa2d888c4cfee12cd27c7`
- `scripts/run_current_player_availability_candidate_prep_v1.py` `1b16c8052f079ba94423b21a755ce4107f60a311`
- Week-1 TeamForm prior wrapper `240f62d018fa419567663cae3c96647483305361`
- PlayerForm strict-prior publication fixture `ee3c88060a3d12b10ada01a0eff6e2e2c5b5d1d5`
- candidate workflow `51362ee87a164824cc96dc186aeb9f1f66973f4a`.

## Preserved Full Slate mechanical failures

Run 1 — Week-1 TeamForm source selection:
- run `34439714153`, job `102752015236`
- artifact `10137497418`
- digest `sha256:0f3f5195fce8bc6514804b411e0b312475a379cfeb722eb5f5280237f8ff50f2`
- record: `docs/operations/CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_RUN1_QB_CONTEXT_MECHANICAL_FAILURE.md`.

Run 2 — PlayerForm history publication contract:
- run `34443710690`, job `102763847787`
- artifact `10138897760`
- digest `sha256:165c0e431e4771eba05472b17d6680457fedac37d5769bfc5c0b478aad830b1f`
- record: `docs/operations/CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_RUN2_PLAYER_HISTORY_PUBLICATION_MECHANICAL_FAILURE.md`.

Both were plumbing failures and received only value-neutral repairs; scientific model semantics were unchanged.

## First mechanically valid no-odds availability candidate — SUCCESS

Candidate branch `ops-current-player-availability-full-slate-v1`, head `9800254f3ab208ab42501faac83a0d6e5fe3b93d`.

Run `34447900206`, Job `102776660124`:
- artifact `10140425929`
- digest `sha256:6b31ae40d648780673b7953b57323d509b302ea7a82fe99801b024ede2603f37`
- conclusion `success`
- availability rows `468`
- definitive unavailable `1` (`TreVeyon Henderson`, NE, weekly injury report)
- uncertain `1`; unknown `0`
- production-eligible games `15`
- withheld games `1` (`NE-SEA`, already kicked off)
- production-eligible active-role rows `437`
- eligible teams `30`
- sportsbook inputs used for availability/opportunity `0`
- disposition `CURRENT_PLAYER_AVAILABILITY_NO_ODDS_FULL_SLATE_CANDIDATE_COMPLETED`
- production promoted `false`.

This satisfies frozen gate 26 only; it is not the 35-gate integration result.

## Eligible-team downstream coverage seam — REGRESSION PASS

The valid 30-team candidate exposed two legacy current-coverage guards that assumed an untouched 32-team slate. They were frozen before implementation:
- plan `docs/operations/CURRENT_PLAYER_AVAILABILITY_ELIGIBLE_TEAM_COVERAGE_SEAM_V1_FROZEN_PLAN.md`
- plan commit `5a7b3c7d2cb1dc81fc442abf4b304f366965f1d9`
- plan blob `d0a7f74cbc8a4fcb8aa46a0b8a0aaf4987c72006`.

Frozen implementation uses a surgical, blob-locked CI/runtime transformation rather than an unreviewed rewrite of the two large protected source files:
- helper `scripts/utils/eligible_team_set_v1.py` blob `77b591e431378ec984c51e8a032262e673d4c843`
- transformer `scripts/operations/apply_current_availability_eligible_team_seam_v1.py` blob `b64ec5ccd59728121a250433e40e77e3e1013a05`
- regression `scripts/operations/test_current_availability_eligible_team_seam_v1.py` blob `778753400f9f1caa6831f2a71e121a13c268356c`
- implementation lock commit `7bccb4960ab0d2e538fd8c0e179ceedf4b1fb458`
- implementation lock blob `ae77e8772ef30ae813b960196b6ebed219e3333b`
- protected full-universe source blob `f8429ea5b6dd730f054460493facde4ab21b0998`
- protected R26 source blob `0c7528a3ca9e750d3b9ef2f08ef9721949b3e7fc`.

First seam regression:
- branch/head `ops-current-player-availability-35gate-cert-v1` / `c4529c1dcb21f807b0e539db5bad53f68bdf4dfb`
- Run `34453027002`
- Job `102792905910`
- Artifact `10142304020`
- digest `sha256:071f791c916d5c17c655b62d6858ca2adfbfd276009878281898608c2d9d3cc0`
- result record commit `30558db704f81b6336514024e3ed0c5c75291fcc`
- disposition `ELIGIBLE_TEAM_COVERAGE_SEAM_REGRESSION_PASS`.

Frozen regression passed all prerequisites:
1. no-availability mode still requires 32 teams;
2. explicit availability mode accepts exactly the certified eligible set;
3. missing eligible team rejected;
4. extra withheld team rejected;
5. transformation touched only the two declared downstream coverage-source files in CI;
6. transformed full-universe and R26 modules compiled;
7. no 35-gate result was produced.

This removes the known 32-team plumbing blocker only. R26 assets/coefficients/vacancy set/CIN control/redistribution math, R22, M38, TE-R5P, WR-R15, QB C2 and sportsbook boundaries remain unchanged.

## Exact next action

1. Stay on `ops-current-player-availability-35gate-cert-v1`.
2. Build a sportsbook-independent certification football universe from immutable Run `34447900206` / Artifact `10140425929`; do not regenerate the live roster/availability snapshot.
3. Apply the locked eligible-team seam transformation in the certification workspace.
4. Execute the real M38 finite entitlement -> TE-R5P -> WR-R15 -> R26 -> R22 football path on only the 30 certified eligible teams / 15 eligible games.
5. Construct the already-frozen fixture candidate paths: RB1 OUT, QB1 inactive, WR/TE unavailable, including conservation checks.
6. Complete the frozen 35-gate evaluator and result serializer.
7. Lock exact evaluator/fixture/workflow blobs before the first 35-gate execution.
8. Record the first valid integration disposition. Promotion is authorized only for exact `CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_PASS_READY_FOR_PROMOTION` with 35/35 PASS.
9. If 35/35 PASS, use a separately frozen production-promotion implementation and then run post-promotion Full Slate verification. Any other valid disposition leaves protected production unchanged.
10. Update this handoff at every material commit/run/job/artifact/digest/disposition.

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
