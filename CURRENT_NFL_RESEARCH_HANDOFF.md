# CURRENT NFL RESEARCH HANDOFF — READ FIRST

**Repository:** `dkaps6/imtiredofthis`  
**Protected production-code authority:** `bb76ba9eabb08e2f0875a9af49301c3877f4141f`  
**Production stack before availability promotion:** QB M89/M90 + mean-neutral C2; WR M38 WR1 + WR-R15 WR2+; TE TE-R5P; RB rushing P3; RB receptions R26; RB receiving-yard tails R22; sportsbook downstream only.  
**RB receiving-yard mean lane:** CLOSED at a defensible scientific frontier; no new mean integration authorized.  
**Active lane:** current roster / late-week availability production promotion and clean Full Slate verification.  
**Production availability has NOT yet been merged to main.**

GitHub is canonical; chat memory is secondary. Preserve all mechanical failures and first valid scientific/integration results. Never lower frozen gates, post-hoc route losing cohorts, allow sportsbook to define football, or mutate R26/R22/scientific model parameters without separately frozen authority.

## Prior handoff authority

The immediately prior detailed handoff is commit `d370725666c6783f21a9dc728646e189b613d78c`, blob `31ffe649b4e545a26fd98f53e764bd6a56218c58`. It contains the complete lineage through 35-gate Run2 and the locked Run3 retry. All of that history remains canonical and is not superseded except where explicitly updated below.

## RB receiving-yard mean lane — CLOSED

Final R27D strict-prior YACOE residual result:
- branch `research-rb-r27d-yacoe-residual-v1`
- run/job `34436178615` / `102741600329`
- artifact `10136250846`
- digest `sha256:b975511dc54e961c7745e9d1422ac48be4f867c65d3733adfaeabe93af9951e6`
- disposition `R27D_STRICT_PRIOR_YACOE_RESIDUAL_MIXED_OR_FAIL_NO_INTEGRATION`
- 22/31 total gates; 4/13 scientific gates; vacancy RB1 B1 MAE `14.709395` -> C1 `14.710366`; 2023 `13.992139` -> `14.055947`; only 3/6 seasons improved.

No R27D integration/retuning/router. Reopen only for genuinely new pregame information.

## Availability authority

Frozen Full Slate integration plan:
- commit `91ee6aa3ad3813c7d285f6f3163368205937eb09`
- blob `54eb4629c48062fcaef3153918b2069238584d0a`
- exactly 35 predeclared gates; all required before promotion.

Canonical T-75 timing authority:
- Run `34437715931`, Job `102746163583`
- >75 min: `NOT_YET_REQUIRED`, eligible
- <=75 min with complete official pre-kickoff inactive sections: `REQUIRED_AND_CERTIFIED`
- <=75 min missing/incomplete official sections: `REQUIRED_MISSING_FAIL_CLOSED` for that game
- at/after kickoff: `KICKED_OFF_LOCKED`.

Locked availability core blobs:
- Ourlads sidecar `c115816ea8aa4ba7150a635c3115546d43f94b3c`
- resolver `9a1b0a672db7854ff52764ad93e5fcd89f9cb0ea`
- NFL official inactives `0d67316b6d7b7b9aa9d3637a07da4cd2b171639e`
- T-75 validator `d67ae30ed6e837f62098671c499d05462fe9d837`
- reconciled active roles `b2d05882cada048337f1f5f8b8db8ec7f9eef001`.

## First mechanically valid no-odds availability candidate — SUCCESS

- branch/head `ops-current-player-availability-full-slate-v1` / `9800254f3ab208ab42501faac83a0d6e5fe3b93d`
- run/job `34447900206` / `102776660124`
- artifact `10140425929`
- artifact name `current-player-availability-full-slate-candidate-v1`
- digest `sha256:6b31ae40d648780673b7953b57323d509b302ea7a82fe99801b024ede2603f37`
- 468 availability rows
- 1 definitive unavailable, 1 uncertain, 0 unknown
- 15 production-eligible games; 1 kicked-off game withheld
- 437 production-eligible active-role rows across 30 teams
- sportsbook inputs to availability/opportunity `0`
- disposition `CURRENT_PLAYER_AVAILABILITY_NO_ODDS_FULL_SLATE_CANDIDATE_COMPLETED`
- production promoted `false`.

Preserved candidate plumbing failures remain canonical:
- Run1 `34439714153`, Job `102752015236`, Artifact `10137497418`, digest `sha256:0f3f5195fce8bc6514804b411e0b312475a379cfeb722eb5f5280237f8ff50f2` — Week1 TeamForm source selection.
- Run2 `34443710690`, Job `102763847787`, Artifact `10138897760`, digest `sha256:165c0e431e4771eba05472b17d6680457fedac37d5769bfc5c0b478aad830b1f` — PlayerForm history publication contract.

## Availability-aware current-team coverage seams

Full-universe/R26 eligible-team seam:
- frozen plan commit `5a7b3c7d2cb1dc81fc442abf4b304f366965f1d9`
- helper blob `77b591e431378ec984c51e8a032262e673d4c843`
- transformer blob `b64ec5ccd59728121a250433e40e77e3e1013a05`
- regression Run `34453027002`, Job `102792905910`, Artifact `10142304020`
- digest `sha256:071f791c916d5c17c655b62d6858ca2adfbfd276009878281898608c2d9d3cc0`
- disposition `ELIGIBLE_TEAM_COVERAGE_SEAM_REGRESSION_PASS`.

QB C2 current-output seams:
- starter-audit transformer `c7569c54cda779eb04bed7dbf2b22b9ec4fb526b`
- primary-frame transformer `fbb7d34b54aefe98e95d8c097c7542c7d6490b52`
- sequential regression Run `34460546690` — SUCCESS.

Semantics are frozen: legacy mode still requires 32 teams; explicit `ACTIVE_ROLES_CSV` mode requires exactly the certified eligible team set. The separate complete 32-team QB state-context source-integrity guard remains unchanged.

## 35-gate certification failures — PRESERVED

Run1:
- head `eb7f37fa6e2a9019ed0ec8f3bbe6fe73be202699`
- run/job `34459655725` / `102814178762`
- failure before gates: QB C2 starter authority hardcoded 32 teams
- gates evaluated `0/35`
- disposition `MECHANICAL_FAILURE_NO_DECISION`.

Run2:
- head `1d38953995842aa0236edda7124a79665d0b8628`
- run/job `34460227422` / `102816052394`
- 30-team M38 -> TE-R5P -> WR-R15 stage passed
- failure before gates: second QB C2 primary-frame hardcoded 32-team assertion
- gates evaluated `0/35`
- disposition `MECHANICAL_FAILURE_NO_DECISION`.

## 35-gate Run3 — FIRST VALID GATES 1-34 EVIDENCE

Run3 was frozen separately:
- retry-plan commit `8324754389bae0162455a565bd86ee78aee6a91e`
- workflow blob `8d60c7fb4ff301aab9abcb3c2630601c5d9a8ea9`
- lock/head `76d01dd8e7b26ef8921cd70c18f27957da46c560`
- run/job `34461561636` / `102820358570`.

Run3 completed all substantive football/integration work:
- exact immutable candidate staged
- locked eligible-team seams applied
- baseline M38 -> TE-R5P -> WR-R15 -> QB C2 -> R22 -> R26 -> outer P3 conservation completed
- RB1 OUT fixture completed
- QB1 inactive fixture completed
- WR/TE unavailable fixture completed
- frozen gates 1 through 34 evaluated
- **34/34 PASS**

Immutable gates 1-34 artifact:
- artifact `10145975346`
- name `current-player-availability-35gate-evidence-v1-run3`
- digest `sha256:dd32b45f6746176911ca68aa1d73a8a75325f92cf7b0951c335ae5484752e0ed`.

The Run3 workflow then failed only at the post-upload gate-35 lineage finalization step. Final result upload was skipped. That failure is preserved as finalization plumbing, not a football/scientific failure. Result-record commit on certification branch: `4646883a7eb9c60cbe556ee93bfc619acce368ae`.

## Run4A — GATE 35 FINALIZED AGAINST IMMUTABLE RUN3 EVIDENCE — SUCCESS

To avoid rerunning already immutable 34/34 football evidence, a separately frozen evidence-only finalization was used.

Frozen lineage:
- Run4 generic repair-plan commit `2a54b5868dd58eb7c52af8487c9e3509459b0e09`
- conservative Run4A post-upload plan commit `0854ad71163d9f3b6cb7219d307ae3069106d0df`
- Run4A plan blob `e0b4e76e2143f85a27cdb1de8b97aadc379b9072`
- Run4A workflow blob `a4c68c00c2a4d37131969104ace9b7f64958a6db`
- unchanged gate35 finalizer blob `a5302186eadb748863c70b175421d064885dde60`
- Run4A lock/head `8609a2ad503b93cc372a72ab3421bc8f16ef7961`.

Execution:
- wrapper run `34463888613` — SUCCESS
- Run4A verified exact Run3 artifact ID/name/digest from GitHub API
- downloaded exact Run3 evidence artifact
- asserted `all_1_34_pass=true`, `evaluated_gate_count=34`, `passed_1_34=34`, `failed_1_34=0`
- invoked the exact unchanged gate-35 finalizer using Run3 lineage values
- no football stack, fixture, gate 1-34, candidate or production recomputation occurred.

Finalized result:
- artifact `10146675272`
- name `current-player-availability-35gate-finalized-run3-v1`
- digest `sha256:dcc0fec6f07c62542e7115a49c1c45e9a185d18fd3baa2e423e0bd3dd08d0286`
- source evidence lineage remains Run3: branch `ops-current-player-availability-35gate-cert-v1`, head `76d01dd8e7b26ef8921cd70c18f27957da46c560`, run `34461561636`, job `102820358570`, artifact `10145975346`, digest `sha256:dd32b45f6746176911ca68aa1d73a8a75325f92cf7b0951c335ae5484752e0ed`
- **passed gates: 35/35**
- failed gates: `0`
- disposition `CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_PASS_READY_FOR_PROMOTION`
- `production_promoted=false`
- `sportsbook_used_to_define_football=false`.

This is now the complete immutable certification authority.

## CURRENT LIVE STATE — separate production promotion frozen

Dedicated promotion branch:
- `ops-current-player-availability-production-promotion-v1`
- based from current `main`
- frozen promotion-plan commit `9b3241996c50f1a05d124acd370fa09f0bc07e35`
- plan file `docs/operations/CURRENT_PLAYER_AVAILABILITY_PRODUCTION_PROMOTION_V1_FROZEN_PLAN.md`.

The promotion plan authorizes only the certified availability/current-role input plumbing and Full Slate ordering/coverage changes. It explicitly prohibits changes to M89/M90/C2 science, M38, WR-R15, TE-R5P, P3, R26 science, R22 tails, historical research, T-75 semantics, QUESTIONABLE/DOUBTFUL semantics, and sportsbook-to-football boundaries.

No production implementation files have yet been changed on the promotion branch. `main` still does not contain the availability promotion.

### Exact next action

1. On `ops-current-player-availability-production-promotion-v1`, implement only the frozen promotion plan using the already-certified provider/build/current-role/helper seams.
2. Wire canonical `full-slate.yml` so availability is resolved before PlayerForm/opportunity and `ACTIVE_ROLES_CSV=data/roles_current_production_eligible_v1.csv` drives current-role consumers.
3. Preserve the complete 32-team QB state-context source-integrity guard; make only current-output team-count checks availability-aware using the exact certified eligible-team semantics.
4. Keep sportsbook downstream; an unavailable/withheld player/game must never be resurrected by odds matching.
5. Lock exact promotion implementation blobs before the first post-promotion verification run.
6. Run a dedicated **no-odds clean-checkout Full Slate production verification**. Any mechanical failure must be preserved and minimally repaired under a separately frozen repair; any semantic/scientific invariant failure means no promotion.
7. Only after that branch verification PASS may the implementation be promoted to `main`.
8. After main promotion, run Full Slate again from clean `main` with odds disabled and preserve exact run/job/artifact/digest before declaring availability lane complete.
9. Update this handoff at every material checkpoint.

## Remaining roadmap after availability

1. Finish current roster / late-week availability production promotion + clean-main verification — ACTIVE.
2. Grade sealed R26Q with exact locked R26S once authoritative Week1 outcomes exist.
3. QB opportunity/efficiency: attempts/dropbacks/pass rate/YPA/sacks/scrambles, building on M89/M90.
4. Selective unresolved WR/TE opportunity/efficiency/distribution while preserving M38/WR-R15 and TE-R5P.
5. Shared QB↔receiver conservation.
6. Unified game simulation.
7. Anytime TD modeling.
8. Game ML/spread/total from football simulation rather than sportsbook imitation.
9. Final operational package and prospective grading.
