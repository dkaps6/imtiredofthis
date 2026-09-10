# CURRENT NFL RESEARCH HANDOFF — READ FIRST

**Repository:** `dkaps6/imtiredofthis`  
**Main head before this handoff update:** `8c83c1cc8db3e80f869fab66adcc869574aa6a3c`  
**Protected production-code authority before availability promotion:** `bb76ba9eabb08e2f0875a9af49301c3877f4141f`  
**Production stack remains:** QB M89/M90 + mean-neutral C2; WR M38 WR1 + WR-R15 WR2+; TE TE-R5P; RB rushing P3; RB receptions R26; existing RB receiving-yard mean; RB receiving-yard tails R22; sportsbook downstream only.  
**RB receiving-yard mean lane:** CLOSED at a defensible scientific frontier; no new mean integration authorized.  
**Active lane:** current roster / late-week availability production promotion.  
**IMPORTANT:** availability is **NOT** promoted to `main`; first locked production-branch verification failed mechanically at the strict repository-audit stage.

GitHub is canonical; chat memory is secondary. Preserve first valid science/integration results and every mechanical failure. No post-result gate changes, no sportsbook-defined football, no target/future leakage, and no R26/R22/model-science mutation without separately frozen authority.

## Historical detailed handoff authority

For the complete R27/R27D history, source audits, candidate failures, availability source/timing/candidate lineage, eligible-team/QB seams, 35-gate fixture history, and all older run artifacts, read main commit `93b974acf9563c5b875d7d08476b339fcd5549c6`, handoff blob `0ef4a1f1443ee0327344b5d6ae7002a577dd9d0e`.

## RB receiving-yard mean lane — CLOSED / NO INTEGRATION

Final R27D strict-prior YACOE residual result:
- branch `research-rb-r27d-yacoe-residual-v1`
- first valid head `641b25419c4f5ff3c234d1c000222fb4909ef940`
- run/job `34436178615` / `102741600329`
- artifact `10136250846`
- digest `sha256:b975511dc54e961c7745e9d1422ac48be4f867c65d3733adfaeabe93af9951e6`
- result-record commit `a20e757a1b85806ae5e244cc42d29cfbc7ed2742`
- disposition `R27D_STRICT_PRIOR_YACOE_RESIDUAL_MIXED_OR_FAIL_NO_INTEGRATION`
- integrity 18/18 PASS; total scorecard 22/31 PASS; scientific gates 4/13 PASS
- vacancy RB1 B0 MAE `14.305919`; B1 `14.709395`; C1 `14.710366`
- 2023 vacancy RB1 B0 `12.349471`; B1 `13.992139`; C1 `14.055947`
- only 3/6 seasons improved C1 versus B1.

Conclusion: strict-prior relative xYAC/YACOE persistence did not provide a reliable RB1 receiving-yard mean correction. No R27D integration, retuning, or post-hoc router is authorized. R26 receptions and R22 tails remain protected. Future RB receiving-mean work requires genuinely new pregame information/mechanism.

Preserved R27D non-scientific failures:
- Run `34435661834`, Job `102740069405`: duplicate deterministic Week1 column collision; no scientific result.
- Run `34435897671`, Job `102740771822`, Artifact `10136169669`, digest `sha256:d00eab566113e0c43502bf75f07fe477295dbef6a7ce20389e23ea941ec6d845`: null-outcome scoring/integrity failure; emitted result explicitly rejected as invalid.

## Availability certification authority — 35/35 PASS

Frozen Full Slate availability integration plan:
- commit `91ee6aa3ad3813c7d285f6f3163368205937eb09`
- blob `54eb4629c48062fcaef3153918b2069238584d0a`
- exactly 35 predeclared gates; all required before promotion.

Canonical T-75 timing authority:
- Run `34437715931`, Job `102746163583`
- >75 min: `NOT_YET_REQUIRED`, eligible
- <=75 min with complete official pre-kickoff inactive sections: `REQUIRED_AND_CERTIFIED`
- <=75 min missing/incomplete official sections: `REQUIRED_MISSING_FAIL_CLOSED`
- at/after kickoff: `KICKED_OFF_LOCKED`.

First mechanically valid no-odds availability candidate:
- branch/head `ops-current-player-availability-full-slate-v1` / `9800254f3ab208ab42501faac83a0d6e5fe3b93d`
- run/job `34447900206` / `102776660124`
- artifact `10140425929`
- digest `sha256:6b31ae40d648780673b7953b57323d509b302ea7a82fe99801b024ede2603f37`
- 468 availability rows; 1 definitive unavailable; 1 uncertain; 0 unknown
- 15 production-eligible games; 1 already-kicked-off game withheld
- 437 production-eligible active-role rows across 30 teams
- sportsbook inputs to availability/opportunity `0`
- disposition `CURRENT_PLAYER_AVAILABILITY_NO_ODDS_FULL_SLATE_CANDIDATE_COMPLETED`
- production promoted `false`.

Preserved candidate plumbing failures:
- Run `34439714153`, Job `102752015236`, Artifact `10137497418`, digest `sha256:0f3f5195fce8bc6514804b411e0b312475a379cfeb722eb5f5280237f8ff50f2` — Week1 TeamForm source-selection failure.
- Run `34443710690`, Job `102763847787`, Artifact `10138897760`, digest `sha256:165c0e431e4771eba05472b17d6680457fedac37d5769bfc5c0b478aad830b1f` — PlayerForm history-publication contract failure.

Availability-aware seam regression:
- full-universe/R26 seam plan `5a7b3c7d2cb1dc81fc442abf4b304f366965f1d9`
- regression Run `34453027002`, Job `102792905910`, Artifact `10142304020`
- digest `sha256:071f791c916d5c17c655b62d6858ca2adfbfd276009878281898608c2d9d3cc0`
- disposition `ELIGIBLE_TEAM_COVERAGE_SEAM_REGRESSION_PASS`.
- QB C2 sequential current-output seam regression Run `34460546690` — SUCCESS.

35-gate certification:
- Run1 `34459655725` / `102814178762`: mechanical QB C2 starter 32-team current-output assertion; 0/35 evaluated.
- Run2 `34460227422` / `102816052394`: mechanical QB C2 primary-frame 32-team current-output assertion; 0/35 evaluated.
- Run3 first valid substantive evidence: head `76d01dd8e7b26ef8921cd70c18f27957da46c560`, run/job `34461561636` / `102820358570`; gates 1-34 **34/34 PASS**; artifact `10145975346`, digest `sha256:dd32b45f6746176911ca68aa1d73a8a75325f92cf7b0951c335ae5484752e0ed`.
- Run3 then failed only in post-upload gate-35 lineage finalization; preserved as finalization plumbing.
- Evidence-only Run4A wrapper `34463888613` verified exact Run3 artifact/digest and finalized unchanged gate 35 without recomputing football.
- final 35/35 artifact `10146675272`
- digest `sha256:dcc0fec6f07c62542e7115a49c1c45e9a185d18fd3baa2e423e0bd3dd08d0286`
- disposition `CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_PASS_READY_FOR_PROMOTION`
- production promoted in certification result: `false`.

## Frozen production-promotion implementation

Dedicated branch: `ops-current-player-availability-production-promotion-v1`.

Promotion plan:
- commit `9b3241996c50f1a05d124acd370fa09f0bc07e35`
- blob `2d350379377ae65b2fb094504624da0591f53f7e`.

Dynamic production-verification plan:
- commit `3a560ad4c73adfd992035d4c650e406c050ed309`
- blob `a304f4bcf977c421aef7b7eb1a960579f0c2150c`.

Candidate Full Slate:
- `.github/workflows/full-slate.yml`
- wiring commit `753a301881361a71c25a9b3850eea8adf65fee25`
- blob `41bacd4b756c32279169c80e0ff603a4780b7bee`.

Dynamic verifier:
- `scripts/operations/run_current_availability_production_verify_v1.py`
- commit `7fb0349587a6ad95c3486321edf6b8f751c2c803`
- blob `0a84913d850b44504271fba1985f053b2838b970`.

Verification launcher:
- `.github/workflows/current-player-availability-production-verify-v1.yml`
- commit `cfb2cb1d7e917d82affecfb0bcf2da50b4cf60a9`
- blob `8f1bd9d931cf588574cd45c0879708092487fa39`.

Formal implementation lock/head:
- `6e96c7db16ea3f5cf349d6cae006e122007bdbaf`
- lock file `docs/operations/CURRENT_PLAYER_AVAILABILITY_PRODUCTION_PROMOTION_V1_IMPLEMENTATION_LOCK.md`.

Locked semantics:
- availability resolves before opportunity;
- `ACTIVE_ROLES_CSV=data/roles_current_production_eligible_v1.csv` is current-role authority;
- Week1 PlayerForm/TeamForm use certified strict-prior wrappers;
- complete QB C2 state-context source remains exactly 32 teams;
- only current-output coverage follows exact certified eligible-team set;
- sportsbook acquisition/matching occurs only after football eligibility and cannot resurrect withheld games/players;
- no M89/M90/C2 science, M38, WR-R15, TE-R5P, P3, R26 science, R22, model artifact, or historical result may change.

## FIRST LOCKED PRODUCTION-BRANCH VERIFICATION — PRESERVED FAILURE / NO PROMOTION

Parent launcher:
- Run `34470549704`
- Job `102849180219`
- head `6e96c7db16ea3f5cf349d6cae006e122007bdbaf`
- frozen implementation verification: PASS
- child Full Slate dispatch: PASS
- final parent conclusion: FAILURE because child Full Slate failed
- parent evidence artifact `10149440197`
- digest `sha256:355a8eb8340bdc3281461457cef39a205ffaeedddbe0543b9a291cf1d3db38ae`
- dynamic production verification was correctly skipped after child failure.

Canonical child Full Slate:
- Run `34470613780`
- Job `102849391150`
- run number `560`
- head `6e96c7db16ea3f5cf349d6cae006e122007bdbaf`
- `FETCH_LIVE_ODDS=false`
- conclusion: **FAILURE**
- artifact `10149436841`
- digest `sha256:62fbde8ff296d8df8fbe2c40b5fa3d04a281640281c53dea84be8d14a751f946`.

Child steps that PASSED before failure:
- raw Ourlads roles
- authoritative team-week map
- weather/injuries
- **current availability before opportunity**
- Week1 strict-prior TeamForm
- promoted QB M89/M90 context
- Coverage v2 and optional PBP
- PlayerForm from production-eligible current roles with no current/future Week1 history
- provider/team context, Bayesian, ML, Markov, rules and ensemble bridges
- RB P3 from production-eligible current roles
- complete 32-team QB C2 football state context
- certified availability-aware current-output seams.

Sportsbook/pricing steps were skipped because live odds were disabled. The failing step was **Strict repository audits**. The child still uploaded its full data/output artifact for diagnosis.

Current live availability snapshot from the child run:
- 468 availability rows
- 1 definitive unavailable
- 1 uncertain
- 0 unknown
- 16 games total; 15 production-eligible; 1 already kicked off and locked
- withheld teams: NE and SEA
- 437 production-eligible active-role rows across 30 teams
- sportsbook inputs used to availability/opportunity: 0.

### Preliminary mechanical diagnosis — NOT YET A FROZEN REPAIR

Static inspection after the failure identifies a deterministic audit-contract mismatch candidate:
- `scripts/utils/audit_repo.py` still requires the literal Full Slate workflow token `scripts/run_player_form_v2_loader.py`.
- The certified availability workflow intentionally routes Week1 PlayerForm through `scripts/run_player_form_current_roles_v1.py`, which is the new strict-prior/current-role wrapper and therefore the old literal token is absent from the workflow.
- The child successfully completed the PlayerForm current-role/strict-prior step before the later static audit failed.

This strongly suggests a **mechanical static-audit compatibility failure**, not evidence that the availability semantics failed. However this diagnosis is not yet authoritative until a separately frozen minimum repair is written and the branch is reverified. Do not alter semantic gates or availability behavior based on this preliminary diagnosis.

## Exact next action

1. Preserve parent Run `34470549704` and child Run `34470613780` exactly as the first production-branch verification failure; classify final disposition only after the audit failure is formally reproduced/diagnosed.
2. Freeze a minimum audit-compatibility repair plan before changing code. The repair may only update static audit expectations to recognize the certified current-role PlayerForm wrapper while continuing to protect the underlying production/load/history contracts. It may not change availability semantics, T-75 rules, eligible-team semantics, model science, R26/R22, or sportsbook ordering.
3. Lock the repaired audit implementation before retry.
4. Rerun the exact production-branch verification with odds disabled. Any semantic/scientific invariant failure means NO promotion.
5. Only `CURRENT_PLAYER_AVAILABILITY_PRODUCTION_BRANCH_VERIFY_PASS_READY_FOR_MAIN_PROMOTION` may authorize merging the exact verified implementation to `main`.
6. After main promotion, run a clean-main no-odds Full Slate and preserve exact run/job/artifact/digest before declaring availability complete.
7. Then proceed to the next roadmap item.

## Remaining roadmap after availability

1. Finish availability branch verification + main promotion + clean-main verification — ACTIVE.
2. Grade sealed R26Q with exact locked R26S once authoritative Week1 outcomes exist.
3. QB opportunity/efficiency: attempts/dropbacks/pass rate/YPA/sacks/scrambles, building on M89/M90.
4. Selective unresolved WR/TE opportunity/efficiency/distribution while preserving M38/WR-R15 and TE-R5P.
5. Shared QB↔receiver conservation.
6. Unified game simulation.
7. Anytime TD modeling.
8. Game ML/spread/total from football simulation rather than sportsbook imitation.
9. Final operational package and prospective grading.
