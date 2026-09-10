# CURRENT NFL RESEARCH HANDOFF — READ FIRST

**Repository:** `dkaps6/imtiredofthis`  
**Canonical operational production authority before this documentation-only handoff commit:** `3e49e0e3a155c0fe74b134542eb359d6444cd8c8`  
**Protected scientific/model authority remains:** `bb76ba9eabb08e2f0875a9af49301c3877f4141f`  
**Availability/current-role production promotion:** **COMPLETE / PROMOTED / CLEAN-MAIN VERIFIED**  
**RB receiving-yard mean research lane:** **CLOSED / NO NEW MEAN INTEGRATION AUTHORIZED**  
**Production stack:** QB M89/M90 + mean-neutral C2; WR M38 WR1 + WR-R15 WR2+; TE TE-R5P; RB rushing P3; RB receptions R26; existing RB receiving-yard mean; RB receiving-yard tails R22; sportsbook downstream only.

GitHub is canonical; chat memory is secondary. Preserve first valid science/integration results and every mechanical failure. No post-result gate changes, no sportsbook-defined football, no target/future leakage, and no silent mutation of protected model science.

## Historical detailed handoff authority

The immediately prior handoff is preserved at operational main commit `3e49e0e3a155c0fe74b134542eb359d6444cd8c8`, file blob `cb2d09e2c4c968d4280d81078d0e5cbd9a288f85`. It contains the detailed availability candidate/certification history, Run1 failure diagnosis, R27/R27D history, and older roadmap detail.

For deeper pre-promotion history, also preserve main commit `93b974acf9563c5b875d7d08476b339fcd5549c6`, handoff blob `0ef4a1f1443ee0327344b5d6ae7002a577dd9d0e`.

## Scientific production boundary

Availability promotion changed operational roster/current-role plumbing only. It did **not** alter M89/M90/C2 scientific parameters, M38/WR-R15, TE-R5P, P3 science, R26 science, R22 tail science, trained model artifacts, or historical research results.

Protected scientific/model authority therefore remains `bb76ba9eabb08e2f0875a9af49301c3877f4141f`. The operational production authority including availability plumbing and the final CI test-contract cleanup is `3e49e0e3a155c0fe74b134542eb359d6444cd8c8`.

## RB receiving-yard mean lane — CLOSED / NO INTEGRATION

Final R27D strict-prior YACOE residual result:
- branch `research-rb-r27d-yacoe-residual-v1`
- first valid head `641b25419c4f5ff3c234d1c000222fb4909ef940`
- run/job `34436178615` / `102741600329`
- artifact `10136250846`
- digest `sha256:b975511dc54e961c7745e9d1422ac48be4f867c65d3733adfaeabe93af9951e6`
- result-record commit `a20e757a1b85806ae5e244cc42d29cfbc7ed2742`
- disposition `R27D_STRICT_PRIOR_YACOE_RESIDUAL_MIXED_OR_FAIL_NO_INTEGRATION`
- integrity `18/18 PASS`
- total scorecard `22/31 PASS`
- scientific gates `4/13 PASS`
- vacancy RB1 B0 MAE `14.305919`; B1 `14.709395`; C1 `14.710366`
- 2023 vacancy RB1 B0 `12.349471`; B1 `13.992139`; C1 `14.055947`
- only `3/6` seasons improved C1 versus B1.

Conclusion: strict-prior relative xYAC/YACOE persistence did not provide a reliable RB1 receiving-yard mean correction. No R27D integration, retuning, or post-hoc router is authorized. R26 receptions and R22 tails remain protected. Future RB receiving-mean work requires genuinely new pregame information/mechanism.

Preserved R27D non-scientific failures:
- Run `34435661834`, Job `102740069405`: duplicate deterministic Week1 column collision; no scientific result.
- Run `34435897671`, Job `102740771822`, Artifact `10136169669`, digest `sha256:d00eab566113e0c43502bf75f07fe477295dbef6a7ce20389e23ea941ec6d845`: null-outcome scoring/integrity failure; emitted result rejected as invalid.

## Availability certification authority — 35/35 PASS

Frozen Full Slate availability integration plan:
- commit `91ee6aa3ad3813c7d285f6f3163368205937eb09`
- blob `54eb4629c48062fcaef3153918b2069238584d0a`
- exactly 35 predeclared gates.

Canonical T-75 timing authority:
- Run `34437715931`, Job `102746163583`
- `>75 min`: `NOT_YET_REQUIRED`, eligible
- `<=75 min` with complete official pre-kickoff inactive sections: `REQUIRED_AND_CERTIFIED`
- `<=75 min` missing/incomplete official sections: `REQUIRED_MISSING_FAIL_CLOSED`
- at/after kickoff: `KICKED_OFF_LOCKED`.

First mechanically valid no-odds candidate:
- head `9800254f3ab208ab42501faac83a0d6e5fe3b93d`
- run/job `34447900206` / `102776660124`
- artifact `10140425929`
- digest `sha256:6b31ae40d648780673b7953b57323d509b302ea7a82fe99801b024ede2603f37`
- 468 availability rows; 1 definitive unavailable; 1 uncertain; 0 unknown
- 15 production-eligible games; 1 kicked-off game withheld
- 437 production-eligible active-role rows across 30 teams
- sportsbook inputs to availability/opportunity `0`.

Final 35/35 certification:
- first valid substantive head `76d01dd8e7b26ef8921cd70c18f27957da46c560`
- Run/Job `34461561636` / `102820358570`
- gates 1-34: `34/34 PASS`
- artifact `10145975346`
- digest `sha256:dd32b45f6746176911ca68aa1d73a8a75325f92cf7b0951c335ae5484752e0ed`
- gate-35 evidence-only finalizer Run `34463888613`
- final 35/35 artifact `10146675272`
- digest `sha256:dcc0fec6f07c62542e7115a49c1c45e9a185d18fd3baa2e423e0bd3dd08d0286`
- disposition `CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_PASS_READY_FOR_PROMOTION`.

## Availability production-promotion frozen authority

Dedicated promotion branch: `ops-current-player-availability-production-promotion-v1`.

Promotion plan:
- commit `9b3241996c50f1a05d124acd370fa09f0bc07e35`
- blob `2d350379377ae65b2fb094504624da0591f53f7e`.

Dynamic production-verification plan:
- commit `3a560ad4c73adfd992035d4c650e406c050ed309`
- blob `a304f4bcf977c421aef7b7eb1a960579f0c2150c`.

Canonical Full Slate workflow blob:
- `.github/workflows/full-slate.yml`
- `41bacd4b756c32279169c80e0ff603a4780b7bee`.

Locked production semantics:
- availability resolves before opportunity;
- `ACTIVE_ROLES_CSV=data/roles_current_production_eligible_v1.csv` is current-role authority;
- Week1 PlayerForm/TeamForm use certified strict-prior wrappers;
- complete QB C2 state-context source remains 32 teams;
- current-output coverage follows exact certified eligible-team set;
- sportsbook acquisition/matching occurs only after football eligibility and cannot resurrect withheld games/players;
- no sportsbook input defines carries, targets, receptions, passing opportunity, roster eligibility, or roles.

## Production branch verification lineage

### Run1 — preserved mechanical static-audit failure

Parent:
- Run `34470549704`
- Job `102849180219`
- head `6e96c7db16ea3f5cf349d6cae006e122007bdbaf`
- evidence artifact `10149440197`
- digest `sha256:355a8eb8340bdc3281461457cef39a205ffaeedddbe0543b9a291cf1d3db38ae`.

Child Full Slate:
- Run `34470613780`
- Job `102849391150`
- artifact `10149436841`
- digest `sha256:62fbde8ff296d8df8fbe2c40b5fa3d04a281640281c53dea84be8d14a751f946`
- failed only at strict repository audit because the old audit required a direct literal invocation of `scripts/run_player_form_v2_loader.py`; certified production uses `scripts/run_player_form_current_roles_v1.py` as a strict-prior/current-role wrapper.

Run1 disposition: `CURRENT_PLAYER_AVAILABILITY_PRODUCTION_BRANCH_VERIFY_MECHANICAL_FAILURE_NO_DECISION`.

Frozen Run1 repair:
- plan commit `90a297ab94b75ffbe5da7d050928d01f9da60e1d`
- plan blob `9b7ffb85d6bfb0895abaca97d4d2642ae3a07288`
- static audit compatibility only; no football/availability/science semantic change.

### Run2 — preserved mechanical R22 disposition-name verifier failure

Parent:
- Run `34477957076`
- Job `102873279264`
- head `944426bae1a6d9292fded5dcf41f8951e7357467`
- artifact `10152500791`
- digest `sha256:0f24bbf30094aaf8e22133eb70669d7f4e29275faa0d9b29ae48c051e66d6c14`
- conclusion `FAILURE`.

Child Full Slate:
- Run `34478009290`
- artifact `10152451377`
- digest `sha256:dd7a2cace7a5ec014ad020d9959238b7896e372ad3b9c748c14e58ec077fdeb3`
- conclusion `SUCCESS`.

Exact failure: dynamic verifier accepted only obsolete R22 disposition literals, while production correctly emitted canonical `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_ADAPTER_PASS`. Child R22 audit simultaneously had `integration_valid=true`, sportsbook inputs `0`, future outcomes `0`, max mean delta `3.552713678800501e-15`, and all R22 gates true.

Frozen Run2 repair document:
- `docs/operations/CURRENT_PLAYER_AVAILABILITY_PRODUCTION_BRANCH_RUN2_R22_DISPOSITION_MECHANICAL_REPAIR_V1.md`
- repair-plan commit `538c984eb5c06a72e8b4423a65b564ff6af8d7f2`
- minimum implementation: add canonical current R22 adapter PASS literal to verifier accepted set; no threshold/model/tail change.

Run2 disposition: `CURRENT_PLAYER_AVAILABILITY_PRODUCTION_BRANCH_VERIFY_MECHANICAL_FAILURE_NO_DECISION`.

### Run3 — FIRST SUCCESSFUL LOCKED PRODUCTION-BRANCH VERIFICATION

Final relocked head:
- `f55b14b99844d5d4de9899db91bc9a7abdc15bf3`.

Parent verification:
- Run `34497510664`
- Job `102939616416`
- conclusion `SUCCESS`
- artifact `10160648036`
- digest `sha256:e1fbd76f5f5189383acea770ff3b669d22d40313c0042cb07c02fe85b92b41ed`.

Clean branch child Full Slate:
- Run `34497578081`
- conclusion `SUCCESS`
- artifact `10160594516`
- digest `sha256:656a2ede23db107002abf75dcb568ecf26afcabc987f3b0b1bbdeba1a568e7db`.

Run3 passed:
- frozen implementation boundary
- clean no-odds Full Slate
- current availability before opportunity
- exact eligible-team/current-role semantics
- strict-prior PlayerForm/TeamForm
- M89/M90 and QB C2 current output
- RB P3
- M38/WR-R15/TE-R5P conservation
- R22 mean neutrality
- R26 football-only usage
- zero definitive unavailable PlayerForm rows
- zero definitive unavailable simulation arrays
- sportsbook inputs to football/verification `0`
- strict repository audit
- 2026 production-readiness audit.

Terminal disposition:
`CURRENT_PLAYER_AVAILABILITY_PRODUCTION_BRANCH_VERIFY_PASS_READY_FOR_MAIN_PROMOTION`.

## Promotion to main — COMPLETE

Promotion PR:
- PR `#513`
- title `Promote certified current player availability to Full Slate production`
- exact verified branch head `f55b14b99844d5d4de9899db91bc9a7abdc15bf3`
- merged to main as `f813f85ed814cc7c231a459e2301170171b8ed10`.

This was not a force move; newer main documentation/checkpoint commits were preserved through the merge.

### Required first clean-main no-odds proof

- Run `34498365769`
- Job `102942531505`
- head `f813f85ed814cc7c231a459e2301170171b8ed10`
- conclusion `SUCCESS`
- artifact `10160866044`
- digest `sha256:89f36a3d1f100e4aefeb2167873485f271faf40b86d0ac2415b05da5f7b2e94b`.

Passed on clean main:
- availability before opportunity
- production-eligible current-role PlayerForm
- RB P3
- 32-team QB C2 state-context source
- availability-aware current-output seams
- strict repository audits.

Odds/pricing were correctly skipped because this was a no-odds production verification.

## Post-promotion CI cleanup — COMPLETE

First Repo CI after PR #513:
- Run `34498365855`
- Job `102942532249`
- compile PASS
- static repo audit PASS
- unit tests `205 passed, 1 skipped, 1 failed`.

Sole failure:
`tests/test_rb_pricing_adapter_v1.py::test_canonical_pricing_and_full_slate_are_wired_to_p3`

The test expected obsolete workflow label:
`Build promoted RB P3 football-only context`

Production now correctly uses:
`Build promoted RB P3 from production-eligible current roles`

Frozen minimum repair:
- branch `ops-post-promotion-ci-p3-contract-repair-v1`
- repair document commit `a9f1adce38a94b0ee1bc2f840fd3903221f1a28e`
- repair document `docs/operations/CURRENT_PLAYER_AVAILABILITY_POST_PROMOTION_CI_P3_TEST_CONTRACT_REPAIR_V1.md`
- one-line test repair commit/head `b0a40ddda9c8d0d6a7e231c1ae420c4598a56f8e`
- diff from `f813f85...` contained exactly two files: frozen repair document + one test literal line; no runtime/model file changed.

Repair PR:
- PR `#514`
- PR Repo CI Run `34500788405`
- Job `102950759957`
- compile PASS
- static repo audit PASS
- unit tests PASS
- merged to main as `3e49e0e3a155c0fe74b134542eb359d6444cd8c8`.

Final main Repo CI:
- Run `34500901575`
- Job `102951129505`
- head `3e49e0e3a155c0fe74b134542eb359d6444cd8c8`
- conclusion `SUCCESS`
- compile PASS
- static repo audit PASS
- unit tests PASS.

Final post-repair main no-odds Full Slate:
- Run `34500901695`
- Job `102951133264`
- head `3e49e0e3a155c0fe74b134542eb359d6444cd8c8`
- conclusion `SUCCESS`
- artifact `10161915727`
- digest `sha256:238202b4f355fc86dd2da3ad340f4f7824d0401921b2fa0f4ff03a886dcda99b`
- availability-before-opportunity PASS
- production-eligible PlayerForm PASS
- RB P3 PASS
- QB C2 PASS
- availability-aware seams PASS
- strict repository audits PASS
- sportsbook/pricing steps correctly skipped.

## Availability lane terminal status

**CLOSED / PRODUCTION COMPLETE.**

Current Full Slate now has certified current-roster/current-availability plumbing on `main`, verified before and after promotion, with fail-closed T-75 semantics and sportsbook downstream only.

Do not reopen this lane or retune any availability thresholds merely because future slates look different. Normal future weekly operation should run the production path as designed. Any future operational bug must be preserved and repaired separately; any proposed semantic change requires a new frozen authority.

## Sealed R26 Week1 prospective obligation — PENDING FUTURE OUTCOMES

R26Q remains immutable; do not recompute:
- Run `34400524030`
- Job `102630996205`
- Artifact `10123251043`
- Digest `sha256:dd3ec0e8e3831ab7f2255c2e5abf343cda8a7943d33a1d4863e52372d6f858a1`
- disposition `R26Q_2026_WEEK1_RECEPTIONS_PROSPECTIVE_SEAL_PASS_READY_FOR_OBSERVATION`
- NPZ SHA `7933bd7629d0e7108fe181e7a5474f8c8f6e8512b6a1b35081d3a8d71af8fe06`.

R26S exact evaluator remains frozen. Once authoritative Week1 outcomes are complete/available, rerun the exact locked evaluator unchanged. Do not rebuild R26Q from hindsight.

Canonical R26S pregame dry run:
- Run `34411889262`
- Job `102667966181`
- Artifact `10127562840`
- digest `sha256:e9ecd949816eafd813a50e60e8af28ce387e668848483ffcda3dfadf7614919e`
- disposition `R26S_2026_WEEK1_POSTGAME_PROSPECTIVE_EVALUATION_INCOMPLETE_NO_DECISION` because outcomes were unavailable at the time.

## Exact next research action

Availability is no longer the active research lane.

Unless the user explicitly chooses a different roadmap item, the next active model-development lane is **QB opportunity/efficiency**, building on the protected M89/M90 + mean-neutral C2 production stack. The research question should focus on actual football prediction: attempts/dropbacks/pass rate/YPA/sacks/scrambles and their strict-pregame context, while preserving the existing QB production authority until a separately frozen candidate qualifies.

Before implementing a new QB candidate:
1. audit existing QB migrations/research so we do not reinvent prior work;
2. identify unresolved error decomposition/opportunity/efficiency gaps already demonstrated by repo evidence;
3. freeze the next scientific question, comparators, features, walk-forward rules, cohorts and gates before results;
4. keep sportsbook completely downstream of football prediction and use market only as an external benchmark where appropriate;
5. preserve all first valid results, pass or fail.

R26S prospective grading remains a parallel future obligation as soon as complete authoritative Week1 outcomes exist; it does not require pausing QB research while outcomes are incomplete.

## Remaining roadmap

1. QB opportunity/efficiency — next active research lane unless user redirects.
2. Grade sealed R26Q with exact locked R26S once authoritative Week1 outcomes are complete.
3. Selective unresolved WR/TE opportunity/efficiency/distribution while preserving M38/WR-R15 and TE-R5P.
4. Shared QB↔receiver conservation.
5. Unified game simulation.
6. Anytime TD modeling.
7. Game ML/spread/total from football simulation rather than sportsbook imitation.
8. Final operational package and prospective grading.

Long-term architecture remains a coherent NFL game model with conservation/context rather than siloed prop predictors. Primary target is actual football outcomes; Vegas is a benchmark/opponent, never the source of football truth.
