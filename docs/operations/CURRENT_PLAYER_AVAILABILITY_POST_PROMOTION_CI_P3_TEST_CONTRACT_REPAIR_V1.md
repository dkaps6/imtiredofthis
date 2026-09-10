# Current Player Availability — Post-Promotion CI P3 Test-Contract Repair V1

Status: `FROZEN_BEFORE_TEST_CONTRACT_REPAIR_IMPLEMENTATION`

## Preserved production authority

Availability production promotion merged to `main` at:
- PR `#513`
- merge commit `f813f85ed814cc7c231a459e2301170171b8ed10`
- verified branch head `f55b14b99844d5d4de9899db91bc9a7abdc15bf3`.

First successful locked production-branch verification:
- parent run/job `34497510664` / `102939616416`
- parent artifact `10160648036`
- parent digest `sha256:e1fbd76f5f5189383acea770ff3b669d22d40313c0042cb07c02fe85b92b41ed`
- child Full Slate run `34497578081`
- child artifact `10160594516`
- child digest `sha256:656a2ede23db107002abf75dcb568ecf26afcabc987f3b0b1bbdeba1a568e7db`
- disposition `CURRENT_PLAYER_AVAILABILITY_PRODUCTION_BRANCH_VERIFY_PASS_READY_FOR_MAIN_PROMOTION`.

Clean-main no-odds Full Slate:
- run/job `34498365769` / `102942531505`
- head `f813f85ed814cc7c231a459e2301170171b8ed10`
- artifact `10160866044`
- digest `sha256:89f36a3d1f100e4aefeb2167873485f271faf40b86d0ac2415b05da5f7b2e94b`
- conclusion `SUCCESS`
- availability before opportunity PASS
- PlayerForm from production-eligible current roles PASS
- promoted RB P3 current-role step PASS
- QB C2 state-context PASS
- availability-aware current-output seams PASS
- strict repository audits PASS.

## Preserved CI failure

Repo CI run/job:
- run `34498365855`
- job `102942532249`
- head `f813f85ed814cc7c231a459e2301170171b8ed10`
- compile production modules PASS
- static repo audit PASS
- unit tests: `205 passed, 1 skipped, 1 failed`.

The sole failure is:
`tests/test_rb_pricing_adapter_v1.py::test_canonical_pricing_and_full_slate_are_wired_to_p3`.

The failing assertion expects the historical Full Slate step label:
`Build promoted RB P3 football-only context`.

The promoted availability-aware Full Slate intentionally and successfully executes the renamed step:
`Build promoted RB P3 from production-eligible current roles`.

The clean-main Full Slate run proves that this renamed step executes successfully and the remaining P3 wiring assertions in the same unit test continue to pass. This is therefore a stale literal test-contract failure, not a P3 model, pricing, availability, opportunity, or semantic failure.

## Frozen minimum repair

Authorized change:
- change exactly the stale workflow-label assertion in `tests/test_rb_pricing_adapter_v1.py` from `Build promoted RB P3 football-only context` to `Build promoted RB P3 from production-eligible current roles`.

No production workflow, model, pricing, availability, role, T-75, eligible-team, sportsbook ordering, R26, R22, P3 science, QB C2 science, WR/TE science, trained artifact, or historical result may change under this repair.

## Acceptance

A valid repair requires:
1. branch diff from `f813f85ed814cc7c231a459e2301170171b8ed10` contains only this frozen repair document and the one-line test literal change;
2. Repo CI on the repair PR passes compile, static repo audit, and the full unit-test suite;
3. after merge, Repo CI on `main` is green;
4. the canonical handoff is updated with the full Run1/Run2/Run3/promotion/clean-main/CI lineage.

This repair does not reopen availability science or any RB scientific lane.