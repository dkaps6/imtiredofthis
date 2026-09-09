# RB R26O Run 1 Mechanical Repair V1

## Scope

This note records the first mechanical execution failure of the frozen R26O 2026 Week-1 receptions-only shadow-integration compatibility study.

The frozen R26O scientific plan, evaluator, 38 gates, Monte Carlo tolerance, parent artifacts, and authority ceiling are unchanged.

## Failed execution

- Workflow run: `34397293311`
- Job: `102620027890`
- Head: `9c207635db8ded30076b9237e99a4fb6f77437ee`
- Failure classification: **MECHANICAL / NO SCIENTIFIC DISPOSITION**

All upstream safeguards passed before the crash:

- frozen plan / evaluator verification
- protected production boundary
- immutable R26N artifact digest
- immutable post-merge Full Slate artifact digest and head
- immutable R22 artifact digest
- 468-row exact production identity staging

The evaluator then entered the exact current V3 production simulation and failed before any R26O structural gate could be evaluated.

## Exact blocker

The staged Full Slate artifact root did not contain the repository-committed QB C2 selector authority:

`model/qb_distribution_state_selector_v1.json`

`full_v3._simulate_promoted_stack(...)` calls the protected QB C2 production selector, whose loader resolves that file relative to the active production root. The Actions Full Slate artifact persists runtime `data/` assets but does not persist this repo-committed `model/` file. Therefore the isolated staged production root was incomplete even though the production code itself was unchanged.

Traceback terminus:

`FileNotFoundError: [Errno 2] No such file or directory: 'model/qb_distribution_state_selector_v1.json'`

## Authorized repair

Before rerunning the identical frozen evaluator, stage **only** the exact protected production-base selector file into the isolated production root:

- source commit: `f8417f55b04ce0e19baf260e9d532765034c47f1`
- source path: `model/qb_distribution_state_selector_v1.json`
- SHA-256: `04946438c0cdbcddc7ce4f95f88d37a582daf3affe9f04b1991149e7ce98c3b1`

The workflow must obtain/verify that exact file from the protected production base, copy it to `staged_production/model/qb_distribution_state_selector_v1.json`, and verify the staged SHA-256 before execution.

No other model file, data row, football value, parent artifact, R26N candidate value, R22 asset, R26O gate, threshold, seed, iteration count, or evaluator line may change.

## Authority

This repair does not create a new R26O candidate and does not count as a scientific retry. It completes the isolated production staging environment required to execute the already-frozen study.
