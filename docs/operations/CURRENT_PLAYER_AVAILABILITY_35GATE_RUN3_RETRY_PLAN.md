# Current Player Availability 35-Gate — Run3 Retry Plan

Status: `FROZEN_BEFORE_RUN3_WORKFLOW_LOCK / MECHANICAL RETRY ONLY`

## Preserved prior executions

Run1: `34459655725` / Job `102814178762` — `MECHANICAL_FAILURE_NO_DECISION`, zero gates evaluated.

Run2: `34460227422` / Job `102816052394` — `MECHANICAL_FAILURE_NO_DECISION`, zero gates evaluated. Exact exception: `QB C2 production adapter did not resolve exactly one primary QB per team`.

Immutable candidate remains Artifact `10140425929`, digest `sha256:6b31ae40d648780673b7953b57323d509b302ea7a82fe99801b024ede2603f37`.

## Canonical Run2 repair authority

Use the earlier already-frozen Run2 repair lineage as authority:
- repair plan commit `2518f8366a8acbad5eb9d91de6134a1923a2381b`
- protected QB C2 blob `7b677470b27b6776055c75c924a0ddf22d724a44`
- Run1 starter-audit transformer blob `c7569c54cda779eb04bed7dbf2b22b9ec4fb526b`
- Run2 primary-frame transformer blob `fbb7d34b54aefe98e95d8c097c7542c7d6490b52`
- shared helper blob `77b591e431378ec984c51e8a032262e673d4c843`
- canonical sequential regression Run `34460546690` — SUCCESS.

A later duplicate regression (`34461154452`) also passed but is non-authoritative because the earlier frozen lineage already existed.

## Frozen Run3 change versus Run2 workflow

Create a separate Run3 workflow rather than modifying the preserved Run1/Run2 workflow. It must retain the same:
- immutable candidate and digest;
- football-stack runner;
- fixture builder;
- gates 1-34 evaluator;
- gate-35 finalizer;
- full-universe/R26 seam;
- first QB starter-audit seam;
- baseline/fixture order and evidence definitions;
- no-odds / sportsbook-zero boundary.

The ONLY Run3 mechanical addition is:
1. verify canonical Run2 transformer blob `fbb7d34b54aefe98e95d8c097c7542c7d6490b52`;
2. apply `scripts/operations/apply_current_availability_qb_c2_primary_team_seam_v1.py` immediately after `apply_current_availability_qb_c2_eligible_team_seam_v1.py` in baseline workspace setup and each isolated fixture worktree;
3. compile the transformed QB C2 adapter.

No gate definition, fixture target, model parameter, model asset, role logic, starter-selection logic, T-75 logic, R26/R22/P3, M38/WR-R15/TE-R5P, QB C2 science or sportsbook boundary may change.

Run3 remains eligible to become the first valid immutable 35-gate result because Run1 and Run2 both stopped before `Evaluate frozen gates 1 through 34`.
