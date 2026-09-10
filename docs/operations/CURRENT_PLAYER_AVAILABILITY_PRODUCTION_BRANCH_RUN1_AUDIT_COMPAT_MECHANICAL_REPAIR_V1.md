# Current Player Availability — Production Branch Run1 Audit Compatibility Mechanical Repair V1

Status: `FROZEN_BEFORE_REPAIR_IMPLEMENTATION_AND_RETRY`

## Preserved failed verification authority

Parent production-verification launcher:
- run `34470549704`
- job `102849180219`
- head `6e96c7db16ea3f5cf349d6cae006e122007bdbaf`
- parent evidence artifact `10149440197`
- digest `sha256:355a8eb8340bdc3281461457cef39a205ffaeedddbe0543b9a291cf1d3db38ae`

Canonical child Full Slate:
- run `34470613780`
- job `102849391150`
- head `6e96c7db16ea3f5cf349d6cae006e122007bdbaf`
- artifact `10149436841`
- digest `sha256:62fbde8ff296d8df8fbe2c40b5fa3d04a281640281c53dea84be8d14a751f946`
- live odds disabled
- all football-building stages through availability-aware current-output seams passed
- failure occurred at `Strict repository audits`
- no main promotion occurred

Disposition for this run remains `CURRENT_PLAYER_AVAILABILITY_PRODUCTION_BRANCH_VERIFY_MECHANICAL_FAILURE_NO_DECISION` pending repaired verification.

## Reproduced root cause

Both static production audits still require the canonical Full Slate workflow to contain the literal direct invocation token `scripts/run_player_form_v2_loader.py`.

The certified availability-aware Full Slate intentionally invokes `scripts/run_player_form_current_roles_v1.py` instead. That wrapper:
- resolves the explicit certified current-role file;
- imports `scripts.run_player_form_v2_loader` as the protected PlayerForm authority;
- calls `loader.main()` unchanged for the underlying PlayerForm computation;
- republishes only strict-prior legal history after the protected loader completes;
- does not change PlayerForm identity/history/blend formulas.

Therefore the old literal workflow-token requirement is stale after the certified operational wrapper was introduced. This is an audit-contract compatibility defect, not authorization to change football semantics.

## Frozen minimum repair

Only these audit expectations may change:
- `scripts/utils/audit_repo.py`
- `scripts/audit_2026_production_readiness.py`

The repaired audits must accept the certified workflow wrapper `scripts/run_player_form_current_roles_v1.py` in place of a direct workflow call to `scripts/run_player_form_v2_loader.py` only if the wrapper itself is present and proves all of the following static contract tokens:
- import/delegation to `scripts.run_player_form_v2_loader`;
- `loader.main()` execution;
- explicit current-role resolution via `resolve_current_roles_path`;
- strict-prior history publication via `strict_prior_logs` / `publish_strict_prior_history`;
- an explicit guard rejecting target-week-or-future active-season rows.

The underlying `scripts/run_player_form_v2_loader.py` remains a protected/material production dependency and must continue to be audited for its existing identity/v3 contracts. The repair may not remove that dependency from production-runtime presence/literal checks.

## Prohibited changes

This repair may NOT change:
- `.github/workflows/full-slate.yml` football semantics or ordering;
- availability source precedence or T-75 timing;
- current-role eligibility/re-ranking semantics;
- M89/M90/C2, M38, WR-R15, TE-R5P, P3, R26 or R22 science;
- any model artifact or frozen historical research result;
- sportsbook ordering or sportsbook-to-football separation;
- the immutable 35/35 certification evidence or gates.

## Verification requirement

After implementing this exact audit-only repair:
1. compile the two repaired audit scripts;
2. run both strict audits against the promotion branch;
3. create a repaired production implementation lock pinning the two new audit blobs in addition to every previously locked production blob;
4. rerun the same dedicated no-odds production-branch verification;
5. preserve the first repaired verification result exactly.

Only `CURRENT_PLAYER_AVAILABILITY_PRODUCTION_BRANCH_VERIFY_PASS_READY_FOR_MAIN_PROMOTION` may authorize promotion to `main`.
