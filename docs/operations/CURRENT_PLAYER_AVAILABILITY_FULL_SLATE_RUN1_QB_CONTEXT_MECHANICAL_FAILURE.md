# Current Player Availability Full Slate — Run 1 Mechanical Failure

Status: `MECHANICAL_FAILURE_BEFORE_AVAILABILITY_EXECUTION_NO_INTEGRATION_RESULT`

## Immutable failed lineage

- candidate branch: `ops-current-player-availability-full-slate-v1`
- locked candidate head: `9147bde6d08c05249415ec4cd364db482d812a35`
- first lock-triggered run: `34439714153`
- job: `102752015236`
- artifact: `10137497418`
- artifact digest: `sha256:0f3f5195fce8bc6514804b411e0b312475a379cfeb722eb5f5280237f8ff50f2`
- failed step: `Build team context and promoted QB context`

The artifact proves that Sharp inputs and `data/team_form.csv` were built before failure, while `data/qb_promoted_team_context.csv` was not produced. Therefore the failure occurred in the promoted QB-context command after team-form construction and before any current-availability, PlayerForm, RB P3, C2, static-audit, or 35-gate integration execution.

This run is not a scientific/integration FAIL and must not be scored against the frozen availability design.

## Value-neutral repair

Capture stdout/stderr from the existing `scripts/run_qb_promoted_context.py --season 2026 --prior-season 2025` command into an artifact log while preserving shell `pipefail`. No model code, feature semantics, source hierarchy, T-75 rule, role logic, availability logic, production code, R22/R26 behavior, or frozen integration gate changes are authorized.

The repaired run remains pre-integration plumbing evidence until the no-odds Full Slate reaches substantive completion.
