# RB R21 Phase B — Pre-Outcome Implementation Record

Date: 2026-09-08/09 UTC
Branch: `research-cross-position-catastrophic-casebook-v1`
Frozen governing plan: `docs/migrations/RB_R21_2026_PROSPECTIVE_TAIL_FORECAST_LOCK_AND_GRADE_V1_PLAN.md`
Frozen plan commit: `cdd503919e56860c6f17df12e01f4f34072d888c`
Week-1 forecast-lock run: `34294227588`
Week-1 forecast-lock artifact: `10082525892`
Week-1 forecast-lock digest: `sha256:302df98f83c461d8abd74da9985dacc80c9477596b8bba85cc4d4a27c4fbc6f3`

## Purpose

This record proves that the Week-1 outcome-grading implementation was written and committed before the first 2026 regular-season outcome. The grading implementation therefore cannot be retroactively designed around Week-1 results without creating a new, visible post-outcome version/commit.

## Grader implementation

File: `scripts/backtest/grade_rb_r21_week1_prospective_tail_v1.py`
Commit: `dfd74e186043f349255e8ce010c5d798cda2e803`
Commit message: `Implement pre-outcome RB R21 Week 1 grader`

The grader:

- consumes the sealed R21 CONTROL/SHADOW draw matrices only;
- pins the Week-1 lock run/artifact/digest/head and exact CONTROL/SHADOW/ledger hashes;
- never regenerates Week-1 forecasts;
- waits for all 16 Week-1 regular-season games to be final;
- loads governed weekly player statistics and snap counts through `nflreadpy`;
- resolves player outcomes fail-closed using locked team/player identity;
- requires at least 90% locked-row outcome coverage;
- computes the frozen CRPS, Brier30, Brier50, q90 pinball, q95 pinball, 80% coverage and 90% coverage comparisons;
- applies the already-frozen Week-1 R21 support gates without changing them;
- writes auditable source snapshots, match audit, player casebook, metrics, and result JSON;
- adds no sportsbook input upstream and changes no production parameter.

Week-1 PASS disposition is `RB_R21_WEEK1_PROSPECTIVE_GRADE_PASS_CONTINUE_SHADOW`.

Week-1 FAIL disposition is `RB_R21_WEEK1_PROSPECTIVE_GRADE_FAIL_DIAGNOSE`.

Neither disposition promotes the tail adapter into production.

## Manual grading workflow

Workflow: `.github/workflows/research-rb-r21-week1-prospective-tail-grade-v1.yml`
Workflow commit: `3312bd7c5d509568b6eccb52758c82d14a9c64bb`
Commit message: `Add pre-outcome RB R21 Week 1 grading workflow`

The workflow is intentionally `workflow_dispatch` only. It does not auto-run on push, schedule, or any data update. This prevents an accidental premature grade before Week-1 games are final.

At dispatch it:

1. downloads the immutable R21 Week-1 forecast-lock artifact from run `34294227588`;
2. captures GitHub artifact metadata;
3. verifies the sealed lock result, ledger and draw archive are present;
4. executes the frozen pre-outcome grader;
5. uploads the complete Week-1 grading evidence artifact.

## Governance boundary

Do not modify the frozen plan or Week-1 gates after outcomes are visible.

If the committed grader contains a true mechanical defect discovered after outcomes are visible, document the defect explicitly. Do not silently repair it and call the repaired grade the original prospective test. Any scientifically material rule/identity/scoring change must be versioned and its evidence status stated clearly.

R21 Week-1 remains SHADOW evidence only. Production eligibility still requires the frozen cumulative evidence floor (4 completed weeks, 250 eligible locked RB player-games, 15 observed 30+ events, 5 observed 50+ events) and then a separate governed production-promotion decision.
