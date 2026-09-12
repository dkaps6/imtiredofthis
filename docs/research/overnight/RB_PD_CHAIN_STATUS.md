# RB PD Chain Status (PD2 -> PD3 -> PD4 -> PD5 -> PD6)

**STATUS: RESEARCH ONLY — NOT PROMOTED.**

This document recovers and reconciles the RB player-error-persistence / residual-calibration
research chain from GitHub Actions run evidence and branch history. No production file, model
weight, or threshold was touched to produce it. No local compute was run; all dispositions below
are quoted from CI job logs (`mcp__github__get_job_logs`, `return_content: true`) or from frozen
docs committed to the branches.

Branches involved (chronological by commit, all on 2026-09-07, `dkaps6/imtiredofthis`):
- `origin/research-rb-pd2-player-error-persistence`
- `origin/research-rb-pd3-player-residual-calibration` (also carries the PD4 experiment — see below)
- `origin/research-rb-pd5-carry-only-residual-calibration`
- `origin/research-rb-pd6-positive-entitlement-residual-calibration`

**Naming-drift warning confirmed**: `.github/workflows/backtest-rb-pd2-player-error-persistence.yml`
and `docs/migrations/RB_PD2_PLAYER_ERROR_PERSISTENCE_PLAN.md` are reused/overwritten in place across
PD2, PD3, and PD4 — the file names do not track the experiment number. PD3 and PD4 both live on the
`research-rb-pd3-player-residual-calibration` branch, sequentially, as separate commits that rewrite
the same plan file and the same workflow file's `name:` field.

## What PD2 authorized

Source: `docs/migrations/RB_PD2_PLAYER_ERROR_PERSISTENCE_RESULT.md` @
`origin/research-rb-pd2-player-error-persistence` (final commit `a4173a7`).

Disposition: `RB_PLAYER_ERROR_PERSISTENCE_DETECTED`. All four frozen carry/yard persistence
diagnostics passed on strict walk-forward history (last 8 strictly-prior same-player games, minimum
4; 882 scoreable rows of 1,393 canonical 2025 rows; 148 players). Run `34064637295`, job
`101571138337`.

The doc's exact authorized next step (quoted in full):

> The next legitimate step is a separately frozen full-stack calibration test with conservative
> shrinkage/minimum-sample protection:
> - prior carry bias as a candidate carry-mean calibration input;
> - prior yard bias only at the natural efficiency/yardage layer after carry effects are accounted
>   for;
> - prior carry/yard difficulty as uncertainty/MC-width calibration rather than a blind mean offset.
>
> No player-specific correction is promoted from this diagnostic alone.

Note: nothing downstream (PD3/PD4/PD5/PD6) implemented the third bullet (difficulty -> MC-width /
uncertainty calibration). All four downstream experiments only ever touched carry-mean and
yard-efficiency-mean calibration, never the uncertainty-calibration leg PD2 authorized.

## PD3/PD5/PD6 walkthrough

### PD3 — Player Residual Calibration (carry-mean + yard-efficiency-mean, both directions)

Branch `research-rb-pd3-player-residual-calibration`, commits `6b5a791` (plan freeze) ->
`aa1b735` (evaluator) -> `fc24d34` (launch). Candidate: carry correction
`clip(0.25 * prior8_carry_bias, -2, +2)`; yard correction from prior8 efficiency-residual error,
capped at +/-8 yards. Implements PD2's first two bullets exactly; still a two-sided (up and down)
correction.

Actually executed: run `34089486097`, job `101639902906`, completed 2026-09-07 06:08 UTC, CI
conclusion `success`. Real disposition from job log:

```
"disposition": "RB_PD3_PLAYER_RESIDUAL_CALIBRATION_FAIL"
scoreable_rows: 882
scientific_gates: 11 of 12 true; only "yard_p90_not_worse": false
```

Pooled yard p90 absolute error moved from 48.783 (baseline) to 50.514 (candidate) — the sole
failing gate. Carry MAE improved (3.706 -> 3.617), yard MAE improved (21.833 -> 21.381), all other
guards (early/late season, top-carry-quartile, miss-rate guards) passed. This exact FAIL disposition
and run/job ID are also recorded inline in the PD4 plan's "Lineage" section
(`docs/migrations/RB_PD2_PLAYER_ERROR_PERSISTENCE_PLAN.md` @ PD3 tip), confirming the doc and the
live CI log agree.

**This PD3 result was never written up as its own doc** — no `RB_PD3_..._RESULT.md` exists on any
branch. It is written up here for the first time from the job log, exactly as done for QB-PD3/TE-R1
earlier tonight.

### PD4 — Role-Stability-Gated Calibration (same branch, later same day)

Still on `research-rb-pd3-player-residual-calibration`: commit `44a732c` ("Freeze PD3 branch before
PD4 experiment") -> `d37905a` (PD4 plan, overwrites the same `RB_PD2_..._PLAN.md` file) ->
`b6a4e5e` (evaluator) -> `5be73d0` (launch). Candidate: apply the exact frozen PD3 correction only
to player-games where a 5-condition pregame "stable role" gate is met (>=4 prior games, last game
within 2 weeks, mean carries >=8.0, CV<=0.35, baseline within 4.0 carries of recent mean); PD3
correction otherwise never applied (candidate = baseline).

Actually executed: run `34098419744`, job `101667112210`, completed 2026-09-07 08:02 UTC, CI
conclusion `success`. Real disposition from job log:

```
"disposition": "RB_PD4_ROLE_STABILITY_GATED_CALIBRATION_FAIL"
scoreable_rows: 882, stable_role_rows: 258
scientific_gates: 12 of 14 true; failing:
  "carry_mae_improve_ge_0_03": false
  "late_yard_p90_not_worse": false   (and "yard_p90_not_worse": false, pooled)
```

Stable-role subset MAE did improve for both carries and yards (per
`stable_carry_mae_better`/`stable_yard_mae_better` = true), and top-carry-quartile guards held, but
the pooled carry-MAE-improvement threshold (>=0.03) was missed and both the pooled and Weeks 13-18
rushing-yard p90 guards still failed. **Also never written up as its own doc** — recovered here from
the job log for the first time.

### PD5 — Carry-Only Residual Calibration

Branches from the PD4 tip (`5be73d0`) into `research-rb-pd5-carry-only-residual-calibration`:
`3a04228` (plan) -> `4a8169e` (evaluator) -> `b70ed8b` (launch). Hypothesis, stated in the plan
(`docs/migrations/RB_PD5_CARRY_ONLY_RESIDUAL_CALIBRATION_PLAN.md`): the yard-efficiency-residual leg
is the likely source of PD3/PD4's persistent yard-tail failure, so drop it entirely — carry-only
correction, rushing yards recomputed from corrected carries via baseline YPC, no independent yard
correction. Plan explicitly (and, per the PD6 branch's discrepancy doc, **inaccurately**) says to
use "the same canonical 2020-2025 RB evidence ... as PD3/PD4."

Actually executed: run `34109390989`, job `101701846765`, completed 2026-09-07 10:04 UTC, CI
conclusion `success`. Real disposition from job log:

```
"disposition": "RB_PD5_CARRY_ONLY_RESIDUAL_CALIBRATION_FAIL"
scoreable_rows: 1393, eligible_rows: 882
scientific_gates: 11 of 12 true; only "yard_p90_not_worse": false
```

This fixed the PD3/PD4 late-season yard-p90 problem (`late_yard_p90_not_worse: true`) and improved
carry MAE, yard MAE, bias, and catastrophic-miss rates across the board, but the **pooled eligible**
rushing-yard p90 still narrowly worsened (48.783 -> 49.098 among the 882-row eligible subset — see
below). Also never written up as its own PD5 `_RESULT.md`.

**The cohort-execution discrepancy** (`docs/migrations/RB_PD5_COHORT_EXECUTION_DISCREPANCY.md`,
committed `2a774c7`, 08:02 UTC on `research-rb-pd6-positive-entitlement-residual-calibration`,
2 minutes after the PD6 plan itself was frozen): the PD5 plan's prose claimed a 2020-2025 evaluation
cohort, but the evaluator and workflow (`EXPECTED_ROWS = 1393`, `season == 2025` filter, downloads
of `stack1_2025_rb_trace.csv` / 2025 STACK2 casebook) only ever ran 2025. The actual run is a valid
execution of the inherited 2025 PD3/PD4 cohort but not of the literal text in the frozen PD5 plan.
The doc treats the 2025 result as real exploratory evidence but explicitly **not** a valid
multi-season confirmation, and lays out five numbered remediation steps (verbatim):

> 1. Preserve RB P3 production unchanged.
> 2. Do not relax or rewrite PD5 gates.
> 3. Do not launch RB-PD6 against the same 2025 cohort as a purported independent confirmation
>    after seeing PD5 results.
> 4. Build or locate a canonical multi-season RB P3-equivalent evidence set using seasons outside
>    the already-observed 2025 cohort.
> 5. Freeze a separate replication/confirmation plan before evaluating those unseen seasons.
> 6. Only after that replication is dispositioned may PD6 or another residual mechanism advance as
>    a confirmatory candidate.

### PD6 — Positive-Entitlement Residual Calibration: **frozen, never implemented, never launched**

Branches from the PD5 tip (`b70ed8b`) into
`research-rb-pd6-positive-entitlement-residual-calibration`. Only two commits exist beyond PD5:

1. `48e01fe` (08:00:48 UTC) — freezes
   `docs/migrations/RB_PD6_POSITIVE_ENTITLEMENT_RESIDUAL_CALIBRATION_PLAN.md`. Hypothesis: make the
   PD5 carry correction one-sided — only raise projected carries when `prior8_carry_bias < 0`
   (history says the model underprojected), never lower them for positive bias. Plan text again
   says "the exact same canonical 2020-2025 RB evidence... as PD5" — i.e. it inherits the exact same
   inaccurate cohort description PD5 had.
2. `2a774c7` (08:02:13 UTC, **2 minutes later**) — adds the cohort-discrepancy doc above, which
   explicitly forbids launching PD6 against the 2025 cohort.

There is **no** "Implement frozen RB-PD6 evaluator" commit and **no** "Launch frozen RB-PD6"
commit on this branch — confirmed by `git diff` between the PD5 tip and the PD6 tip on both
`scripts/` and `.github/workflows/`, which is empty. The evaluator script and the workflow file on
the PD6 branch are byte-identical to PD5's; they implement PD5's carry-only (two-sided) logic, not
PD6's one-sided positive-entitlement logic described in the plan. `mcp__github__actions_list`
(`list_workflow_runs`, branch=`research-rb-pd6-positive-entitlement-residual-calibration`) returns
**`total_count: 0`** — this branch has never triggered any workflow run, successful or otherwise.
There is no completed-but-unwritten PD6 result to recover; PD6 genuinely never ran.

## Current status

**Genuinely blocked — not resolved, and there is no ungated next step ready to run today.**

- PD3, PD4, and PD5 all reached real `..._FAIL` dispositions (confirmed from live CI job logs, not
  inferred). Production (RB P3) is untouched by all three; each disposition's own JSON confirms
  `"production_changed": false`.
- PD5's fix for the earlier late-season yard-p90 failure is real, but PD5 itself still failed on
  pooled eligible yard p90, and its own plan's cohort description is inaccurate for what actually
  ran (2025-only, not 2020-2025) per the PD6-branch discrepancy doc.
- The discrepancy doc's own rule 3 explicitly forbids launching PD6 on the 2025 cohort as an
  "independent confirmation," and no PD6 evaluator/workflow update exists to run anything else. The
  PD6 plan document itself still carries the same unresolved "2020-2025" cohort description that
  triggered the block in the first place — the plan was frozen and then immediately (2 minutes
  later, same author) followed by the block, and nothing since has revised the PD6 plan or built the
  required multi-season evidence.
- Remediation steps 4-6 from the discrepancy doc (build/locate a multi-season RB P3-equivalent
  evidence set outside the 2025 cohort, freeze a replication plan against it, disposition that
  replication) have **not** been done on any of these four branches. I did not locate a completed,
  dispositioned multi-season RB replication elsewhere in the repository's history in the course of
  this investigation, though I did not exhaustively audit the full RB experiment history (M91-M96,
  RB-STACK1-6, RB-ND1/2, RB-ENV1, etc.) for a pre-existing qualifying multi-season baseline dataset —
  that audit, and the decision of whether any existing dataset satisfies remediation step 4, is a
  substantive judgment call left to the repo owner.
- There is no dispatchable, already-frozen workflow that would correctly test the PD6 hypothesis:
  the only workflow_dispatch-enabled RB-PD workflow reachable from the PD6 branch
  (`backtest-rb-pd2-player-error-persistence.yml`) is byte-identical to PD5's, still targets the
  2025-only cohort, and does not implement PD6's one-sided candidate logic. Dispatching it would
  either (a) silently re-run PD5's already-dispositioned two-sided experiment under a misleading
  PD6 branch label, or (b) if treated as a PD6 confirmation, violate the discrepancy doc's rule 3
  outright. **I did not dispatch it**, per the task's explicit instruction to hold when there is any
  doubt about whether dispatching is safe/correct.

## Recommendation

This needs the repo owner's judgment call — it is not a mechanical "run the frozen workflow" case
like QB-PD3/TE-R1 were.

1. Do not promote PD3, PD4, or PD5 to production integration testing — all three are dispositioned
   FAIL on their own frozen gates. If the two now-unwritten results (PD3, PD4) are wanted as formal
   docs, this file plus the quoted job-log JSON above is sufficient source material to draft
   `RB_PD3_..._RESULT.md` and `RB_PD4_..._RESULT.md` in the normal docs/migrations location — that
   authoring step was intentionally left to the repo owner rather than done here, since
   `docs/migrations/` is off-limits on this working branch.
2. Before any further RB residual-calibration experiment (PD6 or otherwise) is launched, the owner
   needs to decide how to satisfy the discrepancy doc's remediation steps 4-6: either locate an
   existing multi-season (non-2025) RB P3-equivalent evidence set already produced somewhere in this
   repo's extensive RB experiment history, or commission a new one, then freeze a dedicated
   replication plan against it and disposition that replication first.
3. Only after that replication is dispositioned does the discrepancy doc authorize PD6 (or another
   mechanism) to advance. At that point PD6's evaluator/workflow still need to be implemented
   (currently they are just PD5's files under a new branch name) before it can be meaningfully
   launched at all.
4. Separately, note that PD2's third authorized leg — prior carry/yard difficulty as MC-width /
   uncertainty calibration — was never attempted in PD3-PD6. That remains open scientific real
   estate PD2 explicitly authorized and nothing since has touched.
