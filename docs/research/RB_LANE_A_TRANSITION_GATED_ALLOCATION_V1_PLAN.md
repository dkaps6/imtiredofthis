# RB Lane A — Transition-Gated Carry-Share Reallocation V1 (Gate 0 + Frozen Plan)

**FROZEN BEFORE ANY CANDIDATE RESULT. RESEARCH ONLY. NO PRODUCTION CHANGE. NOTHING IN
THIS DOCUMENT HAS BEEN RUN.**

Per Issue #535 (GPT-5.6, comment `5690752971`): a plan-only Gate-0/frozen-plan draft
for the transition-gated allocation candidate identified in the Lane A audit (comment
`5690735640`). Nothing here authorizes building or running a candidate -- this is the
plan to be adversarially reviewed first, matching the discipline #562's own plan/
Amendment docs went through before implementation.

## Why this candidate, and why it is not a disguised STACK2 retest

The Lane A audit established: STACK2 (`scripts/backtest/evaluate_rb_stack2_enriched_
allocation.py`, **on main**) already pooled rolling carry/snap share, room composition,
depth rank, injury/practice status, backfield HHI concentration, and tenure/draft
context into one HistGradientBoosting regressor fit on 2024 and scored on 2025. That
model became P3's Weeks-2-18 carry/allocation half, and `RB_FINAL_QUALIFICATION_
RESULTS.md` shows the resulting P3 formula passing 6/7 gates while failing the
high-workload/ceiling gate specifically: at actual carries>=20 (n=98) P3 MAE 43.741 vs
M94C baseline 40.005 (worse by 3.736 yd); at actual carries>=25 (n=24) P3 57.365 vs
49.310; at actual rushing yards>=100 (n=95) P3 72.096 vs 66.440 (worse by 5.656 yd) --
both models underproject every member of this slice.

STACK2 and its STACK6B follow-on (`RB_STACK6B_COMPACT_ROLE_MODEL_RESULTS.md`) both fit
**one pooled model across every team-week**, transition and stable alike. Neither ever
isolated the discrete-transition population as its own estimation problem, despite
ND1's own Shapley forensic atlas (`RB_ND1_FORENSIC_FAILURE_ATLAS_RESULTS.md`) showing
"role-collapse" (n=59, MAE 32.59) and "new-role initialization" (n=8, MAE 41.81) as the
highest-error-per-row classes in the entire atlas.

This candidate's novelty is **state segmentation**, not new raw data: split every
team-week into `transition` vs `stable`, leave `stable` weeks exactly on today's
existing baseline route, and apply a targeted reallocation rule **only** on `transition`
weeks. If this candidate cannot beat baseline specifically on the transition
subpopulation while leaving the stable subpopulation unregressed, it does not qualify --
there is no pooled-average rescue path.

## Gate 0 — depth-chart schema harmonizer (must pass before any candidate science)

The Lane A audit found `nflreadpy.load_depth_charts` returns a week-tagged schema for
2016-2024 (`season, week, club_code, depth_team, gsis_id, position, depth_position,
full_name, ...`) but an entirely different **date-stamped, non-week-tagged** ESPN
roster-snapshot schema for 2025+ (`dt, team, player_name, espn_id, gsis_id, pos_grp,
pos_slot, pos_rank`, 221 unique snapshot timestamps spanning 2025-08-03 through
2026-03-14). `scripts/backtest/historical_inputs.py::_weekly_depth_lookup` already
fails closed on this today -- it requires `{"season","week"}` in columns and returns an
**empty** frame otherwise, silently dropping to blank depth fields (exactly what ND2B's
own as-of-join workaround was built to avoid, but that workaround was never ported into
the general pipeline).

Before any candidate feature is computed, build one deterministic harmonizer:

1. **Historical side (2016-2024):** consume `load_depth_charts` natively; a team's
   depth-chart state for `(season, week)` is the native week-tagged row set, unchanged.
2. **Live side (2025+):** for a given team and a given game's kickoff timestamp
   `kickoff_utc`, the team's depth-chart state is the **most recent snapshot with
   `dt < kickoff_utc`** (matching ND2B's proven as-of convention, median snapshot age
   10.77h, p90 17.90h, 100% team-game pregame coverage in ND2B's own audit). No
   snapshot with `dt >= kickoff_utc` may ever enter a scored row for that game --
   enforced as a hard assertion, not a convention.
3. **Semantic parity proof:** for every 2023-2024 team-week where both a native
   week-tagged row and an as-of-joined snapshot (using that same team-week's actual
   historical kickoff time, reconstructed from `schedule_history.csv`) are available,
   compute agreement between the two methods' derived `depth_position`/`pos_rank`
   ordering for that team's RB room. Report exact agreement rate; this is a disclosed
   fact, not a pass/fail bar in itself, but a harmonizer with materially low agreement
   (informally, below the same neighborhood as ND2B's own coverage numbers) blocks
   proceeding to candidate science pending a separate root-cause review.
4. **Coverage/missingness disclosure by season:** report, per season 2016-2025, the
   fraction of team-weeks with a resolvable depth state under each method's own native
   contract. Fail closed (do not impute) for any team-week without a resolvable state.

Gate 0 is a **data-integrity gate, scored and reported before any candidate outcome is
computed or inspected** -- exactly the same ordering discipline #562 used for its own
reconstruction checksum.

## Transition definition (exact, leakage-safe)

A team-week `(team, season, week)` is a **transition week** for RB purposes if, using
only information available strictly before that week's kickoff, any of the following
holds relative to the same team's immediately preceding resolvable depth/status state:

1. **Depth-rank change**: the harmonized `pos_rank` (RB room only) of the player
   previously ranked RB1 (by the prior state) differs from the current state's RB1, OR
   any player's `pos_rank` moves by more than one position between the two states.
2. **Status-onset change**: a player who was fully available (no `OUT`/`DOUBTFUL`/`IR`/
   `PUP` designation) in the prior state now carries one of those designations in the
   current state, or vice versa (a previously unavailable player returns).
3. **Roster-membership change**: the active RB room's membership (via
   `ACTIVE_ROLES_CSV`, matching the current production availability contract) differs
   from the prior state's membership.

All three checks use **only** the harmonized depth/status state as of strictly before
the current week's kickoff compared against the harmonized state as of strictly before
the previous state's kickoff -- never same-week or postgame information. A team-week
that is not a transition week under any of the three checks is a **stable week**.

## Candidate mechanism (transition weeks only)

On a transition week, for each RB on the team's active room:

1. Compute each back's trailing role weight from `prior3_rb_share` and
   `prior3_snap_pct` (STACK2's own existing rolling-share construction, reused
   unchanged -- not redefined here).
2. Identify departed share: the summed trailing role weight of any back who is no
   longer in the active room, or who newly carries an `OUT`/`DOUBTFUL`/`IR`/`PUP`
   designation this week.
3. Reallocate departed share among the remaining active backs, weighted by their own
   trailing role weight, and **dampened by `prior_backfield_hhi`** (STACK2's existing
   concentration index): a high-concentration (workhorse-dominant) room reallocates
   more of the departed share to the single next-most-used back; a low-concentration
   (committee) room spreads it more evenly across the remaining room.
4. **Hard conservation identity**: reallocated shares, applied to the team's actual
   rush-attempt total for that week, must sum to exactly the team total --
   `max(abs(sum(reallocated_carries) - team_rush_attempts)) == 0.0` -- identical
   contract to P3's own conservation check in `RB_FINAL_QUALIFICATION_RESULTS.md`.

On a stable week, the candidate's output is defined to be **identical** to the existing
baseline route (today's production RB projection) -- no reallocation logic executes.

## Baseline

The baseline for every comparison is the current production RB rushing route (P3 for
Week 1, the existing non-P3 fallback route for Weeks 2-18, matching what
`scripts/modeling/rb_rush_synthesis_v1.py` actually emits today). The candidate is
graded as a **replacement of the Weeks-2-18 fallback route on transition weeks only** --
Week 1 and stable Weeks-2-18 weeks are unchanged by definition.

## Two-rotation temporal design (per GPT-5.6's correction: two rotations, not one)

Matching the QB M89/M90 rotation-confirmation pattern
(`scripts/backtest/run_m89_pregame_synthesis.py`, `run_m90_qb_synthesis_confirmation.py`,
**on main** -- exact hard check `test_season == train_season + 1`, no feature
engineering after rotation, `BOOT_N=10000`, `BOOTSTRAP_GATE=0.90`):

- **Rotation 1 (discovery)**: fit the transition-detection thresholds and reallocation
  weighting exactly as specified above on season 2023; freeze; score OOS on season
  2024. No parameter is fit beyond what is explicitly specified in the "Candidate
  mechanism" section above -- there is no free hyperparameter search.
- **Rotation 2 (confirmation)**: the exact same frozen architecture (unchanged from
  Rotation 1) fit on 2024, scored OOS on 2025.
- No feature, threshold, or cohort definition may change between rotations. If
  Rotation 1 and Rotation 2 disagree materially, that is a genuine finding (season
  instability), not a cue to retune Rotation 2.

## Protected cohorts and gates

Scored separately on **(a) the transition subpopulation** and **(b) the stable
subpopulation**, each season, each rotation:

- **Transition subpopulation must win**: pooled MAE strictly better than baseline on
  transition-week rows, in both Rotation 1 (2024 OOS) and Rotation 2 (2025 OOS)
  independently -- not just pooled across rotations.
- **Stable subpopulation must not regress**: since stable-week output is defined
  identical to baseline, this is a code-identity assertion (`max abs diff == 0.0`
  between candidate and baseline on stable rows), not a statistical gate.
- **Protected P3-failure cohorts, reused exactly** from `RB_FINAL_QUALIFICATION_
  RESULTS.md` for direct comparability: actual carries>=20, actual carries>=25, actual
  rushing yards>=100. Candidate MAE on each cohort, restricted to transition-week rows
  within that cohort, must be non-worse than baseline; if the transition-week
  sub-sample within a protected cohort is too small to score (informally, below
  STACK6-family's own established minimum-n conventions for a slice), report as
  insufficient rather than pass/fail.
- **Per-season non-regression**: both OOS seasons (2024 and 2025) individually, not
  just pooled -- a candidate that wins pooled but loses one season does not qualify.
- **Catastrophic/p90 protection**: 90th-percentile absolute error on the transition
  subpopulation must be non-worse than baseline (protects against the reallocation rule
  creating rare large misses even while improving mean error).
- **Clustered bootstrap support**: reuse the repo's own twice-validated
  `BOOT_N=10000`/`BOOTSTRAP_GATE=0.90` convention from M89/M90, applied to the paired
  transition-subpopulation MAE delta (candidate vs. baseline), player-clustered.
- **Conservation identity**: `max(abs(sum(reallocated_carries) - team_rush_attempts))
  == 0.0`, every transition team-week, both rotations.
- **No sportsbook inputs**: `sportsbook_inputs_used == 0` contract field, matching the
  existing runtime assertion in `rb_pricing_adapter_v1.py`.
- **No same-week/postgame information**: every transition-detection and reallocation
  input is drawn from state strictly before that week's kickoff (enforced by Gate 0's
  as-of contract).

## Promotion disposition rule

- **All gates above pass, both rotations** ->
  `RB_LANE_A_TRANSITION_ALLOCATION_QUALIFIED`. Eligible for a separate,
  independently-reviewed production-integration PR (this plan does not itself
  authorize enabling anything) -- and, per Issue #535's standing policy, enters
  permanent all-season shadow monitoring after any such promotion, same as every other
  qualified authority.
- **Gate 0 (schema harmonizer) fails or discloses materially low agreement** ->
  `RB_LANE_A_TRANSITION_ALLOCATION_GATE0_BLOCKED`. Candidate science does not proceed;
  the harmonizer itself is the thing needing a separate fix/review first.
  Explicitly does not permit reverting to a coarser identity/leakage-relaxed
  harmonizer to unblock -- the fix is to the harmonizer, not the gate.
- **Any protected-cohort/bootstrap/conservation/per-season gate fails** ->
  `RB_LANE_A_TRANSITION_ALLOCATION_NOT_QUALIFIED`. No rescue tuning, no re-test with
  adjusted thresholds, no subset search. Same stop-rule discipline as every other
  closed RB lane in this program (STACK6, STACK6B, M95T, Role-Order-Remap-V1).

## Explicitly out of scope

- No change to stable-week RB output -- by construction, identical to baseline.
- No change to WR/TE/QB, receiving markets, or any non-rushing RB market.
- Does not reopen STACK6 team-rush-context slicing, direct depth-rank carry
  assignment, or pooled-whole-season secondary-role features (STACK6B) -- this
  candidate is structurally distinct from all three (see "Why this candidate" above).
- Does not fold in #562's yard-difficulty MC-width work -- that is a separate,
  already-qualified, already-forward-confirming lane; this candidate concerns the
  rushing-yard **mean/allocation**, not distribution width.
