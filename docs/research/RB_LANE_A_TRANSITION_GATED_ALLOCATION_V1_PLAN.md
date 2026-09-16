# RB Lane A — Transition-Gated Carry-Share Reallocation V1 (Gate 0 + Frozen Plan)

**FROZEN BEFORE ANY CANDIDATE RESULT. RESEARCH ONLY. NO PRODUCTION CHANGE. NOTHING IN
THIS DOCUMENT HAS BEEN RUN.**

Per Issue #535 (GPT-5.6, comment `5690752971`): a plan-only Gate-0/frozen-plan draft
for the transition-gated allocation candidate identified in the Lane A audit (comment
`5690735640`). Nothing here authorizes building or running a candidate -- this is the
plan to be adversarially reviewed first, matching the discipline #562's own plan/
Amendment docs went through before implementation.

**Amendment 1** (this revision): incorporates GPT-5.6's ten prospective corrections
from Issue #535 comment `5690848753`, before any candidate is built or run. The most
important of these is #1 below: the original draft's conservation identity used
**actual (postgame) team rush-attempt totals**, which cannot legally construct a
pregame candidate. That is fixed here by conserving to a **predicted** pregame pool
instead, using the exact same leakage-safe precedent STACK2 itself already
established. No candidate code exists under any prior revision of this document;
nothing here is a post-implementation change.

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

## Gate 0 — schema/timing harmonizer for every transition-trigger source (must pass before any candidate science)

Amendment 1 point #4: the original draft only certified depth-chart timing. Transition
detection also uses injury/status and roster-membership sources, each with its own
timing-integrity risk, so Gate 0 now covers all three independently.

### 0.1 Depth-chart source

The Lane A audit found `nflreadpy.load_depth_charts` returns a week-tagged schema for
2016-2024 (`season, week, club_code, depth_team, gsis_id, position, depth_position,
full_name, ...`) but an entirely different **date-stamped, non-week-tagged** ESPN
roster-snapshot schema for 2025+ (`dt, team, player_name, espn_id, gsis_id, pos_grp,
pos_slot, pos_rank`, 221 unique snapshot timestamps spanning 2025-08-03 through
2026-03-14). `scripts/backtest/historical_inputs.py::_weekly_depth_lookup` already
fails closed on this today -- it requires `{"season","week"}` in columns and returns an
**empty** frame otherwise, silently dropping to blank depth fields.

Harmonizer, built before any candidate feature is computed:

1. **Historical side (2016-2024):** consume `load_depth_charts` natively; a team's
   depth-chart state for `(season, week)` is the native week-tagged row set, unchanged.
2. **Live side (2025+):** for a given team and a given game's kickoff timestamp
   `kickoff_utc`, the team's depth-chart state is the **most recent snapshot with
   `dt < kickoff_utc`** (matching ND2B's proven as-of convention, median snapshot age
   10.77h, p90 17.90h, 100% team-game pregame coverage in ND2B's own audit). No
   snapshot with `dt >= kickoff_utc` may ever enter a scored row for that game --
   enforced as a hard assertion, not a convention.
3. **Semantic parity proof, exact numeric bar (Amendment 1 point #5):** for every
   2023-2024 team-week where both a native week-tagged row and an as-of-joined
   snapshot (using that team-week's actual historical kickoff time, reconstructed from
   `schedule_history.csv`) are available, compute the RB-room `pos_rank`-ordering
   agreement rate between the two methods. **Pass bar: agreement rate `>= 0.90`, and
   pregame-coverage rate (fraction of team-weeks with a resolvable as-of state)
   `>= 0.95`** -- adopted from ND2B's own disclosed coverage figures (100% pregame
   depth coverage, 78.6-83.7% prior-week snap coverage) as the nearest existing
   precedent in this repo for this exact source family. Below either bar: **fail
   closed**, do not proceed to candidate science.
4. **Coverage/missingness disclosure by season:** report, per season 2016-2025, the
   fraction of team-weeks with a resolvable depth state under each method's own native
   contract.

### 0.2 Injury/status source

- **Historical side (2016-2024):** `nflreadpy.load_injuries`, week-tagged
  (`report_status`, `practice_status`), consumed natively.
- **Live side (2025+):** must be proven week-tagged or given the same as-of treatment
  as depth charts before use; if `load_injuries` for 2025+ is date-stamped rather than
  week-tagged, build the identical `dt < kickoff_utc`-gated as-of join used for depth,
  with its own agreement/coverage disclosure against the 2023-2024 overlap. **Same pass
  bars as 0.1**: agreement `>= 0.90`, coverage `>= 0.95`.
- **Fail closed** if 2025+ injury-report timing cannot be proven pregame-safe by either
  a native week-tag or a validated as-of join.

### 0.3 Roster-membership source

- `ACTIVE_ROLES_CSV` is named in the audit as the current production availability
  contract, but per GPT-5.6's correction it **cannot be assumed timestamp-safe for
  2023-2025 reconstruction without proof**. Before use in transition detection,
  reconstruct `ACTIVE_ROLES_CSV`'s own generation lineage for 2023-2025 and confirm
  each week's active-room snapshot reflects only information available strictly before
  that week's kickoff (no same-week or postgame roster-move leakage). Report the exact
  audit method and result. **Fail closed** if this cannot be proven; do not substitute
  an unverified proxy silently.

### 0.4 Gate 0 disposition

Gate 0 is scored and reported **before any candidate outcome is computed or
inspected** -- exactly the same ordering discipline #562 used for its own
reconstruction checksum. All three sub-gates (0.1, 0.2, 0.3) must independently pass
their stated numeric bars. Any one failing routes the whole plan to
`RB_LANE_A_TRANSITION_ALLOCATION_GATE0_BLOCKED` (see Promotion disposition rule) --
never a silent fallback to a coarser, less leakage-safe source.

## Transition definition (exact, leakage-safe)

A team-week `(team, season, week)` is a **transition week** for RB purposes if, using
only information available strictly before that week's kickoff (per the Gate-0
harmonized sources above), any of the following holds relative to the same team's
immediately preceding resolvable depth/status state:

1. **Depth-rank change**: the harmonized `pos_rank` (RB room only) of the player
   previously ranked RB1 (by the prior state) differs from the current state's RB1, OR
   any player's `pos_rank` moves by more than one position between the two states.
2. **Status-onset change**: a player who was fully available (no `OUT`/`DOUBTFUL`/`IR`/
   `PUP` designation) in the prior state now carries one of those designations in the
   current state, or vice versa (a previously unavailable player returns).
3. **Roster-membership change**: the active RB room's membership (via the Gate-0.3
   timestamp-proven active-roster source) differs from the prior state's membership.

All three checks use **only** the harmonized depth/status state as of strictly before
the current week's kickoff compared against the harmonized state as of strictly before
the previous state's kickoff -- never same-week or postgame information. A team-week
that is not a transition week under any of the three checks is a **stable week**.

## Candidate mechanism (transition weeks only)

### Predicted pregame conservation pool (Amendment 1 point #1 -- fatal-leakage fix)

The original draft conserved reallocated carries to "the team's actual rush-attempt
total for that week" -- **actual attempts are postgame information and cannot
construct a pregame candidate.** This is corrected by reusing STACK2's own existing,
already-leakage-safe pool construction exactly:

`scripts/backtest/evaluate_rb_stack2_enriched_allocation.py::add_projection_arms`
(**on main**, lines ~321-330) builds `m94c_att = candidate_rush_att` -- M94C's own
**pregame** per-player predicted rush-attempt projection, not a realized value -- then
computes `pool = x.groupby(["season","week","team"])["m94c_att"].transform("sum")`,
the team's RB-room predicted rush-attempt pool, and allocates `enriched_att =
enriched_share * pool`. STACK2 itself has never used actual attempts as its
conservation target; this candidate reuses that exact pattern.

For this candidate, the conservation pool for a team-week is therefore the sum, over
the team's active RB room for that week, of each player's **pregame STACK1 rush-
attempt projection** (`stack_att`, the same field `rb_rush_synthesis_v1.py` already
consumes as `stack_att`/`stack_yards` for the Week-1 route and for Weeks-2-18's
efficiency term) -- **never realized/actual attempts.**

### Exact frozen HHI-dampened reallocation formula (Amendment 1 point #3)

On a transition team-week, let the team's active RB room (post-transition) be indexed
`i = 1..N`, and let `pool` be the predicted pregame pool defined above (summed over
the **pre-transition** room, i.e. including any departing player's `stack_att`, so
that the full predicted opportunity is preserved and only its allocation changes).

1. **Role weight**: `w_i = prior3_rb_share_i` (STACK2's own existing rolling-share
   feature, reused unchanged -- no new blending of snap share is introduced, to keep
   the formula fully specified with a single source per weight).
2. **Departed share**: `R = sum(w_j)` over every player `j` who was in the
   pre-transition room and is not in the post-transition room, or who newly carries an
   `OUT`/`DOUBTFUL`/`IR`/`PUP` designation this week.
3. **HHI**: `H = prior_backfield_hhi` (STACK2's own existing pre-transition concentration
   index, `sum(w_j^2)` over the pre-transition room), reused unchanged.
4. **Concentration exponent**: `p = 1 + 2*H` (frozen constant `2`, chosen prospectively
   and disclosed as a design choice, not fit to any data -- at `H=0` (perfectly even
   committee), `p=1` (plain share-proportional redistribution); at `H` approaching `1`
   (single-back monopoly), `p` approaches `3` (redistribution concentrates sharply
   toward the single highest-share remaining back). This exponent is not retuned
   between Rotation 1 and Rotation 2.
5. **Recipient weights**: for each remaining active back `i`, `v_i = w_i^p`.
6. **Zero-history/new-player behavior**: if a remaining back has no resolvable
   `prior3_rb_share` (rookie/no-history), `w_i = 0` for role-weight purposes, `v_i = 0`
   -- new/unproven backs receive no reallocated share under this formula (they may
   still carry their own independently-projected `stack_att`, untouched). If this
   yields all-zero `v_i` across the remaining room (every remaining back is history-less),
   fail closed: no reallocation is performed for that team-week and it is excluded from
   the transition subpopulation (reported, not silently dropped).
7. **Normalization and floor**: `recipient_share_i = v_i / sum(v_j)` over remaining
   backs with `v_j > 0`. Ties (`identical v_i`) split the departed share equally among
   the tied backs by construction of the formula (no special-case tie-break needed).
8. **Reallocated predicted carries**: for each remaining back,
   `candidate_att_i = (w_i * pool) + (R * recipient_share_i * pool)`. For a departed
   player, `candidate_att = 0`.
9. **Conservation identity**: `sum(candidate_att_i over remaining room) == pool`
   exactly (by construction of steps 1-8; asserted as a hard check, not merely implied
   by the algebra) -- `max(abs(sum(candidate_att_i) - pool)) == 0.0`.

### Rush-yard translation (Amendment 1 point #2 -- freeze the complete endpoint, not just carries)

Since qualification must ultimately be on rushing-yard accuracy, the candidate's final
projection is defined explicitly, reusing the exact existing P3 efficiency seam
(`scripts/modeling/rb_rush_synthesis_v1.py::compose_p3_row`, **on main**) unchanged:

`candidate_rush_yards_i = candidate_att_i * ypc_i`, where `ypc_i = stack_yards_i /
stack_att_i` when `stack_att_i > 0.20`, else the existing M94C implied-YPC fallback --
the identical rule and threshold already frozen in production P3. **No YPC
learning/tuning belongs in Lane A**; efficiency is untouched, only opportunity
allocation changes.

On a stable week, the candidate's output is defined to be **identical** to the existing
baseline route (today's production RB projection) -- no reallocation logic executes.

## Authority-exact baseline reconstruction (Amendment 1 point #6)

Before any candidate-vs-baseline comparison is scored, the exact Weeks-2-18 baseline
route being challenged (`enriched_att * stack_implied_ypc`, per `rb_rush_synthesis_v1.py`)
must itself be reconstructed for both OOS seasons and proven to match its own canonical
historical lineage row-for-row and value-for-value (same identity/value-parity
discipline #562 used for its own parent-panel check). Report row-count and max-abs-value
delta against the canonical STACK2/P3 casebook. **Fail closed** if reconstruction
does not reproduce to near-machine precision -- a candidate "win" against a
mis-reconstructed baseline is not evidence. Stable-week rows remain byte-identical by
construction and do not need this proof independently; transition-week baseline rows
do.

## Two-rotation temporal design

Matching the QB M89/M90 rotation-confirmation pattern
(`scripts/backtest/run_m89_pregame_synthesis.py`, `run_m90_qb_synthesis_confirmation.py`,
**on main** -- exact hard check `test_season == train_season + 1`, no feature
engineering after rotation, `BOOT_N=10000`, `BOOTSTRAP_GATE=0.90`):

- **Rotation 1 (discovery)**: the transition-detection rules and reallocation formula
  exactly as specified above (nothing is "fit" -- every constant is already frozen in
  this document) evaluated on 2023 -> OOS 2024.
- **Rotation 2 (confirmation)**: the exact same frozen architecture (unchanged from
  Rotation 1) evaluated on 2024 -> OOS 2025.
- No feature, threshold, weight, or cohort definition may change between rotations. If
  Rotation 1 and Rotation 2 disagree materially, that is a genuine finding (season
  instability), not a cue to retune Rotation 2.

## Protected cohorts and gates

Scored separately on **(a) the transition subpopulation** and **(b) the stable
subpopulation**, each season, each rotation. Per Amendment 1 point #7, both the
allocation mechanism and the final rush-yard production endpoint are scored, and the
**production endpoint is decisive** for qualification.

- **Mechanism evidence (informative, not itself sufficient)**: carry MAE / allocation
  error on transition rows, candidate vs. baseline.
- **Production endpoint (decisive)**: pooled **rushing-yard MAE** on transition rows
  must be strictly better than baseline, independently in Rotation 1 (2024 OOS) and
  Rotation 2 (2025 OOS). A carry-allocation improvement that does not improve
  rushing-yard MAE does **not** qualify.
- **Stable subpopulation must not regress**: since stable-week output is defined
  identical to baseline, this is a code-identity assertion (`max abs diff == 0.0`
  between candidate and baseline on stable rows), not a statistical gate.
- **Protected P3-failure cohorts, reused exactly** from `RB_FINAL_QUALIFICATION_
  RESULTS.md` for direct comparability: actual carries>=20, actual carries>=25, actual
  rushing yards>=100. **These are evaluation-only slices** (Amendment 1 point #10):
  `actual carries`/`actual rush_yards` are used exclusively to define which rows fall
  into these cohorts for grading after the fact -- they never enter transition
  detection, the reallocation formula, pool construction, or any other pregame
  candidate-construction step. Candidate rushing-yard MAE on each cohort, restricted to
  transition-week rows within that cohort, must be non-worse than baseline.
- **Exact adequacy counts (Amendment 1 point #8)**: a protected-cohort or transition
  subpopulation slice is scoreable only if it has `n>=30` transition-week rows in that
  OOS season (frozen here explicitly as a new minimum for this candidate -- the earlier
  draft's "STACK6-family convention" reference could not be verified against a citable
  exact number and is replaced with this explicit bar rather than left informal).
  Additionally, **each OOS season's total transition subpopulation must have `n>=30`**
  for a scientific disposition to be issued for that season at all; below that, report
  as insufficient rather than pass/fail for that season specifically.
- **Per-season non-regression**: both OOS seasons (2024 and 2025) individually, not
  just pooled -- a candidate that wins pooled but loses one season does not qualify.
- **Catastrophic/p90 protection**: 90th-percentile absolute rushing-yard error on the
  transition subpopulation must be non-worse than baseline (protects against the
  reallocation rule creating rare large misses even while improving mean error).
- **Dependence-aware bootstrap support (Amendment 1 point #9)**: both required,
  neither can rescue the other's failure:
  - the planned player-cluster paired bootstrap (reuse the repo's own
    `BOOT_N=10000`/`BOOTSTRAP_GATE=0.90` convention from M89/M90), applied to the
    paired transition-subpopulation rushing-yard MAE delta (candidate vs. baseline),
    player-clustered; and
  - a dependence-aware team/game-clustered bootstrap, adapting #562's already-reviewed
    crossed player x game construction (`crossed_player_game_bootstrap_probability` in
    `scripts/research/evaluate_rb_pd2_yard_difficulty_mc_width_v1.py`, mechanics
    unchanged) to this candidate's paired rushing-yard MAE delta, because multiple RB
    rows on the same team-game are mechanically coupled by the conservation/
    reallocation identity. Same `>=0.90` threshold, applied consistently.
- **Conservation identity**: `max(abs(sum(candidate_att_i) - pool)) == 0.0`, every
  transition team-week, both rotations (see mechanism section above).
- **No sportsbook inputs**: `sportsbook_inputs_used == 0` contract field, matching the
  existing runtime assertion in `rb_pricing_adapter_v1.py`.
- **No same-week/postgame information**: every transition-detection and reallocation
  input is drawn from state strictly before that week's kickoff (enforced by Gate 0's
  as-of contracts across all three source families).

## Promotion disposition rule

- **Gate 0 (all three sub-gates) passes, authority-exact baseline reconstruction
  passes, both rotations clear every protected-cohort/bootstrap/conservation/
  per-season gate on the rushing-yard production endpoint** ->
  `RB_LANE_A_TRANSITION_ALLOCATION_QUALIFIED`. Eligible for a separate,
  independently-reviewed production-integration PR (this plan does not itself
  authorize enabling anything) -- and, per Issue #535's standing policy, enters
  permanent all-season shadow monitoring after any such promotion, same as every other
  qualified authority.
- **Any Gate-0 sub-gate fails, or discloses agreement/coverage below its frozen
  numeric bar** -> `RB_LANE_A_TRANSITION_ALLOCATION_GATE0_BLOCKED`. Candidate science
  does not proceed; the harmonizer itself is the thing needing a separate fix/review
  first. Explicitly does not permit reverting to a coarser identity/leakage-relaxed
  harmonizer to unblock -- the fix is to the harmonizer, not the gate.
- **Authority-exact baseline reconstruction fails** ->
  `RB_LANE_A_TRANSITION_ALLOCATION_BASELINE_RECONSTRUCTION_FAILURE`. Candidate science
  does not proceed until the baseline reconstruction itself is fixed and re-verified.
- **Any protected-cohort/bootstrap/conservation/per-season gate fails on the
  production endpoint** -> `RB_LANE_A_TRANSITION_ALLOCATION_NOT_QUALIFIED`. No rescue
  tuning, no re-test with adjusted thresholds, no subset search. Same stop-rule
  discipline as every other closed RB lane in this program (STACK6, STACK6B, M95T,
  Role-Order-Remap-V1).
- **A protected cohort or an OOS season's transition subpopulation is below the
  `n>=30` adequacy bar** -> report as insufficient for that specific slice/season;
  does not by itself block an overall disposition if every other required gate is
  otherwise scoreable and clears, but any season/cohort that cannot be scored is
  disclosed explicitly, never silently omitted.

## Explicitly out of scope

- No change to stable-week RB output -- by construction, identical to baseline.
- No change to WR/TE/QB, receiving markets, or any non-rushing RB market.
- Does not reopen STACK6 team-rush-context slicing, direct depth-rank carry
  assignment, or pooled-whole-season secondary-role features (STACK6B) -- this
  candidate is structurally distinct from all three (see "Why this candidate" above).
- Does not fold in #562's yard-difficulty MC-width work -- that is a separate,
  already-qualified, already-forward-confirming lane; this candidate concerns the
  rushing-yard **mean/allocation**, not distribution width.
- No YPC/efficiency learning or tuning -- efficiency is inherited unchanged from the
  existing P3 seam.
