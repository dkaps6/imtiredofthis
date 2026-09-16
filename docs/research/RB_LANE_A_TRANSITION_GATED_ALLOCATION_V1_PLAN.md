# RB Lane A — Transition-Gated Carry-Share Reallocation V1 (Gate 0 + Frozen Plan)

**FROZEN BEFORE ANY CANDIDATE RESULT. RESEARCH ONLY. NO PRODUCTION CHANGE. NOTHING IN
THIS DOCUMENT HAS BEEN RUN.**

Per Issue #535 (GPT-5.6, comment `5690752971`): a plan-only Gate-0/frozen-plan draft
for the transition-gated allocation candidate identified in the Lane A audit (comment
`5690735640`). Nothing here authorizes building or running a candidate -- this is the
plan to be adversarially reviewed first, matching the discipline #562's own plan/
Amendment docs went through before implementation.

**Amendment 1**: incorporated GPT-5.6's ten prospective corrections from Issue #535
comment `5690848753`. The most important was the original draft's conservation
identity using **actual (postgame) team rush-attempt totals**, which cannot legally
construct a pregame candidate -- fixed by conserving to a predicted pregame pool using
STACK2's own leakage-safe precedent.

**Amendment 2** (this revision): incorporates GPT-5.6's four further prospective
corrections from Issue #535 comment `5698948964`, before any candidate is built or
run. The most important: (1) the plan's baseline was P3/STACK2, but production does
**not** actually use that route for Weeks 2-18 -- `run_pricing_v2.py` applies RB P3
only when `week == 1`; Weeks 2-18 prices from the plain calibrated ensemble mean. A
second, **decisive promotion comparator** is added against that real production route.
(2)-(3) the reallocation pool/weighting is redesigned to be a team-level,
player-identity-independent figure with weights that are explicitly normalized (the
prior version's conservation identity was not algebraically guaranteed), which also
resolves the departed-player-has-no-row problem. (4) adequacy/disposition language is
made fail-closed and internally consistent. No candidate code exists under any prior
revision of this document; nothing here is a post-implementation change.

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
team-week into `transition` vs `stable`, leave `stable` weeks exactly on the mechanism
comparator's route, and apply a targeted reallocation rule **only** on `transition`
weeks. If this candidate cannot beat the real production (promotion) comparator
specifically on the transition subpopulation while leaving the stable subpopulation
unregressed, it does not qualify -- there is no pooled-average rescue path. (See "Two
comparators" below for why beating P3/STACK2 alone is not sufficient.)

**Scope correction (Amendment 2)**: V1 targets the **role-collapse** error class
only -- transitions driven by a departure/status change among backs who already have
resolvable prior-history role weights. Per GPT-5.6's correction, the mechanism
formula's zero-history/new-entrant handling cannot simultaneously claim to give new
entrants a role weight of zero *and* claim to fix "new-role initialization" -- those
are contradictory. V1 explicitly does **not** claim to address ND1's "new-role
initialization" error class (n=8); a team-week whose entire remaining active room is
history-less is excluded from V1's transition subpopulation and reported, not scored.
A future V2 could target that class separately with its own frozen mechanism.

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
3. **Overlap-existence proof, required before the parity check (Amendment 2):** before
   relying on any 2023-2024 native-vs-as-of comparison, first confirm a date-stamped
   (non-week-tagged) depth source actually exists for that overlap window -- the Lane A
   audit only confirmed `load_depth_charts` returns the native week-tagged schema for
   2022-2024 with no date-stamped variant observed. If no such historical as-of source
   exists in `nflreadpy`, check whether ND2B's own preserved source/lineage (branch
   `6a01c631`) retained a usable date-stamped snapshot for any overlap season. If
   neither exists, **the parity check as originally specified cannot be constructed --
   do not manufacture a synthetic overlap.** In that case Gate 0.1 fails closed
   pending a different verification method (e.g. auditing ND2B's own disclosed
   coverage/timing evidence directly, without a fresh parity re-derivation) rather than
   silently skipping the requirement.
4. **Semantic parity proof, exact numeric bar (Amendment 1 point #5), only if step 3
   confirms a real overlap exists:** for every 2023-2024 team-week where both a native
   week-tagged row and an as-of-joined snapshot (using that team-week's actual
   historical kickoff time, reconstructed from `schedule_history.csv`) are available,
   compute the RB-room `pos_rank`-ordering
   agreement rate between the two methods. **Pass bar: agreement rate `>= 0.90`, and
   pregame-coverage rate (fraction of team-weeks with a resolvable as-of state)
   `>= 0.95`** -- adopted from ND2B's own disclosed coverage figures (100% pregame
   depth coverage, 78.6-83.7% prior-week snap coverage) as the nearest existing
   precedent in this repo for this exact source family. Below either bar: **fail
   closed**, do not proceed to candidate science.
5. **Coverage/missingness disclosure by season:** report, per season 2016-2025, the
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

### Predicted pregame conservation pool (Amendment 1 point #1, redesigned under Amendment 2 points #2-3)

The original draft conserved reallocated carries to "the team's actual rush-attempt
total for that week" -- **actual attempts are postgame information and cannot
construct a pregame candidate.** Amendment 1 fixed this using STACK2's own
`m94c_att`-summed pool pattern, but GPT-5.6's Amendment-2 review found that pattern
still had two unresolved problems here: (a) summing `stack_att` over "the pre-
transition room including a departing player" assumes that departing player still has
a resolvable current-game STACK1 row, which is not guaranteed once they are marked
`OUT`/inactive; and (b) the reallocation algebra using raw, un-normalized
`prior3_rb_share` values does not algebraically guarantee the stated conservation
identity, since rolling per-player shares are not guaranteed to sum to exactly `1.0`
across a room.

Both are fixed by sourcing the pool at the **team level**, independent of any
individual player's row, and by **explicitly normalizing** role weights before
redistribution:

- **Pool**: `pool = mc_projected_plays * (1 - mc_dropback_rate) *
  historical_rb_room_rush_share`, where `mc_projected_plays` and `mc_dropback_rate`
  are the team's existing pregame play-volume/pass-rate projections (already computed
  in `scripts/backtest/component_predictions.py::build_mc_predictions`, **on main**,
  fields `mc_projected_plays`/`mc_dropback_rate`) and `historical_rb_room_rush_share`
  is a strictly-prior trailing 3-game average of the RB/FB room's share of the team's
  realized rush attempts (same trailing-window convention as STACK2's own `prior3_*`
  features, reused for consistency). This pool is defined at the team level for
  **every** team-week (stable and transition alike) and never depends on any specific
  player's row existing -- it fully resolves the departed-player-missing-row problem,
  because no individual player's `stack_att` is summed to build it.

### Exact frozen HHI-dampened reallocation formula (Amendment 1 point #3, normalization fixed under Amendment 2)

On a transition team-week, let the team's **active (post-transition) RB room** be
indexed `i = 1..N`, and let `pool` be the team-level pool defined above.

1. **Raw role weight**: `raw_w_i = prior3_rb_share_i` (STACK2's own existing
   rolling-share feature, reused unchanged) for each active back `i`; `raw_w_i = 0` if
   unresolvable (rookie/no history).
2. **Explicit normalization (Amendment 2 fix)**: `w_i = raw_w_i / sum(raw_w_j over
   the active room)` if `sum(raw_w_j) > 0`, so that `sum(w_i) == 1.0` exactly over the
   active room **by construction**, asserted as a hard check -- this is computed only
   over the active room, never referencing a departed player's weight, so no departing
   player's row is required to exist.
3. **Fail-closed on all-zero weights**: if `sum(raw_w_j) == 0` across the entire active
   room (every active back is history-less), no reallocation is performed for that
   team-week and it is excluded from the transition subpopulation (reported, not
   silently dropped) -- consistent with V1's explicit exclusion of the "new-role
   initialization" class above.
4. **HHI**: `H = prior_backfield_hhi` (STACK2's own existing pre-transition
   concentration index, `sum(raw_w_j^2)` over the **pre-transition** room, computed
   from history alone and not dependent on any current-week row), reused unchanged.
5. **Concentration exponent**: `p = 1 + 2*H` (frozen constant `2`, chosen
   prospectively and disclosed as a design choice, not fit to any data -- at `H=0`
   (perfectly even committee), `p=1` (plain share-proportional split); as `H`
   approaches `1` (single-back monopoly), `p` approaches `3`, concentrating the pool
   sharply toward the highest-share remaining back). Not retuned between rotations.
6. **Recipient weights**: `v_i = w_i^p` for each active back with `w_i > 0`.
7. **Final allocation shares**: `recipient_share_i = v_i / sum(v_j over the active
   room)`. Ties (`identical v_i`) split proportionally by construction -- no
   special-case tie-break needed.
8. **Reallocated predicted carries**: `candidate_att_i = recipient_share_i * pool` for
   every active back. A departed/inactive player is not in the active room and
   receives no row (`candidate_att` undefined for them, not zero-valued -- they simply
   are not part of this week's scored universe, matching how the pregame RB universe
   already excludes inactive players elsewhere in this pipeline).
9. **Conservation identity**: `sum(candidate_att_i over the active room) == pool`
   exactly, by construction of steps 1-8 (`sum(recipient_share_i) == 1.0` follows
   directly from step 7's normalization) -- asserted as a hard check every transition
   team-week: `max(abs(sum(candidate_att_i) - pool)) == 0.0`.

### Rush-yard translation (Amendment 1 point #2 -- freeze the complete endpoint, not just carries)

Since qualification must ultimately be on rushing-yard accuracy, the candidate's final
projection is defined explicitly, reusing the exact existing P3 efficiency seam
(`scripts/modeling/rb_rush_synthesis_v1.py::compose_p3_row`, **on main**) unchanged:

`candidate_rush_yards_i = candidate_att_i * ypc_i`, where `ypc_i = stack_yards_i /
stack_att_i` when `stack_att_i > 0.20`, else the existing M94C implied-YPC fallback --
the identical rule and threshold already frozen in production P3. **No YPC
learning/tuning belongs in Lane A**; efficiency is untouched, only opportunity
allocation changes.

On a stable week, the candidate's **mechanism-track** output is defined to be
**identical** to the mechanism comparator (below) -- no reallocation logic executes.

## Two comparators: mechanism vs. promotion (Amendment 2 point #1)

GPT-5.6's Amendment-2 review found the original single "baseline" conflated two
different things. `docs/production/RB_P3_WEEK1_PROMOTION_2026_09_05.md` and
`scripts/run_pricing_v2.py` (the RB P3 override applies only when `week == 1`, **on
main**) both confirm: **P3/STACK2's enriched-allocation route is not the live
production Weeks-2-18 authority.** For Weeks 2-18, `run_pricing_v2.py` prices
`rush_yards` from `target_mean = ensemble_proj` -- the plain calibrated MC+ML+State
ensemble mean (`scripts/modeling/ensemble_v2.py::apply_ensemble`, weighted by
`data/model_ensemble_weights.csv`), with no RB-specific override at all. Beating
P3/STACK2 alone therefore does not make this candidate production-eligible; it has to
beat what production actually does today.

Two comparators are frozen, both reconstructed identically across every OOS
team-week (stable and transition):

- **Mechanism comparator** (diagnostic only, not itself sufficient for qualification):
  the P3/STACK2 enriched-allocation route (`enriched_att * stack_implied_ypc`, per
  `rb_rush_synthesis_v1.py`) -- because this is the specific allocation failure Lane A
  is trying to repair, and comparing against it isolates whether the reallocation
  mechanism itself is an improvement over STACK2's pooled model.
- **Promotion comparator** (decisive for qualification): the actual historical
  reconstruction of `ensemble_proj` for Weeks 2-18, built via the same frozen-parent
  pattern already established in PR #615 -- `scripts/modeling/ensemble_v2.py`
  (blob `41e809b32e4596b8cf18bedbf2b940a2aa3b80b2` at #562's merge commit `91afb3a5`),
  `scripts/backtest/component_predictions.py` (blob
  `18f7289515b88c84a91479da18526df4cd7f5398`), and `data/model_ensemble_weights.csv`
  (blob `baade160a124e5cd8ecd415c0276622d4b60953f`), pinned identically for both OOS
  rotations. **Qualification requires beating the promotion comparator on the
  decisive rushing-yard endpoint** (see Protected cohorts and gates); beating only the
  mechanism comparator is necessary evidence that the reallocation mechanism itself
  works, but is not sufficient by itself for a `QUALIFIED` disposition.

## Authority-exact baseline reconstruction (Amendment 1 point #6)

Before any candidate-vs-comparator comparison is scored, **both** comparators above
must themselves be reconstructed for both OOS seasons and proven to match their own
canonical historical lineage row-for-row and value-for-value (same identity/value-
parity discipline #562 used for its own parent-panel check): the mechanism comparator
against the canonical STACK2/P3 casebook, and the promotion comparator against the
canonical historical `component_predictions.csv`/ensemble-output lineage already used
elsewhere in this repo's backtest machinery. Report row-count and max-abs-value delta
against each canonical source. **Fail closed** if either reconstruction does not
reproduce to near-machine precision -- a candidate "win" against a mis-reconstructed
comparator is not evidence. Stable-week mechanism-comparator rows remain byte-identical
by construction and do not need this proof independently; transition-week rows and the
promotion comparator (scored on every row, stable and transition alike) do.

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
subpopulation**, each season, each rotation, against **both** comparators. Per
Amendment 1 point #7 and Amendment 2 point #1, both the allocation mechanism and the
final rush-yard production endpoint are scored, and **the promotion comparator on the
production endpoint is decisive** for qualification.

- **Mechanism evidence (informative, against the mechanism comparator only -- not
  itself sufficient)**: carry MAE / allocation error on transition rows, candidate vs.
  P3/STACK2.
- **Production endpoint (decisive, against the promotion comparator)**: pooled
  **rushing-yard MAE** on transition rows must be strictly better than the promotion
  comparator (`ensemble_proj`), independently in Rotation 1 (2024 OOS) and Rotation 2
  (2025 OOS). A carry-allocation improvement over the mechanism comparator that does
  not improve rushing-yard MAE over the **promotion** comparator does **not** qualify.
- **Stable subpopulation must not regress**: mechanism-track stable-week output is
  defined identical to the mechanism comparator by construction (`max abs diff ==
  0.0`), so this is a code-identity assertion, not a statistical gate. Against the
  promotion comparator, stable-week rows are scored exactly the same as transition-week
  rows (the promotion comparator makes no stable/transition distinction).
- **Protected P3-failure cohorts, reused exactly** from `RB_FINAL_QUALIFICATION_
  RESULTS.md` for direct comparability: actual carries>=20, actual rushing yards>=100
  are **required** gates (each non-worse than the promotion comparator, transition-week
  rows within the cohort). Actual carries>=25 is **designated disclosure-only now**
  (Amendment 2 point #4): its original full-season n was already only 24 in
  `RB_FINAL_QUALIFICATION_RESULTS.md`, so a transition-week-only subset is predictably
  too sparse to support a required gate at this candidate's adequacy bar -- reported for
  transparency, never blocking or granting qualification by itself. **These are
  evaluation-only slices** (Amendment 1 point #10): `actual carries`/`actual
  rush_yards` are used exclusively to define which rows fall into these cohorts for
  grading after the fact -- they never enter transition detection, the reallocation
  formula, pool construction, or any other pregame candidate-construction step.
- **Exact adequacy counts, fail-closed (Amendment 1 point #8, contradiction fixed under
  Amendment 2 point #4)**: a required protected-cohort or the overall transition
  subpopulation is scoreable only if it has `n>=30` transition-week rows in that OOS
  season (frozen here explicitly as a new minimum for this candidate). **Both OOS
  rotations must independently have an adequate (`n>=30`) overall transition
  subpopulation AND adequate support on both required cohorts (carries>=20,
  yards>=100) for a `QUALIFIED` disposition to be possible at all.** If either
  rotation's overall transition subpopulation, or either required cohort within it, is
  below `n>=30`, the disposition is `RB_LANE_A_TRANSITION_ALLOCATION_INSUFFICIENT_
  EVIDENCE` for that gate -- this is fail-closed, not a partial pass, and does not by
  itself grant `QUALIFIED` even if every scoreable gate happened to clear.
- **Per-season non-regression**: both OOS seasons (2024 and 2025) individually, not
  just pooled -- a candidate that wins pooled but loses one season does not qualify.
- **Catastrophic/p90 protection**: 90th-percentile absolute rushing-yard error on the
  transition subpopulation, vs. the promotion comparator, must be non-worse (protects
  against the reallocation rule creating rare large misses even while improving mean
  error).
- **Dependence-aware bootstrap support (Amendment 1 point #9)**: both required,
  neither can rescue the other's failure, both applied to the paired transition-
  subpopulation rushing-yard MAE delta against the **promotion** comparator:
  - the planned player-cluster paired bootstrap (reuse the repo's own
    `BOOT_N=10000`/`BOOTSTRAP_GATE=0.90` convention from M89/M90), player-clustered;
    and
  - a dependence-aware team/game-clustered bootstrap, adapting #562's already-reviewed
    crossed player x game construction (`crossed_player_game_bootstrap_probability` in
    `scripts/research/evaluate_rb_pd2_yard_difficulty_mc_width_v1.py`, mechanics
    unchanged), because multiple RB rows on the same team-game are mechanically
    coupled by the conservation/reallocation identity. Same `>=0.90` threshold.
- **Conservation identity**: `max(abs(sum(candidate_att_i) - pool)) == 0.0`, every
  transition team-week, both rotations (see mechanism section above).
- **No sportsbook inputs**: `sportsbook_inputs_used == 0` contract field, matching the
  existing runtime assertion in `rb_pricing_adapter_v1.py`.
- **No same-week/postgame information**: every transition-detection and reallocation
  input is drawn from state strictly before that week's kickoff (enforced by Gate 0's
  as-of contracts across all three source families).

## Promotion disposition rule

- **Gate 0 (all three sub-gates) passes; authority-exact reconstruction of both
  comparators passes; both rotations have adequate (`n>=30`) support on the overall
  transition subpopulation and both required protected cohorts; both rotations clear
  every required gate on the rushing-yard production endpoint against the promotion
  comparator** -> `RB_LANE_A_TRANSITION_ALLOCATION_QUALIFIED`. Eligible for a separate,
  independently-reviewed production-integration PR (this plan does not itself
  authorize enabling anything) -- and, per Issue #535's standing policy, enters
  permanent all-season shadow monitoring after any such promotion, same as every other
  qualified authority.
- **Any Gate-0 sub-gate fails, or discloses agreement/coverage below its frozen
  numeric bar (including the Gate-0.1 overlap-existence proof failing to find a
  constructible parity check)** -> `RB_LANE_A_TRANSITION_ALLOCATION_GATE0_BLOCKED`.
  Candidate science does not proceed; the harmonizer itself is the thing needing a
  separate fix/review first. Explicitly does not permit reverting to a coarser
  identity/leakage-relaxed harmonizer to unblock -- the fix is to the harmonizer, not
  the gate.
- **Authority-exact reconstruction of either comparator fails** ->
  `RB_LANE_A_TRANSITION_ALLOCATION_BASELINE_RECONSTRUCTION_FAILURE`. Candidate science
  does not proceed until the failing reconstruction is fixed and re-verified.
- **Either OOS rotation's overall transition subpopulation, or either required
  protected cohort (carries>=20, yards>=100) within it, is below the `n>=30` adequacy
  bar** -> `RB_LANE_A_TRANSITION_ALLOCATION_INSUFFICIENT_EVIDENCE`, fail-closed --
  this alone prevents `QUALIFIED` regardless of how any scoreable gate performs.
  Disclosure-only slices (carries>=25) never trigger this regardless of their count.
- **Gate 0, both reconstructions, and adequacy all clear, but any required
  protected-cohort/bootstrap/conservation/per-season gate fails against the promotion
  comparator** -> `RB_LANE_A_TRANSITION_ALLOCATION_NOT_QUALIFIED`. No rescue tuning, no
  re-test with adjusted thresholds, no subset search. Same stop-rule discipline as
  every other closed RB lane in this program (STACK6, STACK6B, M95T,
  Role-Order-Remap-V1).

## Explicitly out of scope

- No change to stable-week mechanism-track RB output -- by construction, identical to
  the mechanism comparator (P3/STACK2).
- No change to WR/TE/QB, receiving markets, or any non-rushing RB market.
- Does not reopen STACK6 team-rush-context slicing, direct depth-rank carry
  assignment, or pooled-whole-season secondary-role features (STACK6B) -- this
  candidate is structurally distinct from all three (see "Why this candidate" above).
- Does not fold in #562's yard-difficulty MC-width work -- that is a separate,
  already-qualified, already-forward-confirming lane; this candidate concerns the
  rushing-yard **mean/allocation**, not distribution width.
- No YPC/efficiency learning or tuning -- efficiency is inherited unchanged from the
  existing P3 seam.
