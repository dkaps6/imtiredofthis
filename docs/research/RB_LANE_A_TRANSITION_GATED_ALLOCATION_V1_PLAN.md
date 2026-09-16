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

**Amendment 3** (this revision): incorporates GPT-5.6's cross-audit of Amendment 2
from Issue #535 comment `5699148456`, before any candidate is built or run. Two
fixes: (1) **fatal temporal leak in the decisive promotion comparator** -- Amendment
2 pinned `data/model_ensemble_weights.csv` blob `baade160...` identically for both
OOS rotations, but that file's `rush_yards` row has `fit_scope =
all_2024_oos_frozen_for_2025` (fit using 2024 outcomes). Applying it to Rotation 1's
2024 test set would let the comparator's own calibration season leak into its test
season -- the same class of defect the M89/M90 rotation discipline exists to
prevent. Fixed by using a genuinely pre-2024 `rush_yards` weight authority for
Rotation 1 specifically (see "Two comparators" and "Authority-exact baseline
reconstruction" below), with a hard per-rotation provenance assertion. (2) **scope/
trigger mismatch** -- V1 claims the ND1 "role-collapse" cohort, but the original
three-way transition definition also scored role-expansion/re-entry events (a
previously-unavailable player returning, a roster addition) under the same trigger.
Fixed by splitting "detected transition" (broad, all three checks, used only for
Gate-0 harmonizer testing and disclosure) from "scored V1 transition" (narrow: loss/
vacancy events only), so the scored population matches the claimed cohort honestly.

**Amendment 4** (this revision): incorporates GPT-5.6's cross-audit of Amendment 3
from Issue #535 comment `5700382360`, before any candidate is built or run.
Amendment 3's temporal-comparator and scope fixes were verified correct, but one
deployment-architecture defect remained: the plan defined stable-week candidate
output as identical to the **mechanism comparator (P3/STACK2)** -- an unpromoted
diagnostic route -- rather than to the **promotion comparator (`ensemble_proj`)**,
the actual live Weeks-2-18 production route Amendment 2 itself established as
decisive. Left uncorrected, a qualified Lane A module could have silently replaced
today's real stable-week production output with an unpromoted research route on
every non-transition week. Fixed by splitting the candidate into two explicit,
non-interchangeable arms -- `deployable_candidate` (byte-identical to the promotion
comparator on every stable/non-scored week; the transition reallocation only on
scored V1 weeks) and `mechanism_diagnostic` (transition reallocation vs. P3/STACK2,
diagnostic-only, never defines production output) -- with a new hard stable-identity
gate and a whole-season integration-sanity check added to the protected gates. See
"Two candidate arms" below.

**Amendment 5** (this revision): adjudicated by GPT-5.6 on Issue #535 during
implementation itself (comment `5701881512`), resolving three empirical findings
surfaced while building Gate 0, before any candidate output exists. (1) **Gate 0.1
disposition corrected**: since no date-stamped 2023-2024 depth source exists
anywhere (confirmed empirically against `nflreadpy` and ND2B's preserved lineage),
the originally-specified semantic-parity check is recorded as
`NOT_CONSTRUCTIBLE_NO_OVERLAP`, not silently treated as a pass. This does **not**
block scored V1 science: per Amendment 3, the scored V1 trigger is loss/vacancy
only (status-onset-loss, roster-membership-shrink) and never depends on depth
rank; depth-chart harmonization remains available only for the broad
`detected_transition` disclosure population, unchanged. (2) **Gate 0.2 confirmed
PASS**: empirically verified `nflreadpy.load_injuries(seasons=[2025])` is natively
`season`/`week`-tagged (`report_status`/`practice_status` present, matching
2016-2024), so the plan's as-of-join contingency never triggers. (3) **Gate 0.3
source formally replaced**: `ACTIVE_ROLES_CSV` cannot reconstruct any past week
(current-snapshot-only generation, no historical-replay mode) and is replaced with
`nflreadpy.load_rosters_weekly` -- the same canonical weekly pregame-universe
source `scripts/backtest/historical_inputs.py` already uses, under seven frozen
requirements (see revised Gate 0.3 below). (4) **HHI prose corrected**: the
mechanism's step 4 said `H = sum(raw_w_j^2)`; STACK2's actual `prior_backfield_hhi`
field computes `sum((ownshares/ownshares.sum())^2)` -- normalized, not raw. Prose
corrected to match; the existing field is reused byte-for-byte, unchanged, no new
calculation.

**Amendment 6** (this revision): adjudicated by GPT-5.6 on Issue #535 (comment
`5704381940`), replacing the promotion comparator's cross-run parity requirement
before any candidate output exists. During implementation, a cross-run parity check
of a fresh Rotation-1 (2024) reconstruction against run `35032590321` (initially
believed to be the canonical M91 authority artifact) surfaced a residual `mc_proj`-
only delta on Chris-Godwin-affected weeks (1-7) after two rounds of invocation-
fidelity and universe-membership diagnostics (Issue #535 comments `5703806811`,
`5703860966`, `5704046915`, `5704073674`, `5704161947`, `5704337661`) ruled out
invocation mismatch and universe-membership disagreement as the cause (the A/B
deterministic-trace test matched canonical to machine epsilon under the correct
universe assumption, isolating the residual to somewhere inside `simulate()`/the MC
layer itself, or to a field not yet compared). GPT-5.6's audit then found the deeper
problem: **`35032590321` is not the original M91 authority artifact at all** -- it
is itself a 2026-09-15 reconstruction (`M91_EXACT_RUN_ID=33348554748` in its own
workflow inputs); the true original run (`33348554748`, 2026-08-31) has an expired
artifact. Cross-run parity against a non-authoritative, now-partially-unreproducible
reconstruction is therefore **not a legitimate blocking requirement** -- it correctly
caught real invocation bugs earlier in this process, but continued cross-run
archaeology against it cannot be a permanent gate. **The "Authority-exact baseline
reconstruction" section's promotion-comparator requirement (Amendment 1 point #6) is
formally replaced** with the same-job double-build authority contract specified
below ("Same-job promotion-comparator authority (Amendment 6 replacement)"). The
mechanism-comparator reconstruction requirement (against the canonical STACK2/P3
casebook, which is not affected by this cross-run-artifact problem) is unchanged.

**Amendment 7** (this revision): resolves a structural gap found while starting the
mechanism-comparator reconstruction (Issue #535 comment `5704854605`), before any
candidate output exists. `scripts/backtest/evaluate_rb_stack2_enriched_allocation.py`
-- the source of the mechanism comparator's `enriched_att` input -- is entirely
hardcoded to a single fit/eval pair (its own docstring: "2024 is the only fit season.
2025 is evaluation"; literal `2025`-suffixed input/output filenames; `rosters=
load_rosters([2024,2025])`). No `2023`-fit/`2024`-eval STACK2 casebook exists
anywhere in this repo's history, and nothing in that script is parameterized to
produce one -- refitting STACK2 for a new rotation would itself be new modeling
requiring its own separate review, not a reconstruction. GPT-5.6's comment
`5704913012` (posted concurrently with the gap report, addressing it via already-
established plan language rather than a point-by-point reply) confirmed the
mechanism comparator is "diagnostic only, non-gating for promotion" -- consistent
with "Two comparators" (Amendment 2) and Amendment 4's framing of
`mechanism_diagnostic` as "informative only, never a candidate for production" and
"not itself sufficient for qualification," and with `mechanism_diagnostic` playing no
role anywhere in the Promotion disposition rule's `QUALIFIED` path (only
`deployable_candidate` vs. the **promotion** comparator does). Per that reading:
**the mechanism comparator's authority-exact reconstruction is required for Rotation
2 (2024-fit/2025-eval, the existing canonical STACK2 casebook) only.** Rotation 1's
`mechanism_diagnostic` arm is disclosed as `NOT_CONSTRUCTIBLE_NO_CASEBOOK` --
same fail-closed-but-non-blocking disclosure pattern Amendment 5 established for
Gate 0.1's impossible 2023-2024 parity check -- and does not trigger
`BASELINE_RECONSTRUCTION_FAILURE` on its own. This interpretation is flagged
explicitly, not silently assumed: if GPT-5.6 intended something else (e.g. building a
genuinely new, separately-reviewed 2023-fit STACK2 casebook per option 1 in comment
`5704854605`), that supersedes this amendment on say-so, before any candidate
science is interpreted against it.

**Amendment 8** (this revision): adjudicated by GPT-5.6 on Issue #535 (comment
`5705529323`), replacing the "Rush-yard translation" section's dependency on the P3
efficiency seam (STACK1 `stack_yards/stack_att`, M94C implied-YPC fallback) before
any candidate output exists. While building the candidate, tracing that seam's full
parent chain (Issue #535 comment `5705499168`) found it unconstructible for Rotation
1: STACK1's own evaluator fits weights on 2024 components but scores **2025 only**
(one trace file, `stack1_2025_rb_trace.csv`, no 2024 equivalent ever produced); its
M94C input is likewise 2025-only; and M94C's own M94B input is single-rotation **by
architecture**, not just an unexercised code path -- its docstring bakes 2024 in as
the one-time architecture-selection/blend-holdout season, so substituting an earlier
season would mean redoing that model-family selection, i.e. new modeling across
three already-frozen tiers (M94B, M94C, STACK1), not a reconstruction. Unlike the
mechanism comparator (Amendment 7), this blocked the **decisive** candidate output
itself, not a diagnostic-only arm.

GPT-5.6 rejected both rebuilding that three-tier lineage for 2024 (unnecessary new
modeling, would confound the Lane-A question) and collapsing to a single 2025
rotation (weakens the evidence standard when a cleaner constructible source already
exists in this repo). **The "Rush-yard translation" section below is replaced** with
a hold-incumbent-efficiency-fixed design: since Lane A's stated novelty is
allocation/state segmentation, not efficiency, the candidate reconstructs each
player's *incumbent production ensemble* mean for **both** `rush_att` and
`rush_yards` (same Amendment-6 Build-A component source, same per-rotation frozen
weight rows already used for the promotion comparator, same
`calibration_season < test_season` hard rule extended to both markets), derives
`incumbent_ypc_i = promotion_rush_yards_i / promotion_rush_att_i` per player
per scored-transition row, and translates candidate carries through that held-fixed
efficiency: `candidate_rush_yards_i = candidate_att_i * incumbent_ypc_i`. This makes
the *only* scored-transition change carry allocation -- comparator and candidate
share the same component source and the same per-player efficiency, isolating
exactly the mechanism this candidate claims to improve. Fully constructible both
rotations (no new M94-tier model, no sportsbook/postgame input). A new fail-closed
**constructibility check** (see revised section below) runs before any outcome is
scored: every active player row in every scored-V1 transition team-week must have
finite `promotion_rush_yards_i` and finite `promotion_rush_att_i > 0.20`, or the
disposition is `RUSH_YARD_TRANSLATION_CONSTRUCTIBILITY_FAILURE` -- no dropping,
imputing, clipping, or inventing a fallback after seeing coverage. Stable-week
`deployable_candidate` output is unchanged (still exactly `promotion_rush_yards`,
Amendment 4). All previously frozen adequacy/protected-cohort/bootstrap/per-season/
p90-cat/stable-identity/whole-season gates are unchanged; no new tuning parameter is
introduced.

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
team-week into `transition` vs `stable`, leave `stable` weeks exactly on today's real
production route (the promotion comparator -- see "Two candidate arms" below,
Amendment 4), and apply a targeted reallocation rule **only** on scored V1
`transition` weeks. If this candidate cannot beat the real production (promotion)
comparator specifically on the transition subpopulation while leaving the stable
subpopulation unregressed, it does not qualify -- there is no pooled-average rescue
path. (See "Two comparators" below for why beating P3/STACK2 alone is not
sufficient.)

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
   `6a01c631`) retained a usable date-stamped snapshot for any overlap season.
   **Amendment 5 records the result of this check empirically, done during
   implementation**: neither `nflreadpy` (confirmed via `evaluate_rb_stack2_enriched_
   allocation.py::depth_tables`, native week-tagged 2016-2024, date-stamped only from
   2025) nor ND2B's preserved lineage (`origin/research-rb-nd2b-allocation-env-atlas`
   at `6a01c631`, whose own disclosed audit confirms its date-stamped coverage is
   2025-only) provides a 2023-2024 date-stamped source. **The parity check as
   originally specified cannot be constructed -- no synthetic overlap is manufactured.**
4. **Disposition when no overlap exists (Amendment 5, replaces the original step 4
   numeric-bar check for this case):** GPT-5.6's adjudication (Issue #535 comment
   `5701881512`) is that this is **not** treated as a Gate-0.1 pass via a
   post-hoc-invented numeric bar, and does **not** block scored V1 science either.
   Instead: (a) the 2023-2024 semantic-parity check's disposition is recorded exactly
   as `NOT_CONSTRUCTIBLE_NO_OVERLAP`, disclosed alongside Gate 0's report, never
   silently upgraded to a pass; (b) depth-chart harmonization (both the native
   2016-2024 side and the as-of 2025+ side) remains available **only** for the broad
   `detected_transition` disclosure population defined in "Transition definition"
   below -- it was never an input to the scored V1 trigger, the candidate mechanism,
   any adequacy cohort, or any promotion gate (per Amendment 3's scope narrowing to
   loss/vacancy events, which never depend on depth rank), so this disclosure-only
   limitation carries no scientific cost to the decisive science; (c) any 2025+ depth
   row used anywhere, even for disclosure, must still satisfy the strict `dt <
   kickoff_utc` hard assertion from step 2 above, unchanged; (d) no depth-derived
   field may be introduced into the scored V1 population, candidate formula, or any
   gate later without a new preregistered experiment -- this is a scope/dependency
   correction made prospectively, not a loosening of a gate after seeing results.
5. **Coverage/missingness disclosure by season:** report, per season 2016-2025, the
   fraction of team-weeks with a resolvable depth state under each method's own native
   contract, alongside the `NOT_CONSTRUCTIBLE_NO_OVERLAP` disposition from step 4.

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
- **Resolved PASS (Amendment 5):** empirically verified during implementation --
  `nfl.load_injuries(seasons=[2025])` returns natively `season`/`week`-tagged rows
  (weeks 1-22 incl. playoffs, zero null season/week, `report_status`/`practice_status`
  present) -- identical schema shape to 2016-2024, not date-stamped. The as-of-join
  contingency above never triggers; Gate 0.2 passes on the native week-tag contract
  alone for every season 2016-2025. Preserve this schema/coverage confirmation in the
  Gate-0 report.

### 0.3 Roster-membership source (Amendment 5 -- source formally replaced)

`ACTIVE_ROLES_CSV` was originally named as the audit target, but empirically (during
implementation) its default resolution (`scripts/utils/current_roles_v1.py` ->
`data/roles_ourlads_active_v1.csv`, generated by `scripts/build/
build_production_eligible_active_roles_v1.py`) turns out to be a **current-snapshot-
only** artifact -- its generator has no season/week parameter and no historical-
replay mode, so it cannot reconstruct what active-room state existed for any past
2023-2025 week. It is the correct source for live production, structurally the wrong
source for this candidate's backtest. Per GPT-5.6's adjudication (Issue #535 comment
`5701881512`), this is a **formal source-contract amendment**, not a silent
substitution: `ACTIVE_ROLES_CSV` is replaced by **`nflreadpy.load_rosters_weekly`**
-- the same canonical weekly pregame-universe source
`scripts/backtest/historical_inputs.py::build_pregame_universe_for_week` already
uses, under the explicit contract that target-week box scores are never used to
decide universe membership. Reuse that contract exactly; do not invent a third
source.

Seven frozen requirements, all verified and reported before any candidate outcome is
computed or inspected:

1. Seasons 2023-2025 must each expose `season`, `week`, `team`, `position`, and a
   resolvable player-name identity from `load_rosters_weekly`.
2. RB-room membership is the canonical weekly-roster universe using the same
   team/position/status normalization already used by `build_pregame_universe_for_
   week` (`RB`/`FB`/`HB` subset; same accepted roster-status rule where present,
   e.g. `ALLOWED_ROSTER_STATUS = {"ACT", "INA"}`).
3. Zero duplicate `(season, week, team, player_key)` identities after
   canonicalization.
4. Every regular-season scheduled team-week in **both** OOS test seasons (2024 and
   2025) must have a resolvable RB-room roster state; a team-week with no resolvable
   state is **not** silently dropped -- it routes Gate 0.3 to fail-closed.
5. Every **scored** loss/vacancy event (per "Scored V1 transition" below) must have
   both the current and immediately previous resolvable roster state available from
   this same source lineage -- no substitute or interpolated state.
6. Zero target-week statistics or outcomes may enter membership construction, at any
   step.
7. Report exact source coverage, missingness, duplicate counts, and identity lineage
   for every season 2023-2025 before any candidate outcome is computed or inspected.
- **Fail closed** if requirements 1, 3, 4, 5, or 6 cannot be satisfied; do not
  substitute an unverified proxy silently, and do not drop a team-week to force a
  pass.

### 0.4 Gate 0 disposition (Amendment 5 revision)

Gate 0 is scored and reported **before any candidate outcome is computed or
inspected** -- exactly the same ordering discipline #562 used for its own
reconstruction checksum. The three sub-gates are no longer symmetric under
Amendment 5:

- **0.2 and 0.3 must independently pass their stated numeric bars/requirements.**
  Either failing routes the whole plan to
  `RB_LANE_A_TRANSITION_ALLOCATION_GATE0_BLOCKED` (see Promotion disposition rule)
  -- never a silent fallback to a coarser, less leakage-safe source.
- **0.1 is disclosure-only and does not gate Gate 0's pass/fail disposition.** Its
  step-2 live-side `dt < kickoff_utc` hard assertion still must hold whenever any
  2025+ depth row is used for disclosure (a violation there is a real leakage bug
  and still fails closed), but the step-4 `NOT_CONSTRUCTIBLE_NO_OVERLAP` semantic-
  parity disposition itself does not block Gate 0 as a whole, per Amendment 5 --
  because no scored V1 gate, cohort, or candidate-construction step depends on
  depth rank (see "Detected vs. scored V1 transition" above).

So: `RB_LANE_A_TRANSITION_ALLOCATION_GATE0_BLOCKED` is triggered by a 0.2 or 0.3
failure, or by a 0.1 live-side `dt < kickoff_utc` assertion violation -- never by
0.1's `NOT_CONSTRUCTIBLE_NO_OVERLAP` disposition alone.

## Transition definition (exact, leakage-safe)

### Detected transition (broad; disclosure and Gate-0 testing only, not itself scored)

A team-week `(team, season, week)` is a **detected transition week** if, using only
information available strictly before that week's kickoff (per the Gate-0 harmonized
sources above), any of the following holds relative to the same team's immediately
preceding resolvable depth/status state:

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
that is not a detected transition week under any of the three checks is a **stable
week**. The detected-transition population is reported in full (rate by season, by
trigger type) for disclosure and is the population Gate 0's harmonizers are tested
against, but **it is not the population the V1 candidate is scored on** -- see below.

### Scored V1 transition (narrow; role-collapse/vacancy only -- Amendment 3 fix)

GPT-5.6's Amendment-3 review found the broad detected-transition population above
mixes two directionally opposite football events under one label: a back **losing**
role (departure, new unavailability) and a back **gaining/regaining** role (a return
from `OUT`/`DOUBTFUL`/`IR`/`PUP`, a roster addition). V1 claims only ND1's
"role-collapse" cohort (`RB_ND1_FORENSIC_FAILURE_ATLAS_RESULTS.md`, n=59, MAE 32.59)
-- a vacancy in the room causing the *remaining* backs' roles to redistribute -- not
the broader mixed population. Scoring the broad population under a role-collapse
label would misrepresent what was actually tested.

A team-week is a **scored V1 transition week** if and only if it is a detected
transition week **and** the specific trigger is a **loss/vacancy event** relative to
the immediately preceding resolvable state:

- a player who was fully available now carries `OUT`/`DOUBTFUL`/`IR`/`PUP` (the
  "return" direction of check 2 above is excluded from scoring), OR
- the active RB room's membership **shrinks** by the departure of a previously-
  rostered player (the "addition" direction of check 3 above is excluded from
  scoring).

A depth-rank change (check 1) is **not** an independent scored trigger -- it is
downstream evidence of a loss/vacancy event, not a standalone one (a depth-rank
change absent a co-occurring loss/vacancy event, e.g. a pure coaching-decision
reordering, is a detected transition but not a scored V1 transition; it is disclosed,
not scored, and is explicitly out of V1's claimed scope). The reallocation mechanism
below executes **only** on scored V1 transition weeks; a detected-but-not-scored
transition week is treated identically to a stable week by the candidate mechanism
(no reallocation), and is reported separately from both the stable and scored-
transition subpopulations in every disclosure table.

## Candidate mechanism (scored V1 transition weeks only -- Amendment 3 scope)

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

On a scored V1 (loss/vacancy) transition team-week, let the team's **active
(post-transition) RB room** be indexed `i = 1..N`, and let `pool` be the team-level
pool defined above.

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
4. **HHI (prose corrected, Amendment 5)**: `H = prior_backfield_hhi` (STACK2's own
   existing pre-transition concentration index, `sum((ownshares/ownshares.sum())^2)`
   over the **pre-transition** room -- i.e. the sum of *squared, normalized* shares,
   not raw un-normalized shares as earlier revisions of this document stated --
   computed from history alone and not dependent on any current-week row), reused
   byte-for-byte unchanged; no new HHI calculation or tuning.
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

### Rush-yard translation (Amendment 1 point #2, replaced under Amendment 8 -- hold incumbent efficiency fixed)

Since qualification must ultimately be on rushing-yard accuracy, the candidate's final
projection is defined explicitly. Amendment 8 replaces the original P3-efficiency-seam
design (unconstructible for Rotation 1, see Amendment 8 above) with a
hold-incumbent-efficiency-fixed design, so that carry allocation is the *only* thing
that differs between candidate and comparator on scored rows:

1. **Reconstruct both incumbent-production means** from the same Amendment-6 Build-A
   component source already used for the promotion comparator, for **both**
   `rush_att` and `rush_yards`, using the correct per-rotation frozen weight row for
   each market (Rotation 1: `docs/research/overnight/ensemble_weights_2023_fit_v1.csv`
   -- `rush_att` mc `.260264`/ml `.660877`/state `.078859`, `rush_yards` mc
   `.396067`/ml `.559625`/state `.044308`; Rotation 2: `data/model_ensemble_
   weights.csv` -- `rush_att` mc `.3164919683016017`/ml `.6528957474344519`/state
   `.030612284263946517`, `rush_yards` mc `.5569542426070742`/ml
   `.4430457573929258`/state `0`), producing `promotion_rush_att_i` and
   `promotion_rush_yards_i`. The `calibration_season < test_season` hard rule
   (Amendment 3) applies to both markets, not just `rush_yards`.
2. **Join** the two incumbent means on exact player identity `(season, week, team,
   player_clean_key)`; assert zero duplicates and zero ambiguous joins.
3. **Constructibility check (fail-closed, before any outcome is scored)**: every
   active player row in every scored-V1 transition team-week must have finite
   `promotion_rush_yards_i` and finite `promotion_rush_att_i > 0.20`. If any required
   scored row fails this, disposition is `RUSH_YARD_TRANSLATION_CONSTRUCTIBILITY_
   FAILURE` -- do not drop the row, impute, clip, change the threshold, or invent a
   fallback after seeing coverage.
4. **Held-fixed incumbent efficiency**: `incumbent_ypc_i = promotion_rush_yards_i /
   promotion_rush_att_i`, no fitting or clipping.
5. **Candidate endpoint**: `candidate_rush_yards_i = candidate_att_i *
   incumbent_ypc_i`, equivalently `candidate_rush_yards_i = promotion_rush_yards_i *
   (candidate_att_i / promotion_rush_att_i)`. **No YPC learning/tuning belongs in
   Lane A**; efficiency is untouched, only opportunity allocation changes.

This reallocation-plus-translation formula defines the candidate's output **only on
scored V1 transition weeks**. What happens on every other week -- and which comparator
that output must match -- is specified explicitly in "Two candidate arms" below
(Amendment 4); it is **not** the mechanism comparator, per GPT-5.6's Amendment-4
correction. On scored rows, `promotion_rush_yards_i` (from step 1) is exactly the
`ensemble_proj` promotion comparator value already required elsewhere in this
document -- no separate reconstruction, same value reused.

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
  (blob `41e809b32e4596b8cf18bedbf2b940a2aa3b80b2` at #562's merge commit `91afb3a5`)
  and `scripts/backtest/component_predictions.py` (blob
  `18f7289515b88c84a91479da18526df4cd7f5398`), pinned identically for both OOS
  rotations. **Qualification requires beating the promotion comparator on the
  decisive rushing-yard endpoint** (see Protected cohorts and gates); beating only the
  mechanism comparator is necessary evidence that the reallocation mechanism itself
  works, but is not sufficient by itself for a `QUALIFIED` disposition.

  **Rush-yard ensemble weights, per-rotation (Amendment 3 fix -- temporal leak
  correction):** GPT-5.6's cross-audit (`5699148456`) found `data/model_
  ensemble_weights.csv`'s `rush_yards` row (blob `baade160a124e5cd8ecd415c0276622
  d4b60953f`) is `fit_scope = all_2024_oos_frozen_for_2025` -- fit using 2024
  outcomes. That row is a legitimate pre-2025 authority (2024 < 2025), so it remains
  the correct weight source for **Rotation 2 (test = 2025)**. It is **not** a
  legitimate pre-2024 authority, so it may **not** be used for **Rotation 1
  (test = 2024)** -- doing so would let the comparator's own 2024 calibration leak
  into its 2024 test row, defeating the OOS design for the comparator itself. Instead,
  Rotation 1's promotion comparator uses the already-existing, already-merged
  2023-only frozen fit: `docs/research/overnight/ensemble_weights_2023_fit_v1.csv`
  (blob `b9c193b9b9d11578f3dd17ae6da241715878bea5`, produced by `scripts/research/
  fit_2023_ensemble_weights_v1.py` per `.github/workflows/backtest-ensemble-weight-
  2023-fit-v1.yml`, trained on 2023 component predictions with `prior_season = 2022`,
  merged via PR #545 (commit `3f75c8ba`), unchanged since): `rush_yards` mc_weight
  `0.396067`, ml_weight `0.559625`, state_weight `0.044308`, `calibration_rows=3180`,
  method `nonnegative_oos_linear_blend_v2` -- the identical fitting method as the
  production file's row, differing only in which season it was fit on. This mirrors
  the pattern the production file itself already uses for `rec_yards`/`receptions`
  (`fit_2023_only_blind_holdout_2024_2025`), just applied to the market this
  candidate actually needs.

  **Hard per-rotation provenance assertion:** before scoring, assert
  `max(calibration_season_used) < test_season` for the `rush_yards` weight row
  actually applied in each rotation's promotion-comparator reconstruction (Rotation 1:
  `2023 < 2024`; Rotation 2: `2024 < 2025`). If this cannot be proven for either
  rotation from the weight file's own recorded provenance fields, the reconstruction
  fails closed -- see Promotion disposition rule -- rather than silently reusing a
  wrong-season weight file. No new weight-fitting is performed to satisfy this: only
  the already-frozen, already-reviewed 2023-only artifact is reused, unchanged, exactly
  as GPT-5.6's finding required ("Do not silently backfit one after looking at
  candidate results").

## Two candidate arms: deployable vs. mechanism-diagnostic (Amendment 4)

GPT-5.6's Amendment-4 cross-audit found that defining stable-week candidate output as
identical to the **mechanism comparator** (P3/STACK2, an unpromoted diagnostic route)
was a deployment-architecture defect: it would let a qualified Lane A module silently
replace today's real stable-week production route with an unpromoted research route,
even though "Two comparators" above already establishes the promotion comparator
(`ensemble_proj`) as the actual Weeks-2-18 production authority. Two explicit,
non-interchangeable arms are frozen instead:

- **`deployable_candidate`** (the only arm with any bearing on a future production
  question): on a **scored V1 transition week**, equal to the transition reallocation
  candidate's `candidate_rush_yards_i` as defined in "Candidate mechanism" above. On
  **every other week** (stable weeks, and detected-but-not-scored transition weeks
  alike), equal to the **promotion comparator** (`ensemble_proj`, per its frozen
  per-rotation reconstruction above) -- **byte-identical, by construction, never the
  mechanism comparator.** This is the only arm any future production-integration PR
  could ever propose enabling.
- **`mechanism_diagnostic`** (informative only, never a candidate for production): on
  scored V1 transition weeks only, the same transition reallocation candidate compared
  directly against the **mechanism comparator** (P3/STACK2). Its sole purpose is
  answering whether the reallocation mechanism itself improved on STACK2's pooled
  model on exactly the population it failed (see "Why this candidate" above) -- it
  never defines stable-week output and is not itself sufficient for qualification (see
  "Two comparators" above, unchanged from Amendment 2).

**Hard stable-identity gate (new, Amendment 4):** for every non-scored row (stable
weeks and detected-but-not-scored transition weeks) in both OOS rotations,
`max(abs(deployable_candidate_rush_yards - promotion_comparator_rush_yards)) == 0.0`
-- a code-identity assertion, not a statistical gate, exactly analogous to the
conservation identity check already required in "Candidate mechanism" above. A
failure here means the deployable arm was built wrong, not that the candidate is
weak; see Promotion disposition rule.

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
comparator is not evidence. **Scope corrected under Amendment 4**: the mechanism
comparator feeds only the diagnostic-only `mechanism_diagnostic` arm, which scores
scored V1 transition rows exclusively (see "Two candidate arms"), so its
reconstruction proof is required on scored V1 transition rows only -- stable-week
mechanism-comparator values are not used anywhere and need no reconstruction proof.
The promotion comparator, by contrast, is required on **every** row (stable and
transition alike): transition rows need it as the decisive qualification comparator,
and stable/non-scored rows need it as the exact value the `deployable_candidate` arm
must reproduce under the hard stable-identity gate.

**Rotation scope corrected under Amendment 7**: no canonical STACK2/P3 casebook
exists for a 2023-fit/2024-eval rotation (`evaluate_rb_stack2_enriched_allocation.py`
is hardcoded to the single existing 2024-fit/2025-eval casebook). The mechanism
comparator's authority-exact reconstruction proof is therefore required for
**Rotation 2 only**. Rotation 1's `mechanism_diagnostic` arm is disclosed as
`NOT_CONSTRUCTIBLE_NO_CASEBOOK` -- non-blocking, since `mechanism_diagnostic` never
gates the `QUALIFIED` disposition (see "Two candidate arms," Amendment 4). The
promotion comparator's every-row, both-rotations requirement is unchanged.

**Per-rotation weight-file provenance check (Amendment 3 addition):** as part of this
same reconstruction proof, confirm which `rush_yards` weight row was actually applied
for each rotation's promotion-comparator reconstruction -- Rotation 1 must resolve to
`docs/research/overnight/ensemble_weights_2023_fit_v1.csv` (blob
`b9c193b9b9d11578f3dd17ae6da241715878bea5`), Rotation 2 must resolve to `data/
model_ensemble_weights.csv` (blob `baade160a124e5cd8ecd415c0276622d4b60953f`) -- and
assert `max(calibration_season_used) < test_season` holds for the row actually used
in each. A reconstruction that resolves to the wrong weight file for either rotation
(e.g. accidentally applying the 2024-fit row to the 2024 test rotation) is treated
identically to any other baseline-reconstruction failure below -- it fails closed,
not silently corrected after the fact.

## Same-job promotion-comparator authority (Amendment 6 replacement)

Replaces this document's promotion-comparator cross-run parity requirement (the
promotion-comparator half of "Authority-exact baseline reconstruction" above), per
Amendment 6 and GPT-5.6's Issue #535 comment `5704381940`. The mechanism-comparator
reconstruction requirement (against the canonical STACK2/P3 casebook) is unaffected
and still applies exactly as specified above.

For each rotation, the promotion comparator's `component_predictions.csv` is
established by:

1. Building the exact M91 invocation (the same helper-script sequence already
   verified in this branch: `build_historical_inputs.py` -> `enrich_historical_
   defense.py` -> `validate_historical_inputs.py` -> `historical_player_logs_m95q.py`
   -> `build_historical_injuries.py` -> `build_historical_weather.py` ->
   `walk_forward.py --injuries --weather --iterations 2000`) **twice, in the same CI
   job**, from one frozen input snapshot/manifest (i.e. build A and build B both read
   the identical downloaded/generated historical-input files -- not two independent
   re-fetches -- isolating this check to the deterministic code path, not to whether
   an upstream data source can change between two separate fetches).
2. Persisting, as evidence: SHA256 of every historical-input file that feeds the
   build (player/team-weekly/schedule history, injuries, weather, pregame universe
   per week), the repo code SHA, the `scripts/simulation_v2.py` blob SHA
   specifically, the Python/numpy/pandas versions, the seed policy (`42 + week`), and
   `iterations=2000`.
3. Requiring **exact row identity** between build A and build B on `(season, week,
   team, player_clean_key, market)` -- zero rows unique to either build.
4. Requiring `mc_proj`, `ml_proj`, and `state_proj` equality between build A and
   build B to machine tolerance (`<=1e-6`; same-seed/same-input reruns are expected
   to reproduce far tighter than this).
5. Preserving the raw pre-actual-filter simulation-universe identity/order manifest
   (every player row entering `simulate()`, in order, per week) as part of the
   evidence, so a membership/order ambiguity of the kind this amendment resolves
   cannot recur undetected.
6. Designating **build A as the single canonical component-prediction source**
   consumed by both (a) the promotion comparator used for candidate qualification,
   and (b) the Lane-A candidate's own `ensemble_proj` construction -- the critical
   paired-input guarantee: candidate and comparator are never built from two
   separately-reconstructed component-prediction sources, so no future cross-run
   discrepancy of this kind can silently bias a candidate-vs-comparator comparison.
7. Disposition naming: a pass is recorded as `SAME_JOB_AUTHORITY_RECONSTRUCTION_
   PASS`, never as "parity passed" against `35032590321` or any other cross-run
   artifact. The `35032590321` cross-run discrepancy remains fully disclosed in this
   document's history (Amendment 6 above) as a historical reconstruction diagnostic
   that correctly caught real invocation bugs earlier in this process -- it is not
   deleted or hidden, only no longer treated as a blocking requirement. A failure of
   this same-job contract (row-identity mismatch, or any of `mc_proj`/`ml_proj`/
   `state_proj` exceeding `1e-6` between build A and build B) stops the process the
   same as any other baseline-reconstruction failure below -- it is a genuine
   nondeterminism/environment defect requiring its own fix, not something to tune
   around.
8. Per-rotation weight-file provenance (Amendment 3) is unchanged by this
   replacement: Rotation 1 still resolves to `docs/research/overnight/
   ensemble_weights_2023_fit_v1.csv`, Rotation 2 still resolves to `data/
   model_ensemble_weights.csv`, both asserted exactly as specified above.

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

Scored separately on **(a) the scored V1 transition subpopulation** (loss/vacancy
events only, per Amendment 3) and **(b) the stable subpopulation**, each season, each
rotation, against **both** comparators. The detected-but-not-scored transition
population (returns, roster additions, pure depth-rank reordering absent a
co-occurring loss/vacancy) is disclosed separately and gates nothing. Per Amendment 1
point #7 and Amendment 2 point #1, both the allocation mechanism and the final
rush-yard production endpoint are scored, and **the promotion comparator on the
production endpoint is decisive** for qualification.

- **Mechanism evidence (informative, against the mechanism comparator only -- not
  itself sufficient)**: carry MAE / allocation error on transition rows, candidate vs.
  P3/STACK2.
- **Production endpoint (decisive, against the promotion comparator)**: pooled
  **rushing-yard MAE** on transition rows must be strictly better than the promotion
  comparator (`ensemble_proj`), independently in Rotation 1 (2024 OOS) and Rotation 2
  (2025 OOS). A carry-allocation improvement over the mechanism comparator that does
  not improve rushing-yard MAE over the **promotion** comparator does **not** qualify.
- **Stable subpopulation must not regress (corrected, Amendment 4)**: the
  `deployable_candidate` arm's non-scored-week output is defined identical to the
  **promotion comparator** by construction (`max abs diff == 0.0`, the hard
  stable-identity gate in "Two candidate arms" above) -- a code-identity assertion,
  not a statistical gate. This replaces the prior, incorrect Amendment-1/2/3 wording
  that compared stable weeks to the mechanism comparator.
- **Deployable whole-season safety check (new, Amendment 4)**: because non-scored
  rows are byte-identical to the promotion comparator by construction, the
  `deployable_candidate` arm's **whole-season** (every row, stable and scored
  transition alike) rushing-yard MAE must be non-worse than the promotion comparator,
  independently in each OOS rotation -- report the exact delta. This is an
  integration-sanity gate confirming the hybrid arm as a whole is safe to consider for
  deployment; per GPT-5.6's framing it is **required** but cannot by itself rescue a
  failure on the decisive scored-transition-only gate above -- both must pass.
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

- **Gate 0 (all three sub-gates) passes; the mechanism comparator's authority-exact
  reconstruction passes for Rotation 2 (Amendment 7 -- Rotation 1 discloses
  `NOT_CONSTRUCTIBLE_NO_CASEBOOK`, non-blocking) and the promotion comparator's
  same-job authority contract (Amendment 6) reaches
  `SAME_JOB_AUTHORITY_RECONSTRUCTION_PASS` for both rotations; both rotations have
  adequate (`n>=30`) support on the overall transition
  subpopulation and both required protected cohorts; both rotations clear every
  required gate on the rushing-yard production endpoint against the promotion
  comparator; the Amendment-4 stable-identity gate holds exactly on every non-scored
  row in both rotations; and the Amendment-4 whole-season deployable safety check is
  non-worse than the promotion comparator in both rotations** ->
  `RB_LANE_A_TRANSITION_ALLOCATION_QUALIFIED`. Eligible for a separate,
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
- **The mechanism comparator's authority-exact reconstruction fails for Rotation 2
  (a Rotation-1 `NOT_CONSTRUCTIBLE_NO_CASEBOOK` disclosure per Amendment 7 does
  *not* trigger this disposition on its own), or the promotion comparator's
  same-job authority contract (Amendment 6) fails (row-identity mismatch between
  build A/B, or any of `mc_proj`/`ml_proj`/`state_proj` exceeding `1e-6` between
  build A/B), or the Amendment-3 per-rotation weight-file provenance check fails
  (wrong weight file resolved for a rotation, or
  `max(calibration_season_used) < test_season` cannot be proven)** ->
  `RB_LANE_A_TRANSITION_ALLOCATION_BASELINE_RECONSTRUCTION_FAILURE`. Candidate
  science does not proceed until the failing reconstruction is fixed and re-verified.
- **The Amendment-8 rush-yard translation constructibility check fails (any active
  player row in any scored-V1 transition team-week, either rotation, lacks a finite
  `promotion_rush_yards_i` or a finite `promotion_rush_att_i > 0.20`)** ->
  `RUSH_YARD_TRANSLATION_CONSTRUCTIBILITY_FAILURE`. Candidate science does not
  proceed; no row is dropped, imputed, clipped, or given an invented fallback to
  route around the failure.
- **Either OOS rotation's overall transition subpopulation, or either required
  protected cohort (carries>=20, yards>=100) within it, is below the `n>=30` adequacy
  bar** -> `RB_LANE_A_TRANSITION_ALLOCATION_INSUFFICIENT_EVIDENCE`, fail-closed --
  this alone prevents `QUALIFIED` regardless of how any scoreable gate performs.
  Disclosure-only slices (carries>=25) never trigger this regardless of their count.
- **Gate 0, both reconstructions, and adequacy all clear, but any required
  protected-cohort/bootstrap/conservation/stable-identity/whole-season-safety/
  per-season gate fails against the promotion comparator** ->
  `RB_LANE_A_TRANSITION_ALLOCATION_NOT_QUALIFIED`. No rescue tuning, no re-test with
  adjusted thresholds, no subset search. Same stop-rule discipline as every other
  closed RB lane in this program (STACK6, STACK6B, M95T, Role-Order-Remap-V1). A
  stable-identity-gate failure specifically (Amendment 4) indicates a
  `deployable_candidate` construction bug, not candidate weakness -- it is fixed and
  re-verified before any other gate is re-scored, not tuned around.

## Explicitly out of scope

- No change to stable-week (or detected-but-not-scored transition-week) RB output --
  by construction, the `deployable_candidate` arm is identical to the **promotion
  comparator** (`ensemble_proj`, today's real Weeks-2-18 production route) on every
  such row (Amendment 4). The mechanism comparator (P3/STACK2) is used only inside the
  diagnostic-only `mechanism_diagnostic` arm and never defines any candidate output.
- No change to WR/TE/QB, receiving markets, or any non-rushing RB market.
- Does not reopen STACK6 team-rush-context slicing, direct depth-rank carry
  assignment, or pooled-whole-season secondary-role features (STACK6B) -- this
  candidate is structurally distinct from all three (see "Why this candidate" above).
- Does not fold in #562's yard-difficulty MC-width work -- that is a separate,
  already-qualified, already-forward-confirming lane; this candidate concerns the
  rushing-yard **mean/allocation**, not distribution width.
- No YPC/efficiency learning or tuning -- efficiency is inherited unchanged from the
  existing P3 seam.
