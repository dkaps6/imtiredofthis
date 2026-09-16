# RB-PD2 Yard-Difficulty MC-Width V1 — Forward/Shadow Confirmation Plan

**FROZEN BEFORE ANY WEEK-2-OR-LATER OUTCOME IS OBSERVED. RESEARCH ONLY. NO PRODUCTION CHANGE.**

**Amendment 1** (this revision): incorporates GPT-5.6's five prospective corrections
from Issue #535 comment `5690752971`, before any shadow implementation or Week-2
outcome exposure -- crossed player x game bootstrap restored, Week 8 stripped of
scientific values, evidence-count-driven (not calendar-driven) stopping rule,
structural job isolation, and tightened sportsbook wording. No implementation existed
under the prior revision; nothing here is a post-implementation change.

This is the separately-frozen forward-confirmation contract required by PR #562's own
hard-gate section before any production promotion of
`RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED` (merged `91afb3a5`, evaluator
`scripts/research/evaluate_rb_pd2_yard_difficulty_mc_width_v1.py`). Per Issue #535
(GPT-5.6, comments `5690509775` and `5690752971`): shadow capture only, zero effect on
published pricing, predeclared gates, predeclared evidence horizon, no week-to-week
auto-tuning, no outcome-driven stopping.

## What runs, and when

The shadow computation is a **separate, downstream job that runs only after that
slate's production football/pricing artifacts are already fully materialized**. It
reads those frozen outputs; it has **no write path back into any pricing/workbook
artifact**, and no production job reads its path -- isolation is structural (a
different job/step boundary in the pipeline), not merely a contractual promise in
this doc.

For every RB `rush_yards` row in that slate's already-priced production output,
compute and persist **both**:

- **baseline**: the exact current-production Monte Carlo distribution/fair-probability
  outputs, read from the already-materialized production artifacts, unchanged.
- **candidate**: the identical #562 transform applied to that same distribution --
  `strict_prior_difficulty_scores` -> `width_multiplier` -> `widen_mean_neutral`, with
  the frozen `0.50` onset / `1.30x` cap / `0.30` width-cap coefficient, byte-identical to
  the merged evaluator code. No retuning, no re-derivation of the difficulty-score
  reference logic. **The transform consumes the football distribution and strictly-prior
  same-player difficulty history only -- sportsbook lines/odds are never inputs to the
  shadow transform itself.** Any later probability-to-market-line comparison is a
  downstream evaluation step, not part of the transform.

Both are written to an immutable, pre-kickoff sidecar artifact
(`data/backtests/rb_pd2_yard_width_forward_shadow/{season}_week_{week:02d}.csv`, one row
per RB rush-yards identity, columns: identity keys, `prior8_yard_mae`,
`difficulty_score`, `width_mult`, baseline/candidate quantiles needed for CRPS and
50/75/100-yard tail probabilities, `baseline_mean`, `candidate_mean`), alongside a
**lineage manifest** (source production artifact identity/hash, sidecar write
timestamp, confirmation that the write occurred strictly before that slate's earliest
kickoff). The sidecar and manifest are written once, before kickoff, and are never
edited afterward -- outcome grading later *reads* them; it never modifies them.

## What does not change

- `outputs/props_pricing_offers.csv`, the master betting workbook, `fair_prob`,
  `ev_roi`, `best_side`, `HAS EDGE`/`PASS` labels, and every other published/priced
  artifact are computed exactly as they are today, by the exact same code path,
  before the shadow job even starts. The candidate distribution is never read by any
  pricing/decision code path, structurally (job ordering), not just by omission.
- No sportsbook input feeds the shadow transform (see above). Still football-only,
  matching #562's own contract.
- The width coefficient (`0.30`), onset (`0.50`), cap, percentile convention, and
  eligibility rule (`prior_games>=4`, `>=100` strictly-prior reference rows) are frozen
  for the entire confirmation window. No mid-window edits.

## Evidence horizon: sample-size-driven, not calendar-driven

#562's own backtest treated a season-slice as adequate at `>=100` high-difficulty rows.
Rather than picking a calendar week and risking an awkward "extend or not" call once
outcomes are already visible, the stopping rule is frozen now as a fixed **count**
condition:

- **No scientific disposition is issued before the high-difficulty adequacy count is
  reached.** The first and only scientific look occurs at the first weekly lock where
  cumulative captured high-difficulty rows (`difficulty_score` >= the running global
  Q75 across all captured shadow rows to that point) reaches `n>=100`.
- **Hard cap at Week 18** (end of regular season): if `n>=100` has still not been
  reached by then, the disposition is
  `RB_YARD_DIFFICULTY_MC_WIDTH_FORWARD_INSUFFICIENT_EVIDENCE` -- not a pass, not an
  extension.
- **Week 8 operational-integrity checkpoint** (fixed, non-scientific): a status report
  to Issue #535 containing only mechanical/operational facts -- rows captured to date,
  missingness, identity-uniqueness pass/fail, pre-kickoff timestamp proof, sidecar
  immutability verification. **No CRPS, coverage, Brier, bootstrap probability, or any
  candidate-vs-baseline comparison value is computed or reported at this checkpoint.**
  It cannot function as an informal stopping signal because no scientific value exists
  yet to look at.

If the evidence-count condition or the Week-18 cap need to change, that requires a
**new** frozen amendment posted to Issue #535 before any further data is inspected --
never an ad hoc call once results are visible.

## Predeclared forward-confirmation gates (mirrors #562's own hard gates exactly)

Scored exactly once, at the evidence-count-triggered checkpoint above, on real 2026
pregame candidate/baseline distributions joined to real game outcomes:

- `FA_reconstruction_mean_neutral`: candidate mean equals baseline mean within `1e-8`
  on every shadow row (confirms the live transform stayed mean-neutral in production
  conditions, not just backtest conditions).
- `FA_point_mae_identical`: pooled point-MAE delta (candidate vs. baseline) within
  `1e-8`.
- `FC_pooled_crps_improvement_ge_0_5pct`: pooled CRPS improves `>=0.5%` vs. baseline.
- `FC_player_cluster_bootstrap_p_ge_0_95`: paired player-cluster bootstrap (same
  method/seed convention as `player_cluster_bootstrap_probability` in the merged
  evaluator, 10,000 reps) `P(candidate CRPS < baseline) >= 0.95`.
- `FC_crossed_player_game_bootstrap_p_ge_0_95`: the Amendment-3 crossed player x game
  bootstrap, restored per GPT-5.6's correction -- `game_key = (season, week,
  min(team,opponent), max(team,opponent))`, independent player/game resampling,
  multiplicative row weights, 10,000 valid reps, seed `42027`, redraw zero-weight
  reps, `P(weighted candidate CRPS < baseline) >= 0.95`. Required **in addition to**
  `FC_player_cluster_bootstrap_p_ge_0_95`, exactly as #562 itself required both.
- `FD_high_crps_improvement_ge_1pct`: high-difficulty-quartile (global Q75 of
  `difficulty_score` across captured shadow rows) CRPS improves `>=1.0%`.
- `FD_high_coverage80_gap_strictly_better` / `FD_high_coverage90_gap_strictly_better`:
  high-difficulty 80%/90% interval coverage gaps strictly improve.
- `FE_pooled_coverage80_nonworse` / `FE_pooled_coverage90_nonworse`: pooled coverage
  gaps non-worse, at least one strictly better.
- `FF_brier100_strictly_better`: 100-yard tail Brier score strictly better.
  `FF_brier50_nonworse` / `FF_brier75_nonworse`: 50/75-yard non-worse.

## Disposition rule

- **All gates pass** -> `RB_YARD_DIFFICULTY_MC_WIDTH_FORWARD_CONFIRMED`. Eligible for a
  separate production-integration PR (still its own frozen, reviewed change -- this
  plan does not itself authorize enabling anything).
- **Any gate fails** -> `RB_YARD_DIFFICULTY_MC_WIDTH_NOT_FORWARD_CONFIRMED`. The
  candidate stays research/shadow-only indefinitely. No rescue tuning, no re-test with
  adjusted thresholds, no early re-look with a relaxed bar.
- **`n>=100` high-difficulty rows not reached by the Week-18 cap** ->
  `RB_YARD_DIFFICULTY_MC_WIDTH_FORWARD_INSUFFICIENT_EVIDENCE`. Report whatever
  descriptive/operational facts exist; do not issue a scientific disposition; do not
  extend without a new frozen amendment.

## Continuous monitoring after any future promotion

Per Issue #535 (comment `5690598737`): if this candidate is later forward-confirmed
and promoted, shadow capture continues for the rest of the 2026 season regardless --
production authority and permanent shadow laboratory are not mutually exclusive. That
continuation is out of scope for this plan (which only covers reaching the first
confirmation disposition) and will be specified in the promotion PR itself.

## Explicitly out of scope

- Carry-width remains a separate, unauthorized lane -- not folded into this
  confirmation.
- No change to any other market, position, or production pathway.
- No new predictor/feature -- this confirms the already-qualified #562 mechanism only.
