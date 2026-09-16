# RB-PD2 Yard-Difficulty MC-Width V1 — Forward/Shadow Confirmation Plan

**FROZEN BEFORE ANY WEEK-2-OR-LATER OUTCOME IS OBSERVED. RESEARCH ONLY. NO PRODUCTION CHANGE.**

This is the separately-frozen forward-confirmation contract required by PR #562's own
hard-gate section before any production promotion of
`RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED` (merged `91afb3a5`, evaluator
`scripts/research/evaluate_rb_pd2_yard_difficulty_mc_width_v1.py`). Per Issue #535
(GPT-5.6, comment `5690509775`): shadow capture only, zero effect on published pricing,
predeclared gates, predeclared evidence horizon, no week-to-week auto-tuning, no
outcome-driven stopping.

## What runs

For every live pregame RB `rush_yards` row the production Full Slate pipeline prices,
compute and persist **both**:

- **baseline**: the exact current-production Monte Carlo distribution/fair-probability
  outputs, unchanged.
- **candidate**: the identical #562 transform applied to that same distribution --
  `strict_prior_difficulty_scores` -> `width_multiplier` -> `widen_mean_neutral`, with
  the frozen `0.50` onset / `1.30x` cap / `0.30` width-cap coefficient, byte-identical to
  the merged evaluator code. No retuning, no re-derivation of the difficulty-score
  reference logic.

Both are written to an immutable, pre-kickoff sidecar artifact
(`data/backtests/rb_pd2_yard_width_forward_shadow/{season}_week_{week:02d}.csv`, one row
per RB rush-yards identity, columns: identity keys, `prior8_yard_mae`,
`difficulty_score`, `width_mult`, baseline/candidate quantiles needed for CRPS and
50/75/100-yard tail probabilities, `baseline_mean`, `candidate_mean`). The sidecar is
written **before** that week's games kick off and is never edited afterward.

## What does not change

- `outputs/props_pricing_offers.csv`, the master betting workbook, `fair_prob`,
  `ev_roi`, `best_side`, `HAS EDGE`/`PASS` labels, and every other published/priced
  artifact are computed exactly as they are today. The candidate distribution is
  never read by any pricing/decision code path.
- No sportsbook input feeds the shadow computation beyond what baseline projections
  already use -- still football-only, matching #562's own contract.
- The width coefficient (`0.30`), onset (`0.50`), cap, percentile convention, and
  eligibility rule (`prior_games>=4`, `>=100` strictly-prior reference rows) are frozen
  for the entire confirmation window. No mid-window edits.

## Evidence horizon (fixed now, before Week 2 exists)

#562's own backtest treated a season-slice as adequate at `>=100` high-difficulty rows.
This repo's historical seasons run roughly 30-35 RB `rush_yards` rows/week
league-wide, with the high-difficulty quartile at ~25% of that (~8/week) --
reaching `n>=100` in the high-difficulty slice alone takes on the order of 12-13 weeks.
Two fixed calendar checkpoints are predeclared now, neither adjustable once shadow
capture starts:

- **Week 8 interim checkpoint**: descriptive-only status report to Issue #535 (rows
  captured, gate values as computed so far). **No promotion/no-promotion decision is
  permitted at this checkpoint under any outcome** -- it exists for transparency, not
  as a stopping rule, and is explicitly barred from being used as one.
- **Week 14 primary confirmation checkpoint**: the only point a disposition is issued,
  scored against the gates below on all shadow rows captured through that week.
  Chosen as the earliest point the high-difficulty cohort plausibly reaches the
  same `n>=100` adequacy bar #562 itself required, with buffer before playoff-week
  roster/usage volatility.

If real accumulated evidence forces an earlier or later look, that requires a **new**
frozen amendment posted to Issue #535 before the data is inspected -- not an ad hoc
call once results are visible.

## Predeclared forward-confirmation gates (mirrors #562's own hard gates)

Scored once, at the Week-14 checkpoint, on real 2026 pregame candidate/baseline
distributions joined to real game outcomes:

- `FA_reconstruction_mean_neutral`: candidate mean equals baseline mean within `1e-8`
  on every shadow row (confirms the live transform stayed mean-neutral in production
  conditions, not just backtest conditions).
- `FA_point_mae_identical`: pooled point-MAE delta (candidate vs. baseline) within
  `1e-8`.
- `FC_pooled_crps_improvement_ge_0_5pct`: pooled CRPS improves `>=0.5%` vs. baseline.
- `FC_player_cluster_bootstrap_p_ge_0_95`: paired player-cluster bootstrap (same
  method/seed convention as `player_cluster_bootstrap_probability` in the merged
  evaluator, 10,000 reps) `P(candidate CRPS < baseline) >= 0.95`.
- `FD_high_crps_improvement_ge_1pct`: high-difficulty-quartile (global Q75 of
  `difficulty_score` across captured shadow rows) CRPS improves `>=1.0%`.
- `FD_high_coverage80_gap_strictly_better` / `FD_high_coverage90_gap_strictly_better`:
  high-difficulty 80%/90% interval coverage gaps strictly improve.
- `FE_pooled_coverage80_nonworse` / `FE_pooled_coverage90_nonworse`: pooled coverage
  gaps non-worse, at least one strictly better.
- `FF_brier100_strictly_better`: 100-yard tail Brier score strictly better.
  `FF_brier50_nonworse` / `FF_brier75_nonworse`: 50/75-yard non-worse.

All fail-closed: if fewer than 100 high-difficulty rows have accumulated by Week 14,
the high-difficulty gates (`FD_*`) cannot be scored and the disposition is
`RB_YARD_DIFFICULTY_MC_WIDTH_FORWARD_INSUFFICIENT_EVIDENCE`, not a pass.

## Disposition rule

- **All gates pass** -> `RB_YARD_DIFFICULTY_MC_WIDTH_FORWARD_CONFIRMED`. Eligible for a
  separate production-integration PR (still its own frozen, reviewed change -- this
  plan does not itself authorize enabling anything).
- **Any gate fails** -> `RB_YARD_DIFFICULTY_MC_WIDTH_NOT_FORWARD_CONFIRMED`. The
  candidate stays research/shadow-only indefinitely. No rescue tuning, no re-test with
  adjusted thresholds, no early re-look with a relaxed bar.
- **Insufficient captured rows at Week 14** -> report the insufficiency and the
  gates that could be scored; do not extend the window informally -- an extension
  requires a new frozen amendment.

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
