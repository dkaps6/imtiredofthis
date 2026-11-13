# RB-PD2 Yard-Difficulty MC-Width V1 — Forward/Shadow Confirmation Plan

**FROZEN BEFORE ANY WEEK-2-OR-LATER OUTCOME IS OBSERVED. RESEARCH ONLY. NO PRODUCTION CHANGE.**

**Amendment 1**: incorporated GPT-5.6's five prospective corrections from Issue #535
comment `5690752971` -- crossed player x game bootstrap restored, Week 8 stripped of
scientific values, evidence-count-driven (not calendar-driven) stopping rule,
structural job isolation, and tightened sportsbook wording.

**Amendment 2** (this revision): incorporates GPT-5.6's two further prospective
corrections from Issue #535 comment `5690848753`, before any shadow implementation or
Week-2 outcome exposure -- a frozen qualification-parent track (so the confirmation
cannot silently mix pre- and post-change production authorities if RB production
changes mid-window) and a durable, content-hashed, append-only evidence storage
contract. No implementation exists under any prior revision; nothing here is a
post-implementation change.

This is the separately-frozen forward-confirmation contract required by PR #562's own
hard-gate section before any production promotion of
`RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED` (merged `91afb3a5`, evaluator
`scripts/research/evaluate_rb_pd2_yard_difficulty_mc_width_v1.py`). Per Issue #535
(GPT-5.6, comments `5690509775` and `5690752971`): shadow capture only, zero effect on
published pricing, predeclared gates, predeclared evidence horizon, no week-to-week
auto-tuning, no outcome-driven stopping.

## Frozen qualification-parent track (Amendment 2)

`RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED` was earned against a specific production route,
not against "whatever production happens to be running this week." If RB production
changes during the confirmation window (e.g. a Lane A allocation candidate is promoted
before this window closes), letting the shadow baseline silently track the new
production route would mix two different mean authorities into one `prior8_yard_mae`
difficulty history and one CRPS/coverage comparison -- contaminating the exact
experiment #562 qualified.

The parent is therefore pinned, not re-resolved weekly:

- **Frozen parent identity**: the RB rushing-projection code as it existed at #562's
  merge commit `91afb3a5` -- specifically `scripts/modeling/rb_rush_synthesis_v1.py`
  (blob `f5cf574144faf4cc527f2ef6627511ae774d2431`), `scripts/modeling/rb_pricing_
  adapter_v1.py` (blob `b9ed94dc39fc0c8397675859fd1c659ae689ff14`),
  `scripts/backtest/component_predictions.py` (blob
  `18f7289515b88c84a91479da18526df4cd7f5398`), and `scripts/modeling/ensemble_v2.py`
  (blob `41e809b32e4596b8cf18bedbf2b940a2aa3b80b2`). None of these files were modified
  by #562 itself (confirmed: #562's merge touched only `scripts/research/`,
  `scripts/backtest/historical_player_logs.py`, and
  `scripts/backtest/persist_historical_simulated_outcomes_v1.py`).
- **Every week of the confirmation window**, the baseline and candidate distributions
  for the scientific track are built by invoking this exact frozen-parent code
  (pinned at these blob SHAs, checked out into an isolated execution context -- not
  whatever `main` currently contains) against that week's real pregame inputs. The
  frozen parent's own `prior8_yard_mae`/difficulty-score history accumulates only from
  this same frozen-parent lineage for the entire window -- never mixed with a
  different production authority's errors.
- **Parent version/hash is recorded in every sidecar row and manifest** (see storage
  contract below), so any future audit can verify which code produced each row.
- If production RB projections change mid-window, a **separate, additional
  contemporaneous-live-authority companion track** may be captured for operational
  learning, but it is informational only and never replaces or mixes into the
  frozen-parent rows used for #562's scientific confirmation disposition.

## What runs, and when

The shadow computation is a **separate, downstream job that runs only after that
slate's production football/pricing artifacts are already fully materialized**. It
reads those frozen outputs; it has **no write path back into any pricing/workbook
artifact**, and no production job reads its path -- isolation is structural (a
different job/step boundary in the pipeline), not merely a contractual promise in
this doc.

For every RB `rush_yards` row in that slate's already-priced production output,
compute and persist **both**, using the frozen-parent lineage above:

- **baseline**: the frozen-parent Monte Carlo distribution/fair-probability outputs
  for that row, built from that week's real pregame inputs via the pinned parent code.
- **candidate**: the identical #562 transform applied to that same distribution --
  `strict_prior_difficulty_scores` -> `width_multiplier` -> `widen_mean_neutral`, with
  the frozen `0.50` onset / `1.30x` cap / `0.30` width-cap coefficient, byte-identical to
  the merged evaluator code. No retuning, no re-derivation of the difficulty-score
  reference logic. **The transform consumes the football distribution and strictly-prior
  same-player difficulty history only -- sportsbook lines/odds are never inputs to the
  shadow transform itself.** Any later probability-to-market-line comparison is a
  downstream evaluation step, not part of the transform.

### Durable evidence storage (Amendment 2)

A short-lived CI artifact is not sufficient -- the sidecar must still exist and be
independently auditable at the evidence-count-triggered checkpoint (which may be as
late as Week 18) and after the season ends. Storage contract:

- **One immutable pre-kickoff file + manifest per slate**, committed into the repo at
  `data/backtests/rb_pd2_yard_width_forward_shadow/{season}_week_{week:02d}.csv` plus
  `{season}_week_{week:02d}_manifest.json`, one row per RB rush-yards identity.
  Sidecar columns: identity keys, `prior8_yard_mae`, `difficulty_score`, `width_mult`,
  baseline/candidate quantiles needed for CRPS and 50/75/100-yard tail probabilities,
  `baseline_mean`, `candidate_mean`, frozen-parent blob-SHA columns (one per pinned
  file above).
- **Manifest contents**: `content_sha256` of the sidecar file, the frozen-parent blob
  SHAs used to build that week's rows, the git commit/tag identity of the shadow-job
  code itself, and a pre-kickoff timestamp proof (the write timestamp and that slate's
  earliest kickoff timestamp, with the write strictly earlier).
- **Fail if the `(season, week)` evidence object already exists.** The write step
  hard-checks for an existing sidecar/manifest pair for that identity and refuses to
  proceed (rather than overwrite) if one is found -- append-only, no overwrite/rewrite
  path, ever.
- **Retention**: committed to the repository (not a time-limited Actions artifact
  retention window), so it survives through at least the end-of-season audit and
  indefinitely thereafter. Outcome grading later *reads* the committed sidecar/
  manifest pair; it never edits them -- a grading result is its own separate,
  additional file, never a mutation of the pre-kickoff evidence.

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
