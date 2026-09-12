STATUS: RESEARCH ONLY — NOT PROMOTED.

# WR-R3 Combined Candidate — Status Check

## What R3 authorized

`docs/migrations/WR_R3_PLAYER_ERROR_PERSISTENCE_RESULT.md` (branch
`research-wr-r3-player-error-persistence`, run `34064572328`, job `101570966426`,
disposition `WR_PLAYER_ERROR_PERSISTENCE_DETECTED`) ran a strict walk-forward
diagnostic on the 12,396-row M38 WR receiving-yard source (10,675 scoreable rows,
478 players, last-8/min-4 strictly-prior same-player history) and passed all three
frozen individual-error diagnostics:

1. Directional bias persistence (Spearman .08889, +9.55 signed-yard quartile gap,
   55.56% sign agreement, 6/6 positive seasons).
2. Individual difficulty persistence (Spearman .25793, +13.90 absolute-yard
   quartile gap, 6/6 positive seasons).
3. Extreme-miss persistence (1.368x next-game 30+ miss enrichment, 6/6 positive
   seasons).

It explicitly named two next integration lanes and said a later **predeclared
combined full-stack candidate is legitimate**, but only if tested "through the
exact M38 production architecture" and passing its own **frozen aggregate gates +
individual-error gates**:

- (a) prior signed bias → a conservative shrunk mean-calibration layer;
- (b) prior difficulty / extreme-miss history → player-specific Monte Carlo
  uncertainty / tail calibration.

No production change was authorized by R3 itself.

## What I found

**The exact combined candidate R3 describes was never built.** No workflow file,
evaluator script, or plan/result doc on any `research-wr-*` branch implements a
full-stack integration of lanes (a)+(b) using R3's own strict walk-forward
features. This is a genuine gap, not an unwritten-up completed run.

### Key structural finding: the branch that continued past R3 is a different R3

`research-wr-r3-player-error-persistence` (the branch with the
`WR_PLAYER_ERROR_PERSISTENCE_DETECTED` result above) is an orphaned sibling —
nothing in the repository branches forward from it. All later WR work
(`research-wr-r4-individual-mechanism-decomposition` through
`-r11-strict-prior-ngs-target-model`, `-nd1` through `-nd6`, and
`-post-m38-error-decomposition`) instead branches from a **different, same-named
R3 thread**: `research-wr-r3-player-bias-persistence`. The two R3 branches share
only the R2 ancestor commit (`9c047ea`) and diverge from there — they are parallel
research lines, not sequential steps of one thread.

- `research-wr-r3-player-bias-persistence` froze `docs/migrations/WR_R3_PLAYER_BIAS_PERSISTENCE_PLAN.md`
  and `scripts/backtest/evaluate_wr_r3_player_bias_persistence.py`, defining a
  first-pass, standalone implementation of lane (a) — candidate
  `WR_PLAYER_BIAS_SHRINK_V1` (min 8 prior games, expanding prior-residual mean,
  shrinkage weight 8, correction cap ±20 yards, applied by subtracting the
  shrunk bias directly from the M38 MC output). **This candidate already ran**
  (GH Actions run `34064929300`, job `101571908189`, 2026-09-06, SHA `ed19436`)
  and **completed with a disposition that was never written up as a RESULT.md** —
  the branch instead moved straight to R4 without documenting it. Retrieved via
  job logs:

  ```
  disposition: NO_ACTIONABLE_WR_PLAYER_BIAS_PERSISTENCE
  eligible_rows: 9256, players: 478, qualifying_players: 238
  base:      mae 23.9912  rmse 33.5753  bias -9.9248  median_ae 16.4626  p90_ae 54.2927  miss20 .4245  miss30 .2753  miss40 .1828
  candidate: mae 24.1655  rmse 32.0292  bias -0.7635  median_ae 18.8179  p90_ae 50.8362  miss20 .4719  miss30 .2887  miss40 .1777
  mae_improvement_fraction: -0.00726  (worse)
  gates passed:  eligible_ge6000=true, rmse_nonworse=true, p90_nonworse=true, miss40_nonworse=true
  gates failed:  mae_improve_ge1pct, median_nonworse, miss20_nonworse, miss30_nonworse,
                 four_of_six_seasons, 2024_improve, 2025_improve, median_player_delta_negative
  season MAE deltas: 2020 -0.43, 2021 +0.16, 2022 -0.15, 2023 +0.47, 2024 +0.08, 2025 +0.52
  ```

  8 of 10 frozen gates failed. Note this is a *diagnostic-level* test (shrunk
  bias subtracted directly from M38's MC output on the WR-R1 casebook), not a
  full-stack integration through PlayerForm/TeamForm/Bayesian-baseline/
  simulation_v2 — it never reached the "frozen aggregate gates" stage the
  protocol requires, because it failed the qualifying step first.

- `research-wr-nd6-player-level-explosive-ceiling` is the closest built analog
  to lane (b) (tail/distribution signal from prior history), but it uses
  different features (prior-8 explosive-target-rate per player/defense, not R3's
  prior-difficulty/extreme-miss-persistence features) and it is already
  documented: `NO_ACTIONABLE_EXPLOSIVE_CEILING_SIGNAL` — no candidate signal
  cleared the frozen coverage/Spearman/residual-gap/enrichment gate, so no
  player×defense interaction was authorized. Nothing new to report here; already
  written up in `docs/migrations/WR_ND6_PLAYER_LEVEL_EXPLOSIVE_CEILING_RESULT.md`.

- `research-wr-r4-individual-mechanism-decomposition` (read in full, not just by
  name): confirmed to be a Shapley two-factor decomposition of receiving-yard
  residual into reception-volume vs. yards-per-reception contribution
  (`scripts/backtest/evaluate_wr_r4_individual_mechanism_decomposition.py`).
  Disposition `WR_INDIVIDUAL_MECHANISMS_MAPPED` (226/337 qualifying WRs
  reception-dominant, 83 mixed, 28 YPR-dominant). It is explicitly diagnostic
  ("does not itself change M38... directs subsequent WR research") and routes
  toward target-opportunity mechanics (feeding R5/R6/R8), not toward a
  calibration/shrinkage/MC-uncertainty integration. **It is not the combined
  candidate under a different name** — your original flag was correct.

- `docs/migrations/WR_FULL_STACK_INTEGRATION_PROTOCOL.md` (frozen on the ND6
  branch, carried forward through R7–R11) is the general standing protocol any
  future WR integration branch must follow — it is not R3-specific and was not
  itself an attempt at the combined candidate, but it is the authoritative
  specification for how one must be built (see Design spec below). It fixes the
  exact M38 production baseline to reproduce before testing any candidate:
  receiving-yard rows = 4,647, MC MAE = 17.099904733366, RMSE = 25.196099510686,
  bias = -5.238640833495, correlation = 0.567945850835, WR evaluation rows =
  2,130 — a different (smaller, production-view) population than R3's own
  12,396/10,675-row WR-R1 casebook.

- Commit-message and filename search across all 18 `research-wr-*` branches for
  "combined", "calibrat", "shrink", "MC width", "uncertainty", "Monte Carlo",
  "tail calib" turned up nothing beyond the items above and pre-existing
  general-migration history (M18–M71, unrelated to WR-R3) that predates the WR
  research branches entirely (no common git ancestor with `origin/main`; each
  research branch is its own orphaned lineage seeded from an old snapshot, so
  `--not origin/main` filtering had no effect and manual triage was required).

- No `.github/workflows/*.yml` on any `research-wr-*` branch is named or scoped
  for an R3-combined / full-stack-calibration candidate. There is therefore
  nothing eligible to dispatch under the task's authorization — dispatching
  would mean running new, never-frozen modeling logic, which is out of scope
  here.

**Conclusion: genuinely not built.** The two failed adjacent attempts
(bias-shrink V1, ND6 explosive-ceiling) are informative priors, not a
substitute — they tested different feature constructions than R3's own
strictly-prior-8/min-4 bias and extreme-miss-persistence features, and neither
reached the full-stack (PlayerForm → Bayesian baseline → rules/context →
simulation_v2) integration stage the protocol and R3 both require.

## Design spec

No code below — this is implementation-ready scaffolding for a human or a
dedicated session to build, matching this repo's frozen-protocol conventions.

### Naming and location (matching R3/ND6 conventions)

- Plan: `docs/migrations/WR_R3_COMBINED_CALIBRATION_PLAN.md` (frozen *before*
  any candidate result is visible, per protocol's "Combination rule").
- Evaluator: `scripts/backtest/evaluate_wr_r3_combined_calibration.py`, argparse
  `--wr-r1-root` / `--m38-root` (or equivalent production-baseline artifact) /
  `--out-dir`, matching the `_one()`/`_read()`/`_key()`/`_num()` helper style
  used in `evaluate_wr_r3_player_error_persistence.py` and
  `evaluate_wr_r4_individual_mechanism_decomposition.py`.
- Workflow: `.github/workflows/backtest-wr-r3-combined-calibration.yml`,
  `workflow_dispatch` + push-triggered on the new branch, single `evaluate` job,
  `py_compile` → download prerequisite artifacts (WR-R1 casebook run
  `34058453941`; the M38 production baseline run) → run evaluator → `cat` the
  result JSON → `upload-artifact`, same shape as
  `backtest-wr-r3-player-bias-persistence.yml`.
- Result doc: `docs/migrations/WR_R3_COMBINED_CALIBRATION_RESULT.md`, written
  immediately after the run completes (the gap this whole check exists to catch).

### What it must consume

Two distinct data populations must not be conflated:

1. **Feature source** (to build the two prior-history features): the WR-R1
   paired casebook (run `34058453941`), exact M38 WR `rec_yards` rows = 12,396,
   scoreable rows after strict walk-forward pairing = 10,675 — identical
   construction to `evaluate_wr_r3_player_error_persistence.py`
   (`build_walkforward()`: last-8-game/min-4 strictly-prior history per
   `player_clean_key`, `season`/`week` ordering, explicit leakage assertion
   that no prior game has `season > current` or `season == current and week >=
   current`).
2. **Integration evaluation population**: the exact M38 production baseline
   that `WR_FULL_STACK_INTEGRATION_PROTOCOL.md` freezes — receiving-yard rows
   = 4,647, WR evaluation rows = 2,130 (the same population ND6 and the
   post-M38 error-decomposition diagnostic reproduced). The candidate must
   reproduce this exact baseline (MAE 17.099904733366, RMSE 25.196099510686,
   bias -5.238640833495, correlation 0.567945850835) before any candidate
   number is computed, and must run the two calibration lanes *inside* the
   actual stack (PlayerForm/TeamForm timestamp-safe context → Bayesian
   baseline/shrinkage → canonical rules/context engine → `simulation_v2` Monte
   Carlo → distributions/summary projections), not as a post-hoc patch on MC
   output — the bias-shrink V1 attempt patched MC output directly and failed;
   the protocol's "Mean versus distribution" rule requires mean-lane
   corrections to enter as pre-Monte-Carlo expectation inputs and
   ceiling/tail-lane corrections to enter as distribution-shape/variance
   inputs, never as a second model voting after Monte Carlo runs.

### The two lanes (predeclared combination, per protocol's Combination rule)

Both source diagnostics (directional bias persistence, individual
difficulty/extreme-miss persistence) already independently passed their R3
gates, so a small predeclared combination of both is protocol-eligible. Freeze
both lane definitions together, before running either:

**Lane A — shrunk mean-calibration** (uses R3's directional-bias-persistence
feature, not the failed V1 shrink formula verbatim — re-derive the shrink
weight/cap from this branch's own frozen pre-registration, informed by but not
copied from bias-shrink V1's failure):
- Input: same-player strictly-prior last-8/min-4 signed M38 error
  (`prior8_m38_bias` in R3's own script).
- Apply as an adjustment to the pregame mean input entering the Bayesian
  baseline/shrinkage stage, not to simulation_v2 output.
- Freeze shrinkage weight, minimum prior-game count, and correction cap before
  seeing results; do not retune after seeing the V1 failure's magnitudes.

**Lane B — Monte Carlo uncertainty / tail calibration** (uses R3's
individual-difficulty and extreme-miss-persistence features):
- Inputs: same-player strictly-prior `prior8_m38_mae` (difficulty) and
  `prior8_m38_miss30_rate` (extreme-miss enrichment), both already computed by
  `evaluate_wr_r3_player_error_persistence.py`.
- Apply as a per-player variance/tail-width multiplier on the simulation_v2
  receiving-yard distribution (wider dispersion or fatter upper tail for
  high-difficulty/high-extreme-miss players), not a mean shift — consistent
  with ND6's explicit distinction between ceiling/distribution-shape signals
  and mean signals.
- Freeze the multiplier function and its bounds before seeing results.

**Combined candidate**: both lanes applied together in a single frozen run
(not sequential tuning of one lane against the other's results).

### Two frozen gate families (per R3's own language and the protocol)

**(A) Aggregate M38 architecture gates** — evaluated on the full 2,130-row
production evaluation population, against the reproduced 4,647-row baseline:
- Exact baseline parity reproduction (rows, MAE, RMSE, bias, correlation) before
  any candidate number is trusted.
- Pooled candidate MAE/RMSE/bias/correlation vs. frozen baseline, with a
  pre-registered minimum improvement threshold (do not set it after seeing
  results).
- Phase stability: W2-18 and W13-18 slices must not worsen.
- WR1/WR2/WR3 role-slice behavior must not worsen in at least 2 of 3 tiers
  (mirrors ND6's role-positive-count convention).
- Large-miss tails: 20+/30+/40+ (and, if lane B changes distribution shape,
  100+ actual-yard and 50+ underprediction) miss rates must not worsen.
- For lane B specifically: Monte Carlo distribution/fair-probability behavior
  must be checked directly (calibration of the widened tail), not just point
  MAE — per the protocol's "Required evaluation" section.

**(B) Individual-error gates** — evaluated on the same per-player basis R3 and
bias-shrink V1 used:
- Minimum eligible player-games (bias-shrink V1 used >=6,000; reuse or justify
  a comparable floor).
- Median qualifying-player MAE delta must be negative (bias-shrink V1 failed
  this at `median_player_delta_negative: false`).
- Season-level consistency: improvement in >=4 of 6 seasons, with 2024 and 2025
  (the most recent, highest-weight seasons) both required to improve — both
  failed for bias-shrink V1 and must not be waived for the combined candidate.
- No new systematic miss-tier created (miss20/30/40 non-worse per player
  cohort, not just pooled).

**Disposition naming** (matching repo convention): pass → e.g.
`WR_R3_COMBINED_CALIBRATION_INTEGRATION_WIN`; fail →
`NO_ACTIONABLE_WR_R3_COMBINED_CALIBRATION`. Per the protocol's Production rule,
a pass here still only authorizes proposing production promotion through the
repo's normal migration-gate process — it is not itself a promotion.

## Recommendation

1. Do not dispatch any existing workflow — none implements this candidate; the
   two adjacent ones that exist already ran and already failed at the
   diagnostic-only stage, before ever reaching full-stack integration.
2. Before building this, be aware of (not bound by) the two negative priors:
   a naive post-hoc MC-output bias correction already failed on 8/10 gates
   (worse MAE, worse median AE, worse miss20/miss30, no 2024/2025 improvement),
   and a naive prior-8 explosive-rate tail signal already failed its own gate.
   Both attempted analogs, not identical constructions — genuinely re-deriving
   lanes A/B from R3's own features, applied inside the real stack rather than
   as an output patch, is still a legitimate next step per the protocol, but
   should be built with eyes open to why the nearest attempts failed (both
   over-corrected mean/median without controlling tail behavior, and applied
   corrections downstream of Monte Carlo rather than upstream).
3. Also recommend: write up `WR_R3_PLAYER_BIAS_PERSISTENCE_RESULT.md` for the
   already-completed, already-failed bias-shrink V1 run (data above) — it is a
   real, citable negative result sitting undocumented on
   `research-wr-r3-player-bias-persistence`, independent of whether the combined
   candidate above is ever built.
4. Building lanes A+B properly requires exact fidelity to `simulation_v2`,
   the Bayesian baseline/shrinkage stage, and the frozen M38 parity numbers —
   per the task's own constraint, this should be done by the repo owner or a
   dedicated careful session with write access to `scripts/`, not authored here.
