# RB R21 2026 Prospective Tail Forecast Lock and Grade V1 — Frozen Plan

Date frozen: 2026-09-08
Active branch: `research-cross-position-catastrophic-casebook-v1`
Parent: `RB_R20_REAL_2026_SLATE_SHADOW_INTEGRATION_V1`
Parent run: `34291433027`
Parent artifact: `10081502774`
Parent artifact digest: `sha256:853c0d3aea971c058ae6cc3b80c99ae2a5a0f681fae642835bbfa544da8283ca`
Parent head SHA: `587bf2a89ca16f11361016df3915361390289a7e`
Status: PLAN FROZEN BEFORE 2026 REGULAR-SEASON OUTCOMES

## Timing boundary

The first 2026 regular-season kickoff is frozen as `2026-09-10T00:20:00Z` (Wednesday, September 9 at 8:20 PM ET).

The R21 Week-1 forecast lock must have a GitHub Actions run start time strictly before that cutoff. If the lock is not completed pre-kickoff, Week 1 cannot be used as clean prospective R21 evidence.

## Why R21 exists

R16 showed that large positive RB receiving-yard residuals are partly predictable pregame. R17 converted that signal into a mean-preserving receiving-yard distribution improvement. R18 proved canonical Monte Carlo adapter parity. R19 serialized a strict-prior 2026 scorer. R20 proved the exact chain can score the real 2026 Week-1 full slate without changing production means, target entitlement, non-RB outputs, sportsbook use, or certified-stack behavior.

R21 is not another feature search and is not another retrospective tuning pass.

R21 asks:

> Does the already-frozen R19/R17/R18/R20 SHADOW receiving-yard distribution outperform the canonical CONTROL distribution on genuinely unseen 2026 regular-season outcomes when the forecasts themselves are immutably sealed before kickoff?

## Scope boundary

R21 grades **RB receiving yards distribution only**.

It does not claim that RB receptions are solved or promoted. The R16-R20 tail adapter does not change canonical receptions, target entitlement, or the receiving-yard mean.

Because CONTROL and SHADOW preserve the same receiving-yard mean, R21 is a distribution-shape/calibration test. Mean MAE and mean catastrophic-underprojection counts cannot improve by construction and therefore are not candidate-vs-control promotion metrics for this tail layer.

## Frozen competitors

### CONTROL

The exact canonical RB `rec_yards` Monte Carlo draws produced by the governed full-slate architecture used by R20.

### SHADOW

The exact same canonical simulation after the frozen R19 scorer and R17/R18 mean-preserving tail adapter used by R20.

No football mean, target entitlement, reception distribution, rush distribution, or non-RB market may differ between CONTROL and SHADOW.

## Phase A — pre-outcome forecast lock

R21 Phase A must run before the frozen first-kickoff boundary and must not read any 2026 regular-season outcomes.

It must consume the immutable R20/R19/governed-replay lineage and execute the frozen R20 path without changing R20 logic. The R21 lock wrapper may capture the already-generated canonical and adapted arrays, but it may not alter the scoring chain.

The lock must persist, at minimum:

- exact sorted RB player/event index;
- exact CONTROL receiving-yard draw matrix;
- exact SHADOW receiving-yard draw matrix;
- player-level CONTROL/SHADOW means and q05/q10/q50/q75/q90/q95;
- player-level P(actual >= frozen mean + 30) represented by each distribution;
- player-level P(actual >= frozen mean + 50) represented by each distribution;
- R19 `p30`, `p50`, state probability and identity fields from the R20 casebook;
- SHA256 hashes of the raw float64 CONTROL matrix and SHADOW matrix;
- source/artifact/model/pool/code lineage sufficient to reproduce the lock;
- GitHub run metadata proving the lock preceded the kickoff cutoff.

Future grading must consume the sealed R21 arrays. It may not regenerate a new Week-1 forecast after outcomes exist.

### Frozen Phase-A gates

All must pass:

1. `pre_kickoff_lock`: GitHub Actions run start < `2026-09-10T00:20:00Z`.
2. `r20_parent_exact`: R20 artifact ID/digest/run/head are exact.
3. `r20_reexecution_pass`: the unchanged R20 evaluator passes every frozen R20 gate while the lock is captured.
4. `rb_shape_exact`: exactly 94 adapted RB rows and 10,000 draws per row for Week 1.
5. `player_keys_unique`: `(event_id, player_clean_key)` is unique in the locked RB index.
6. `finite_nonnegative`: all CONTROL and SHADOW receiving-yard draws are finite and nonnegative.
7. `mean_parity`: max absolute CONTROL-vs-SHADOW sample-mean delta <= `1e-8` yards.
8. `draw_hashes_present`: raw float64 CONTROL and SHADOW matrix SHA256 values are recorded.
9. `sportsbook_zero_upstream`: no sportsbook input enters the football distribution/scorer.
10. `outcome_zero`: no 2026 regular-season outcome is read by Phase A.
11. `production_parameters_zero`: no production parameter or canonical output is changed.

A Phase-A PASS has disposition `RB_R21_WEEK1_PROSPECTIVE_FORECAST_LOCK_PASS_SHADOW_ONLY`.

A Phase-A FAIL invalidates Week 1 as clean R21 prospective evidence. Do not repair the forecast after Week-1 outcomes are visible and then call it prospective.

## Phase B — frozen outcome grading

Phase B may run only after the relevant games are final and official/standard 2026 weekly player statistics are available.

The outcome source is the project's existing football-statistics path (`nflreadpy.load_player_stats(..., summary_level="week")` or its governed equivalent) and must be pinned/audited at grading time. No sportsbook result or closing line is an outcome source.

Outcome matching must be fail-closed and one-to-one using locked event/team/player identity information. No manual post-outcome reassignment is permitted. Any unresolved identity/status rows must be listed explicitly rather than guessed.

The grader must report match coverage and exclusion reasons. A scientific grade is invalid if fewer than 90% of locked RBs who can be authoritatively classified for participation/stat status are resolved.

## Frozen scoring

For each eligible locked RB player-game and for aggregate Week 1/cumulative checkpoints, compute from the sealed draw matrices:

- CRPS;
- Brier score for `actual_rec_yards >= frozen_mean + 30`;
- Brier score for `actual_rec_yards >= frozen_mean + 50`;
- q90 pinball loss;
- q95 pinball loss;
- central 80% interval coverage (`q10` to `q90`);
- central 90% interval coverage (`q05` to `q95`);
- upper-quantile exceedance counts/severity for diagnostic casebooks;
- R16 `p30`/`p50` risk-stratified event rates as diagnostics.

The event thresholds use the common frozen mean because CONTROL and SHADOW have the same mean.

Mean MAE/RMSE/bias are still reported for football-model monitoring, but they are identical between CONTROL and SHADOW by construction and are not evidence that the tail adapter improved the mean.

Sportsbook/market performance may be evaluated later as downstream pricing evidence only. It is not an upstream feature and is not part of the R21 football-distribution scientific PASS.

## Frozen Week-1 prospective support gates

Week 1 is an early prospective checkpoint, not sufficient by itself for production promotion.

R21 Week-1 prospective support requires ALL of the following:

1. Phase-A forecast-lock PASS and exact draw hashes.
2. Outcome matching/participation audit valid under the frozen coverage rule.
3. SHADOW combined CRPS <= CONTROL combined CRPS.
4. SHADOW 30+ Brier <= CONTROL 30+ Brier.
5. SHADOW 50+ Brier <= CONTROL 50+ Brier.
6. SHADOW q90 pinball < CONTROL q90 pinball.
7. SHADOW q95 pinball < CONTROL q95 pinball.
8. SHADOW absolute 80% coverage error <= CONTROL absolute 80% coverage error + `0.03`.
9. SHADOW absolute 90% coverage error <= CONTROL absolute 90% coverage error + `0.03`.
10. No sportsbook input added upstream.
11. No production parameter changed.

These gates are frozen before Week-1 outcomes and may not be changed after results are visible.

## Cumulative production-authorization evidence floor

A Week-1 PASS does **not** authorize production.

A later production-promotion decision is not eligible to be made until the shadow ledger contains, at minimum:

- 4 distinct completed 2026 regular-season weeks;
- 250 eligible locked RB player-games;
- 15 observed 30+ underprojection events; and
- 5 observed 50+ underprojection events.

Every included weekly forecast must have been sealed before that week's relevant kickoff/outcomes using the same frozen model/scorer/adapter version or a formally versioned reset that starts a new evidence ledger.

At the first eligible cumulative checkpoint, the same core distribution comparisons remain controlling: CRPS, Brier30, Brier50, q90 pinball, q95 pinball, and interval coverage. Production still requires a separate explicit governed promotion ledger/commit; cumulative eligibility does not auto-promote code.

## Governance

PASS at any R21 checkpoint means prospective evidence supports continuing the frozen SHADOW distribution under observation.

FAIL does not authorize post-outcome threshold/model tuning. Diagnose the failure first. Any changed model or changed scoring rule must receive a new version and a new prospective evidence ledger.

Nothing in R21 may:

- change canonical RB receiving means;
- change RB target entitlement or receptions;
- manufacture RB-room opportunity;
- modify rushing outputs;
- modify WR/TE/QB outputs;
- add sportsbook inputs to football projections; or
- activate the tail adapter in production without a separate governed promotion decision.
