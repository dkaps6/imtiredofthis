# RB-PD2 Multi-Season P3-Equivalent Replication V1 — Frozen Plan

**STATUS: FROZEN BEFORE ANY 2021-2024 PERSISTENCE RESULT. RESEARCH ONLY. NO PRODUCTION CHANGE.**

## Purpose

The RB-PD5 cohort-discrepancy memo requires a valid multi-season, non-2025, P3-equivalent evidence path before any further confirmatory RB residual mechanism can advance. Separately, RB-PD2 detected individual player-error persistence on the already-observed 2025 STACK1 cohort and authorized a still-unattempted difficulty -> MC-width/uncertainty lane.

This experiment is the prerequisite only. It asks:

> Do the original RB-PD2 player-error persistence signals reproduce on a genuinely multi-season, non-2025 panel whose rushing-yard mean is reconstructed with the same P3 architecture: prior-season-frozen STACK1 opportunity/efficiency + prior-season-fit STACK2 allocation + the frozen P3 composition?

It does **not** test a new uncertainty-width multiplier. A PASS only unlocks a separately frozen PD2-width experiment.

## Why 2021-2024

Target seasons are frozen to **2021, 2022, 2023, 2024** before looking at their persistence results.

Reason is architectural, not outcome-based:

- M95Q already established parity-checked temporal M91 component reconstructions through 2024;
- P3-equivalent reconstruction needs a full prior season to fit both the STACK1 ensemble and STACK2 allocation model;
- starting at 2021 lets every target season use an available 2020+ prior-season provider panel without relying on less-certain 2019 roster/depth/injury coverage;
- 2025 is explicitly excluded because it is the already-observed PD2/PD3/PD4/PD5 cohort and cannot act as confirmation.

No 2021-2024 PD2 persistence metric has been inspected before this plan is frozen.

## Canonical historical inputs

### Temporal football components

Use the successful M95Q reconstruction run:

- run `33450395426`;
- M91 rotation artifacts for 2020-2024;
- final M95Q artifact `9779790912`, disposition `M95Q_EXPANDED_PANEL_READY`;
- 2024 M91 universe parity PASS and downstream parity PASS.

The M91 component rows provide the same `mc_proj`, `ml_proj`, `state_proj`, actual, player/team/week identity and historical pregame routing used by STACK1.

### STACK1-equivalent reconstruction

For target season `S` in 2021-2024:

1. fit `fit_market_weights()` on exact season `S-1` component predictions only;
2. freeze those weights;
3. apply them to season `S`;
4. retain RB/HB/FB `rush_att` and `rush_yards` rows;
5. call the resulting projections `stack_att` and `stack_yards`.

This is the exact temporal analogue of 2025 STACK1's `ensemble_2024_frozen` path. No target-season outcome is used to fit target-season STACK1 weights.

### STACK2-equivalent allocation reconstruction

Generalize the already-frozen STACK2 allocation architecture without changing its science:

- same `FULL` feature list from `evaluate_rb_stack2_enriched_allocation.py`;
- same `HistGradientBoostingRegressor`:
  - `loss='squared_error'`
  - `learning_rate=0.05`
  - `max_iter=160`
  - `max_leaf_nodes=15`
  - `min_samples_leaf=30`
  - `l2_regularization=1.0`
  - `random_state=17`;
- same target: actual share of team RB carries;
- same team-score normalization;
- same 50/50 anchor: `enriched_share = 0.5 * stack_share + 0.5 * alloc_full_share`;
- `enriched_att = enriched_share * team_stack_att_pool`.

For each target season `S`, fit the allocator on `S-1` only and freeze it before applying to `S`.

Historical roster, depth, snap and injury features must be strictly pregame/strictly-prior. Missing-provider handling must follow the existing STACK2 semantics; no target-game outcome may enter a feature.

If required historical provider fields cannot be reconstructed with sufficient integrity, the run must fail `P3_EQUIVALENT_INTEGRITY_FAILURE`; do not substitute a simpler allocation model after seeing that failure.

### P3-equivalent composition

Apply the frozen `compose_p3_row()` semantics from `scripts/modeling/rb_rush_synthesis_v1.py`:

- Week 1: `p3_equiv_yards = stack_yards`;
- Weeks 2-18: `p3_equiv_yards = enriched_att * (stack_yards / stack_att)` when `stack_att > 0.20`.

The production M94C efficiency fallback may only be used if an exact historical M94C-equivalent fallback is available from a predeclared source. Otherwise any row requiring the fallback is excluded from the scoreable replication panel and counted explicitly; if fallback-required rows exceed **1%** of otherwise eligible rows in any target season, integrity FAIL.

Carry prediction for the PD2 carry diagnostics is `enriched_att` for Weeks 2-18 and `stack_att` for Week 1, matching the P3 opportunity route. Yard prediction is `p3_equiv_yards`.

## Identity and parity gates

Before any persistence interpretation:

1. target seasons exactly `{2021, 2022, 2023, 2024}`;
2. 2025 rows = 0;
3. no target-game outcomes in any feature used to fit STACK1/STACK2;
4. each target season's STACK1 weights fit only on `S-1`;
5. each target season's STACK2 allocator fit only on `S-1`;
6. M95Q 2024 M91 universe parity source remains PASS;
7. player/team/week identity unique in the final P3-equivalent panel;
8. P3 Week-1 route equals frozen STACK1 exactly to floating-point tolerance;
9. P3 Weeks2-18 arithmetic reproduces `enriched_att * stack_implied_ypc` to <=1e-10;
10. fallback-required share <=1% per season;
11. sportsbook inputs used upstream = 0;
12. production changed = false.

Any failure => `P3_EQUIVALENT_INTEGRITY_FAILURE`; no scientific disposition.

## Frozen PD2 history construction

Reuse the original RB-PD2 contract unchanged:

- same player only;
- chronological completed prior games only;
- last **8** eligible prior games;
- minimum **4** prior games;
- history resets only by chronology, not season boundary: prior completed games may carry across seasons for the same player, but no future/same-game row may enter;
- prior quantities:
  - carry bias = mean(predicted carries - actual carries),
  - carry MAE,
  - yard bias = mean(predicted rushing yards - actual rushing yards),
  - yard MAE.

The replication target is the next game's signed/absolute error.

## Frozen diagnostics — original PD2 gates retained

### A. Carry directional persistence

Pooled 2021-2024 PASS only if all:

- scoreable rows >= 700;
- Spearman(prior carry bias, target carry error) >= +0.08;
- high-low quartile target carry-error gap >= +1.0 carry;
- sign agreement >=55% where |prior carry bias| >=0.5;
- pooled Weeks 5-12 gap >0;
- pooled Weeks 13-18 gap >0.

### B. Carry difficulty persistence

Pooled PASS only if all:

- scoreable rows >=700;
- Spearman(prior carry MAE, target absolute carry error) >= +0.08;
- high-low target absolute-error gap >= +0.75 carry;
- pooled Weeks 5-12 gap >0;
- pooled Weeks 13-18 gap >0.

### C. Yard directional persistence

Pooled PASS only if all:

- scoreable rows >=700;
- Spearman(prior yard bias, target yard error) >= +0.08;
- high-low target yard-error gap >= +6 yards;
- sign agreement >=55% where |prior yard bias| >=3 yards;
- pooled Weeks 5-12 gap >0;
- pooled Weeks 13-18 gap >0.

### D. Yard difficulty persistence

Pooled PASS only if all:

- scoreable rows >=700;
- Spearman(prior yard MAE, target absolute yard error) >= +0.08;
- high-low target absolute-error gap >= +5 yards;
- pooled Weeks 5-12 gap >0;
- pooled Weeks 13-18 gap >0.

## Added multi-season confirmation guard

Because this is specifically a multi-season replication, a pooled diagnostic is `REPLICATED` only when:

- the original pooled PD2 gate above passes; **and**
- both Spearman and quartile-gap direction are positive in at least **3 of 4** target seasons; **and**
- neither 2023 nor 2024 has both Spearman <=0 and quartile gap <=0 for that diagnostic.

This guard is frozen before results and is stricter than the original single-season PD2 disposition; it cannot rescue a pooled failure.

## Dispositions

- integrity failure: `P3_EQUIVALENT_INTEGRITY_FAILURE`;
- zero diagnostics replicate: `NO_MULTISEASON_RB_PLAYER_ERROR_PERSISTENCE`;
- one or more diagnostics replicate: `MULTISEASON_RB_PLAYER_ERROR_PERSISTENCE_REPRODUCED`.

**Authorization boundary for the untouched uncertainty-width lane:**

- Carry-width research is unlocked only if diagnostic **B** replicates.
- Yard-width research is unlocked only if diagnostic **D** replicates.
- Directional A/C results do not by themselves authorize a width experiment.
- Any unlocked width experiment still requires its own separately frozen plan before implementation.

## Forbidden

- no sportsbook/line/odds features;
- no 2025 confirmation rows;
- no threshold/window/model-family tuning after results;
- no alternate historical allocator if this one fails integrity;
- no PD6 launch from this experiment alone;
- no production changes.
