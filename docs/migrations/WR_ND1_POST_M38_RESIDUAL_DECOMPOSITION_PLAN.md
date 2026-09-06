# WR-ND1 — Post-M38 Residual Decomposition Plan

**Branch:** `research-wr-nd1-post-m38-decomposition`

**Base:** `main` at `754d0f4ed34a06a65d2db36f2fb2ca10c58264ed`

**Status:** frozen diagnostic plan; no production change.

## Why this study exists

M31-M38 is already a promoted receiving foundation. M34/M35 established that the pre-M38 receiving problem was dominated by opportunity and, within opportunity, WR target-share allocation rather than team target volume. M36-M38 then sharpened the WR hierarchy and promoted the M37 boundary winner.

The repo does not contain a post-M38 decomposition establishing which error source dominates *after* that correction. Therefore WR-ND1 will diagnose the current M38-derived receiving stack before any new WR predictive architecture is fit.

This is intentionally not another target-pool pruning study, another universal WR1 multiplier search, or another receiver/secondary feature model.

## Frozen population and information contract

- Evaluation season: 2025.
- Target weeks: 1-18.
- Prior season: 2024.
- Historical context must be built strictly before each target week using the existing walk-forward bundle.
- Evaluation rows: WR/LWR/RWR/SWR player-games with canonical pregame projection identity and target-game actuals.
- Sportsbook lines, odds, spreads, totals, implied points, and market-narrowed cohorts are forbidden.
- No fitting, hyperparameter search, coefficient tuning, threshold search, or production mutation.

## Frozen M38 identity

The diagnostic must import the production M38 transform from `scripts/simulation_v2.py` and fail unless:

- `WR_TARGET_HIERARCHY_MULTIPLIERS == (1.40, 1.14, 0.91, 0.78)`;
- the transform preserves each team's pre-transform WR target-share mass within `1e-12`;
- WR rank is derived from leakage-safe pregame target share, never target-game outcomes.

The diagnostic may not retune the M38 multipliers.

## Factorization

For each projected WR player-game, post-M38 expected targets are represented as:

`TEAM_TARGET_VOLUME × WR_TARGET_MASS × WITHIN_WR_ALLOCATION`

where:

- `TEAM_TARGET_VOLUME`: canonical pregame expected team targets/dropbacks under the existing receiving allocation architecture;
- `WR_TARGET_MASS`: post-M38 total WR allocator probability as a fraction of team targets;
- `WITHIN_WR_ALLOCATION`: the player's post-M38 share of the WR room's target probability.

Receiving-yard expectation adds:

`× YARDS_PER_TARGET`

Reception expectation adds:

`× CATCH_RATE`

Target-game truth factors are built only after the pregame factors are frozen:

- actual team targets;
- actual WR targets / actual team targets;
- actual player targets / actual WR targets;
- actual receiving yards / actual targets when targets > 0;
- actual receptions / actual targets when targets > 0.

For zero-target rows, target-game YPT/catch-rate truth is set equal to the frozen pregame value because opportunity truth already drives the fully corrected prediction to zero; this prevents undefined 0/0 values from creating artificial efficiency attribution.

## Exact Shapley decomposition

WR-ND1 will calculate all subsets and exact Shapley MAE attribution.

### Targets — 3 components

1. `TEAM_TARGET_VOLUME`
2. `WR_TARGET_MASS`
3. `WITHIN_WR_ALLOCATION`

The empty subset is the frozen post-M38 deterministic target prediction. The full subset must equal actual targets row-by-row within numerical tolerance. Shapley contributions must sum to the empty-to-full MAE recovery within `1e-9`.

### Receiving yards — 4 components

1. `TEAM_TARGET_VOLUME`
2. `WR_TARGET_MASS`
3. `WITHIN_WR_ALLOCATION`
4. `YARDS_PER_TARGET`

The empty subset is frozen post-M38 deterministic receiving yards. The full subset must equal actual receiving yards row-by-row within `1e-8`. Shapley contributions must sum to empty-to-full MAE recovery within `1e-8`.

### Receptions — 4 components

1. `TEAM_TARGET_VOLUME`
2. `WR_TARGET_MASS`
3. `WITHIN_WR_ALLOCATION`
4. `CATCH_RATE`

The same exact identity requirements apply.

Canonical Monte Carlo receptions/receiving-yard MAE, RMSE, bias, and correlation are reported as reference metrics but are not used to fit or select anything.

## Required slices

Report the same factor attributions for:

- ALL_WR;
- WR1;
- WR2;
- WR3;
- WR4_PLUS;
- base overprojection vs underprojection for receiving yards;
- week phases W1-4, W5-9, W10-13, W14-18;
- actual target tiers: 0-3, 4-6, 7-9, 10+.

WR rank is frozen from the post-M38 pregame hierarchy.

## Diagnostic routing rule

This migration does not promote a model. Its disposition routes the next WR investigation.

For overall receiving-yard Shapley attribution, identify the largest positive component. Then require the same component to be top in at least two of WR1, WR2, and WR3 to call it structurally dominant.

Allowed dispositions:

- `WITHIN_WR_ALLOCATION_DOMINANT`
- `WR_TARGET_MASS_DOMINANT`
- `TEAM_TARGET_VOLUME_DOMINANT`
- `YARDS_PER_TARGET_DOMINANT`
- `MIXED_POST_M38_WR_ERROR`

A component is called dominant only if it is the largest positive overall attribution, accounts for at least 40% of the sum of positive overall Shapley attribution, and is top in at least two of WR1/WR2/WR3. Otherwise disposition is `MIXED_POST_M38_WR_ERROR`.

The target-only and reception Shapley results are supporting diagnostics and cannot override the receiving-yard routing rule.

## What each disposition means next

- `WITHIN_WR_ALLOCATION_DOMINANT`: investigate dynamic WR entitlement/role allocation using genuinely new strict-prior role information; do not retune universal rank multipliers.
- `WR_TARGET_MASS_DOMINANT`: investigate WR-vs-TE/RB positional target allocation and personnel usage.
- `TEAM_TARGET_VOLUME_DOMINANT`: return to receiving-side team passing opportunity only if the result materially contradicts M35 under the post-M38 stack.
- `YARDS_PER_TARGET_DOMINANT`: shift to WR efficiency/route-depth/YAC architecture, excluding already-failed M72/M75 proxy families unless new information is added.
- `MIXED_POST_M38_WR_ERROR`: design a joint generative receiving architecture rather than a single-factor patch.

## Explicit anti-duplication rules

Do not use WR-ND1 to reopen:

- generic receiving-pool pruning from M31-M32;
- WR identity/coverage as the primary issue from M33;
- pre-M38 target-share oracle conclusions without remeasurement;
- universal hierarchy-strength search from M36-M38;
- M72 aggregate explosive-weapon x defense interactions;
- M75 NGS separation/cushion/aDOT/YACOE + PFR secondary aggregates or another algorithm over those same features;
- fake WR-CB assignments inferred from generic participation;
- current-only WR-CB webpages as historical training data;
- sportsbook information upstream of football projection.

## Required artifacts

- `wr_nd1_player_factors.csv`
- `wr_nd1_stage_metrics.csv`
- `wr_nd1_shapley_summary.csv`
- `wr_nd1_slice_shapley.csv`
- `wr_nd1_summary.json`

No production file may be changed by WR-ND1.
