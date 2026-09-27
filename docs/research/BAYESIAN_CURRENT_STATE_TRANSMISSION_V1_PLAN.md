# Bayesian Current-State Transmission V1 — Frozen Audit Plan

Status: **FROZEN BEFORE HISTORICAL SCORING**

Branch:
`research-bayesian-current-state-transmission-v1`

Parent main:
`37deff5b51b5ec48117556b91d12dba1d4ca1fad`

## Why this audit exists

Current-Season State Persistence V1 already established, on untouched 2025 replication, that current-season opportunity state contains material next-game information:

- RB rush share MAE: prior-only `0.1486` -> PlayerForm blend `0.1166`;
- WR target share: `0.06773 -> 0.06144`;
- TE target share: `0.05153 -> 0.04553`.

The exact production PlayerForm blend is:

`w_current = current_games / (current_games + 4)`.

Production then feeds the preserved prior/current evidence into `bayesian_v2`, whose rate-metric posterior adds a position-group prior and caps prior-player evidence:

- group strength for target/rush share = `3.0`;
- prior-player cap for target/rush share = `6.0`;
- current games enter with weight `current_games`.

For a veteran with >=6 prior games and exactly two completed current-season games, the current-state weight is therefore:

- PlayerForm blend: `2 / (4 + 2) = 0.333333`;
- production Bayes: `2 / (3 + 6 + 2) = 0.181818`.

Simulation rules prefer `bayes_tgt_share` / `bayes_rush_share` over the PlayerForm blend.

This creates a concrete systems question:

> Does the downstream production Bayesian posterior improve the already-validated fast current-state opportunity update, or does it shrink that useful state back toward older/group information?

This is a **read-only weighting/recombination audit**, not a Bayesian retuning experiment.

## Frozen scope

Metrics only:

1. RB `rush_share`;
2. WR `tgt_share`;
3. TE `tgt_share`.

These are the three strongest opportunity-state metrics from Current-Season State Persistence V1.

Explicitly excluded:
- RB YPC;
- WR/TE YPT or catch rate;
- QB YPA;
- receiving-yard means;
- player-level specialist coefficients;
- sportsbook prices/lines;
- Week-3 outcomes.

## Exact compared authorities

### A. PlayerForm blend

Use the exact production `scripts.player_form_v2._blend` semantics:
- previous-season player total as prior;
- current-season-to-date player total strictly before target week;
- `w_current = current_games / (current_games + 4)`.

### B. Production empirical Bayes

Use the exact production `scripts.modeling.bayesian_v2.build_bayesian_baseline` on the exact same pregame PlayerForm universe.

No Bayes constant may be changed:
- `GROUP_STRENGTH`;
- `PRIOR_PLAYER_CAP`;
- position grouping;
- defaults;
- posterior formula.

## Historical design

Score independently:

- 2024 Weeks 2–18 using 2023 as prior season;
- 2025 Weeks 2–18 using 2024 as prior season.

For each target week:
1. construct the leakage-safe target-week offensive roster from nflverse weekly rosters;
2. build stable identity only from prior-season plus current-season rows strictly before target week;
3. construct exact production prior/current season totals;
4. construct the exact PlayerForm blend;
5. build the exact production Bayesian posterior;
6. join target-week actual player opportunity state only **after** both predictions exist.

Target-week outcomes must never define:
- the player universe;
- identity registry;
- prior/current metrics;
- position-group prior;
- prediction availability.

Sportsbook inputs used = 0.

## Primary populations

For each position/metric/season:
- all rows with finite PlayerForm blend, finite Bayesian posterior, and finite target-game actual;
- separately, the **Week-3 analogue** where `current_games == 2`.

Also report veteran rows with `prior_games >= 6`.

## Metrics

For PlayerForm and Bayes:
- n;
- MAE;
- RMSE;
- bias;
- Pearson correlation;
- Spearman correlation.

Paired diagnostics:
- mean absolute-error delta `Bayes AE - PlayerForm AE`;
- fraction of rows PlayerForm is closer;
- fraction of rows Bayes is closer;
- ties;
- player-clustered paired bootstrap 95% CI for `Bayes AE - PlayerForm AE`.

Positive delta means PlayerForm is more accurate.

## Frozen interpretation

No parameter tuning follows this run.

For each of the three opportunity metrics:
- **BAYES_BETTER** if Bayes MAE is lower in both 2024 and 2025;
- **PLAYERFORM_BETTER** if PlayerForm MAE is lower in both 2024 and 2025;
- otherwise **MIXED**.

Overall disposition:
- `BAYESIAN_CURRENT_STATE_TRANSMISSION_SYSTEMIC_MISMATCH_CONFIRMED` only if all three metrics are `PLAYERFORM_BETTER` and the pooled Week-3-analogue paired AE delta is positive for all three;
- `BAYESIAN_CURRENT_STATE_TRANSMISSION_METRIC_SPECIFIC_MISMATCH` if at least one metric is `PLAYERFORM_BETTER` but the systemic gate is not met;
- `BAYESIAN_CURRENT_STATE_TRANSMISSION_NO_MISMATCH` otherwise.

Bootstrap intervals are descriptive support and may not be used to change this frozen classification after results.

## Integrity gates

All must pass:
1. zero 2026 outcomes;
2. zero sportsbook fields;
3. target/future week rows absent from all feature histories;
4. target-week roster universe comes from weekly roster source, never box-score participation;
5. exact production PlayerForm four-game pseudo-prior semantics;
6. exact production Bayes constants and implementation;
7. PlayerForm and Bayes evaluated on identical rows;
8. stable player identity or explicitly labeled temporary identity; no ambiguous silent name join;
9. no target-game outcome used to calculate the Bayes group prior;
10. no candidate parameter variants.

Any integrity failure yields:
`BAYESIAN_CURRENT_STATE_TRANSMISSION_INTEGRITY_FAILURE`

and no scientific interpretation.

## Anti-retest / stopping rule

This audit does **not** reopen Rush Pool Evidence Guard V1.

Do not use this run to:
- retune Bayesian strengths;
- create RB-only/QB-only/position exceptions;
- change top-N rushing support;
- create depth-chart carveouts;
- fit thresholds from 2024/2025 results;
- use Week-3 outcomes;
- rescue any closed Rush Pool / M96 / WR role-transmission family.

If a mismatch is confirmed, the only authorized next step is a **separately frozen production-order candidate plan** that preserves the scientific boundary established here. No production mutation is authorized by this audit alone.
