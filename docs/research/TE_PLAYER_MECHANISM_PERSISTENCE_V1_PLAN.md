# TE Player Mechanism Persistence V1 — Frozen Diagnostic Plan

**STATUS: FROZEN BEFORE RESULT. DIAGNOSTIC ONLY. NO PRODUCTION CHANGE.**

## Parent finding

`TE_PLAYER_ERROR_PERSISTENCE_DETECTED`

Authority:
- run `37630274063`
- artifact `11486201614`
- digest `sha256:52b488b4e989ecd18cc061ad5f3697e7aa31ac8ddbb8e55f97d44e1cf0cd14a6`

Exact promoted TE-R5P authority:
- run `34152797603`
- artifact `10029942404`
- digest `sha256:f9951441b748ef72514dbc81adf6fbe9cd023c9bf64ecb52a016840989ab4cdb`

The parent diagnostic established that same-player TE-R5P receiving-yard bias, difficulty, and extreme-miss behavior persist across 2024 and 2025.

This audit asks **where that persistence lives**.

## Purpose

Decompose exact TE-R5P receiving-yard error into two mechanically exact components:

1. opportunity / target-entitlement error;
2. per-target efficiency error.

Then test whether either component itself has stable same-player persistence.

This is not a correction model.

## Exact decomposition

For every exact TE-R5P authority row define:

`pred_targets = candidate_targets_r5p`

`actual_targets = targets`

`pred_ypt = candidate_rec_yards_r5p / candidate_targets_r5p`

When actual targets > 0:

`actual_ypt = rec_yards / targets`

When actual targets = 0:
- set `actual_ypt = 0`;
- the efficiency component is mechanically zero because it is multiplied by actual targets.

Require finite positive `pred_targets`. Otherwise the row is not decomposition-scoreable.

Define:

`opportunity_error = (pred_targets - actual_targets) * pred_ypt`

`efficiency_error = actual_targets * (pred_ypt - actual_ypt)`

Identity:

`opportunity_error + efficiency_error = candidate_rec_yards_r5p - rec_yards`

must hold within `1e-9` for every scoreable row.

This is an accounting decomposition, not a causal claim.

## Optional catch translation diagnostic

Where actual targets > 0 and predicted targets > 0:

`pred_catch_rate = candidate_receptions_r5p / candidate_targets_r5p`

`actual_catch_rate = receptions / targets`

`catch_rate_error = pred_catch_rate - actual_catch_rate`

Catch-rate diagnostics are secondary. They cannot by themselves determine the final mechanism disposition because yard-per-catch is not separately isolated without target-outcome-conditioned denominators.

## Same-player pregame history

Target seasons:
- 2024
- 2025

For each target row and each mechanism:
- use only the same player's exact TE-R5P rows strictly before the target;
- history may cross seasons;
- latest up to 8 prior player-games;
- minimum 4 prior player-games.

Features:
- `prior8_opportunity_error_mean`
- `prior8_abs_opportunity_error_mean`
- `prior8_efficiency_error_mean`
- `prior8_abs_efficiency_error_mean`
- `prior8_catch_rate_error_mean` where available

No target/future row enters its own feature.

## Frozen mechanism tests

### O1 — Opportunity signed persistence
Spearman(prior mean opportunity error, current opportunity error).

PASS only if:
- > 0 in 2024;
- > 0 in 2025;
- pooled player-cluster bootstrap P(positive) >= 0.95.

### O2 — Opportunity difficulty persistence
Spearman(prior mean absolute opportunity error, current absolute opportunity error).

Same gates as O1.

### E1 — Efficiency signed persistence
Spearman(prior mean efficiency error, current efficiency error).

Same gates.

### E2 — Efficiency difficulty persistence
Spearman(prior mean absolute efficiency error, current absolute efficiency error).

Same gates.

### C1 — Catch-rate signed persistence
Spearman(prior mean catch-rate error, current catch-rate error).

Secondary:
- report by season and pooled;
- bootstrap;
- does not control final disposition.

## Error-mass diagnostics

Per 2024, 2025, pooled report:
- mean absolute total receiving-yard error;
- mean absolute opportunity component;
- mean absolute efficiency component;
- normalized opportunity mass:
  `abs(opportunity) / (abs(opportunity)+abs(efficiency))`;
- normalized efficiency mass;
- component sign cancellation rate.

This is descriptive only.

## Support

Each 2024 and 2025 target season must have:
- >=400 scoreable decomposition rows;
- >=50 distinct players.

## Bootstrap

- cluster unit = `player_clean_key`;
- 5,000 replicates;
- seed `20261007`;
- sample player clusters with replacement;
- preserve all target rows from each sampled player;
- separately grade O1, O2, E1, E2, C1.

## Disposition

`TE_PLAYER_PERSISTENCE_EFFICIENCY_DOMINANT`
if:
- support passes;
- E1 and/or E2 passes;
- neither O1 nor O2 passes.

`TE_PLAYER_PERSISTENCE_OPPORTUNITY_DOMINANT`
if:
- support passes;
- O1 and/or O2 passes;
- neither E1 nor E2 passes.

`TE_PLAYER_PERSISTENCE_MIXED_MECHANISM`
if:
- support passes;
- at least one opportunity family and at least one efficiency family pass.

`TE_PLAYER_PERSISTENCE_MECHANISM_UNRESOLVED`
if:
- support passes;
- none of O1/O2/E1/E2 pass.

`TE_PLAYER_PERSISTENCE_MECHANISM_SOURCE_LIMITED`
if support fails.

## Scientific boundaries

This audit does not authorize:
- target-share correction;
- YPT correction;
- same-player bias shrink;
- width/SD changes;
- TE1-only routing;
- target thresholds;
- sportsbook filters.

TE-R5P Width V2 remains failed/closed regardless of this result.

If efficiency persistence survives, the next question must use **genuinely player-specific football efficiency state**, not direct historical model-error patching.

If opportunity persistence survives despite TE-R5P, the next question must ask what player-state entitlement variable TE-R5P is systematically missing, not simply feed residual error back into targets.

Models fit: **0**
Sportsbook inputs: **0**
2026 outcomes: **0**
Production mutations: **0**
