# PLAYER OPPORTUNITY SHARE VOLATILITY — FUTURE-WEEK LOCK V1

Frozen: 2026-10-12

Status: **PROSPECTIVE FEATURE LOCK ONLY. NO MEAN CHANGE. NO DISTRIBUTION CHANGE.**

Scientific authority:
- `PLAYER_OPPORTUNITY_SHARE_STABILITY_SIGNAL_CONFIRMED` in 2024-2025;
- `PLAYER_OPPORTUNITY_SHARE_STABILITY_THREE_SEASON_CONFIRMED` after untouched
  2023 holdout.

## Purpose

Preserve the replicated raw player-specific opportunity-volatility feature
prospectively before target-week outcomes.

This lock does not interpret the feature as a betting rule or alter model
projections.

## Parent player-state authority

Input:
- exact future-week player-state rows/result from
  `PLAYER_STATE_FUTURE_WEEK_COVERAGE_V1`.

Require:
- parent disposition = `PLAYER_STATE_FUTURE_WEEK_READY`;
- exact target-week roster source;
- complete immediately-prior-week stats/snaps;
- chronology violations = 0.

## Feature semantics

Use normalized nflverse / PlayerForm weekly stats from the target season only.

For each target-week player:
- same stable identity;
- same current team;
- games strictly before target week;
- latest up to 8 completed prior games;
- require >=4 finite-share games.

### RB

Preserve:
- `prior_rush_share_mean`
- `prior_rush_share_sd`
- `prior_rush_share_min`
- `prior_rush_share_max`
- `prior_target_share_mean`
- `prior_target_share_sd`
- `prior_target_share_min`
- `prior_target_share_max`

Shares:
- player rushes / team rushes;
- player targets / team targets.

### WR

Preserve:
- prior target-share mean / SD / min / max.

### TE

Preserve:
- prior target-share mean / SD / min / max.

QB is carried for identity continuity but receives no feature from this V1.

## Unsupported rows

Do not impute or pool.

If a player has <4 qualifying same-season same-team games:
- support flag = false;
- volatility feature remains missing.

No prior-season substitution.

## Integrity

Require:
- target-week player identity exactly matches parent snapshot;
- source max week < target week;
- same-team history only;
- no target/future stats;
- no sportsbook inputs;
- no OddsAPI;
- no model residuals;
- no fitted coefficients;
- no thresholds;
- no projection changes;
- production changed = false.

## Advancement

The raw lock may feed a separately frozen uncertainty research candidate.

It may not:
- change mean entitlement;
- gate players from the model;
- create a bet/pass threshold;
- change production confidence labels by itself.

Production use requires separate prospective 2026 validation.
