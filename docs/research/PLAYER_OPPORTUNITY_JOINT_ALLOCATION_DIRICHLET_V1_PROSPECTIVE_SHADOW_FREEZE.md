# PLAYER OPPORTUNITY JOINT ALLOCATION — DIRICHLET V1 PROSPECTIVE SHADOW FREEZE

Frozen: 2026-10-13

Status: **PROSPECTIVE RESEARCH SHADOW ONLY. NO PRODUCTION PRICING OR BOARD CHANGE. FAIL CLOSED UNTIL EXACT TARGET-WEEK PLAYER STATE EXISTS.**

Historical authority:
- `docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_HISTORICAL_RESULT.md`
- run `38401069806`
- artifact `11815490905`
- digest `sha256:2b336114dc5ef0145c0a40706266da850a0ff1d25c8935958481b996333fce29`

Historically authorized cells only:
- `RB_CARRY_SHARE`
- `WR_TARGET_SHARE`

Not authorized:
- RB target share
- TE target share

## Purpose

Lock the joint latent-role distribution **before target-week outcomes** using:
1. exact target-week player identity;
2. strictly-prior player excess-share variance;
3. canonical target-week model mean allocation probabilities.

This prospective shadow exists to test distribution quality. It must not alter
the production model, prices, recommendations, or sportsbook board.

## Required pregame authorities

For season S / target week W:

1. target-week player-state result:
   - `PLAYER_STATE_FUTURE_WEEK_READY`
   - `target_week_roster_source_ready = true`
   - exact roster source week = W
   - complete immediately-prior-week stats/snaps

2. opportunity-share volatility lock:
   - status `WEEK{W}_OPPORTUNITY_SHARE_VOLATILITY_LOCK_FROZEN`
   - chronology violations = 0
   - no target-week outcomes
   - authorized candidate fields only:
     - `rb_carry_betabinomial_eligible`
     - `wr_target_betabinomial_eligible`

3. leakage-safe target-week model context:
   - canonical pregame player universe;
   - canonical rules-applied mean shares;
   - no sportsbook input required.

If any authority is missing, shadow state is NOT created.

## Canonical mean probabilities

RB carry:
- `rules_rush_share`;
- canonical top-five rushing pool;
- canonical 0.95 modeled-player cap + residual.

WR target:
- `rules_tgt_share`;
- canonical M38 hierarchy sharpening;
- canonical 0.95 modeled-player cap + residual.

These are the shadow mean probabilities q_i.

Historical raw share means never replace q_i.

## Volatility transport

For an eligible player:

`rho_i = excess_share_var / (prior_share_mean*(1-prior_share_mean))`

using only the locked strictly-prior feature.

Require 0 < rho_i < 1 and 0 < q_i < 1.

Invalid rows remain baseline / ineligible. No imputation.

## Joint room lock

For each team/cell:

Eligible players E use the historical PASS law:

`rho_room = SUM[q_i(1-q_i)rho_i] / SUM[q_i(1-q_i)]`

`kappa = 1/rho_room - 1`

First-stage categories:
- every eligible player separately;
- one complement containing all noneligible modeled players + residual.

Lock:
- q_i
- rho_i
- rho_room
- kappa
- alpha_i = q_i*kappa
- complement q / alpha
- player identity
- feature source max week
- target week
- canonical-model source metadata

No target-game team opportunity count N is needed pregame.

## Mean/conservation invariants

Before the shadow is frozen:
- room probabilities sum to 1 within 1e-12;
- alpha/alpha0 equals canonical q within 1e-12;
- max expected-share gap = 0 within numerical tolerance;
- no eligible player appears twice;
- no source feature week >= target week.

## Prospective grading

After games settle, and only after pregame lock bytes are frozen:

For actual target-game team count N:
- RB carry: team rushes
- WR target: team targets

Realized vector:
- each locked eligible player count;
- complement = N - sum eligible counts.

Score:
1. baseline Multinomial joint log score;
2. locked Dirichlet-Multinomial joint log score;
3. eligible-player marginal Binomial vs Beta-Binomial CRPS.

Do not rebuild q/rho after the game.

## Prospective evidence rule

Do not promote from one week.

Accumulate immutable future weeks until at least:
- 4 locked weeks;
- >=100 candidate-applied team-game rooms per cell;
- >=50 distinct eligible players per cell.

Report each week separately plus pooled descriptive evidence.

Production promotion requires a separately frozen gate after prospective sample
requirements are met.

## Integrity

- no sportsbook / OddsAPI;
- no target-week outcomes at lock time;
- no fitted coefficients;
- no threshold search;
- no mean changes;
- no production simulator edit;
- no production board/pricing change.

## Week-6 status at freeze

Week 6 is currently **NOT LOCK-READY**:
- Week-5 stats/snaps complete;
- target-week roster source still reports Week 5;
- `target_week_roster_source_ready = false`.

Do not substitute the Week-5 roster.
