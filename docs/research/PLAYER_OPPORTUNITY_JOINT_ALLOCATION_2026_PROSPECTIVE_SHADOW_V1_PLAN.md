# PLAYER OPPORTUNITY JOINT ALLOCATION — 2026 PROSPECTIVE SHADOW V1 PLAN

Frozen: 2026-10-13

Status: **PROSPECTIVE COMPONENT SHADOW ONLY. NO PRODUCTION CHANGE. NO OUTCOME GRADING BEFORE PREGAME LOCK.**

Historical authority:
- `docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_HISTORICAL_RESULT.md`
- run `38401069806`
- artifact `11815490905`
- digest `sha256:2b336114dc5ef0145c0a40706266da850a0ff1d25c8935958481b996333fce29`

Historical dispositions:
- RB carry share:
  `JOINT_ALLOCATION_DIRICHLET_HISTORICAL_PASS_FREEZE_FORWARD_SHADOW`
- WR target share:
  `JOINT_ALLOCATION_DIRICHLET_HISTORICAL_PASS_FREEZE_FORWARD_SHADOW`

Not authorized:
- RB target share
- TE target share

## Purpose

Confirm prospectively that the historically validated player-specific
opportunity-uncertainty law improves allocation calibration in 2026 without
changing any player's expected opportunity share.

This shadow intentionally evaluates the allocation component before any
production simulator integration.

It does not alter:
- player mean share;
- team pass/rush volume;
- efficiency;
- model projections;
- sportsbook pricing;
- production outputs.

## Pregame parent authorities

A target-week shadow may lock only when BOTH are available:

1. exact target-week player-state:
   `PLAYER_STATE_FUTURE_WEEK_READY`

2. exact target-week opportunity-volatility feature lock:
   `WEEK{week}_OPPORTUNITY_SHARE_VOLATILITY_LOCK_FROZEN`

Require:
- exact target-week roster source;
- immediately prior scheduled-week stats/snaps complete;
- source max week < target week;
- no target-week outcomes;
- no sportsbook input.

If either parent is not ready:
`JOINT_ALLOCATION_PROSPECTIVE_SHADOW_NOT_READY`

No stale-roster fallback.

## Canonical mean-share authority

The shadow must consume the same target-week canonical football metrics used by
the model before sportsbook comparison.

### RB carry

Per team:
1. take `rules_rush_share`;
2. apply canonical stable top-five rushing pool;
3. apply canonical 0.95 modeled-player cap;
4. preserve canonical residual bucket.

Resulting player probability:
`q_i`

### WR target

Per team:
1. take `rules_tgt_share`;
2. apply canonical M38 WR hierarchy sharpening;
3. apply canonical 0.95 modeled-player cap;
4. preserve canonical residual receiver bucket.

Resulting player probability:
`q_i`

The shadow must not substitute prior realized share mean for `q_i`.

## Identity bridge

Primary prospective identity remains stable GSIS identity from the target-week
player-state / volatility lock.

The model metric bridge may use canonical team + canonical player-name key only
when:
- exactly one target-week stable identity maps to that team/name key;
- exactly one canonical metric player maps to the same key.

Any ambiguity or collision fails the affected room closed.

Report:
- stable-ID rows;
- canonical-q matched rows;
- match rate;
- all unmatched/ambiguous identities.

No result-driven remapping.

## Eligible cells

### RB carry share

A player is candidate-eligible only when:
- position RB/HB/FB;
- `rb_carry_betabinomial_eligible == true`;
- finite positive `prior_rush_excess_share_var`;
- finite canonical `q_i` strictly between 0 and 1.

### WR target share

A player is candidate-eligible only when:
- WR-family;
- `wr_target_betabinomial_eligible == true`;
- finite positive `prior_target_excess_share_var`;
- finite canonical `q_i` strictly between 0 and 1.

Unsupported players remain inside the complement at baseline relative weights.

## Frozen latent uncertainty transport

For an eligible player:

Use the strictly-prior feature-lock values:

`mu_i = prior_share_mean`

`v_i = excess_share_var`

derive:

`rho_i = v_i / (mu_i * (1 - mu_i))`

only when:
- 0 < mu_i < 1
- 0 < v_i < mu_i(1-mu_i)
- 0 < rho_i < 1

No coefficient, shrinkage, cap, floor, threshold or bucket is fitted.

## Frozen joint law

For a room with >=1 eligible player:

- preserve each eligible player's canonical model probability `q_i`;
- collapse all ineligible players + canonical residual into complement
  probability `q_c`.

Room intraclass correlation:

`rho_room = SUM[q_i(1-q_i)rho_i] / SUM[q_i(1-q_i)]`

over eligible players.

`kappa = 1/rho_room - 1`

First-stage probabilities:

`p = [q_1, ..., q_m, q_c]`

`alpha = p * kappa`

Candidate conditional allocation after a team total N is known:

`Y ~ DirichletMultinomial(N, alpha)`

Baseline remains:

`Y ~ Multinomial(N, p)`

Expected counts are identical:
`E[Y_i] = N*q_i`

## Pregame lock artifact

Before target-week kickoff, preserve one immutable artifact containing:

Per room:
- season/week/game/team/cell;
- authoritative model/code SHA;
- stable player identity keys;
- canonical player-name keys;
- canonical `q_i`;
- canonical complement probability;
- eligible-player flags;
- frozen prior mean;
- frozen excess variance;
- `rho_i`;
- `rho_room`;
- `kappa`;
- candidate alpha vector;
- source-max week;
- parent artifact identifiers/digests;
- creation timestamp;
- row / room SHA256 digests.

Do NOT include:
- target-game N;
- target-game player opportunity counts;
- target-game outcomes;
- sportsbook lines.

## Prospective grading

Only after a valid pregame lock exists may the game be graded.

For each locked room, read target-game realized team total N and player counts y.

Primary:
- conditional joint log-score improvement:
  candidate Dirichlet-Multinomial vs baseline Multinomial.

Protection:
- eligible-player marginal count CRPS;
- exact expected-count identity.

This preserves the historical component-isolation design.

## Prospective accumulation gate

Do not promote from one week.

Require at minimum:
- >=4 prospectively locked weeks;
- RB carry:
  - >=100 locked team-game rooms;
  - >=50 distinct eligible RB identities;
- WR target:
  - >=100 locked team-game rooms;
  - >=75 distinct eligible WR identities.

For each cell independently require:
- pooled mean joint log-score improvement >0;
- game-cluster bootstrap 95% lower bound >0;
- team-cluster bootstrap 95% lower bound >0;
- marginal CRPS non-worse;
- expected-count gap <=1e-12;
- no production mutation.

Insufficient support:
`JOINT_ALLOCATION_PROSPECTIVE_INSUFFICIENT_SUPPORT`

Supported failure:
`JOINT_ALLOCATION_PROSPECTIVE_FAIL_CLOSED`

Supported pass:
`JOINT_ALLOCATION_PROSPECTIVE_CONFIRMED_FOR_INTEGRATION_TEST`

Even a PASS authorizes only a separately frozen full-simulator integration test.

## Week-6 boundary

As of the current 2026-10-13 readiness authority:
- Week-5 stats complete;
- Week-5 snaps complete;
- exact Week-6 roster authority unavailable;
- roster source remains Week 5.

Therefore Week-6 prospective shadow is currently:
`JOINT_ALLOCATION_PROSPECTIVE_SHADOW_NOT_READY`

Do not weaken this gate.
