# PRISM RB Stage-2 Player-Specific Mechanism Authority Bridge V1 — Frozen Before Scoring

Date: 2026-10-18

Status: **FROZEN BEFORE BRIDGE SCORING / RB RUSHING + RB RECEIVING ONLY / NO PRODUCTION CHANGE**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_RB_STAGE2_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_V1`

Scientific parent:
`PRISM_STAGE2_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_QUALIFIED_PARTIAL`

Signal authority:
- run `39099164620` — SUCCESS;
- exact head `36b34a1b392d4d6196b72116b0ef5b88273906bd`;
- artifact `12125359950`;
- digest `sha256:efa919e2590e7409e1d989b59cafe293d6d4831dfb434635df8fb5d15143e95b`.

Universal Stage-2 parent:
- run `38957575645` — SUCCESS;
- artifact `12070774378`;
- digest `sha256:e07edbad1bf97a4917ba95c58156c8941448e7b2668bd0eae2e9bcad5d4945b0`.

## 1. Frozen question

For RB rushing and RB receiving only:

> Can PRISM use the qualified named-player mechanism-utility signal to decide
> whether this supported RB's own active-role uncertainty should participate in
> Stage-2, while preserving exact finite opportunity and improving the actual
> named-player opportunity distribution over universal Stage-2?

This is the first projection-level test of:

> **The player should have the rules. The position should not have the player.**

It does not fit a new carry-share or target-share model.

## 2. Cells

Eligible:
- RB rushing;
- RB receiving.

Not eligible:
- TE receiving — player-specific eligibility V1 CLOSED on tail harm;
- QB passing — Stage-2 CLOSED;
- WR receiving — Stage-2 source-blocked.

No cell pooling for qualification.

## 3. Predictor authority

Consume only pregame fields from the immutable signal artifact:

- `player_prior_positive_q_rows`;
- `hierarchical_expected_benefit`;
- current `authority_q`;
- current Stage-1 `p_active`;
- cell/player/game identity.

The signal artifact's target `delta_crps` may not be used to choose target-row
authority. It is ignored by the bridge predictor.

## 4. Frozen player authority rule

For every canonical-positive named RB:

### Low support

If:

`player_prior_positive_q_rows < 3`

then:

`p_effective = p_active`.

The player has not earned deviation from the universal qualified Stage-2 rule.

### Supported player, predicted nonnegative Stage-2 utility

If:

`player_prior_positive_q_rows >= 3`

and:

`hierarchical_expected_benefit > 0`

then:

`p_effective = p_active`.

Stage-2 remains fully active for that named player.

### Supported player, predicted nonpositive Stage-2 utility

If:

`player_prior_positive_q_rows >= 3`

and:

`hierarchical_expected_benefit <= 0`

then:

`p_effective = 1.0`.

Interpretation:
that named player is treated as deterministically active **for Stage-2
mechanism purposes only**. His own active-role uncertainty is disabled.

This does not remove him.
It does not change his canonical entitlement.
It does not create opportunity.

No alternate threshold is tested.
Zero is the already-frozen natural benefit boundary.

## 5. Conservation law

For each RB room:

- preserve canonical `q_i`;
- replace only `p_active_i` with `p_effective_i` above;
- exactly enumerate active-set states;
- condition canonical entitlement on each active set;
- retain the existing anonymous complement;
- retain the existing all-off fallback;
- conserve the finite team opportunity total in every state.

Exact `p_effective = 1.0` is allowed and means the player is active in every
positive-weight state.

The candidate must not independently blend player means after room allocation.
Player-specific authority acts **inside** the same conservation system.

## 6. Historical scoring

Seasons:
- 2024 development;
- 2025 independent confirmation.

Primary population:
all canonical-positive RB player rows in the eligible cell.

Low-support rows remain in the evaluation because they are part of the deployed
fallback population; their own Stage-2 rule remains universal.

Primary comparator:
the already-qualified **universal Stage-2** candidate, not the older canonical
baseline.

Primary endpoint:

`delta_personalized_crps = universal_stage2_crps - personalized_stage2_crps`.

Positive means player-specific authority improves over universal Stage-2.

Protection endpoints:
- expected opportunity absolute error vs universal Stage-2;
- room-level allocation error vs universal Stage-2;
- finite-mass integrity.

## 7. Support floors

Per season:

RB rushing:
- >=1,250 canonical-positive target rows;
- >=140 distinct RBs;
- >=400 team rooms;
- >=28 teams.

RB receiving:
- >=1,350 canonical-positive target rows;
- >=145 distinct RBs;
- >=400 team rooms;
- >=28 teams.

Changed-authority support must also be disclosed:
- rows where the player's own `p_effective` becomes 1;
- distinct affected players;
- affected rooms.

No minimum changed-row gate is used to rescue or kill the mechanism after
observing performance; the counts are diagnostics.

## 8. Player-aware inference

Frozen bootstrap:
- 5,000 replicates;
- seed `20261018`;
- player cluster;
- game cluster;
- equal-player.

Equal-player frame:
- >=5 canonical-positive scored target rows for the named player in the season.

Report:
- equal-player mean CRPS improvement;
- median;
- improved / tied / worsened player shares;
- p10 / p25 / p50 / p75 / p90;
- worst named-player mean change;
- results by player prior-support tier;
- low-support fallback rows separately.

## 9. Frozen qualification gates

A cell qualifies only if **all** pass.

### 2024 development

1. support floors pass;
2. mean personalized CRPS improvement over universal Stage-2 > 0;
3. expected-count AE improvement >= 0;
4. room-allocation error improvement >= 0;
5. max state-weight / candidate-mass conservation gaps <= 1e-10.

### 2025 independent confirmation

6. support floors pass;
7. mean personalized CRPS improvement > 0;
8. player-cluster 95% CI lower bound > 0;
9. game-cluster 95% CI lower bound > 0;
10. equal-player mean improvement > 0;
11. equal-player 95% CI lower bound > 0;
12. median named-player improvement >= 0;
13. improved-player share > worsened-player share;
14. expected-count AE improvement >= 0;
15. room-allocation error improvement >= 0;
16. low-support fallback-row mean CRPS improvement >= 0;
17. tail-harm guard:
    `p10_named_player_improvement >= -0.25 * universal_stage2_equal_player_crps`;
18. max state-weight / candidate-mass conservation gaps <= 1e-10.

A pooled result cannot rescue a failed cell.

## 10. If qualified

Qualification means:
- named-player mechanism authority can improve a qualified Stage-2 cell while
  preserving finite opportunity;
- the player-specific rule is historically supported for that exact RB market.

It still does **not** authorize production deployment.

A separate prospective/deployment contract would be required.

## 11. If closed

Close the exact V1 bridge for that cell.

Do not rescue with:
- alternate zero threshold;
- pseudo-count changes;
- support floors;
- top-RB carveouts;
- recent-only benefit history;
- soft authority coefficients;
- player-name exceptions;
- room-size subsets;
- alternate loss targets;
- sportsbook or 2026 outcomes.

A future player-specific attempt must be materially distinct.

## 12. Protected boundaries

Unchanged:
- universal TE Stage-2 remains qualified;
- TE personalized signal V1 remains CLOSED;
- QB Stage-2 CLOSED;
- WR Stage-2 source-blocked;
- finite opportunity conservation mandatory;
- no sportsbook upstream;
- no OddsAPI;
- no production mutation.

## 13. Integrity at freeze

At freeze:
- personalized Stage-2 rows scored: **0**;
- projection values changed: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- Week-6 outcomes used: **0**;
- sportsbook inputs: **0**;
- OddsAPI calls: **0**;
- production changed: **false**.

Disposition at freeze:

`PRISM_RB_STAGE2_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_V1_FROZEN_BEFORE_SCORING`.
