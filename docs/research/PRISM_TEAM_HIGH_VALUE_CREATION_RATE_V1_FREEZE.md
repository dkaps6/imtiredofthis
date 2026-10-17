# PRISM Team High-Value Opportunity Creation Rate V1 — Frozen Mechanism Contract

Frozen: 2026-10-16

Status: **FROZEN BEFORE SCORING / RESEARCH ONLY / NO PRODUCTION CHANGE**

Model:
**PRISM — Player Role & Individual Situation Model**

Architecture position:

`PLAYER IDENTITY -> CURRENT ROLE/STATE -> TEAMMATE COMPETITION -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

## 1. Scientific question

Conditional on PRISM's already-qualified canonical pregame parent opportunity
quantity, does the team's own strictly-prior same-season high-value-opportunity
creation state improve the next-game high-value-opportunity count?

This is the first scored PRISM **opportunity-quality / team-creation** mechanism.

It is not a team model replacing the player model. The named player remains the
scientific object downstream. Team high-value creation is only the finite
environment through which the existing canonical player entitlement is mapped.

## 2. Immutable parents

Canonical parent denominator persistence:
- run `38991915503` — SUCCESS;
- exact head `02e96ad9faabee62c1ecb31303b48746e4945a8c`;
- artifact `12085194208`;
- digest `sha256:c2be00311f9cf2149ebef36d419851214add08cae83bbc3b947990ce60a5e569`;
- disposition `PRISM_PARENT_DENOMINATOR_PERSISTENCE_PARITY_PASS`;
- numeric field `authority_denominator_mean`.

Raw finite team high-value source:
- run `38991605252` — SUCCESS;
- exact head `e590ce4754da2c7f7aa01beab2ee148aa5b613a0`;
- artifact `12085367606`;
- digest `sha256:68b680c745ae8b8479b3dbaf8f41dcc4f6cb728a0f39b0ac10b77dc3ac8e4b90`;
- disposition `PRISM_OPPORTUNITY_QUALITY_TEAM_VOLUME_SOURCE_READY`.

Exact source reconciliation:
- run `38991737241` — SUCCESS;
- artifact `12085278127`;
- digest `sha256:0be3d571129876ce0a8df755762215e147faec2ff1ac1822e05cbd84508a12bb`;
- disposition `PRISM_OPPORTUNITY_QUALITY_SOURCE_RECONCILED_READY_ALL`.

Qualified player-side parents:
- Stage 1 active-role: `PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED_ALL`;
- Stage 2 conditional allocation: qualified for RB rushing, RB receiving, TE receiving.

Terminal anti-retest parent:
`PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_CLOSED`.

That closed result is not a candidate in this mechanism.

## 3. Frozen cells

Only these cells are scored:

1. `RB_RUSH_INSIDE10`
   - parent opportunity = canonical PRISM team rushing-opportunity mean;
   - high-value team count = all team rushing opportunities with opponent yardline <= 10;
   - player allocation = existing canonical RB rushing entitlement `q_i`.

2. `RB_RECEIVING_REDZONE`
   - parent opportunity = canonical PRISM team target-opportunity mean;
   - high-value team count = all team targets with opponent yardline <= 20;
   - player allocation = existing canonical RB receiving entitlement `q_i`.

3. `TE_RECEIVING_REDZONE`
   - parent opportunity = canonical PRISM team target-opportunity mean;
   - high-value team count = all team targets with opponent yardline <= 20;
   - player allocation = existing canonical TE receiving entitlement `q_i`.

QB is excluded because exact Stage-2 V1 is closed.
WR is excluded because 2025 independent Stage-2 OOS authority is blocked.

## 4. Chronology

Target seasons:
- **2024 development**
- **2025 independent confirmation**

Target weeks:
regular-season Weeks **5-18** only.

Candidate team history:
- same team;
- same season;
- last **4** completed prior games;
- minimum **3** valid prior games;
- source-present parent and high-value channels only.

League baseline history:
- same season;
- all source-valid team-games with week strictly less than the target week.

No same-week target information enters either state.
No 2026 outcome is read.

2022/2023 are not scored and cannot rescue a failed 2025 confirmation.

## 5. Frozen rates

For each target team-game and cell, define strictly-prior league rate:

`p_league = sum(prior league high-value counts) / sum(prior league parent opportunities)`.

Define the team's own last-4 rate:

`p_team = sum(last-4 team high-value counts) / sum(last-4 team parent opportunities)`.

No fitted coefficient.
No blend.
No shrinkage strength.
No opponent allowance.
No threshold search.
No recent player high-value share.

Both rates are clipped only for numerical integrity to `[0,1]`; any raw rate
outside that interval by more than `1e-12` is an integrity failure.

## 6. Conditional parent quantity

Let:

`mu = authority_denominator_mean`

from the immutable canonical Stage-2 source.

This value is not refit, rescaled, or replaced.

Baseline expected team high-value count:

`E0[H] = mu * p_league`.

Candidate expected team high-value count:

`E1[H] = mu * p_team`.

The experiment therefore changes only the team's strictly-prior high-value
creation rate. Parent opportunity quantity is held fixed.

## 7. Frozen finite-count law

The canonical parent opportunity mean `mu` is continuous. To preserve that
mean exactly while retaining finite count support, use a two-adjacent-binomial
mixture.

Let:
- `n0 = floor(mu)`;
- `n1 = ceil(mu)`;
- `w = mu - n0`.

For high-value probability `p`:

`H ~ (1-w) * Binomial(n0, p) + w * Binomial(n1, p)`.

If `mu` is an integer, this reduces to `Binomial(mu, p)`.

Properties required by contract:
- finite support `0..ceil(mu)`;
- `E[H] = mu * p` exactly to numerical tolerance;
- high-value team count can never exceed the finite parent support.

No alternative Poisson, negative-binomial, dispersion fit, or tail rescue may be
introduced after scoring.

## 8. Frozen named-player mapping

Player-side allocation is not refit.

For named player `i`, use exact canonical Stage-2 entitlement `q_i`.

Each parent opportunity has frozen categorical probabilities:
- high-value opportunity assigned to player `i`: `p * q_i`;
- high-value opportunity assigned to room complement:
  `p * (1 - sum(q_i in scored family room))`;
- non-high-value opportunity: `1 - p`.

The marginal named-player count therefore follows the same adjacent-binomial
mixture with probability:

`p_i = p * q_i`.

Expected named-player high-value count:

`E[H_i] = mu * p * q_i`.

This is an intentional independence assumption for V1: high-value topology is
created at team level and the current canonical overall entitlement allocates it.
No recent player scoring-area share is allowed to modify `q_i`.

Room conservation requirements:
- every `q_i` finite and in `[0,1]`;
- family-room `sum(q_i) <= 1 + 1e-10`;
- complement mass nonnegative within tolerance.

## 9. Scoring

### Team level

For every scored team-game:
- discrete CRPS under the finite count distribution;
- expected-count absolute error.

Gain is always:

`baseline_error - candidate_error`.

Positive = candidate better.

### Named-player level

For every Stage-2 authority-covered named-player row:
- discrete CRPS for actual player high-value count;
- expected-count absolute error.

Zero high-value player counts remain valid labels when the team source is present.
Players are not selected based on recording a target-game high-value event.

### Heterogeneity

For 2025 confirmation, among players with at least **4** scored target games:
- per-player mean CRPS gain;
- median per-player CRPS gain;
- share of supported players with positive CRPS gain;
- quartiles.

Position-average results are downstream QA only and cannot rescue a failed
named-player gate.

## 10. Frozen support floors

Per cell, per scored season:

Team:
- >= **400** scored team-games;
- >= **30** distinct teams;
- >= **300** team-games with at least one observed high-value opportunity.

Player:
- RB rushing: >= **800** player-games / >= **80** players;
- RB receiving: >= **800** player-games / >= **80** players;
- TE receiving: >= **700** player-games / >= **75** players.

Additional:
- >= 95% of candidate team-games have >=3 valid strictly-prior same-team games;
- league prior available for 100% of scored rows;
- canonical parent denominator finite for >=99% of authority-covered rows;
- target-game topology source present for >=99% of scheduled target team-games;
- chronology violations = 0.

No support floor may be lowered after scoring.

## 11. Dependence-aware inference

Use **5,000** bootstrap replicates for every frozen interval.

Fixed seeds:
- RB rush: base `2026101701`;
- RB receiving: base `2026101711`;
- TE receiving: base `2026101721`.

2025 team CRPS gain:
- team-cluster bootstrap;
- NFL game-cluster bootstrap.

2025 named-player CRPS gain:
- player-cluster bootstrap;
- team-game cluster bootstrap;
- equal-player bootstrap, where each supported player contributes one mean gain.

95% percentile intervals.

No IID-row confidence interval may substitute for these.

## 12. Frozen qualification gates

A cell qualifies only if **all** gates pass.

Integrity/support:
1. immutable parent artifact identities/digests pass;
2. chronology violations = 0;
3. source and finite-pool conservation pass;
4. frozen support floors pass in 2024 and 2025;
5. canonical `q_i` and parent denominator integrity pass;
6. no player recent high-value-share predictor is used.

Development direction, 2024:
7. team CRPS gain > 0;
8. team expected-count AE gain >= 0;
9. named-player CRPS gain > 0;
10. named-player expected-count AE gain >= 0.

Independent confirmation, 2025:
11. team CRPS gain > 0;
12. team expected-count AE gain >= 0;
13. named-player CRPS gain > 0;
14. named-player expected-count AE gain >= 0;
15. 2025 team-cluster CRPS-gain CI lower bound > 0;
16. 2025 game-cluster CRPS-gain CI lower bound > 0;
17. 2025 player-cluster named-player CRPS-gain CI lower bound > 0;
18. 2025 team-game-cluster named-player CRPS-gain CI lower bound > 0;
19. 2025 equal-player CRPS-gain CI lower bound > 0;
20. 2025 median supported-player CRPS gain > 0.

Integrity boundary:
21. sportsbook inputs = 0;
22. OddsAPI calls = 0;
23. 2026 outcomes = 0;
24. fitted predictive coefficients = 0;
25. production changed = false.

A team-level win cannot rescue a named-player failure.
A pooled player average cannot rescue failed dependence-aware confirmation.

## 13. Dispositions

Per cell:
- `RB_RUSH_INSIDE10_PRISM_TEAM_HIGH_VALUE_CREATION_QUALIFIED`
- `RB_RECEIVING_REDZONE_PRISM_TEAM_HIGH_VALUE_CREATION_QUALIFIED`
- `TE_RECEIVING_REDZONE_PRISM_TEAM_HIGH_VALUE_CREATION_QUALIFIED`

Otherwise:
- `<CELL>_PRISM_TEAM_HIGH_VALUE_CREATION_CLOSED`

Overall:
- all qualify: `PRISM_TEAM_HIGH_VALUE_CREATION_QUALIFIED_ALL`;
- some qualify: `PRISM_TEAM_HIGH_VALUE_CREATION_QUALIFIED_PARTIAL`;
- none qualify: `PRISM_TEAM_HIGH_VALUE_CREATION_CLOSED`;
- integrity failure: `PRISM_TEAM_HIGH_VALUE_CREATION_INTEGRITY_FAILURE`.

Historical qualification remains research evidence only. It is not production
promotion.

## 14. Anti-rescue contract

After scoring, do **not**:
- change last-4/min-3;
- switch RB inside-10 to inside-20 or inside-5;
- switch receiving red-zone to inside-10/end-zone;
- add opponent allowance or game environment;
- add drives/possessions;
- shrink/blend `p_team` toward league;
- fit a coefficient or calibration;
- add player recent scoring-area share;
- carve lead backs / TE1 / high-volume subgroups;
- lower support floors;
- use touchdown outcomes;
- use sportsbook lines/odds;
- use 2026 outcomes;
- pool a failed cell with a passing cell;
- change the finite-count distribution;
- relax a dependence-aware inference gate.

If the exact V1 fails, close it and move to the next genuinely different
architectural hypothesis.

## 15. Integrity at freeze

At this freeze:
- predictive candidates scored: **0**;
- target-game outcomes read by this mechanism: **0**;
- player high-value-share rescue attempts: **0**;
- opponent/environment predictors used: **0**;
- fitted coefficients: **0**;
- sportsbook inputs: **0**;
- OddsAPI calls: **0**;
- 2026 outcomes: **0**;
- production changed: **false**.
