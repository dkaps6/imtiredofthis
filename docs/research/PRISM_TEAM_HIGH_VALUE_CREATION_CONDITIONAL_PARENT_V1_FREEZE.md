# PRISM Team High-Value Opportunity Creation Conditional Parent V1 — Frozen Mechanism Contract

Date: 2026-10-16

Status: **FROZEN BEFORE SCORING / RESEARCH ONLY / NO PRODUCTION CHANGE**

Model:
**PRISM — Player Role & Individual Situation Model**

Architecture position:

`PLAYER IDENTITY -> CURRENT ROLE/STATE -> TEAMMATE COMPETITION -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

## Scientific parents

Required immutable parents:

- Stage 1 active-role participation:
  `PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED_ALL`
  - run `38907459996`
  - artifact `12049120604`
  - digest `sha256:41b1acd3f1b8d77abb53980db160ece32f8fc40351939d4827248b9caad0c64b`

- Stage 2 conditional allocation:
  `PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_PARTIAL`
  - run `38957575645`
  - artifact `12070774378`
  - digest `sha256:e07edbad1bf97a4917ba95c58156c8941448e7b2668bd0eae2e9bcad5d4945b0`

- Raw finite team high-value topology:
  `PRISM_OPPORTUNITY_QUALITY_TEAM_VOLUME_SOURCE_READY`
  - run `38991605252`
  - artifact `12085367606`
  - digest `sha256:68b680c745ae8b8479b3dbaf8f41dcc4f6cb728a0f39b0ac10b77dc3ac8e4b90`

- Reconciled opportunity-quality source:
  `PRISM_OPPORTUNITY_QUALITY_SOURCE_RECONCILED_READY_ALL`
  - run `38991737241`
  - artifact `12085278127`
  - digest `sha256:0be3d571129876ce0a8df755762215e147faec2ff1ac1822e05cbd84508a12bb`

- Canonical parent denominator persistence:
  `PRISM_PARENT_DENOMINATOR_PERSISTENCE_PARITY_PASS`
  - run `38991915503`
  - artifact `12085194208`
  - digest `sha256:c2be00311f9cf2149ebef36d419851214add08cae83bbc3b947990ce60a5e569`

Terminal anti-retest parent:
`PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_CLOSED`.

That player-side recent scoring-area-share candidate remains closed and is not reused here.

## Exact frozen question

> Conditional on PRISM's already-existing canonical pregame parent opportunity quantity, does the team's own strictly-prior high-value opportunity creation state improve next-game high-value opportunity count?

This V1 isolates **team creation** only.

It does not add opponent/environment transmission.
It does not fit touchdown conversion.
It does not use player recent high-value share.
It does not change the Stage-1 or Stage-2 player allocation law.

## Eligible cells

Only the already-qualified Stage-2 cells are scored:

1. `RB_RUSH_INSIDE10`
   - Stage-2 parent: `RB_RUSHING`
   - actual team high-value count: all team rushing attempts with opponent yardline <= 10
   - historical parent opportunity: all team rush attempts
   - canonical pregame parent mean: persisted `authority_denominator_mean`

2. `RB_RECEIVING_REDZONE`
   - Stage-2 parent: `RB_RECEIVING`
   - actual team high-value count: all team targets with opponent yardline <= 20
   - historical parent opportunity: all team targets
   - canonical pregame parent mean: persisted `authority_denominator_mean`

3. `TE_RECEIVING_REDZONE`
   - Stage-2 parent: `TE_RECEIVING`
   - same finite team red-zone-target pool as RB receiving
   - same canonical pregame team target mean
   - TE player allocation remains TE's qualified Stage-2 allocation

QB is excluded because QB Stage-2 V1 is closed.
WR is excluded because 2025 independent OOS Stage-2 entitlement authority is blocked.

## Chronology

Target seasons:
- **2024 development**
- **2025 independent confirmation**

Target weeks:
- regular-season Weeks **5-18**.

No 2026 outcomes.

For every target team-game, all historical state must come from the same season and from weeks strictly less than the target week. Same-week earlier games are not eligible history.

## Frozen historical team state

Candidate history:
- same team;
- same season;
- last up to **4** source-present completed games;
- minimum **3** prior games;
- no opponent variables.

For cell (c), let:
- (H_{T,t}) = sum of the team's high-value opportunities across that frozen prior window;
- (N_{T,t}) = sum of the corresponding parent opportunities across that window.

Frozen team rate:

`r_team = H_T / N_T`.

No smoothing coefficient.
No fitted weight.
No band search.
No alternate window.

If fewer than 3 prior source-present games exist, or (N_T <= 0), the target team-game is not scoreable.

## Frozen baseline

The non-team-specific baseline is the strictly-prior same-season league environment.

For the target week:
- include every source-present team-game in the same season with week < target week;
- sum high-value opportunities and parent opportunities across those team-games.

`r_league = H_league_prior / N_league_prior`.

This baseline uses the exact same canonical pregame parent opportunity mean as the candidate. The only difference is whether the high-value rate is league-generic or team-specific.

## Canonical parent opportunity quantity

For every target team-game/cell, the pregame parent mean is the already-persisted:

`lambda_parent = authority_denominator_mean`.

Required integrity:
- finite and > 0;
- one unique value within numerical tolerance across the player rows in the same team-game/cell.

No denominator is refit or reconstructed from the target outcome.

## Frozen team count law

The parent opportunity quantity is treated as the mean of the existing finite-opportunity process. High-value opportunities are a thinning of that parent opportunity process.

Baseline:

`lambda_HV_baseline = lambda_parent * r_league`.

Candidate:

`lambda_HV_candidate = lambda_parent * r_team`.

Frozen count law:

`Y_team_high_value ~ Poisson(lambda_HV)`.

This is a count-distribution comparison conditional on the same PRISM parent opportunity quantity. No opponent state or touchdown outcome enters either mean.

## Frozen player allocation

The player side is held fixed.

For each team-game/cell:
1. consume the canonical Stage-2 `authority_q`;
2. consume the already-qualified Stage-1 `p_active`;
3. apply the exact already-qualified Stage-2 independent-Bernoulli active-set enumeration and anonymous-complement fallback;
4. take the Stage-2 expected named-player share (q_i^*).

No new player share is fit.

Player expected high-value count:

Baseline:
`mu_i_baseline = lambda_HV_baseline * q_i_star`.

Candidate:
`mu_i_candidate = lambda_HV_candidate * q_i_star`.

Player count law:

`Y_i_high_value ~ Poisson(mu_i)`.

The same (q_i^*) is used on both sides. Therefore every player-level difference is attributable only to the team high-value creation state.

## Target labels

Team:
- observed finite team high-value opportunity count in the target game.

Player:
- observed named-player high-value opportunity count from the raw finite topology source;
- a missing positive-event topology row is zero only when the target team-game source is present and the player exists in the canonical Stage-2 player universe.

Target-game counts are labels only. They never enter the pregame rates or allocation.

## Primary scoring

Primary proper score:
**Poisson CRPS**.

Report separately:
- team baseline CRPS;
- team candidate CRPS;
- team CRPS gain = baseline - candidate;
- player baseline CRPS;
- player candidate CRPS;
- player CRPS gain = baseline - candidate.

Positive gain is improvement.

Mean/count protection:
- absolute error of expected team high-value count;
- absolute error of expected named-player high-value count.

No touchdown metric may rescue a count failure.

## Frozen support floors per season and cell

Team:
- >= **400** scoreable target team-games;
- >= **300** target team-games with at least one observed high-value opportunity.

Player:
- RB rushing: >= **700** player-games and >= **80** players;
- RB receiving: >= **700** player-games and >= **80** players;
- TE receiving: >= **700** player-games and >= **75** players.

Player heterogeneity frame:
- named players with >= **4** scored games in the season.

No support floor may be lowered after scoring.

## Dependence-aware inference

Frozen bootstrap replicates:
**5,000**.

2025 confirmation team CRPS gain:
- team-cluster bootstrap;
- game-cluster bootstrap.

2025 confirmation player CRPS gain:
- player-cluster bootstrap;
- game-cluster bootstrap;
- equal-player bootstrap over each supported player's mean CRPS gain.

Fixed seeds are implementation constants declared before scoring.

No naive row-IID confidence interval may substitute for these gates.

## Frozen qualification gates

A cell qualifies only if **all** gates pass:

1. all immutable parent/source dispositions match the authorities above;
2. chronology violations = 0;
3. no sportsbook input, no OddsAPI call, no 2026 outcomes, no production mutation;
4. support floors pass in 2024;
5. support floors pass in 2025;
6. 2024 development team mean CRPS gain > 0;
7. 2024 development team expected-count AE gain >= 0;
8. 2024 development player mean CRPS gain > 0;
9. 2024 development player expected-count AE gain >= 0;
10. 2025 confirmation team mean CRPS gain > 0;
11. 2025 confirmation team expected-count AE gain >= 0;
12. 2025 team-cluster CRPS-gain 95% CI lower bound > 0;
13. 2025 game-cluster team CRPS-gain 95% CI lower bound > 0;
14. 2025 confirmation player mean CRPS gain > 0;
15. 2025 confirmation player expected-count AE gain >= 0;
16. 2025 player-cluster CRPS-gain 95% CI lower bound > 0;
17. 2025 game-cluster player CRPS-gain 95% CI lower bound > 0;
18. 2025 equal-player mean CRPS gain > 0;
19. 2025 equal-player CRPS-gain 95% CI lower bound > 0;
20. parent denominator is unchanged between baseline and candidate;
21. Stage-2 expected player share is unchanged between baseline and candidate.

Per-cell disposition:
- `<CELL>_PRISM_TEAM_HIGH_VALUE_CREATION_QUALIFIED`
- otherwise `<CELL>_PRISM_TEAM_HIGH_VALUE_CREATION_CLOSED`.

Overall:
- all three qualify -> `PRISM_TEAM_HIGH_VALUE_CREATION_QUALIFIED_ALL`;
- some qualify -> `PRISM_TEAM_HIGH_VALUE_CREATION_QUALIFIED_PARTIAL`;
- none qualify -> `PRISM_TEAM_HIGH_VALUE_CREATION_CLOSED`;
- any integrity failure -> `PRISM_TEAM_HIGH_VALUE_CREATION_INTEGRITY_FAILURE`.

No pooled cell result may rescue a failed cell.

## Mandatory reporting

For each season/cell:
- team-games and positive-high-value team-games;
- player-games, distinct players, supported heterogeneity players;
- mean baseline/candidate team rate and high-value count mean;
- team CRPS and expected-count AE;
- player CRPS and expected-count AE;
- share of player rows improved;
- per-player median/mean gain and positive-player share;
- all 2025 dependence-aware confidence intervals;
- exact failed gates.

## Anti-rescue contract

After the first score, do **not**:
- change last-4/min-3;
- add shrinkage to the team rate;
- tune a blend between team and league;
- switch RB rushing to inside-20 or inside-5;
- switch receiving to inside-10 or end-zone targets;
- subset lead backs / TE1 / favored players;
- pool RB receiving with TE receiving to rescue either cell;
- add opponent allowance;
- add drive state;
- add touchdown outcomes;
- add sportsbook variables;
- change Stage-1 participation;
- change Stage-2 allocation;
- use 2026 outcomes;
- tune thresholds, windows, bands, or subgroups after results.

If this exact team-creation V1 fails, close it and move to the next genuinely different opportunity-quality mechanism.

## Integrity boundary at freeze

At this freeze:
- predictive candidates scored: **0**;
- target-game outcomes read for scoring: **0**;
- player recent high-value share used as candidate: **0**;
- opponent/environment variables used: **0**;
- fitted coefficients: **0**;
- sportsbook inputs: **0**;
- OddsAPI calls: **0**;
- 2026 outcomes: **0**;
- production changed: **false**.
