# PRISM Team Scoring-Territory Drive Creation V1 — Frozen Mechanism Contract

Date: 2026-10-16

Status: **FROZEN BEFORE SCORING / RESEARCH ONLY / NO PRODUCTION CHANGE**

Model:
**PRISM — Player Role & Individual Situation Model**

Architecture position:

`PLAYER IDENTITY -> CURRENT ROLE/STATE -> TEAMMATE COMPETITION -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

## Scientific parents

Immutable parents remain:
- Stage 1 active-role participation: `PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED_ALL`;
- Stage 2 conditional allocation: qualified for RB rushing / RB receiving / TE receiving;
- raw opportunity-quality source: `PRISM_OPPORTUNITY_QUALITY_TEAM_VOLUME_SOURCE_READY`;
- reconciled opportunity-quality source: `PRISM_OPPORTUNITY_QUALITY_SOURCE_RECONCILED_READY_ALL`;
- canonical parent denominator persistence: `PRISM_PARENT_DENOMINATOR_PERSISTENCE_PARITY_PASS`.

Canonical denominator authority:
- run `38991915503`;
- artifact `12085194208`;
- digest `sha256:c2be00311f9cf2149ebef36d419851214add08cae83bbc3b947990ce60a5e569`.

Raw topology authority:
- run `38991605252`;
- artifact `12085367606`;
- digest `sha256:68b680c745ae8b8479b3dbaf8f41dcc4f6cb728a0f39b0ac10b77dc3ac8e4b90`.

Terminal anti-retest parents:
- `PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_CLOSED`;
- `PRISM_TEAM_HIGH_VALUE_CREATION_CLOSED`.

The direct recent-player high-value-share mechanism and direct recent-team high-value-event-rate persistence mechanism are both closed and are not reused as candidates here.

## Exact frozen question

> Conditional on PRISM's existing canonical pregame parent opportunity quantity, does the team's own strictly-prior ability to create drives that reach the relevant scoring territory improve next-game high-value opportunity count?

This mechanism changes the upstream football state from **recent high-value event rate** to **territory-entry drive creation**.

It does not use:
- the team's recent inside-10 rush rate as candidate;
- the team's recent red-zone target rate as candidate;
- player recent high-value share;
- opponent/environment state;
- touchdown outcomes;
- sportsbook information.

## Eligible cells and territory state

1. `RB_RUSH_INSIDE10`
   - Stage-2 parent: `RB_RUSHING`;
   - target high-value count: team inside-10 rushing opportunities;
   - parent opportunity: team rushing opportunities;
   - territory state: offensive drives reaching opponent 10.

2. `RB_RECEIVING_REDZONE`
   - Stage-2 parent: `RB_RECEIVING`;
   - target high-value count: team red-zone targets;
   - parent opportunity: team targets;
   - territory state: offensive drives reaching opponent 20.

3. `TE_RECEIVING_REDZONE`
   - Stage-2 parent: `TE_RECEIVING`;
   - same finite team red-zone-target pool;
   - territory state: offensive drives reaching opponent 20.

QB remains excluded. WR remains source-blocked/not scored.

## Chronology

Target seasons:
- **2024 development**
- **2025 independent confirmation**

Target weeks:
- regular-season Weeks **5-18**.

No 2026 outcomes.

Every candidate and baseline state uses only same-season games with week strictly less than the target week. Same-week games never enter history.

## Frozen team drive state

Candidate team history:
- same team;
- same season;
- last up to **4** drive-source-present completed games;
- minimum **3** prior games.

For the relevant territory (z):
- (D_{T,t}) = prior offensive drives;
- (R_{T,t,z}) = prior offensive drives reaching territory (z).

Frozen team territory-entry rate:

`g_team = R_team / D_team`.

No smoothing.
No fitted coefficient.
No alternate window.
No switching the territory threshold after scoring.

For RB rushing, (z=10).
For RB/TE receiving, (z=20).

## Frozen league reference states

For every target week, using all source-present team-games in the same season with week < target:

High-value baseline rate:

`r_league_hv = H_league_prior / N_league_parent_prior`.

League drive-entry rate for the relevant territory:

`g_league = R_league_prior / D_league_prior`.

These are strict-prior same-season league states only.

## Frozen candidate construction

The candidate uses the team's relative scoring-territory drive creation propensity as a multiplicative quality transmission on the league high-value opportunity rate:

`territory_multiplier = g_team / g_league`.

`r_candidate_raw = r_league_hv * territory_multiplier`.

`r_candidate = min(1, max(0, r_candidate_raw))`.

Baseline:

`r_baseline = r_league_hv`.

This is the only candidate transformation.

The candidate does **not** contain the team's own prior high-value event count or prior high-value event rate.

## Canonical parent opportunity quantity

For each target team-game/cell:

`lambda_parent = authority_denominator_mean`.

It must be finite, >0, and unique within numerical tolerance across named-player rows in the room.

Baseline expected team high-value count:

`lambda_HV_baseline = lambda_parent * r_baseline`.

Candidate expected team high-value count:

`lambda_HV_candidate = lambda_parent * r_candidate`.

The parent mean is identical on both sides.

## Frozen count law

Keep the same count law used by the first binding opportunity-quality V1 so this mechanism changes only the football state:

`Y_team_high_value ~ Poisson(lambda_HV)`.

No count-law change is allowed after scoring.

## Frozen player allocation

Player allocation is identical baseline vs candidate and remains the already-qualified Stage-1 + Stage-2 exact active-set expected named share (q_i^*).

`mu_i_baseline = lambda_HV_baseline * q_i_star`.

`mu_i_candidate = lambda_HV_candidate * q_i_star`.

`Y_i_high_value ~ Poisson(mu_i)`.

No new player share model is fit.

## Labels

Team target:
- observed next-game finite team inside-10 rush count or red-zone target count.

Player target:
- observed named-player count from the raw finite topology source;
- missing positive-event topology rows are zero only when the target team-game source is present and the player exists in the canonical Stage-2 universe.

Target-game labels never enter candidate construction.

## Primary scoring

Primary proper score:
**Poisson CRPS**.

Protection metric:
expected-count absolute error.

Positive gain = baseline minus candidate, so positive values are improvement.

Report both team and named-player levels.

## Frozen support floors per season and cell

Team:
- >= **400** scoreable target team-games;
- >= **300** positive high-value target team-games;
- >= **400** target team-games with valid drive-history candidate state.

Player:
- RB rushing >= **700** player-games and >= **80** players;
- RB receiving >= **700** player-games and >= **80** players;
- TE receiving >= **700** player-games and >= **75** players.

Heterogeneity frame:
- named players with >= **4** scored games in the season.

Do not lower support floors after scoring.

## Dependence-aware inference

Frozen bootstrap replicates:
**5,000**.

2025 team CRPS gain:
- team-cluster bootstrap;
- game-cluster bootstrap.

2025 player CRPS gain:
- player-cluster bootstrap;
- game-cluster bootstrap;
- equal-player bootstrap over supported players.

No row-IID confidence interval may replace these gates.

## Frozen qualification gates

A cell qualifies only if all of the following pass:

1. immutable parent/source dispositions match;
2. chronology violations = 0;
3. no sportsbook, OddsAPI, 2026 outcome or production mutation;
4. direct recent-team high-value-event-rate candidate used = false;
5. player recent high-value-share candidate used = false;
6. opponent/environment used = false;
7. support floors pass in 2024;
8. support floors pass in 2025;
9. 2024 team CRPS gain > 0;
10. 2024 team AE gain >= 0;
11. 2024 player CRPS gain > 0;
12. 2024 player AE gain >= 0;
13. 2025 team CRPS gain > 0;
14. 2025 team AE gain >= 0;
15. 2025 team-cluster CRPS-gain 95% CI lower bound > 0;
16. 2025 game-cluster team CRPS-gain 95% CI lower bound > 0;
17. 2025 player CRPS gain > 0;
18. 2025 player AE gain >= 0;
19. 2025 player-cluster CRPS-gain 95% CI lower bound > 0;
20. 2025 game-cluster player CRPS-gain 95% CI lower bound > 0;
21. 2025 equal-player mean CRPS gain > 0;
22. 2025 equal-player CRPS-gain 95% CI lower bound > 0;
23. parent denominator unchanged baseline vs candidate;
24. Stage-2 expected player share unchanged baseline vs candidate.

Per-cell disposition:
- `<CELL>_PRISM_TEAM_DRIVE_CREATION_QUALIFIED`;
- otherwise `<CELL>_PRISM_TEAM_DRIVE_CREATION_CLOSED`.

Overall:
- all qualify -> `PRISM_TEAM_DRIVE_CREATION_QUALIFIED_ALL`;
- some -> `PRISM_TEAM_DRIVE_CREATION_QUALIFIED_PARTIAL`;
- none -> `PRISM_TEAM_DRIVE_CREATION_CLOSED`;
- integrity failure -> `PRISM_TEAM_DRIVE_CREATION_INTEGRITY_FAILURE`.

No pooled cell result may rescue a failed cell.

## Mandatory reporting

For each season/cell:
- team-games;
- positive target team-games;
- player-games / players / supported heterogeneity players;
- mean league HV rate;
- mean team and league drive-entry rates;
- mean territory multiplier and clipping frequency;
- baseline/candidate team expected count, CRPS and AE;
- baseline/candidate player CRPS and AE;
- player-row improvement share;
- per-player mean/median gain and positive share;
- all 2025 dependence-aware intervals;
- exact failed gates.

## Anti-rescue contract

After first score, do **not**:
- change last-4/min-3;
- smooth or shrink the drive rate;
- tune the multiplier;
- change multiplicative transmission to additive/logit based on results;
- swap drives-reaching-10 to drives-reaching-20/5 for RB rushing;
- swap drives-reaching-20 to drives-reaching-10 for receiving;
- replace drives with offensive plays to rescue;
- insert direct recent team high-value event rate;
- insert recent player scoring-area share;
- add opponent/environment state;
- add TD outcomes;
- subset lead backs / TE1;
- use sportsbook variables;
- use 2026 outcomes;
- change Stage-1 or Stage-2 player allocation;
- weaken support or inference gates.

If this exact V1 fails, close it and move to the next genuinely different mechanism.

## Integrity boundary at freeze

At this freeze:
- predictive candidates scored: **0**;
- target-game outcomes read for candidate scoring: **0**;
- direct recent-team high-value event-rate candidate used: **false**;
- player recent high-value-share candidate used: **false**;
- opponent/environment variables used: **false**;
- fitted coefficients: **0**;
- threshold searches: **0**;
- sportsbook inputs: **0**;
- OddsAPI calls: **0**;
- 2026 outcomes: **0**;
- production changed: **false**.
