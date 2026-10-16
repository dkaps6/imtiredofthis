# Player High-Value Opportunity Persistence V1 — Frozen Mechanism Contract

Date: 2026-10-15

Status: **FROZEN BEFORE SCORING / RESEARCH ONLY / NO PRODUCTION CHANGE**

Parent:
`PLAYER_HIGH_VALUE_SCORING_OPPORTUNITY_IDENTITY_V1`

Source authority:
- run `38848080453` SUCCESS
- exact head `1448fcce54f803d9737a7b3f7d9ff451b5189add`
- artifact `12023852453`
- digest `sha256:1d8a22bf6240b4d4443ab7432d3224b63eaef97439639e7b47f536aed174adda`
- source disposition `PLAYER_HIGH_VALUE_SCORING_OPPORTUNITY_SOURCE_READY`
- optional end-zone target source also READY.

Version:
`PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_V1`

## Frozen question

For a named player in a named team context:

> Does his own strictly-prior share of **high-value scoring opportunities**
> forecast his next-game share of those opportunities better than his ordinary
> overall opportunity share?

This is an opportunity-entitlement test.

It does not predict touchdown conversion.
It does not fit touchdown probability.
It does not use sportsbook information.

## Channels scored separately

### RB rushing
Primary high-value pool:
**team rush attempts at opponent yardline <= 10**.

Player outcome:
named RB/FB/HB rush attempts <=10 divided by all team rush attempts <=10.

Secondary descriptive band:
inside-5 share.

### RB receiving
Primary:
red-zone targets at opponent yardline <=20 / all team red-zone targets.

### WR receiving
Primary:
red-zone targets at opponent yardline <=20 / all team red-zone targets.

### TE receiving
Primary:
red-zone targets at opponent yardline <=20 / all team red-zone targets.

End-zone targets are source-ready but remain secondary/descriptive in this V1.
Do not change the primary receiving definition after scoring.

## Player-game universe

Target player-games come from weekly roster authority, not from target-game
box-score participation.

Require:
- roster position family matches channel;
- weekly roster status `ACT`;
- stable GSIS identity;
- scheduled team-week represented in the opportunity source.

This preserves zero-opportunity active player-games.

Do not select players because they recorded a target/carry in the target game.

## Strictly-prior history

Same-team only.

Frozen window:
last **4** prior same-team active games.

Minimum:
**3** prior same-team active games.

No previous-team rows after a trade enter the same-team state.

For candidate high-value share, additionally require at least **4 team
high-value opportunities** across the frozen prior window.

No alternate window after scoring.

## Frozen predictors

For each target player-game:

### Baseline
`recent_overall_share`

= player's total relevant opportunities across the last-4 window /
team total relevant opportunities across the same games.

Relevant overall opportunity:
- RB rushing -> all team rush attempts;
- receiving channels -> all team targets.

### Candidate
`recent_high_value_share`

= player's high-value opportunities across the last-4 window /
team total high-value opportunities across the same games.

No fitted coefficient.
No blend.
No threshold search.
No position-average replacement.

Room competition is inherently represented by the denominator: the player must
earn a larger or smaller share of the same finite high-value team pool.

Report room concentration/H​​HI separately; it does not enter the V1 predictor.

## Target outcome

Only score games where the target team generated at least one high-value
opportunity in the relevant channel.

Primary outcome:
`actual_high_value_share`.

Primary errors:
- baseline absolute share error;
- candidate absolute share error.

Gain:
`baseline_abs_error - candidate_abs_error`.

Positive = named-player high-value identity improves the forecast.

## Frozen seasons

- 2022 development
- 2023 replication
- 2024 replication
- 2025 independent confirmation

No 2026 outcomes.

No fitting occurs on any season.

## Frozen support floors per season

RB rushing:
- >= 500 scored player-games
- >= 80 distinct RB/FB/HB players

RB receiving:
- >= 250 scored player-games
- >= 70 distinct RB/FB/HB players

WR receiving:
- >= 800 scored player-games
- >= 150 distinct WRs

TE receiving:
- >= 300 scored player-games
- >= 70 distinct TEs

No lowering floors after scoring.

## Dependence-aware inference

For each channel:

Pooled:
- 5,000 player-cluster bootstrap replicates;
- 5,000 game/team-week cluster bootstrap replicates.

2025 confirmation:
- 5,000 player-cluster replicates;
- 5,000 game/team-week cluster replicates.

Bootstrap statistic:
mean share-error gain.

Fixed seeds by channel.

## Named-player heterogeneity

For 2025 confirmation, calculate per-player MAE gain for players with at least
4 scored target games.

Report:
- median per-player MAE gain;
- share of supported players with positive MAE gain;
- distribution quartiles.

This is mandatory interpretation, not optional presentation.

## Frozen qualification gates

A channel qualifies only if ALL pass:

1. source/integrity passes;
2. support floors pass in all four seasons;
3. mean candidate MAE < baseline MAE in 2022;
4. same in 2023;
5. same in 2024;
6. same in 2025;
7. pooled player-cluster 95% CI lower bound for gain > 0;
8. pooled game/team-week cluster 95% CI lower bound > 0;
9. 2025 player-cluster 95% CI lower bound > 0;
10. 2025 game/team-week cluster 95% CI lower bound > 0;
11. 2025 median per-player MAE gain > 0;
12. sportsbook inputs = 0;
13. OddsAPI calls = 0;
14. 2026 outcomes = 0;
15. predictive models fit = 0;
16. production changed = false.

No pooled-position average may rescue a failed channel.

## Dispositions

Per channel:
- `RB_RUSH_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_QUALIFIED`
- `RB_RECEIVING_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_QUALIFIED`
- `WR_RECEIVING_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_QUALIFIED`
- `TE_RECEIVING_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_QUALIFIED`

Otherwise corresponding:
`NO_ACTIONABLE_*_HIGH_VALUE_OPPORTUNITY_PERSISTENCE`.

Overall:
- all four: `PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_ALL_QUALIFIED`
- some: `PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_PARTIAL`
- none: `PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_CLOSED`
- integrity failure:
  `PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_INTEGRITY_FAILURE`.

## Anti-rescue

After scoring do not:
- swap RB primary from inside-10 to inside-20 or inside-5;
- swap receiving primary from inside-20 to inside-10/end-zone;
- change last-4/min-3;
- lower the 4 prior high-value opportunity support;
- select only lead backs / WR1s / TE1s to rescue;
- pool channels;
- fit a coefficient;
- add TD outcomes;
- add sportsbook odds;
- use 2026 results.

## If qualified

Qualification means:

the named player's scoring-area entitlement is a repeatable component of player
identity and can advance to a separately frozen integration layer.

Next stage would combine:
1. team scoring-opportunity forecast;
2. named-player high-value entitlement;
3. touchdown conversion/calibration.

That future stage remains separate.

No production mutation from V1 alone.
