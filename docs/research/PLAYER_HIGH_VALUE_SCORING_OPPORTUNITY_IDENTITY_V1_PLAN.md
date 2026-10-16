# Player High-Value Scoring Opportunity Identity V1 — Frozen Source / Mechanism Plan

Date: 2026-10-15

Status: **FROZEN BEFORE SCIENTIFIC SCORING / RESEARCH ONLY / NO PRODUCTION CHANGE**

Version:
`PLAYER_HIGH_VALUE_SCORING_OPPORTUNITY_IDENTITY_V1`

## Why this lane exists

The repository is a named-player projection model.

The SEA-DEN Dobbins reasoning exposed a useful distinction:
- rushing-yard opportunity and rushing efficiency can be mediocre;
- touchdown opportunity can still be strong if the named player owns the
  offense's valuable scoring-area touches.

That is a market-specific player identity question.

The current generic ATD execution already contains football-only player state,
but it is intentionally uncertified. Current lineage is approximately:
- player historical offensive TD/game rate, with position-prior fallback;
- season/blended red-zone target share;
- season/blended red-zone carry share;
- `rz_share = max(rz_tgt_share, rz_carry_share)`;
- generic `0.75 + rz_share` multiplier;
- small `team_wp` script modifier;
- Bernoulli touchdown draw.

Canonical validator status remains:
`DEDICATED_ATD_WALK_FORWARD_CERTIFICATION_REQUIRED`.

Its own declared next research requirement is:
`football-only team TD opportunity -> red-zone/end-zone pool -> player entitlement -> probability calibration`.

This V1 starts with the **player entitlement / opportunity** piece.
It does not jump directly to touchdown probability.

## Frozen player-first question

> Can strictly-prior named-player high-value opportunity identity forecast
> which player is likely to receive a team's scoring-area rushing/receiving
> opportunities in the next game?

Primary scientific unit:
**named player-game**, not position average.

Position aggregates are support/calibration summaries only.

## Anti-retest boundary

Do NOT reopen or disguise:

- the closed situational third-down/red-zone/two-minute target-residual V1;
  that experiment asked whether broad situational target state directly
  explained receiving-yard residual;
- generic historical TD/game rate;
- the current generic `rz_share` ATD modifier;
- RB Vacancy Opportunity V1 deterministic vacated-share transfer;
- retrospective M96 RB routing/threshold searches;
- generic position-level touchdown rate fitting;
- sportsbook-implied TD probability as a football feature.

This lane is genuinely different:
it forecasts **named-player high-value opportunity entitlement** before any
touchdown-conversion model is considered.

## Source contract

Historical source:
nflverse/nflfastR play-by-play through the repo's canonical `get_pbp` path.

Target source-audit seasons:
- 2022
- 2023
- 2024
- 2025 regular seasons.

No 2026 outcomes.

Required core PBP fields:
- season
- week
- posteam
- rush_attempt
- pass_attempt
- yardline_100
- rusher player identity
- receiver player identity

Position identity:
weekly roster / player-stat GSIS identity only.

Optional end-zone target derivation may use `air_yards` only if source
coverage passes a separately reported source gate. Failure of optional
end-zone-target coverage does not authorize an invented proxy.

## Player opportunity topology

Keep rushing and receiving channels separate.

### Rushing

For each named rusher and team-game:
- total rush attempts;
- rush attempts inside opponent 20;
- rush attempts inside opponent 10;
- rush attempts inside opponent 5;
- goal-to-go rush attempts where source-supported;
- player share of team rush attempts in each scoring-area band.

### Receiving

For each named receiver and team-game:
- total targets;
- targets inside opponent 20;
- targets inside opponent 10;
- scoring-area target share;
- optional end-zone targets only if the source gate passes.

Do not collapse rushing and receiving into one `max()` state in this
mechanism test.

## Strictly-prior player identity state

For every target player-game, construct from games strictly before target week:

1. recent overall opportunity share;
2. season-to-date overall opportunity share;
3. recent high-value opportunity count;
4. recent high-value opportunity share;
5. season-to-date high-value opportunity share;
6. trajectory in high-value share;
7. room concentration / teammate competition inside the same high-value pool;
8. number of prior games supporting the state.

First frozen recency window:
**last 4 same-team games**, minimum 3.

This is a descriptive prior/state definition, not a fitted coefficient.

Do not alter the window after scoring.

## Room competition

For each team-game and channel, calculate strictly-prior:
- top teammate high-value share;
- gap between player and top teammate;
- room HHI / concentration;
- player's share of the high-value pool.

This operationalizes the "tag-team" part of player identity:
the team can create an opportunity, but the model must estimate **which named
player gets it**.

## Team scoring-opportunity state

Keep separate from player entitlement.

Source-only team context may include strictly-prior:
- red-zone offensive plays / game;
- inside-10 offensive plays / game;
- inside-5 offensive plays / game;
- team scoring-area rush/pass mix.

This V1 does not yet fit team touchdown probability.

Opponent/environment transmission is NOT added in this first mechanism test.
First establish whether the player's own high-value entitlement is real and
persistent.

## Frozen source-audit gates

Before any predictive scoring:

For each season 2022-2025:
1. regular-season PBP nonempty;
2. week/team coverage >= 99%;
3. `yardline_100` coverage on qualifying rush/target rows >= 99%;
4. rusher stable-identity coverage >= 99%;
5. receiver stable-identity coverage >= 99%;
6. RB/WR/TE position-family bridge >= 98%;
7. scoring-area rush events > 0 in every regular-season week;
8. scoring-area target events > 0 in every regular-season week;
9. inside-10 and inside-5 rushing events > 0;
10. duplicate player-play identity contradictions = 0;
11. sportsbook inputs = 0;
12. OddsAPI calls = 0;
13. 2026 outcomes = 0;
14. predictive models fit = 0;
15. production changes = 0.

Optional end-zone target gate:
- `air_yards` coverage on receiver-target rows >= 95%;
- derived end-zone-target semantics must be deterministic and documented.

If optional gate fails, omit end-zone target V1 rather than proxy it.

## Frozen mechanism test after source readiness

The first scientific test predicts **next-game high-value opportunity**, not TD.

Positions/channels scored separately:
- RB rushing;
- RB receiving;
- WR receiving;
- TE receiving.

For each target player-game with >=3 strict-prior same-team games:

Primary outcome:
next-game player share of the relevant team high-value opportunity pool.

Secondary:
next-game count of high-value opportunities.

Frozen baselines:
- recent overall opportunity share only;
- season-to-date overall opportunity share only.

Frozen candidate:
adds the named player's high-value opportunity identity + room competition
state.

No touchdown result enters this first mechanism test.

## Required player-level interpretation

A candidate is not allowed to pass because pooled position MAE looks better
while named-player behavior is incoherent.

Report:
- player-game error;
- per-player error change for supported repeat players;
- player-cluster bootstrap;
- game/team cluster bootstrap;
- concentration strata;
- high-history vs sparse-history strata;
- named-player examples only after aggregate gates are scored.

## Promotion boundary

If high-value opportunity entitlement qualifies, it authorizes only the next
separately frozen stage:

`TEAM SCORING OPPORTUNITY -> PLAYER HIGH-VALUE ENTITLEMENT -> TD CONVERSION CALIBRATION`

It does NOT immediately certify ATD betting.

A dedicated football-only ATD probability model must still be calibrated and
walk-forward validated prospectively before ATD becomes decision-grade.

## Dobbins interpretation

The manual SEA-DEN reasoning becomes a modelable hypothesis:

- current workload establishes that Dobbins is a real active backfield owner;
- scoring-area carry share identifies whether Denver tends to give **him**
  the valuable rush opportunity;
- teammate competition tells us how concentrated that opportunity is;
- team scoring opportunity determines whether Denver is likely to create the
  situation at all;
- rushing efficiency is a separate question.

Thus a weak rushing-yards projection and a strong TD-opportunity projection
can coexist without contradiction.

This game is motivation only.
It is not a training row or pass/fail proof.

No paid OddsAPI.
No production mutation.
Canonical status remains:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.
