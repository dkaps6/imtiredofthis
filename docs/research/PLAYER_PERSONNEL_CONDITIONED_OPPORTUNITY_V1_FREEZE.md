# Player Personnel-Conditioned Opportunity V1 — Frozen Diagnostic

Date frozen: 2026-10-14

Status: **FROZEN BEFORE SCORING / PLAYER-FIRST / HISTORICAL DIAGNOSTIC ONLY / NO PRODUCTION CHANGE**

Exact branch parent at freeze: `91f23daa237991c7083c1c91b858cb6ad099df92`.

## Why this exists

The model projects named players. Position-level aggregates are safety guardrails, not the hypothesis grain.

Two promoted-authority mechanism studies independently found persistent named-player opportunity error after existing individual entitlement science:

- WR: `WR_PLAYER_PERSISTENCE_MIXED_MECHANISM`, run `37634044900`, artifact `11487359185`.
- TE: `TE_PLAYER_PERSISTENCE_MIXED_MECHANISM`, run `37631011196`, artifact `11486082927`.

Both studies explicitly prohibit feeding prior residuals directly back into the next projection. This diagnostic therefore tests a genuinely pregame football mechanism that may explain the persistence: **the named player's own target propensity conditional on offensive personnel, combined with the team's strictly-prior personnel mix.**

## Question

> After the promoted WR/TE opportunity model has set a named player's baseline entitlement, does that same player's strictly-prior on-field target propensity by personnel group, evaluated against the team's strictly-prior projected personnel mix, explain his next opportunity residual?

This is not a defense-v-position multiplier and not a positional mean correction.

## Frozen sources

Parent outcome/residual rows:
- WR: exact `wr_player_mechanism_persistence_rows.csv` from run `37634044900`.
- TE: exact `te_player_mechanism_persistence_rows.csv` from run `37631011196`.

Football-state sources:
- nflverse play-by-play;
- nflverse participation;
- nflverse weekly rosters **for identity mapping only**.

No sportsbook source is permitted.

Historical target seasons:
- WR: 2023 and 2024.
- TE: 2024 and 2025.

Participation is historical/postseason-release data for these seasons. A positive result is therefore diagnostic-only until an equivalent pregame/current source is locked (the existing GSIS Lineup Detail / Formation Usage lane is the intended live bridge).

## Identity

1. Canonical parent identity remains `player_clean_key`.
2. Weekly-roster full names are canonicalized with the repository's existing `canonicalize_player_name_safe` semantics.
3. Canonical name/team identity maps to GSIS/player ID for participation.
4. Ambiguous or missing identity fails that row closed; no fuzzy matching.
5. Report parent-row identity coverage by position and season.

## Play universe

Regular-season offensive dropbacks only.

For each play:
- offense team is canonicalized;
- participation `offense_players` and `offense_positions` define who was on field;
- target is `receiver_player_id`;
- personnel is derived from `offense_personnel`.

Frozen personnel buckets:
- `11`;
- `12`;
- `OTHER` (all remaining valid personnel groupings).

No target-game plays may enter a target row's feature construction.

## Strictly-prior player state

For target player-game `(season, week, team, player)`:

Player history:
- same season;
- same team;
- only games with week < target week;
- most recent 8 games maximum;
- minimum 3 prior games;
- minimum 40 prior on-field dropbacks.

For each personnel bucket compute:
- on-field dropbacks;
- targets;
- target rate = targets / dropbacks.

Frozen sparse-bucket rule:
- if a personnel bucket has <10 prior on-field dropbacks, use the player's unconditional strictly-prior target rate for that bucket;
- no pooling to position peers.

## Strictly-prior team personnel state

For the same target team:
- same season;
- only weeks < target week;
- latest 3 games maximum;
- minimum 2 prior games.

Compute the team's dropback personnel mix over `11 / 12 / OTHER`.

This is the target-game personnel-mix proxy. The actual target-game personnel mix is never used as a feature.

## Frozen player-specific state

`unconditional_rate = prior player targets / prior player on-field dropbacks`

`conditional_rate = sum(team_prior_mix[bucket] * player_prior_target_rate[bucket])`

`personnel_lift = conditional_rate - unconditional_rate`

Interpretation:
- positive `personnel_lift`: the team's expected personnel mix moves toward contexts where THIS player historically earns targets more often;
- negative `personnel_lift`: expected mix moves away from those contexts.

The parent opportunity error is defined as predicted minus actual opportunity in yards-equivalent units. For scoring define:

`realized_opportunity_gain = -current_opportunity_error`

Therefore the expected sign is:

`personnel_lift ↑ -> realized_opportunity_gain ↑`.

No coefficient is fit and no projection is changed.

## Frozen scoring

For WR and TE separately:

1. Spearman(`personnel_lift`, `realized_opportunity_gain`) by target season.
2. Pooled Spearman.
3. 5,000-replicate player-cluster bootstrap of pooled Spearman, seed `20261014`.
4. Quartile diagnostic: mean realized opportunity gain in Q4 `personnel_lift` minus Q1.
5. Report identity/support coverage and player counts.

Position aggregation is a reporting guardrail only. WR and TE are separate scientific cells.

## Frozen support gates

WR:
- each target season >=400 eligible rows and >=75 distinct WRs;
- pooled >=1,000 rows and >=120 WRs.

TE:
- each target season >=200 eligible rows and >=35 distinct TEs;
- pooled >=500 rows and >=60 TEs.

All cells:
- zero same/future-play leakage;
- zero sportsbook inputs;
- zero 2026 outcomes;
- identity coverage among parent rows >=60% in each target season.

## Frozen scientific gates

A position cell confirms only if ALL hold:

1. support passes;
2. expected-sign Spearman >0 in each target season;
3. pooled Spearman >0;
4. player-cluster bootstrap 95% CI lower bound >0;
5. pooled Q4-Q1 realized-opportunity-gain gap >0;
6. integrity passes.

No pooled WR+TE result may rescue a failed position cell.

## Dispositions

- integrity/source failure: `PLAYER_PERSONNEL_CONDITIONED_OPPORTUNITY_INTEGRITY_FAILURE`
- both support-limited: `PLAYER_PERSONNEL_CONDITIONED_OPPORTUNITY_SOURCE_LIMITED`
- WR only confirms: `WR_PERSONNEL_CONDITIONED_OPPORTUNITY_CONFIRMED`
- TE only confirms: `TE_PERSONNEL_CONDITIONED_OPPORTUNITY_CONFIRMED`
- both confirm: `WR_TE_PERSONNEL_CONDITIONED_OPPORTUNITY_CONFIRMED`
- otherwise: `NO_ACTIONABLE_PLAYER_PERSONNEL_CONDITIONED_OPPORTUNITY`

## Anti-rescue

After results are visible do not:
- change personnel buckets;
- use target-game personnel mix;
- lower history/dropback/bucket support;
- pool players to build a position response coefficient;
- feed prior residual error directly into the projection;
- add role carveouts;
- change the latest-8 / latest-3 windows;
- fit a correction coefficient;
- use sportsbook information;
- use 2026 outcomes;
- call historical postseason participation data live-capable.

A positive diagnostic authorizes only a separately frozen room-mass-conserving integration candidate and a current-source/GSIS prospective bridge.

Production changed: **false**
OddsAPI calls: **0**
