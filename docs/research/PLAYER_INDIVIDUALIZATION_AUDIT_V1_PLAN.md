# Player Individualization Audit V1 — Frozen Diagnostic Contract

**STATUS: FROZEN BEFORE AUDIT RESULT. DIAGNOSTIC ONLY. NO PRODUCTION CHANGE.**

## Purpose

Determine whether the current NFL projection stack is sufficiently player-centric at the point where individual historical evidence enters the model, while explicitly preserving all validated production science.

This audit is **not** a replacement-model proposal.

It asks:

> How much of each active player's pregame football baseline comes from that player's own strictly-prior evidence versus position-level population pooling, and where downstream shared role/team/opponent rules cause distinct players to be processed through the same transformations?

The intended use is complementary:

`existing validated stack + stronger player-specific state, if justified`

not:

`discard existing stack and start over`.

## Existing authorities that remain protected

Do not modify or invalidate:
- Player Identity v3;
- PlayerForm v2 leakage controls;
- Bayesian v2 unless a later separately frozen candidate earns a change;
- current RB / WR / TE / QB production authorities;
- RB Rush+Receiving Conservation V2;
- Discrete Count Mean Alignment V1;
- QB-C2 distribution;
- all existing frozen anti-retest results;
- current simulation / full-slate orchestration;
- matchup research closures already recorded in Issue #535.

This audit may describe architectural limitations but authorizes **zero** production mutations.

## Exact questions

### Q1 — Is the stack player-level?
Verify whether:
- stable player identity is the historical grouping key;
- each player's prior-season and current-season evidence is carried separately;
- final sportsbook rows attach player-specific posterior metrics.

### Q2 — How strong is positional shrinkage?
For every eligible historical player-week and every Bayesian metric:
- target share
- rush share
- route rate
- receptions per target
- YPRR
- YPT
- YPC
- YPA

compute the exact empirical-Bayes effective weights implied by current production constants:

`group_weight = GROUP_STRENGTH[metric]`

`prior_player_weight = min(prior_games, PRIOR_PLAYER_CAP[metric])`

`current_player_weight = current_games`

normalized by the sum of available weights.

Report:
- position-level population share;
- prior-player share;
- current-player share;
- total player-specific share = prior + current;
- by season, week, position, metric;
- distribution across player-weeks.

### Q3 — When does the model become mostly player-specific?
For each metric determine the earliest current-season game count at which:
- player-specific evidence exceeds 50%;
- current-season evidence alone exceeds group prior;
- group prior falls below 25%;
- group prior falls below 10%.

This is a mechanical implication of current constants, not a tuned recommendation.

### Q4 — Where does shared processing re-enter?
Trace downstream production code and classify major inputs as:
- PLAYER_SPECIFIC
- ROLE_BUCKET
- POSITION_BUCKET
- TEAM_SPECIFIC
- OPPONENT_TEAM_SPECIFIC
- GENERIC_CONSTANT
- PLAYER_X_ENVIRONMENT_INTERACTION

At minimum trace:
- target share;
- rush share;
- catch rate;
- YPT;
- YPC;
- YPA;
- team plays / pass attempts / rush attempts;
- matchup multipliers;
- injury redistribution;
- volatility/distribution controls.

### Q5 — Does the current stack contain genuine player-by-environment interaction?
A true player-by-environment interaction means the response to the same opponent/team condition can differ because of the player's own historical profile, beyond merely starting from a different baseline.

Do not count as true interaction:
- multiplying every TE by the same TE matchup multiplier;
- multiplying every RB by the same team rush-efficiency multiplier;
- generic position defaults;
- role labels alone.

Count only if the transformation explicitly depends jointly on player-specific evidence and environment.

### Q6 — Are evaluation summaries hiding player heterogeneity?
Describe current evaluation grain:
- player-game rows are scored individually;
- promotion metrics are commonly aggregated by market/position/cohort.

Quantify, using preserved historical projection traces where available:
- number of distinct players per position-market;
- distribution of per-player signed error and MAE for players with >=8 scored games;
- dispersion of per-player error around aggregate positional MAE;
- fraction of players whose error direction differs from the position-level bias direction.

This is diagnostic only. Do not use the result to tune a candidate.

## Historical audit universe

Primary quantitative universe:
- 2025 regular season Weeks 2-18;
- preserved canonical historical player logs from the existing Football Matchup candidate input authority;
- strictly-prior information only for each target week.

Optional cross-check:
- 2024 regular season Weeks 2-18 if already available from the same frozen artifact.

No 2026 outcome is required or authorized.

## Expected dispositions

`PLAYER_INDIVIDUALIZATION_SUFFICIENT`
if:
- player-specific evidence dominates the baseline for established players early enough;
- downstream environment effects materially interact with player-specific state;
- aggregate evaluation is not hiding substantial stable per-player heterogeneity.

`PLAYER_INDIVIDUALIZATION_PARTIAL`
if:
- players have individual histories and projections,
- but meaningful position/role pooling remains and/or environment effects are mostly generic transformations.

`PLAYER_INDIVIDUALIZATION_WEAK`
if:
- most established-player baselines remain dominated by positional priors for much of the season,
- or downstream processing substantially collapses distinct players into shared position/role behavior.

This audit does not itself select a replacement architecture.

## If PARTIAL or WEAK

The only authorized next step is to freeze a **complementary player-state candidate plan** that:
1. keeps the current model as baseline;
2. adds or refines player-specific state upstream or alongside current components;
3. uses strictly-prior player evidence;
4. preserves team/opponent/usage/injury/simulation science;
5. proves incremental value out of sample;
6. does not use sportsbook lines upstream;
7. does not delete validated authorities.

## Explicit prohibitions

- No production edits.
- No changing Bayesian strengths/caps during this audit.
- No replacing the existing model.
- No deleting current matchup/team/usage science.
- No sportsbook input.
- No 2026 target-game outcomes.
- No tuning from audit findings.
- No player-specific candidate fitting yet.

Production mutations authorized: **0**  
Candidate models authorized: **0**  
Sportsbook inputs authorized: **0**  
2026 outcomes authorized: **0**
