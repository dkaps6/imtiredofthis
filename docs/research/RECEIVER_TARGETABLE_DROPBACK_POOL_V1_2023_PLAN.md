# Receiver Targetable-Dropback Pool V1 — 2023 Walk-Forward Plan

Date: 2026-09-25

Status: **FROZEN BEFORE OUTCOME SCORING**

Parent evidence:
- `OPPORTUNITY_PARTITION_SEMANTICS_V1`: receiver targets currently consume a
  dropback pool that includes sacks/scrambles.
- `RECEIVER_OFFICIAL_ATTEMPT_POOL_V1_TEAM_CALIBRATION`: converting dropbacks
  to official attempts failed the full frozen calibration because official-attempt
  forecasting worsened overall, but the same conversion improved **actual team
  receiver target** MAE in both 2024 and 2025.

This candidate is a new football-semantics hypothesis. It is not a rescue of the
official-attempt candidate.

## 1. Football identity

A receiver target can occur only on a **targetable passing play**.

Therefore the receiver opportunity pool should not equal:
- all dropbacks; or
- necessarily all official pass attempts.

Official attempts still include non-targeted plays such as throwaways and spikes.

The directly relevant team quantity is:

`targetable_dropback_rate = actual_team_player_targets / actual_team_dropbacks`

where:
- actual team player targets are the sum of canonical weekly player targets;
- actual team dropbacks come from PBP team opportunity semantics.

This ratio removes:
- sacks;
- QB scrambles;
- official attempts that produce no named player target.

## 2. Why 2023 is tested first

2024-2025 have already been repeatedly exposed elsewhere in the project.

To avoid immediately grading a new hypothesis on the same exposed seasons, the
first evaluation is:

- target season: **2023 regular season**
- prior season: **2022**
- Weeks 1-18 as defined by the canonical historical schedule

No 2024 or 2025 target outcomes may be used in this first run.

If this exact frozen candidate fails 2023, it closes before any 2024-2025
confirmation.

If it passes 2023, the same frozen formula may then be evaluated on 2024-2025
under a separately recorded confirmation step with no rule changes.

## 3. Historical observation construction

For each completed historical team-game:

### Actual team targets
Sum canonical weekly player `targets` across all players on the team.

### Actual team dropbacks
Use the canonical historical PBP team table:
`actual_dropbacks = plays_est * dropback_rate`

The historical builder already defines `dropback_rate` from `qb_dropback`.

### Per-game targetable rate
`game_targetable_rate = actual_team_targets / actual_team_dropbacks`

Hard integrity:
- denominator > 0;
- rate finite;
- rate in [0, 1];
- target-game rate is never used in that same game's forecast.

## 4. Strict-prior forecast authority

For each 2023 target week/team:

1. collect only completed games strictly before the target game;
2. if the team has earlier 2023 games, use the arithmetic mean of its earlier
   2023 `game_targetable_rate`;
3. otherwise use the arithmetic mean of its 2022 regular-season
   `game_targetable_rate`;
4. if neither exists, remain baseline/no-change and report fallback.

This exactly mirrors the current historical team-context source priority:
current-season history when available, otherwise prior-season history.

No fitted decay, shrinkage constant, window length, threshold or league
regression is introduced.

## 5. Forecasts

Baseline:
`B = projected_plays * projected_dropback_rate`

Candidate:
`C = B * strict_prior_targetable_dropback_rate`

The projected plays/dropback rate must come from the existing leakage-safe
historical MC context.

No player share, catch rate, YPT, WR-R15, TE-R5P, QB, rushing or sportsbook
input is changed in this calibration.

## 6. Outcome

Score only against:
`actual_team_targets`

Official pass attempts are not a scientific gate in this experiment because
this candidate is explicitly a target-opportunity model, not a QB-attempt model.

Actual dropbacks may be reported only as a semantic diagnostic.

## 7. Frozen metrics

Report:
- n;
- MAE;
- RMSE;
- bias;
- absolute bias;
- correlation;
- median AE;
- p75 AE;
- p90 AE;
- 5+ miss rate;
- 8+ miss rate;
- 10+ miss rate;
- changed-row candidate closer rate.

Also report:
- targetable-rate min/median/max;
- Week-1 prior-season rows;
- current-season-history rows;
- fallback/no-change rows;
- strict-prior provenance count.

## 8. Frozen 2023 qualification gates

`RECEIVER_TARGETABLE_DROPBACK_POOL_V1_2023_SUPPORTED` requires all:

1. candidate team-target MAE strictly improves;
2. candidate RMSE is nonworse;
3. candidate p90 AE is nonworse;
4. candidate absolute bias strictly improves;
5. changed-row candidate closer rate > 50%;
6. 10+ target miss rate is nonworse;
7. every changed row has explicit strict-prior rate provenance;
8. target-game outcomes used upstream = 0;
9. sportsbook inputs = 0;
10. candidate variants scored = 1;
11. parameters fit = 0.

Any failure:
`RECEIVER_TARGETABLE_DROPBACK_POOL_V1_2023_FAILED_CLOSED`

No rescue tuning.

## 9. If supported

Do not promote to production.

Next:
1. freeze a 2024-2025 confirmation contract with the **identical formula**;
2. score 2024 and 2025 separately and pooled;
3. only if confirmation survives, freeze a player-level full-stack receiver test.

The later player-level test would change only the receiver MC opportunity pool:
- target shares unchanged;
- M38 unchanged;
- WR-R15 unchanged;
- TE-R5P unchanged;
- catch rates unchanged;
- YPT unchanged;
- QB unchanged;
- rushing unchanged;
- dependent RB rush+receiving re-evaluated through current RB V2.

## 10. No-rescue rule

After 2023 results are visible, do not try:
- alternate windows;
- weighted recent games;
- player/position carveouts;
- team carveouts;
- QB carveouts;
- pressure/scramble thresholds;
- caps/floors chosen from outcomes;
- league shrinkage chosen from outcomes;
- combined hierarchical reconciliation;
- combined rushing repair;
- sportsbook-conditioned routing.

Any such idea must be a separately frozen hypothesis.
