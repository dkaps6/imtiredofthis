# Receiver Targetable-Dropback V1 — Temporal Team Calibration Plan

Date: 2026-09-25

Status: **FROZEN BEFORE 2022-2023 OUTCOME SCORING**

Parent evidence:
- `OPPORTUNITY_PARTITION_SEMANTICS_V1_CONFIRMED`
- `RECEIVER_OFFICIAL_ATTEMPT_POOL_V1_TEAM_CALIBRATION_FAILED_CLOSED`

The official-attempt candidate failed its full gate but improved realized team
targets in both 2024 and 2025. This motivates a distinct receiver semantic:
**targetable dropbacks**.

This candidate is not a rescue of the failed official-attempt pool.

## 1. Football hypothesis

A receiver target pool should not equal:
- all QB dropbacks; or
- all official pass attempts.

It should approximate the subset of dropbacks that generate a credited player
target.

Under canonical weekly-stat semantics, team receiver targets are the sum of
player targets.

Define:

`targetable_dropback_rate = team_targets / team_dropbacks`

This naturally excludes:
- sacks;
- QB scrambles;
- official pass attempts that do not generate a credited target.

## 2. Independent temporal screen

The exact candidate is frozen now and first tested on:

- 2022 regular season, using 2021 + completed 2022 games as strict-prior history;
- 2023 regular season, using 2022 + completed 2023 games as strict-prior history.

2024-2025 are not consulted in candidate construction beyond the already-known
discovery that official-attempt conversion improved team-target calibration.

No candidate choice may be changed after 2022-2023 results are visible.

## 3. Strict-prior conversion

For target season S, target week W, team T:

History contains:
- all regular-season games from S-1 for T;
- only completed regular-season games from S with week < W.

For each historical team-game:
- team targets = sum canonical player targets;
- team dropbacks = canonical team PBP dropbacks.

Team conversion:

`R_T = sum(history team targets) / sum(history team dropbacks)`

League fallback, used only if T has zero valid prior dropbacks:

`R_L = sum(all eligible history targets) / sum(all eligible history dropbacks)`

No shrinkage coefficient, recency weight, threshold, clipping search or fitted
model is allowed.

The only fixed sanity bound is:
`0 <= R <= 1`.

## 4. Forecasts

Baseline current receiver opportunity anchor:

`B = projected_plays * 0.57`

Candidate targetable pool:

`C = B * R_T`

No player target shares or player projections are changed in this phase.

## 5. Actual outcome

After B and C are frozen:

`actual_team_targets = sum(targets across all team players in target game)`

Also retain actual target-game dropbacks strictly for semantic diagnostics; they
must never enter B, C or R_T.

## 6. Required output

Each team-game must include:
- season/week/team/opponent;
- projected plays;
- baseline projected dropbacks;
- prior history games;
- prior history dropbacks;
- prior history targets;
- targetable-dropback rate;
- source = team_strict_prior or league_fallback;
- candidate targetable pool;
- actual team targets;
- actual team dropbacks.

Report 2022, 2023 and pooled.

## 7. Frozen metrics

Baseline and candidate vs actual team targets:
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

Diagnostic:
- baseline projected dropbacks vs actual dropbacks;
- distribution of targetable-dropback conversion;
- conversion stability by season;
- fallback count.

## 8. Frozen scientific gates

`RECEIVER_TARGETABLE_DROPBACK_V1_TEAM_CALIBRATION_SUPPORTED` requires all:

1. candidate target MAE improves in 2022;
2. candidate target MAE improves in 2023;
3. pooled target MAE improves;
4. pooled p90 AE is nonworse;
5. pooled absolute bias improves;
6. candidate closer rate >50%;
7. no season candidate p90 worsens by >0.50 opportunities;
8. team-level conversion source exists for >=99% of changed rows;
9. league fallback rate <=1%;
10. all conversion values are finite and in [0,1];
11. target-game outcomes used upstream = 0;
12. sportsbook inputs = 0;
13. parameters fit = 0;
14. candidate variants scored = 1.

Any failure:
`RECEIVER_TARGETABLE_DROPBACK_V1_TEAM_CALIBRATION_FAILED_CLOSED`

No rescue.

## 9. If supported

Freeze a separate 2024-2025 player-level/full-stack candidate before scoring.

That candidate must initially:
- keep fixed 57% team dropback partition unchanged;
- keep QB M89/M90 means unchanged;
- keep C2 unchanged;
- keep WR-R15 / TE-R5P unchanged;
- keep player target entitlement shares unchanged;
- allocate receiver targets from the targetable pool rather than all dropbacks;
- leave rushing unchanged;
- leave catch rate/YPT unchanged;
- leave residual target-share semantics unchanged;
- use no sportsbook input.

Then 2026 prospective capture is still required before production promotion.

## 10. No-rescue rule

Do not alter after result:
- history window;
- recency weights;
- prior-season/current-season weighting;
- shrinkage;
- minimum games;
- team/QB/position carveouts;
- target-rate clipping beyond [0,1];
- throwaway-specific thresholds;
- hierarchical reconciliation;
- dynamic pass-share logic;
- rushing/scramble logic.
