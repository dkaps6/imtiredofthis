# Receiver Targetable-Dropback V1 — 2024-2025 Confirmation Plan

Date: 2026-09-25

Status: **FROZEN BEFORE 2024-2025 CONFIRMATION SCORING**

Parent temporal qualification:
- `RECEIVER_TARGETABLE_DROPBACK_V1_TEAM_CALIBRATION_SUPPORTED`
- run `36144562838`
- artifact `10868338685`
- result commit `ef4474be0902f771b09fa3f7b40c8471c9653480`

The candidate formula is unchanged.

## Exact candidate

For target season S, week W, team T:

Eligible strict-prior history contains:
- all regular-season games from S-1 for T;
- only completed regular-season games from S with week < W.

Team targetable rate:

`R_T = sum(history team targets) / sum(history team dropbacks)`

League fallback only if team prior dropbacks are zero:

`R_L = sum(all eligible history targets) / sum(all eligible history dropbacks)`

Baseline target pool:

`B = projected_plays * 0.57`

Candidate target pool:

`C = B * R_T`

No shrinkage, recency weighting, threshold, minimum-games rule, clipping search,
team/player/QB carveout or fitted parameter is introduced.

## Confirmation seasons

Evaluate separately:
- 2024 regular season using 2023 + completed 2024 history;
- 2025 regular season using 2024 + completed 2025 history;
- pooled 2024-2025.

This is confirmation of the exact 2022-2023-supported candidate.

## Outcome

Score only against actual team receiver targets.

Actual target-game dropbacks are retained only as a semantic diagnostic and may
not enter the candidate forecast.

## Frozen metrics

Report baseline and candidate:
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
- strict-prior team-source rows;
- league-fallback rows;
- baseline projected dropbacks vs actual dropbacks.

## Frozen confirmation gates

`RECEIVER_TARGETABLE_DROPBACK_V1_2024_2025_CONFIRMED` requires all:

1. candidate target MAE improves in 2024;
2. candidate target MAE improves in 2025;
3. pooled target MAE improves;
4. pooled RMSE is nonworse;
5. pooled p90 AE is nonworse;
6. pooled absolute bias improves;
7. changed-row candidate closer rate > 50%;
8. 2024 p90 does not worsen by more than 0.50 opportunities;
9. 2025 p90 does not worsen by more than 0.50 opportunities;
10. strict-prior team conversion source exists for >=99% of rows;
11. league fallback rate <=1%;
12. all conversion values finite and in [0,1];
13. target-game outcomes used upstream = 0;
14. sportsbook inputs = 0;
15. parameters fit = 0;
16. candidate variants scored = 1.

Any failure:
`RECEIVER_TARGETABLE_DROPBACK_V1_2024_2025_FAILED_CLOSED`

No rescue.

## If confirmed

Do not promote directly.

Freeze a separate player-level/full-stack integration candidate:

- receiver target allocation uses the targetable-dropback pool;
- fixed 57% team dropback partition remains unchanged for this first integration;
- player target entitlement shares remain unchanged;
- M38 unchanged;
- WR-R15 unchanged;
- TE-R5P unchanged;
- catch rates unchanged;
- YPT unchanged;
- QB M89/M90 unchanged;
- QB C2 unchanged;
- rushing unchanged;
- residual target-share semantics unchanged;
- RB rush+receiving re-evaluated through current RB V2;
- zero sportsbook input.

Only after player-level historical qualification would prospective 2026 shadow
capture be considered.

## No-rescue rule

Do not modify the targetable-rate formula after 2024-2025 results are visible.
