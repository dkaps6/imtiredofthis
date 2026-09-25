# Receiver Room Targetable-Rate V1 — Temporal Room Calibration Plan

Date: 2026-09-25

Status: **FROZEN BEFORE 2022-2023 ROOM OUTCOME SCORING**

Parent evidence:
- RECEIVER_TARGETABLE_DROPBACK_V1_TEAM_CALIBRATION_SUPPORTED
- RECEIVER_TARGETABLE_DROPBACK_V1_2024_2025_CONFIRMED
- RECEIVER_TARGETABLE_DROPBACK_V1_FULL_STACK_FAILED_CLOSED
- CURRENT_STACK_RECEIVER_COMPENSATION_AUDIT_V1_COMPLETE

This is not a rescue of the failed player-level targetable-dropback candidate.
It is a distinct room-opportunity hypothesis motivated by the compensation audit.

## 1. Diagnostic basis

The current-stack compensation audit showed:

- team target volume improves under targetable-dropback opportunity;
- TE room target MAE improves;
- RB/FB room target MAE improves;
- WR room target MAE worsens;
- WR1 and high-entitlement targets worsen;
- WR2+ targets improve.

The old baseline therefore contains a compensation structure:
total team target volume is too high while the WR room / top-end receiver
allocation is too low.

Uniformly shrinking every player's target probability cannot solve both.

## 2. Anti-retest boundary

C1 group target-mass calibration remains closed.

C1:
- preserved modeled receiver mass;
- estimated WR/TE/RB_FB composition with an 8-game team window;
- used a fixed 105-target league pseudo-count;
- redistributed mass among rooms;
- failed player-level protection.

This V1 does none of those things.

## 3. New football hypothesis

Forecast each receiving room's opportunity directly in **targets per team
dropback**, rather than:

1. forecasting a team target pool; then
2. multiplying it by the current modeled room composition.

For room g in {WR, TE, RB_FB}:

R_g = sum(strict-prior room targets) / sum(strict-prior team dropbacks)

Candidate room targets:

C_g = projected team dropbacks * R_g

The room rates are not normalized against each other. Their sum is the implied
modeled targetable-dropback rate across the three rooms.

## 4. Frozen strict-prior history

First temporal screen:
- 2022 regular season;
- 2023 regular season.

For target season S, week W, team T:

History contains:
- all regular-season games from S-1 for T;
- only completed regular-season games from S with week < W.

Room target counts use canonical completed player logs and target-game position
is never used upstream.

Team dropbacks use the same canonical historical team-dropback authority as the
already-qualified Receiver Targetable-Dropback V1 team study.

For each room:
- numerator = cumulative eligible room targets;
- denominator = cumulative eligible team dropbacks.

If a team has zero prior dropbacks:
- use the corresponding league room-target/dropback rate from the same eligible
  history;
- count the fallback.

No shrinkage, recency, minimum-games threshold, clipping search, or fitted model.

Sanity:
- each room rate must be finite and in [0,1];
- sum of WR+TE+RB_FB room rates must be <=1 for every row.

## 5. Baseline room forecast

Baseline must reflect the current historical receiver stack on the same
team-game:
- current strict-prior football context;
- M38 WR hierarchy;
- fold-safe TE-R5P where authorized;
- fold-safe WR-R15 where authorized;
- current explicit target entitlement.

Baseline room targets are the sum of current player expected targets by room.

No sportsbook input.

## 6. Candidate room forecast

Candidate changes only the **room target total** in this calibration phase:

C_WR = projected_dropbacks * R_WR
C_TE = projected_dropbacks * R_TE
C_RB_FB = projected_dropbacks * R_RB_FB

No player within-room allocation is performed yet.

No receptions, receiving yards, QB, rushing, RB combo, or ATD are scored in
this phase.

## 7. Actual outcome

After forecasts are frozen:

actual room targets = sum target-game player targets for that room

Rooms:
- WR includes WR/LWR/RWR/SWR
- TE includes TE
- RB_FB includes RB/HB/TB/FB

## 8. Required outputs

Per team-game/room:
- season/week/team/opponent
- projected dropbacks
- baseline room targets
- strict-prior room history games
- strict-prior room targets
- strict-prior team dropbacks
- room targetable rate
- source = team_strict_prior or league_fallback
- candidate room targets
- actual room targets
- candidate minus baseline movement

Report:
- 2022
- 2023
- pooled
- each room separately
- macro average across rooms

Metrics:
- MAE
- RMSE
- bias / absolute bias
- correlation
- median AE
- p75 AE
- p90 AE
- candidate closer rate

## 9. Frozen scientific gates

RECEIVER_ROOM_TARGETABLE_RATE_V1_SUPPORTED requires all:

1. pooled macro room-target MAE improves;
2. 2022 macro room-target MAE improves;
3. 2023 macro room-target MAE improves;
4. WR room MAE improves in 2022;
5. WR room MAE improves in 2023;
6. pooled WR room MAE improves;
7. TE pooled room MAE is nonworse;
8. RB_FB pooled room MAE is nonworse;
9. no room pooled p90 worsens by >0.50 targets;
10. pooled macro p90 is nonworse;
11. pooled macro absolute bias is nonworse;
12. pooled candidate closer rate >50%;
13. team strict-prior source rate >=99%;
14. league fallback rate <=1%;
15. all room rates finite/in [0,1];
16. summed room rate <=1 on every row;
17. target-game outcomes used upstream = 0;
18. sportsbook inputs = 0;
19. parameters fit = 0;
20. candidate variants scored = 1.

Any failure:
RECEIVER_ROOM_TARGETABLE_RATE_V1_FAILED_CLOSED

## 10. If supported

Freeze an unchanged 2024-2025 confirmation before any player-level integration.

Only after independent room-level confirmation may a player-level candidate be
specified.

A later player candidate must preserve within-room specialists and allocate each
qualified room pool by the existing within-room proportions unless a separately
frozen diagnostic demonstrates that within-room allocation itself needs a new
mechanism.

## 11. No-rescue rule

After results are visible, do not:
- change history window;
- add recency;
- add shrinkage;
- add league pseudo-count;
- normalize room rates;
- cap/floor a room;
- exempt WR1/Q4;
- use player names/depth labels;
- reuse C1;
- tune position-specific multipliers;
- alter catch rate/YPT;
- alter QB/rushing;
- use sportsbook inputs.

A failure closes this exact formulation.
