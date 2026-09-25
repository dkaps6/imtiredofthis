# Receiver Room Targets-Per-Play V1 — 2024-2025 Confirmation Plan

Date: 2026-09-25

Status: **FROZEN BEFORE 2024-2025 OUTCOME SCORING**

Parent qualification:
- RECEIVER_ROOM_TARGETS_PER_PLAY_V1_SUPPORTED
- 2022-2023 temporal screen run 36172644864
- result commit 602e0c86eaac9e38a2ec65e3dc68e46203dac3e3

## 1. Candidate is unchanged

For room g in {WR, TE, RB_FB}:

R_g_play = sum(strict-prior room targets) / sum(strict-prior offensive plays)

candidate room targets = projected offensive plays * R_g_play

History for target season S, week W, team T:
- all regular-season games from S-1 for T;
- completed regular-season games from S with week < W.

Only if T has zero eligible prior plays, use the corresponding league rate from
the same eligible history.

No shrinkage, recency, pseudo-count, minimum-games threshold, clipping search,
bias offset, fitted coefficient, or candidate variant.

## 2. Confirmation seasons

- 2024 regular season
- 2025 regular season

The 2022-2023 formula and room definitions are unchanged.

## 3. Baseline must match authorized current historical receiving stack

For each target week:
- build current historical football context;
- materialize explicit M38 target entitlement;
- apply fold-safe TE-R5P;
- 2024 only: apply fold-safe WR-R15;
- 2025: WR-R15 retrospective application remains forbidden.

Baseline room targets:

baseline room targets = projected plays * 0.57 * final authorized room entitlement

Specialist conservation must be exact:
- TE room mass unchanged by TE-R5P;
- non-TE unchanged by TE-R5P;
- 2024 WR1 anchor unchanged;
- 2024 WR2+ pool unchanged;
- 2024 WR room mass unchanged;
- non-WR unchanged;
- same/future participation violations = 0;
- 2025 WR-R15 applications = 0.

## 4. Candidate remains room-level only

Candidate room totals ignore the baseline room composition and use only the
strict-prior room targets-per-play rate.

No player within-room allocation is changed or scored in this phase.

No receptions, receiving yards, QB, rushing, ATD, or RB combo outputs are
changed.

## 5. Outcome

After prediction is frozen:

actual room targets = target-game canonical player targets summed by:
- WR = WR/LWR/RWR/SWR
- TE = TE
- RB_FB = RB/HB/TB/FB

## 6. Required metrics

For each room, each season and pooled:
- MAE
- RMSE
- signed bias / absolute bias
- correlation
- median AE
- p75 AE
- p90 AE
- candidate closer rate

Also report macro averages across rooms and summed-room:
- MAE
- RMSE
- bias / absolute bias
- p90
- closer rate

## 7. Frozen confirmation gates

RECEIVER_ROOM_TARGETS_PER_PLAY_V1_2024_2025_CONFIRMED requires all:

1. pooled macro room MAE improves;
2. 2024 macro room MAE improves;
3. 2025 macro room MAE improves;
4. WR room MAE improves in 2024;
5. WR room MAE improves in 2025;
6. pooled WR room MAE improves;
7. pooled TE room MAE nonworse;
8. pooled RB_FB room MAE nonworse;
9. no pooled room p90 worsens by >0.50 targets;
10. pooled macro p90 nonworse;
11. pooled macro absolute bias improves;
12. summed-room pooled MAE improves;
13. summed-room pooled absolute bias improves;
14. summed-room pooled p90 nonworse;
15. pooled all-room-row candidate closer rate >50%;
16. team strict-prior source rate >=99%;
17. league fallback rate <=1%;
18. all room rates finite and in [0,1];
19. summed room rate <=1;
20. TE specialist conservation exact;
21. authorized WR specialist conservation exact;
22. WR same/future participation violations = 0;
23. WR-R15 2025 applications = 0;
24. target-game outcomes upstream = 0;
25. sportsbook inputs = 0;
26. parameters fit = 0;
27. candidate variants scored = 1.

Any failure:
RECEIVER_ROOM_TARGETS_PER_PLAY_V1_2024_2025_FAILED_CLOSED

## 8. If confirmed

Only then freeze a player-level/full-stack integration.

The first player-level candidate must:
- preserve each room total from the confirmed room targets-per-play model;
- allocate within each room using the already-authorized current within-room
  entitlement proportions;
- preserve TE-R5P/WR-R15 ordering and conservation;
- leave catch rate and YPT unchanged;
- leave QB/rushing/ATD unchanged;
- evaluate receptions, receiving yards and RB rush+receiving tails;
- use zero sportsbook inputs.

Production still requires prospective 2026 confirmation.

## 9. No rescue

If confirmation fails, do not:
- alter the formula;
- change history windows;
- add recency/shrinkage;
- blend with fixed 57%;
- add position multipliers;
- exempt WR1/Q4;
- change specialists;
- change efficiency;
- use sportsbook information.

A failure closes this exact formulation.
