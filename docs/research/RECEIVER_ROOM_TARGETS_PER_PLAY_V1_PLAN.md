# Receiver Room Targets-Per-Play V1 — Temporal Calibration Plan

Date: 2026-09-25

Status: **FROZEN BEFORE OUTCOME SCORING**

Parent evidence:
- RECEIVER_TARGETABLE_DROPBACK_V1 team signal replicated in 2022-2025;
- RECEIVER_TARGETABLE_DROPBACK_V1_FULL_STACK_FAILED_CLOSED;
- CURRENT_STACK_RECEIVER_COMPENSATION_AUDIT_V1_COMPLETE;
- RECEIVER_ROOM_TARGETABLE_RATE_V1_FAILED_CLOSED.

## 1. Why this is a new hypothesis

Room Targetable-Rate V1 improved WR, TE and RB_FB room MAE and pooled p90 in 2022 and 2023, but failed the absolute-bias gate.

Post-disposition attribution showed the negative room bias was inherited almost exactly from the upstream fixed-57% projected-dropback anchor:
- 2022 projected-dropback signed bias vs actual dropbacks about -2.85/game;
- 2023 about -3.40/game.

This study asks whether receiver opportunity should be modeled directly per offensive play, without passing through that fixed pass/dropback partition.

## 2. Anti-retest boundary

Migrations 18/20/21 remain closed. They changed or compared team pass-tendency/pass-rate formulations for the broader football simulation.

This candidate does not change team pass rate, QB opportunity, rushing opportunity, or the canonical 57% production state. Offensive plays are used only as the denominator for a receiver-only opportunity forecast.

## 3. Frozen candidate

For room g in {WR, TE, RB_FB}:

R_g_play = sum(strict-prior room targets) / sum(strict-prior offensive plays)

candidate room targets = projected offensive plays * R_g_play

Strict-prior history contains all regular-season games from S-1 for the team plus completed regular-season games from S with week < W.

Projected offensive plays are the exact historical mc_projected_plays authority. No fixed 57% factor enters the candidate.

## 4. Fallback and sanity

Only if a team has zero eligible prior plays, use the corresponding league room-targets/plays rate from the same eligible history.

No shrinkage, pseudo-count, recency, minimum-games threshold, clipping search or fitted coefficient.

Each room rate must be finite and in [0,1], and the sum of WR+TE+RB_FB room rates must be <=1.

## 5. First temporal screen

2022 and 2023 regular season.

Because TE-R5P/WR-R15 do not have authorized fold-safe backcasts to 2022-2023, baseline uses the legitimate pre-specialist/M38 entitlement stack.

Baseline room targets remain: projected plays * 0.57 * baseline room entitlement.

## 6. Actual outcome

After prediction is frozen, actual room targets are the sum of target-game player targets for each room:
- WR = WR/LWR/RWR/SWR
- TE = TE
- RB_FB = RB/HB/TB/FB

## 7. Frozen metrics

Report 2022, 2023 and pooled for each room and macro:
- MAE, RMSE, bias, absolute bias, correlation;
- median, p75 and p90 absolute error;
- candidate closer rate.

Also report summed-room MAE, RMSE, bias/absolute bias and p90.

## 8. Frozen scientific gates

RECEIVER_ROOM_TARGETS_PER_PLAY_V1_SUPPORTED requires all:
1. pooled macro room MAE improves;
2. 2022 macro room MAE improves;
3. 2023 macro room MAE improves;
4. WR MAE improves in 2022;
5. WR MAE improves in 2023;
6. pooled WR MAE improves;
7. pooled TE MAE nonworse;
8. pooled RB_FB MAE nonworse;
9. no room pooled p90 worsens by >0.50 targets;
10. pooled macro p90 nonworse;
11. pooled macro absolute bias improves;
12. summed-room pooled MAE improves;
13. summed-room pooled absolute bias improves;
14. summed-room pooled p90 nonworse;
15. pooled all-room-row candidate closer rate >50%;
16. team source rate >=99%;
17. league fallback rate <=1%;
18. all rates finite/in [0,1];
19. summed room rate <=1;
20. target-game outcomes upstream = 0;
21. sportsbook inputs = 0;
22. parameters fit = 0;
23. variants scored = 1.

Any failure: RECEIVER_ROOM_TARGETS_PER_PLAY_V1_FAILED_CLOSED

## 9. If supported

Freeze an unchanged 2024-2025 confirmation against the authorized specialist order: 2024 TE-R5P + WR-R15; 2025 TE-R5P only, with WR-R15 retrospective application still forbidden.

Only after room-level confirmation may a player-level integration be proposed.

## 10. No-rescue rule

Do not alter history horizon, recency, shrinkage, pseudo-count, room definitions, bias offset, blending with 57%, rate caps/floors, WR1/Q4 exemptions, position multipliers, efficiency, QB/rushing, or sportsbook input after results are visible.

A failure closes this exact formulation.
