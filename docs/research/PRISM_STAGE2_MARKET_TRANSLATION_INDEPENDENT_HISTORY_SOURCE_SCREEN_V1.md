# PRISM Stage-2 -> Market Translation Independent-History Source Screen V1

Date: 2026-10-17

Status: **FROZEN SOURCE / LINEAGE SCREEN ONLY / NO OUTCOME SCORING**

Model:
**PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_STAGE2_MARKET_TRANSLATION_INDEPENDENT_HISTORY_SOURCE_SCREEN_V1`

## 1. Why this source screen exists

The Week-4/5 player-first replay and the Week-5 finite-allocation transmission audit established a concrete architecture gap:

> independently constructed named-player claims can overbook finite team opportunity, so the valid conservation layer can attenuate strong upstream player-role information.

The already-qualified Stage-2 active-role -> conditional-allocation bridge is the current source-qualified player-first architecture for:
- RB rushing;
- RB receiving;
- TE receiving.

Before testing whether coherent Stage-2 opportunity improves market-specific carries/targets/receptions/yards, this screen asks whether a genuinely independent historical season can support that validation without reusing the 2024-development / 2025-confirmation evidence that qualified Stage 2.

Preferred independent target season:
- 2023.

## 2. No-score question

Can 2023 support the exact frozen upstream objects required by the qualified Stage-2 mechanism?

Required:
1. true pregame player universe;
2. stable player identity;
3. exact pregame inactive exclusion semantics consistent with the qualified Stage-1 source contract;
4. strictly-prior participation history sufficient to compute the frozen Stage-1 probability;
5. canonical pregame q_i entitlement for each eligible Stage-2 cell;
6. finite denominator and complement/residual semantics;
7. OOS / non-overlapping specialist authority where required;
8. zero target-game outcome use in source construction.

If any required upstream object is unavailable, the 2023 market-translation validation fails closed before outcome scoring.

## 3. Binding Stage-1 source requirement

Qualified Stage 1 was frozen on a true pregame universe:

`weekly roster ACT + stable identity - exact official inactives`.

Its frozen official-inactive authority is:

`data/backtests/qb_m78_official_inactives_corrected/m78_corrected_candidate_teamweek.csv`

Required SHA-256:

`d39aaf0feea101f3e0d2721ebd4118ef33fb1a4d3c76670e2a4f17734e37b609`.

Known canonical shape:
- 1,088 team-weeks;
- 544 in 2024;
- 544 in 2025.

The source contract explicitly selected 2024/2025 because this exact pregame inactive archive was available for both seasons.

The 2023 screen may not:
- omit the inactive layer;
- infer inactivity from target-game snaps;
- infer inactivity from weekly-stat presence;
- substitute current injury data;
- use sportsbook listing presence;
- weaken the Stage-1 universe definition.

## 4. Stage-2 cells

Screen only:
- RB_RUSHING;
- RB_RECEIVING;
- TE_RECEIVING.

Do not reopen:
- QB exact Stage 2 — CLOSED;
- WR Stage 2 — SOURCE_BLOCKED / NOT SCORED.

## 5. Frozen source gates for 2023

The independent historical season is source-ready only if ALL are true:

1. the exact frozen inactive snapshot SHA matches;
2. 2023 has the same complete regular-season official team-week inactive authority required by Stage 1;
3. 2023 weekly roster authority is available;
4. stable player identity parity can be established;
5. historical offensive-snap source is available with valid team-week coverage;
6. strictly-prior Stage-1 history can be constructed without target-game information;
7. canonical q_i authority is source-valid for the target cell;
8. finite denominator/complement semantics are reconstructable;
9. any specialist entitlement is OOS / non-overlapping for 2023;
10. chronology violations = 0;
11. target-game outcomes used as predictors = 0;
12. sportsbook inputs = 0;
13. OddsAPI calls = 0;
14. models fit = 0;
15. thresholds searched = 0;
16. production changes = 0.

No gate may be lowered after the source screen.

## 6. Early terminal gate

Because the Stage-1 true-pregame universe depends on exact official inactive identity, the source screen stops immediately if the frozen inactive authority has no 2023 coverage.

A failure at this gate is a legitimate source block, not a reason to approximate the player universe.

## 7. If 2023 is blocked

Disposition:

`PRISM_STAGE2_MARKET_TRANSLATION_2023_SOURCE_BLOCKED`.

Then:
- do not score 2023 market outcomes;
- do not substitute 2024/2025 as “independent” evidence;
- do not use target-game snaps to recreate the active set;
- move to a separately frozen prospective mechanics path for future weeks.

## 8. If 2023 is source-ready

Do not score immediately.

Freeze a separate market-specific mechanism test before attaching target outcomes.

Stage-2 mechanics, q_i, active-role probabilities, finite allocation and existing efficiency authorities must remain unchanged.

## 9. Integrity at freeze

- 2023 target outcomes scored: **0**;
- new model fits: **0**;
- threshold searches: **0**;
- sportsbook inputs: **0**;
- OddsAPI calls: **0**;
- Week-6 outcomes read: **0**;
- Week-6 locks changed: **false**;
- production changed: **false**.

Disposition at freeze:
`PRISM_STAGE2_MARKET_TRANSLATION_INDEPENDENT_HISTORY_SOURCE_SCREEN_V1_FROZEN`.
