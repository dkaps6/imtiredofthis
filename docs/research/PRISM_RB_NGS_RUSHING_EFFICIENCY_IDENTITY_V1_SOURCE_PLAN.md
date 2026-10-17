# PRISM RB NGS Rushing Efficiency Identity V1 — Source Plan

Date: 2026-10-17

Status: **FROZEN SOURCE / CHRONOLOGY AUDIT BEFORE ANY PREDICTIVE SCORING**

Model:
**PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_RB_NGS_RUSHING_EFFICIENCY_IDENTITY_V1_SOURCE_AUDIT`

## Player-first question

PRISM already separates rushing opportunity quantity from rushing efficiency.

This source audit asks:

> Can PRISM reconstruct a named RB's strictly-prior Next Gen Stats rushing
> process state with stable identity and enough historical support to test
> whether that player's rushing-efficiency identity adds information beyond
> ordinary box-score YPC?

This is the **INDIVIDUAL EFFICIENCY** layer.

It does not alter carries, team rushing opportunity or teammate allocation.

## Why this is genuinely distinct

Protected/closed work remains protected:
- RB carry/snap allocation;
- RB Vacancy;
- raw current rush-share replacement;
- Stage-1 active-role participation;
- qualified Stage-2 RB rushing allocation;
- generic M95A/M95B role × defense;
- RB run-location × opponent run-location matchup;
- `PLAYER_EFFICIENCY_TAIL_STABILITY_NULL`;
- generic player-residual feedback;
- sportsbook-derived state.

The closed efficiency-tail V1 used box-score efficiency volatility / prior means.
This V1 audits materially different football process information from NFL Next
Gen Stats.

Candidate source fields are player-specific rushing process measures such as:
- rush yards over expected per attempt;
- rush yards over expected;
- rush percent over expected;
- percent of attempts against 8+ defenders in the box;
- average time to line of scrimmage;
- NGS rushing efficiency;
- NGS attempts / yards only as source identity and weighting context.

No opponent multiplier is introduced in the source audit.

## Source authority

Use nflverse / nflreadpy:

`load_nextgen_stats(stat_type="rushing")`

Historical source seasons:
- 2023
- 2024
- 2025

No 2026 NGS rows may be loaded or inspected in this source audit.

Weekly player stats / roster identity may be used only for:
- GSIS identity validation;
- team / position validation;
- source coverage denominators.

## Required source semantics

Required identity/time fields, allowing only documented column aliases:
- season;
- week;
- regular-season marker;
- stable player GSIS ID;
- player display name;
- team;
- player position.

Required NGS process field families:
1. RYOE per attempt;
2. at least one of total RYOE / rush percent over expected;
3. 8+ defenders-in-box attempt percentage;
4. average time to LOS;
5. NGS rushing efficiency.

No target-game row may enter its own predictor history.

## Historical target frame

For source-readiness only, construct target RB weeks in 2024 and 2025.

For each target player-week:
- attach only NGS rows from the same stable player with week < target week;
- same-season history first;
- minimum 3 strictly-prior NGS rushing games for a history-ready row;
- no target-game NGS metrics used as source features.

2023 may provide source coverage / identity evidence and may later serve as
history for early 2024 only if a separately frozen predictive protocol allows
cross-season history. This source audit does not score a predictor.

## Frozen source gates

Adjudicate 2023, 2024 and 2025 source coverage separately.

Every season must pass:

1. regular-season weeks present >=17;
2. stable nonblank GSIS identity >=99%;
3. team identity >=99%;
4. RB/HB/FB position identity >=95%;
5. RYOE/attempt finite coverage >=95% of NGS RB rows;
6. box-exposure finite coverage >=95%;
7. time-to-LOS finite coverage >=95%;
8. NGS efficiency finite coverage >=95%;
9. >=500 usable RB-week rows;
10. >=70 distinct RB identities;
11. >=25 teams represented;
12. duplicate season/week/player identity rows = 0.

Target-history readiness must also pass separately in both 2024 and 2025:

13. >=900 RB target player-weeks with >=3 strictly-prior same-player NGS games;
14. >=80 distinct RBs with >=3-game history;
15. >=28 teams represented;
16. chronology violations = 0.

Integrity:
17. target-game NGS rows used in predictor state = 0;
18. target-game rushing yards/YPC used in predictor state = 0;
19. sportsbook inputs = 0;
20. OddsAPI calls = 0;
21. 2026 outcomes/NGS rows loaded = 0;
22. predictive models fit = 0;
23. thresholds searched = 0;
24. production changes = false.

Do not lower these gates after the audit.

## Dispositions

If all source and target-history gates pass:

`PRISM_RB_NGS_RUSHING_EFFICIENCY_IDENTITY_SOURCE_READY`

Otherwise:

`PRISM_RB_NGS_RUSHING_EFFICIENCY_IDENTITY_SOURCE_BLOCKED`

Integrity failure:

`PRISM_RB_NGS_RUSHING_EFFICIENCY_IDENTITY_SOURCE_INTEGRITY_FAILURE`

## If source-ready

Do not change rushing-yards projections.

Freeze one separate 2024-development / 2025-confirmation predictive efficiency
experiment before scoring.

That experiment must:
- keep canonical rushing opportunity/carries unchanged;
- use the named player's NGS process history only as an efficiency-state signal;
- compare against ordinary strictly-prior player YPC / canonical efficiency;
- preserve player-level heterogeneity;
- use player- and game/team-cluster inference;
- keep opportunity and efficiency contributions separate;
- score rushing efficiency first, rushing yards only under a separately frozen
  market bridge;
- use no 2026 outcomes, sportsbook or OddsAPI.

## If blocked

Fail closed.

Do not:
- lower history/support gates;
- pool positions;
- substitute target-game NGS metrics;
- infer missing player IDs from names after the frozen identity bridge fails;
- turn team NGS aggregates into player identity;
- reuse WR NGS target-model logic;
- fit a box-score YPC rescue.

## Integrity at freeze

At freeze:
- predictive candidates scored: **0**
- NGS 2026 rows loaded: **0**
- sportsbook inputs: **0**
- OddsAPI calls: **0**
- fitted coefficients: **0**
- threshold searches: **0**
- production changes: **0**
