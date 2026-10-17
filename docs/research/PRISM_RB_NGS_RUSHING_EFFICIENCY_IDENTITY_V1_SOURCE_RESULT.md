# PRISM RB NGS Rushing Efficiency Identity V1 — Source Result

Date: 2026-10-17

Status: **SOURCE BLOCKED / DO NOT SCORE**

Model:
**PRISM — Player Role & Individual Situation Model**

Frozen source plan:
`docs/research/PRISM_RB_NGS_RUSHING_EFFICIENCY_IDENTITY_V1_SOURCE_PLAN.md`

Canonical execution:
- run `39068580334` — SUCCESS;
- exact child-PR head `3b39bfce0cdff0f3ecef09518b8b9dd04795cca8`;
- artifact `12113799476`;
- digest `sha256:9511828f360ad01aceb0fae6e9bff4d2b70981ed07d54bca84740bed41d65f00`;
- runner PR `#688`.

Disposition:

`PRISM_RB_NGS_RUSHING_EFFICIENCY_IDENTITY_SOURCE_BLOCKED`

## Source quality

The nflverse Next Gen Stats rushing source itself passed every frozen per-season source gate in 2023, 2024 and 2025.

2023:
- 551 usable RB-week rows;
- 85 distinct RBs;
- 32 teams;
- 18 regular-season weeks;
- stable player ID 100%;
- team identity 100%;
- RB position identity 100%;
- RYOE/attempt finite 100%;
- secondary RYOE family finite 100%;
- 8+ box exposure finite 100%;
- average time-to-LOS finite 100%;
- NGS rushing efficiency finite 100%;
- duplicate player-week identities 0.

2024:
- 533 usable RB-week rows;
- 83 distinct RBs;
- 32 teams;
- all source-quality rates above remain 100%;
- duplicates 0.

2025:
- 569 usable RB-week rows;
- 80 distinct RBs;
- 32 teams;
- all source-quality rates above remain 100%;
- duplicates 0.

Therefore this is **not** a provider-quality or identity failure.

## Frozen history-readiness gates

The source plan required each 2024/2025 target season to have:
- >=900 RB target player-weeks with >=3 strictly-prior same-player NGS rushing games;
- >=80 distinct RBs with >=3-game history;
- >=28 teams;
- chronology violations = 0.

Observed:

2024:
- target pregame RB rows: 1,845;
- history-ready rows: **476**;
- history-ready distinct RBs: **52**;
- teams: 32;
- chronology violations: 0.

2025:
- target pregame RB rows: 1,866;
- history-ready rows: **511**;
- history-ready distinct RBs: **52**;
- teams: 32;
- chronology violations: 0.

The row-support and distinct-player gates fail in both seasons.

## Interpretation

Next Gen Stats rushing process data are high-quality and player-specific, but under the frozen same-season >=3-game history contract they do not provide enough historical OOS support to score the planned individual RB efficiency-state mechanism.

Do not rescue by:
- lowering the >=3-game rule;
- lowering the >=900-row or >=80-player gates;
- pooling seasons after seeing support;
- using target-game NGS metrics;
- fitting a box-score YPC substitute;
- converting team NGS aggregates into player identity;
- using 2026 outcomes.

This result says **source support is insufficient for this exact V1**, not that player-specific rushing process information has no football value.

## Integrity

- chronology violations: 0;
- 2026 outcomes / NGS rows loaded: 0;
- target-game NGS rows used in predictor state: 0;
- target-game rushing yards/YPC used in predictor state: 0;
- sportsbook inputs: 0;
- OddsAPI calls: 0;
- predictive models fit: 0;
- thresholds searched: 0;
- production changed: false.

Final disposition:

`PRISM_RB_NGS_RUSHING_EFFICIENCY_IDENTITY_SOURCE_BLOCKED`.
