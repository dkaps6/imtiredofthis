# Current-Season State Production Audit V1

**Date:** 2026-09-22  
**Branch:** `research-current-season-state-persistence-v1`  
**Status:** source/production-seam audit complete; predictive persistence run pending  
**Production changes authorized:** NO

## Executive finding

The production stack is **not static** and is not blind to the 2026 season.

It already incorporates completed current-season evidence in several important places. The main limitation is that most live inputs enter as blended/rolling **levels**, while explicit **change-of-state** variables (promotion, demotion, vacancy recipient, sharp usage delta, room competition delta) are not represented consistently across positions.

This distinction matters. The next frontier is not "start using current-season data"; it is "use current-season changes more explicitly and validate how quickly those changes should override historical priors."

## PlayerForm V2 — direct current-season evidence

`scripts/player_form_v2.py` builds previous-season totals and current-season-to-date totals using only active-season weeks strictly before the target week.

Its exact blend is:

`w_current = current_games / (current_games + 4)`

The current PlayerForm metrics are:

- target share;
- rush share;
- route rate when supplied;
- YPRR;
- YPT;
- YPC;
- YPA;
- receptions per target.

Implication for an uninterrupted early season:

- after 1 completed game: current weight = 20.0%;
- after 2 completed games: current weight = 33.3%;
- after 3 completed games: current weight = 42.9%;
- after 4 completed games: current weight = 50.0%.

Therefore the Week-3 2026 player baseline already incorporates Weeks 1-2, but only through this fixed shrinkage rule.

The persistence experiment on this branch tests whether that rule is empirically appropriate next-game prediction state.

## QB / team / pass-defense state

`scripts/run_qb_promoted_context.py` and `scripts/team_context_v3.py` implement the promoted M89/M90 semantics.

For each team, the live QB/team context takes the last eight completed games strictly before the target week. During the season, current-year games replace prior-year games naturally.

Confirmed rolling-eight fields include:

- true PROE;
- neutral pace;
- pressure rate allowed/generated;
- pass attempts per dropback;
- offensive pass rate;
- defensive pass rate faced;
- defensive pass EPA allowed;
- defensive pass success allowed;
- defensive YPA allowed;
- offensive YPA;
- offensive pass EPA;
- estimated plays.

The context artifact records exact current-season and prior-season game counts and labels the state `current_season_history` once current games exist.

Therefore QB and passing-defense context already updates meaningfully from completed 2026 football.

Open gap: these are rolling levels. The production stack does not expose a generic cross-position "2026 state minus historical state" layer for role/environment changes.

## RB state

PlayerForm current-season `rush_share`, target share and YPC already influence the generic live player state.

Current Ourlads role is also preserved as `depth_role`, and definitive unavailable players are removed by the promoted current-availability path.

However, the prior Week-1 current-role authority audit established:

`CURRENT_ROLE_PRESERVED_BUT_NOT_DIRECT_ALLOCATION_INPUT`

The effective/model role is rebuilt from historical/blended usage. Current depth hierarchy is contextual evidence but does not directly control rushing allocation.

Historical source work also established that timestamp-safe pregame RB depth context and prior-week offensive snap share are reconstructable. RB-ND2B found 100% 2025 team-game pregame depth coverage and 83.7% prior-week snap coverage on Weeks 2-18 against the M94C cohort.

M95G/M95H further showed that current-week role/depth/availability contains real workload signal, especially for 20+ carries and >=70% RB carry-share ranking, while exact sudden-vacancy successor identity remains harder.

Current production availability removes an unavailable back but does not explicitly transfer the vacated rushing opportunity to a surviving teammate.

This remains a high-priority state-change gap.

## WR state

PlayerForm target share / YPT / catch rate already incorporate completed 2026 weekly stats through the fixed four-game pseudo-prior blend.

WR-R15 then redistributes the WR2+ target pool using strict-prior offensive participation features.

Important live-source gap discovered in this audit:

`wr_r15_entitlement_adapter_v1.py` imports the snap loader from `te_r5p_entitlement_adapter_v1.py`, whose hardcoded `SOURCE_SEASONS` is:

`[2020, 2021, 2022, 2023, 2024, 2025]`

Therefore the live 2026 WR-R15 adapter currently cannot consume Week-1/Week-2 2026 snap counts, even though its feature semantics are strict-prior participation.

A source-readiness audit is included in Current-Season State Persistence V1. No production extension is authorized until that audit is complete.

## TE state

PlayerForm target share / YPT / catch rate already incorporate completed 2026 weekly stats through the same blend.

TE-R5P uses prior-1 and prior-3 offensive snap participation, including same-team and any-team features.

The same hardcoded snap-source list stops at 2025, so the production TE-R5P feature path also cannot consume completed 2026 snaps today.

This is a particularly direct current-state gap because TE-R5P's learned feature contract explicitly expects strict-prior participation.

## Defense state

Passing-side team/defense state is materially current through the promoted M89/M90 rolling-eight context described above.

Some additional Team Context V3 fields remain explicitly tagged as legacy/provider-semantics-pending, including `def_rush_epa` and several box/personnel fields. These should not be treated as equally certified current-state authorities without a separate source audit.

The current-season state lane should distinguish certified promoted rolling context from these legacy fields.

## Existing cross-position state design

The prior branch `data-frontier-cross-position-context-state-v1` froze a correct engineering concept:

`current_environment_state - historical_environment_state_for_player`

It defined role-transition, room-competition, QB-environment, RB-environment, OL/protection and injury-created-vacancy dimensions.

However, that branch contains only the engineering plan and source registry. No generic context-state materializer or predictive experiment was completed there.

Current-Season State Persistence V1 is therefore not duplicating a completed experiment.

## Immediate research sequence

The current branch first measures whether season-to-date player state persists into the next game and whether the existing four-game pseudo-prior blend improves next-game prediction.

In parallel it audits 2026 snap availability against the WR-R15/TE-R5P production source contract.

After the persistence result and Claude's independent Week-2 model-performance grade are both available, the next candidate should be selected from the intersection of:

- live Week-2 weakness;
- historical state persistence;
- a missing production state seam;
- deployable pregame data.

No production model changes are authorized by this audit alone.

## Disposition

`CURRENT_SEASON_STATE_PRODUCTION_AUDIT_V1_COMPLETE`
