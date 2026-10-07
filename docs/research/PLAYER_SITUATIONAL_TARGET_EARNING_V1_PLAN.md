# Player Situational Target Earning V1 — Frozen Source / Nonredundancy Audit

**STATUS: FROZEN BEFORE RESULT. SOURCE/ARCHITECTURE ONLY. NO PREDICTIVE CANDIDATE.**

## Motivation

WR and TE now independently show the same player-level residual structure after their promoted individualized entitlement layers:

- opportunity error persists by player;
- opportunity difficulty persists by player;
- efficiency difficulty persists by player;
- signed efficiency error does not persist.

Authorities:
- WR Player Mechanism Persistence V1: run `37634044900`, disposition `WR_PLAYER_PERSISTENCE_MIXED_MECHANISM`;
- TE Player Mechanism Persistence V1: run `37631011196`, disposition `TE_PLAYER_PERSISTENCE_MIXED_MECHANISM`.

Plain participation/snap hierarchy is not a new lane:
- WR-R15 and TE-R5P already consume participation;
- WR R8 found static participation/depth variables insufficient;
- Receiver Room Targets-per-Play V1 is closed.

This audit therefore asks whether free PBP can expose a **different player-specific opportunity object**:

> how an individual receiver earns his team's targets in different football situations.

## Source

Free public nflverse/nflreadpy regular-season PBP:
- 2022
- 2023
- 2024
- 2025
- 2026 through Week 4 only

Live identity/universe support:
- 2026 weekly roster;
- 2026 Week-5 regular-season schedule.

No sportsbook data.
No Week-5 outcome.
No target projection.
No candidate fit.

## Target-event semantic

A target event is:
- official pass attempt;
- non-empty receiver player ID;
- offense team present;
- sacks and two-point attempts excluded.

## Frozen situational contexts

For every target event classify:

1. `ALL`
2. `EARLY_DOWN`: down in {1,2}
3. `THIRD_DOWN`: down == 3
4. `RED_ZONE`: yardline_100 <= 20
5. `TWO_MINUTE`: half_seconds_remaining <= 120

These are football-state buckets only. No threshold is outcome-optimized.

For every player/team context:

`context_target_share = player context targets / team context targets`

For ALL this is ordinary team target share.

## Historical source inventory

For every season report:
- PBP rows;
- target events;
- receiver-ID coverage;
- target-event counts in every context;
- distinct receivers;
- distinct team-receiver identities.

Historical inventory is source support only.

## Live Week-5 audit

Use only 2026 Weeks 1-4.

Universe:
- Week-5 scheduled WR/TEs from the target-week weekly roster when available;
- otherwise the latest weekly roster strictly before Week 5;
- player must have at least one Weeks 1-4 target to enter the nonredundancy panel.

For each current team/player calculate:
- overall target share;
- early-down target share;
- third-down target share;
- red-zone target share;
- two-minute target share;
- raw target counts and team denominators for each context.

Current team is defined by the target/latest weekly roster, not by target-game outcome.

## Nonredundancy diagnostics

For every situational share vs ALL:
- Spearman correlation across live player rows;
- standard deviation of `context_share - overall_share`;
- count and fraction with absolute difference >= 0.05.

The frozen descriptive criterion for a context to be called **materially nonredundant** is:

- Spearman vs overall < 0.95; AND
- SD(context - overall) >= 0.02; AND
- at least 25 live WR/TE players differ from overall by >= 0.05.

This criterion is only for deciding whether the source object contains information distinct from ordinary target share. It is not a production feature threshold.

## Coverage

Report:
- scheduled teams;
- target-roster WR/TEs;
- live WR/TEs with at least one prior target;
- stable-ID coverage;
- context denominator coverage by team;
- player context-share coverage;
- position splits WR/TE.

## Disposition

`PLAYER_SITUATIONAL_TARGET_EARNING_SOURCE_READY`

requires:
1. 2022-2026 historical/live PBP all non-empty;
2. 2026 max source week <= 4;
3. receiver-ID coverage >= 99% on 2026 target events;
4. 30 Week-5 scheduled teams;
5. >=100 live WR/TEs with at least one prior target;
6. overall target-share coverage 100% for the live panel;
7. each situational context has team-denominator coverage >= 90%;
8. at least **2 of 4** situational contexts are materially nonredundant;
9. zero Week-5 outcomes;
10. zero sportsbook inputs;
11. zero candidate models fit;
12. zero production changes.

If chronology is clean and the source exists but support/nonredundancy misses:
`PLAYER_SITUATIONAL_TARGET_EARNING_SOURCE_PARTIAL`.

If chronology/identity safety fails:
`PLAYER_SITUATIONAL_TARGET_EARNING_SOURCE_NOT_READY`.

## If READY

Only a separately frozen diagnostic may ask whether these strictly-prior player situational target-earning states explain the already-observed WR/TE opportunity residuals.

A future candidate may not:
- replace WR-R15 / TE-R5P wholesale;
- reopen Receiver Room Targets-per-Play;
- reduce to plain snap share;
- use target-game state;
- use sportsbook data.

No production change is authorized.

Models fit: **0**  
Sportsbook inputs: **0**  
Week-5 outcomes: **0**  
Production mutations: **0**
