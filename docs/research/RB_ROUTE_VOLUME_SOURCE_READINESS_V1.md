# RB Route-Volume Source Readiness V1

Date: 2026-09-29  
Status: **COMPLETE — LIVE SOURCE EXISTS, HISTORICAL/LIVE WEEKLY PARITY NOT CLEARED**  
Branch: `research-week3-postmortem-execution-v1`

## Authority / why this audit exists

The frozen `RB_PD2_FORWARD_SHADOW_CONFIRMATION_V1_PLAN.md` authorizes a
separate RB mean-information source program after the first valid prospective
forward lock.

Frozen order:
1. routes-run / route-volume live + historical source readiness;
2. opponent-injury propagation into matchup context;
3. CLV capture architecture as downstream diagnosis only.

Week 3 supplied the first valid RB-PD2 forward observation, so item 1 may now be
audited. This document is source-readiness only. It fits no model and uses no
Week-3 outcome to choose a feature.

## Prior repo state

The 2026-09-21 handoff correctly recorded the state at that time:
- nflverse participation was historical-only for this purpose;
- no live 2026 exact route-run source existed **inside the repo**;
- current PFR/nflverse snap counts were available but were only a coarse
  opportunity proxy;
- do not build a historical route feature that cannot be computed live.

That last rule remains binding.

## Current 2026 source audit

### 1. HeatRadar routes page

Current public 2026 page:
`https://heatradar.app/nfl/routes`

Observed source contract:
- player-level `Routes`;
- `Route %`;
- Targets;
- TPRR;
- YPRR;
- ADOT;
- target share;
- position filter includes RB;
- week buttons are exposed;
- site states weeks are graded independently;
- site states routes/route rate come from a weekly charting export rather than
  public play-by-play.

HeatRadar defines its route percentage as player routes relative to the team's
charted pass-play denominator.

**Readiness:** live/current route-volume source now exists.

**Unresolved:** a stable public historical season/week archive with the same
schema and retrieval contract has not been established. The broader HeatRadar
research page says underlying data are available on request, but this audit did
not obtain or assume access to a historical route archive.

### 2. StatRankings Routes Run

Current page:
`https://statrankings.com/nfl/advanced/players/usage/routes-run`

Observed source contract:
- current 2026 Routes Run;
- historical season selectors 2021-2025 plus 2026;
- page states updates within roughly 24-36 hours postgame;
- player/team/position filters;
- full-season stats are public;
- custom week ranges are explicitly a premium feature.

**Readiness:** strongest same-site historical/current source candidate.

**Blocking limitation for a leakage-safe backtest:** a strict-prior weekly
historical reconstruction needs arbitrary historical week ranges or game-level
rows. The public contract observed here does not expose those arbitrary weekly
ranges without the premium feature. No subscription/purchase was made or
authorized.

## Critical semantic finding: nflverse is not a drop-in historical bridge

nflverse participation is machine-readable and historically valuable, but its
current documentation says:
- 2023+ participation arrives only after the postseason and does not update
  during the season;
- its `route` field is a string describing the route taken by the **primary
  receiver on a play**.

That is not equivalent to a player-level count of every route run on every team
pass play.

Therefore it would be invalid to construct:

> historical nflverse `route` count -> live HeatRadar/StatRankings Routes Run

and call the two the same feature without a separate ground-truth bridge.

The old repo label `nflverse participation / route data` must not be
misinterpreted as proof that historical total player route volume exists there.

## Source-readiness disposition

`LIVE_RB_ROUTE_VOLUME_CONFIRMED_HISTORICAL_WEEKLY_PARITY_NOT_CLEARED`

What changed since 2026-09-21:
- live 2026 exact route-volume information now demonstrably exists publicly.

What did **not** clear:
- one free, machine-reproducible, historical + live weekly source contract;
- an exact semantics bridge between historical nflverse participation and the
  live charted route-count sources;
- a leakage-safe historical backtest panel.

## What is authorized next

### A. Prospective capture — YES

Beginning with the first pregame Week-4+ capture after this audit:
- preserve source timestamp;
- preserve source name/version/page identity;
- preserve player/team/position identity;
- preserve routes and the source's own route-participation definition;
- never overwrite prior captures;
- never infer a missing player as zero;
- keep this research-only and downstream of football generation until science
  clears.

This can build the live temporal history that was missing.

### B. Bounded no-outcome source parity — YES

A parity/semantics check may compare overlapping player/team aggregates between
independent current sources. It may test identity, scale, definition, freshness,
and missingness.

It may **not** use player outcomes to choose a source, threshold, or formula.

### C. Historical predictive model test — NOT YET

Do not freeze a predictive route-volume candidate until one of these occurs:
1. a no-cost week-level historical route-run archive with matching live
   semantics is established;
2. an already-authorized private/licensed source supplied by the user provides
   that contract;
3. enough prospectively captured 2026 weekly history accumulates to support a
   forward-only test.

Do not pay for StatRankings or any other source without explicit user approval.

## Anti-retest / no-substitution rules

- snap share is not exact routes run;
- nflverse primary-receiver route labels are not total player routes run;
- targets are not routes;
- do not reconstruct routes from receptions/targets;
- do not fit a Week-3 route formula after exposure;
- do not use sportsbook data upstream;
- do not reopen closed RB receiving-efficiency transforms under a route label.

## Research consequence

This lane is **not a dead end**. It has moved from
`NO_LIVE_SOURCE` to `LIVE_SOURCE_EXISTS / PARITY_BLOCKED`.

The correct action is prospective acquisition + bounded source-semantic
validation, not a retrospective model fit on mismatched route definitions.
