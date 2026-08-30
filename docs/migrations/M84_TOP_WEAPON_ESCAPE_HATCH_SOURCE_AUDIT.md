# Migration 84 — Top Weapon Escape Hatch Source / Feasibility Audit

## Status

`PREREGISTERED / SOURCE AUDIT ONLY`

M84 does not fit a QB model and does not inspect whether a receiver or QB had a big target-game outcome. It asks whether the project can obtain the **materially new pregame individual matchup information** required to test the M82 `TOP_WEAPON_ESCAPE_HATCH` hypothesis without repeating M72/M75.

Authoritative football-only full-stack benchmark remains M82:

- 884 canonical-v3 QB games
- OOS full-stack pass-yard MAE `56.749517`
- 100+ yard misses `123`

## Exact hypothesis reserved for a later predictive migration

The future hypothesis, if source-qualified, is:

> Conditional on the QB's ordinary macro matchup, does one unusually favorable pregame WR/TE/RB route/responsibility matchup create a repeatable upside escape hatch that explains some otherwise surprising QB overperformance? Conversely, does the absence or removal of a primary weapon edge help identify downside games?

M84 does **not** test this outcome relationship. Realized receiver yards and realized QB passing yards are mechanically linked and may not be used as source qualification evidence.

## What counts as materially new information

At least one source path must provide or permit a trustworthy pregame construction of all required concepts:

1. **weapon identity** — specific WR/TE/RB;
2. **route/alignment exposure** — actual route/alignment history or a trustworthy pregame distribution;
3. **defensive responsibility identity** — the defender or zone/responsibility expected to interact with the weapon, not merely every DB on the field;
4. **defender quality / weakness** — prior coverage performance attributable to that defender/responsibility;
5. **replacement context** — if a starter is absent, identify the likely replacement rather than generic injury burden;
6. **multi-season historical coverage** sufficient for development/validation;
7. **in-season deployability** for 2026 before the target game;
8. **public/free reproducibility** under the current project constraint.

A source may satisfy these through a documented joinable composite, but the responsibility link itself may not be inferred from 'all defenders on field' or nearest-defender-at-catch heuristics and called ground truth.

## Explicit anti-retest boundary

M84 may not reopen these rejected proxies as if they were new:

- M72 aggregate explosive-weapon × defense matchup;
- M75 NGS receiver tracking aggregates;
- M75 PFR DB advanced coverage aggregates;
- M75 generic receiver × secondary interaction features;
- M79 generic official inactive correction;
- M81 contested/catchable/drop process variables;
- generic target share, aDOT, separation, cushion, YAC-over-expectation or team secondary strength without an explicit matchup/responsibility bridge.

## Source families to audit

### 1. nflverse participation

Useful concepts:

- target receiver route for the primary receiver on a play;
- man/zone and coverage shell;
- offense and defense players on field;
- formation/personnel.

Required audit:

- inventory all receiver/target/route/alignment/defender/coverage columns;
- search for explicit defender-to-receiver assignment/responsibility IDs/names;
- confirm current in-season availability contract.

Participation/on-field evidence alone is insufficient for true WR-CB responsibility.

### 2. nflverse FTN charting

Useful concepts include contested/catchable/created-reception/drop and QB process fields and it updates in-season.

Required audit:

- search schema for route/alignment and explicit defender responsibility;
- do not relabel its existing process variables as the new matchup bridge.

### 3. NFL Big Data Bowl 2025 exact-assignment sample

Public competition documentation contains:

- `routeRan`;
- `pff_defensiveCoverageAssignment`;
- `pff_primaryDefensiveCoverageMatchupNflId`;
- `pff_secondaryDefensiveCoverageMatchupNflId`.

This is the closest public example to the exact science target, but it is a competition sample from the first nine weeks of the 2022 regular season, not a multi-season/live feed.

M84 must label this `EXACT_BUT_LIMITED_COMPETITION_SAMPLE`, not deployable.

### 4. NFL Big Data Bowl 2026 tracking research sample

Official competition materials provide tracking from the 2023/2024 seasons and play context including targeted-receiver route, receiver alignment, man/zone and team coverage type. This is useful for route/space research.

It does not automatically constitute a public season-long 2026 pregame feed of NFL Coverage Responsibility assignments. Any inferred nearest-defender construction is a new model and may not be called exact responsibility.

### 5. NFL Next Gen Stats Coverage Responsibility

NFL/AWS documentation states that the production system identifies defender-receiver matchups and coverage assignment frame-by-frame and the defender responsible for the targeted receiver.

This is conceptually the ideal observable.

M84 must separately establish whether a public/free historical bulk feed and a pregame/in-season access path exists. Editorial pages, broadcasts or NFL PRO availability do not by themselves create a reproducible source contract.

### 6. PFR advanced defense + depth chart / availability

PFR advanced defensive stats and public depth-chart information can describe defender quality/replacement context and update in-season.

They do not qualify the escape-hatch hypothesis without an explicit receiver-defender/responsibility exposure bridge.

## Frozen qualification levels

Every source path receives one disposition:

- `QUALIFIED_HISTORICAL_AND_LIVE_EXACT`
- `HISTORICAL_RESEARCH_ONLY`
- `EXACT_BUT_LIMITED_COMPETITION_SAMPLE`
- `LIVE_AUXILIARY_ONLY`
- `NO_EXPLICIT_RESPONSIBILITY`
- `PROPRIETARY_OR_NO_PUBLIC_BULK_CONTRACT`
- `SOURCE_ERROR`

## Predictive advancement gate

M84 may advance a future Top Weapon predictive migration only if at least one public/free reproducible source path satisfies **all** of:

- explicit receiver/weapon identity;
- route/alignment exposure;
- explicit defender/responsibility identity;
- defender quality history joinable by stable player identity;
- replacement context joinable pregame;
- at least the 2024 and 2025 regular seasons available historically, or an equivalent multi-season development/confirmation design approved before outcomes are inspected;
- 2026 in-season updates available before later target games;
- no sportsbook inputs;
- no target-game postgame charting leakage.

If no path qualifies, disposition is:

`SOURCE_BLOCKED_EXACT_TOP_WEAPON_MATCHUP`

and no same-proxy predictive model is opened.

## Historical-research-only interpretation

If exact Big Data Bowl assignment data is accessible but limited to a competition sample, M84 may preserve it as a future mechanism-research asset. It cannot be used to claim the production escape-hatch signal is available or to tune the M82 QB benchmark.

## Next boundary if source blocked

If M84 is source-blocked, the remaining M82 new-information frontiers are:

- route × coverage-shell interaction if a trustworthy historical + live route source becomes available;
- true blocker × true rusher assignment if a complete historical + live assignment contract becomes available.

The project should not return to same-information algorithm search merely because the exact individual matchup data is unavailable.

## Production action

`production_actionable = false`
