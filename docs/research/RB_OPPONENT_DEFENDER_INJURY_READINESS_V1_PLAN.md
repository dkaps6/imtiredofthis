# RB Opponent-Defender Injury Readiness V1 — Frozen Source Audit

Date: 2026-09-26

Status: **FROZEN BEFORE SOURCE-AUDIT EXECUTION**

Branch: `research-rb-opponent-defender-injury-readiness-v1`

## Purpose

Audit whether current/past NFL injury reports can be combined with strictly-prior
defensive participation to create a deployable, leakage-safe opponent-defender
availability surface for future RB research.

This is **source inventory / readiness only**.

It is explicitly allowed before the first RB-PD2 prospective shadow observation.
It does not begin the frozen RB mean-information modeling study and does not
construct or score a projection candidate.

## Why this audit is distinct

Current `TeamContext` contains defense-level historical aggregates such as:
- defensive rush EPA;
- box rates;
- pressure generation;
- coverage rates.

Current `PlayerContext` can carry the offensive player's own injury status.

There is no current `TeamContext` field for:
- target-week unavailable defensive front-seven players;
- prior defensive snap mass lost to target-week injury;
- role-specific defensive replacement burden.

The earlier RB New-Data Readiness V1 audit established that injury reports are
live and keyed, but did not quantify the opponent-defender join/deployability
contract.

## Data sources

Free nflverse / nflreadpy only:

1. `load_injuries` for seasons 2024, 2025, 2026;
2. `load_snap_counts` for seasons 2023, 2024, 2025, 2026;
3. `load_rosters_weekly` when useful for a stable GSIS <-> PFR/name bridge.

No sportsbook data.

No target-game outcomes.

## Frozen chronology contract

For a target `(season, week, team)` injury designation:

- injury designation belongs to the target week and is a pregame source;
- defender participation may use **only weeks strictly before the target week**;
- primary historical load statistic is the player's most recent prior defensive
  snap share / defensive snap count;
- same-week or future participation is forbidden;
- Week 1 may use prior-season strictly-prior participation if identity continuity
  is available; otherwise record unavailable, do not backfill target-week snaps.

No postgame target-week participation may define the injury burden.

## Defensive scope

Audit separately:

- DL/front: DE, DT, NT and provider-equivalent defensive line positions;
- LB: LB, ILB, OLB and provider-equivalent linebacker positions;
- DB: CB, S, DB and provider-equivalent secondary positions;
- FRONT7 = DL + LB;
- ALL_DEFENSE = DL + LB + DB.

No claim is made yet that one group is predictive.

## Identity hierarchy

Prefer stable cross-provider IDs if a roster bridge exposes them.

Allowed hierarchy:

1. direct stable ID bridge;
2. roster-mediated GSIS/PFR bridge;
3. canonicalized `season/team/player name` fallback, audited explicitly.

Any many-to-many or within-team/week collision fails closed for the affected
identity and is reported.

## Readiness outputs

Persist:

1. source schema / season availability;
2. row-level target-week defensive injury rows with:
   - target season/week/team;
   - player;
   - defensive group;
   - report/practice status;
   - stable IDs available;
   - strictly-prior snap source season/week;
   - prior defense snaps / defense_pct;
   - join method;
   - chronology validity;
3. team-week aggregate readiness:
   - injured defender count by group/status;
   - matched prior defensive snap mass by group/status;
   - denominator / coverage diagnostics;
4. season summary.

## Readiness questions

Report descriptively:

- Are 2024, 2025 and live 2026 injury rows available?
- Are target-week defensive injury designations populated?
- What fraction of defensive injury rows can be joined to strictly-prior snaps?
- What fraction of OUT/DOUBTFUL defensive rows can be joined?
- What fraction of FRONT7 OUT/DOUBTFUL rows can be joined?
- How much of the join relies on stable IDs vs name/team fallback?
- Are all 32 teams represented in live 2026 source coverage?
- Does the current repository/model consume any such opponent-defender burden?
- Is the surface deployable today without paid data?

## Integrity gates

This audit is valid only if:

1. sportsbook inputs used = 0;
2. target-game outcomes read = 0;
3. candidate variants constructed/scored = 0 / 0;
4. parameters fit = 0;
5. same/future snap violations = 0;
6. source seasons are exactly the requested historical/live seasons;
7. all identity collisions are counted and never silently resolved;
8. no production file is mutated.

## Disposition vocabulary

This is not a performance qualification.

Allowed dispositions:

- `RB_OPPONENT_DEFENDER_INJURY_SOURCE_READY`
- `RB_OPPONENT_DEFENDER_INJURY_SOURCE_PARTIAL`
- `RB_OPPONENT_DEFENDER_INJURY_SOURCE_NOT_READY`
- `RB_OPPONENT_DEFENDER_INJURY_READINESS_INTEGRITY_FAILURE`

No modeling candidate is authorized by any readiness disposition.

A later candidate may be frozen only after the RB-PD2 forward activation boundary
permits the mean-information lane and only if this source audit is deployable.

Candidate variants scored: **0**  
Parameters fit: **0**  
Sportsbook inputs: **0**  
Target-game outcomes: **0**  
Production mutations: **0**


## Pre-run readiness thresholds

Frozen before source-audit execution:

`RB_OPPONENT_DEFENDER_INJURY_SOURCE_READY` requires all of:

1. injury source returns nonzero rows for 2024, 2025 and 2026;
2. snap source returns nonzero rows for 2023, 2024, 2025 and 2026;
3. live 2026 defensive injury source represents at least **30 NFL teams**;
4. strictly-prior snap join coverage among all defensive injury rows is >= **90%**;
5. strictly-prior snap join coverage among OUT/DOUBTFUL defensive rows is >= **90%**;
6. strictly-prior snap join coverage among OUT/DOUBTFUL FRONT7 rows is >= **90%**;
7. same/future snap violations = **0**;
8. unresolved identity collisions = **0**;
9. at least **80%** of matched defensive rows use a stable-ID or roster-mediated
   identity path rather than raw name/team fallback;
10. all integrity gates pass.

`RB_OPPONENT_DEFENDER_INJURY_SOURCE_PARTIAL` requires:
- all source seasons present;
- chronology/integrity clean;
- all-defensive strictly-prior join coverage >= **70%**;
- but one or more READY gates fail.

Otherwise:
`RB_OPPONENT_DEFENDER_INJURY_SOURCE_NOT_READY`.

These are deployment-readiness thresholds only; they are not predictive-performance gates.
