# RB Opponent-Defender Injury Source Readiness V1 — Frozen Plan

Status: **SOURCE READINESS ONLY — NO PREDICTIVE FIT — NO PRODUCTION CHANGE**

Branch: research-rb-opponent-defender-injury-source-v1

## Why this exists

The frozen RB-PD2 parallel mean-information program explicitly authorized a
source-readiness audit for opponent-injury propagation into matchup context
after the first valid prospective RB-PD2 lock.

That audit was previously assigned but never completed.

The Weeks 1-4 production postmortem now reinforces the need to separate
structural RB rushing-mean error from distribution calibration. This source
audit does **not** use 2026 outcomes to choose a feature or formula.

## Anti-retest boundary

Prior closed work includes:
- generic/static defensive matchup features;
- defensive-front / OL cohesion families;
- exact-personnel discontinuity studies in other model lanes;
- defensive adaptive gameplan studies.

This source audit does not revive any of those.

The only question here is whether a distinct, role-specific, pregame
**opponent defensive availability** state can be constructed with one
historical/live source contract.

No predictive candidate is defined here.

## Candidate source

nflverse / nflreadpy weekly injury reports.

Existing live and historical builders already consume this source, but the
normalized repo injury tables currently discard fields that matter for a
role-specific opponent-defender state, especially player position and GSIS ID.

This audit reads the raw provider schema directly and records whether those
fields are present consistently enough to support a later research-only
normalization.

## Seasons / grain

Audit:
- 2023
- 2024
- 2025
- current 2026

Required grain:
- season
- week
- team
- player
- position
- report/game status
- practice status if available
- stable player identity (gsis_id) if available

No game outcome is loaded.

## Defensive-front descriptive scope

For source-coverage reporting only, normalize provider position strings into a
broad defensive-front bucket when they are unambiguously one of:

- DT / NT / DL
- DE / EDGE
- LB / ILB / OLB

This bucket is **not** a model feature and no position weights are assigned.

Unknown position strings remain unknown; never infer from player name.

## Readiness checks

Per season and week report:
- total injury rows;
- distinct teams;
- rows with finite/nonblank position;
- rows with stable GSIS identity;
- rows with report status;
- rows with practice status;
- defensive-front rows;
- defensive-front rows with stable identity;
- defensive-front rows with report status;
- counts by provider position and report status.

Season-level:
- weeks represented;
- team coverage;
- unresolved/missing identity rate;
- missing-position rate;
- defensive-front row count;
- defensive-front status completeness.

## Live parity gate

The source is SOURCE_READY_FOR_SEPARATE_PREDICTIVE_PLAN only if:
1. 2024 and 2025 each contain >=16 regular-season weeks;
2. 2026 contains every completed regular-season week available at audit time;
3. all 32 teams appear in each full historical season;
4. >=95% of defensive-front rows have stable GSIS identity;
5. >=95% of defensive-front rows have report status;
6. provider position semantics are stable enough to map without fuzzy player-name inference.

Otherwise: SOURCE_PARITY_NOT_CLEARED.

This gate only establishes the data contract. It does not establish that
defensive injuries predict RB rushing.

## Timing caveat

nflverse weekly injury reports are treated as weekly pregame report state, not
official game-day inactive certification.

Any later predictive plan must preserve source timing semantics and must not
reinterpret a Friday OUT designation as T-75 official inactive proof.

If exact publication timestamps are unavailable historically, the later plan
must use only semantics justified by the weekly report archive and document the
limitation explicitly.

## What a source PASS authorizes

Only a separate frozen predictive design that asks whether **strictly pregame,
role-specific opponent defensive availability** adds information beyond the
existing football stack.

That later study must:
- define the aggregation before outcomes are scored;
- use genuine historical holdout / replication;
- avoid generic defense-stat repackaging;
- use no sportsbook inputs upstream;
- make no production change unless separately qualified.

## What this plan forbids

- no injury-count coefficient;
- no DL/LB weighting;
- no status severity weighting;
- no threshold search;
- no 2026 outcome scoring;
- no RB mean adjustment;
- no YPC adjustment;
- no sportsbook data;
- no production mutation.
