# OL Roster Continuity Qualification V1

**Status:** FROZEN PRE-QUALIFICATION — OUTCOME-FREE  
**Parent:** `PERSONNEL_CONTINUITY_CONTRACT_V1_FROZEN`  
**Predictive outcomes authorized:** false  
**Production changes authorized:** false

## Scientific question

Can a broad, pregame-known offensive-line personnel continuity state be materialized
with enough historical coverage, stable identity and incremental information to
deserve a separately frozen predictive experiment?

This is not returning-opportunity continuity. It is roster/personnel identity state
for the offensive line.

## Frozen primary candidate

`ol_roster_continuity_share_prev_game`

For scheduled team-game G:

1. form the current-week OL roster set from stable GSIS IDs;
2. find the same team's previous scheduled regular-season game in the same season;
3. form that prior-game-week OL roster set;
4. compute:

`|current OL IDs ∩ prior-game OL IDs| / |current OL IDs|`

Week 1 / no prior regular-season game is explicit `UNKNOWN_NO_PRIOR_GAME`, not zero.

Supporting diagnostics only:
- current OL roster count
- prior-game OL roster count
- returning OL count
- added OL count
- departed OL count

No alternative continuity formula is inspected in V1.

## Frozen OL eligibility

Use nflverse weekly roster identity only.

A roster row is OL-eligible when either:
- `position` is one of `C,G,T,OT,OG,OL`; or
- `depth_chart_position` is one of `LT,LG,C,RG,RT,OT,OG,OL,G,T`.

Do not use target-game snaps, target-game participation or postgame starting lineups.

Backups remain part of the broad roster state.

## Historical scope

Seasons: **2019–2025**.

Target grain:

`season, week, team` for scheduled regular-season team-games.

Use the repository's canonical schedule builder.

Weekly roster snapshots are the existing nflverse source already used by the
historical pregame-universe pipeline.

## Identity / integrity

- stable GSIS identity coverage >= **0.99**
- ambiguous same-week GSIS/team conflicts = **0**
- duplicate published team-week keys = **0**
- schedule join fanout = **0**
- current-week OL roster count must be >0 for a known state
- unknown is never coerced to zero

Names are diagnostic only.

## Pregame coverage

Broad denominator: all scheduled 2019–2025 regular-season team-games.

Week 1 remains in the denominator as explicit unknown.

Frozen gate:
- eligible team-game rows >= **500**
- known pregame continuity coverage >= **0.80**

No late-season-only rescue.

## Temporal legality

The current weekly roster snapshot is the target game's pregame personnel source.
The comparison roster is the same team's previous scheduled regular-season game week.

Forbidden:
- target-game snaps;
- target-game participation;
- target-game PBP;
- postgame lineup labels;
- future-week rosters.

Temporal violations must be **0**.

## Stability applicability

This is a directly observed personnel-change state, not a latent player/team tendency.
Therefore adjacent-period persistence is **not a hard qualification gate** under the
parent contract's "where appropriate" rule.

For diagnostics only, report:
- adjacent-game Spearman of continuity share;
- median absolute adjacent change.

The candidate cannot fail solely because personnel changes are episodic.

## Outcome-free redundancy

Ask whether OL continuity is reconstructible from already-available canonical
strict-prior team state.

Inputs, all from the prior completed team game only:
- `pressure_rate_allowed`
- `success_rate_off`
- `dropback_rate`
- `plays_est`
- `proe`
- team one-hot
- target week numeric

No target-game PBP or outcomes.

Temporal reconstruction:
- train seasons: **2019–2023**
- holdout seasons: **2024–2025**

Linear least-squares with intercept.
Numeric missingness is imputed with train medians.
Team categories are learned from train only.

Minimum rows:
- train >= **1,000**
- holdout >= **500**

Frozen redundancy interpretation:
- R2 >= **0.90**: `HIGHLY_RECONSTRUCTIBLE_REDUNDANT`
- 0.75 <= R2 < 0.90: `REDUNDANCY_REVIEW`
- R2 < 0.75: `INCREMENTAL_INFORMATION_SURVIVES_REDUNDANCY_GATE`
- insufficient rows: `REDUNDANCY_UNRESOLVED_SOURCE_THIN`

## Qualification disposition

`READY_FOR_FROZEN_EXPERIMENT` requires:
- identity/integrity clean;
- broad coverage >= 0.80;
- eligible rows >= 500;
- direct pregame mechanism;
- redundancy not highly reconstructible;
- no leakage violation.

Possible other dispositions remain:
- `ENGINEERING_READY_SOURCE_THIN`
- `DESCRIPTIVE_ONLY`
- `SOURCE_BLOCKED`
- `REJECTED_INTEGRITY`

## Mechanism

Potential future role:
- pass-protection continuity / uncertainty;
- run-blocking continuity / uncertainty;
- QB efficiency/scramble environment;
- team rushing efficiency context.

No point-mean or distribution change is authorized by qualification.

## No-rescue rules

Do not:
- restrict to starters after seeing coverage;
- use target-game snaps to define the five OL;
- exclude Week 1 from the denominator;
- lower coverage or identity gates;
- replace roster continuity with a favorable post-hoc Jaccard/weighted formula;
- read predictive target outcomes during qualification;
- use sportsbook data;
- touch Issue #535.

## Pre-result disposition

`OL_ROSTER_CONTINUITY_QUALIFICATION_V1_FROZEN_PRE_RESULT`
