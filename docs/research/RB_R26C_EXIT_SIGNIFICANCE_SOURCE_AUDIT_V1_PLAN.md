# RB R26C — Exit Significance Source Audit V1

Status: **FROZEN BEFORE EXIT-SIGNIFICANCE OUTCOME SCORING**
Date: 2026-09-09
Parent forensic authority: `RB_R26B_2023_ROLE_ALLOCATION_FORENSIC_ATLAS_V1`
Parent run: `34360534200`
Parent artifact: `10107626915`
Parent digest: `sha256:29380225446601fef21a46bcca94c496c5bdb689dede76f116c462d6c4ecd7ed`
Production authority protected: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Research branch: `research-rb-r26c-exit-significance-source-audit-v1`

## Why this audit exists

R26 V1 found that a vacancy-gated R9 receiving-identity redistribution improved vacancy-incumbent receptions MAE in 5 of 6 modern seasons and materially improved Week 1, but failed its frozen season-robustness gate because 2023 worsened. R26B then showed that the 2023 loss was overwhelmingly within-room player allocation rather than RB-room total error, and that the failure occurred in Weeks 2+ rather than Week 1. Harm in later-season phases replicated directionally in multiple other seasons.

The current vacancy state is deliberately coarse: any ACT/INA RB/FB identity that existed in the immediately prior leakage-safe roster snapshot but is absent in the current ACT/INA snapshot counts as an exit. That treats a meaningful receiving-role departure and low-usage roster churn identically.

R26C therefore asks a source/measurability question only:

> Can we identify, strictly before the target game, which exited RB/FBs carried meaningful receiving-role evidence before they left?

R26C does **not** score the target-game outcome and does not create a new candidate/router.

## Source contract

For target seasons 2020-2025 and every regular-season target week:

1. Current RB/FB room = target-week canonical weekly roster, statuses `ACT` or `INA` only.
2. Prior RB/FB room = strictly earlier canonical ACT/INA roster snapshot:
   - Week 1: final regular-season snapshot of prior season;
   - Weeks 2+: latest earlier regular-season snapshot in same season.
3. Exited identities = `prior_room - current_room` by normalized player identity.
4. Same-week historical depth is forbidden.
5. Prior depth may come only from the same strictly earlier snapshot used for the prior room.
6. Receiving-role identity may use only player information strictly before target season/week. The production-safe `rb_receiving_identity_runtime_v1` as-of semantics are allowed; exact-match target week is forbidden by that runtime.
7. Sportsbook data are forbidden.
8. Target-game participation, box scores, targets, receptions, yards, snaps, routes, injuries/outcomes are forbidden as inputs to this source audit.
9. No 2026 outcomes are loaded.
10. No production runtime/model file may be changed.

## Required exited-player state

Persist one row per exited RB/FB identity with at least:

- target season/week/team
- prior snapshot season/week
- exited player identity/name
- prior roster position/status
- prior depth position/team/rank-like value when timing-safe and available
- prior-depth availability flag
- strict-prior receiving-identity evidence:
  - prior games
  - prior targets/game
  - prior receptions/game
  - prior target share
  - prior RB-room share
  - last-8 targets/game
  - last-8 receptions/game
  - last-8 target share
  - last-8 RB-room share
  - previous-season targets/game
  - previous-season receptions/game
  - previous-season target share
  - previous-season RB-room share
  - same-team prior targets/game
  - same-team prior RB-room share
  - previous-season availability.

No missing feature may be silently imputed to a nonzero role value. Missing strict-prior evidence must remain distinguishable from true zero where source semantics permit.

## Required team-week vacancy aggregates

For every vacancy-active team-week, aggregate source-only quantities including:

- exits count
- current room size
- prior room size
- max and sum exited prior targets/game
- max and sum exited prior receptions/game
- max and sum exited prior RB-room share
- max and sum exited last-8 targets/game
- max and sum exited last-8 receptions/game
- max and sum exited last-8 RB-room share
- max exited previous-season targets/game
- max exited previous-season RB-room share
- whether any exited player has prior depth evidence
- best available prior depth order among exited players
- exited-player history coverage count/rate.

## Predeclared descriptive bins

These bins are **source inventory bins only**. They do not become a candidate gate or validated threshold merely because they are reported.

### Exited strict-prior targets/game

- no prior receiving history
- `0`
- `(0, 1]`
- `(1, 2]`
- `> 2`

### Exited strict-prior RB-room receiving share

- unavailable/no history
- `< 0.10`
- `[0.10, 0.25)`
- `[0.25, 0.50)`
- `>= 0.50`

### Exited last-8 targets/game

- unavailable/no history
- `0`
- `(0, 1]`
- `(1, 2]`
- `> 2`

### Prior depth order

If `depth_team` can be parsed as an ordinal without outcome information:

- 1
- 2
- 3+
- unavailable/unparseable.

No threshold may be changed after this audit based on later outcome performance.

## Required source diagnostics

Report by season and separately Week 1 / Weeks 2+:

- vacancy team-weeks
- exited-player rows
- strict-prior receiving-history coverage
- prior-depth coverage
- distribution of exits count
- distribution of the predeclared targets/game bins
- distribution of the predeclared RB-room-share bins
- distribution of last-8 targets/game bins
- distribution of prior-depth bins
- fraction of vacancy team-weeks where at least one departed player has positive receiving history
- fraction where at least one departed player exceeds each predeclared descriptive receiving-history band.

Also report whether later-season vacancies differ structurally from Week 1 in the source-only state. This is descriptive only.

## Integrity gates

All must pass before any target-game outcome study can be authorized:

1. target-game outcomes loaded = `0`;
2. target-game participation loaded = `0`;
3. sportsbook inputs = `0`;
4. same-week historical depth used = `false`;
5. current/prior roster statuses restricted to canonical ACT/INA;
6. target/prior snapshot ordering is strictly earlier for every target row;
7. exited identity is present in prior room and absent in current room for every persisted exit row;
8. receiving identity is generated with strict-as-of semantics;
9. vacancy team-week coverage against the R26 room-state definition >= `99.5%`;
10. production protected-file diff is clean.

## Disposition

- `EXIT_SIGNIFICANCE_SOURCE_READY`: every integrity gate passes and meaningful strict-prior receiving-role evidence has sufficient coverage to support a separately frozen retrospective outcome diagnostic.
- `EXIT_SIGNIFICANCE_SOURCE_INSUFFICIENT`: integrity passes but coverage/semantics are too weak to support a meaningful exit-significance test.
- `EXIT_SIGNIFICANCE_SOURCE_FAILURE`: any leakage/source/integrity gate fails.

A READY result authorizes only a new frozen outcome diagnostic. It does not authorize a model, router, prospective shadow, or production change.

## Required artifacts

- frozen plan + SHA256
- source-audit implementation + SHA256
- exited-player source-state CSV
- vacancy team-week aggregate CSV
- season/phase source summary CSV
- source integrity/disposition JSON
- exact run/job/commit/artifact lineage
