# RB Opponent-Defender Injury Source Readiness V1 — Result

Status: **COMPLETE — SOURCE PARITY NOT CLEARED**  
Disposition: **SOURCE_PARITY_NOT_CLEARED**  
Production changed: **false**

## Authority

- branch: research-rb-opponent-defender-injury-source-v1
- frozen plan: docs/research/RB_OPPONENT_DEFENDER_INJURY_SOURCE_READINESS_V1_PLAN.md
- canonical successful audit run: 37494717496
- artifact: 11426662979
- digest: sha256:8c2d28a674b92d4462eb26cd4285306e711306ebdf56d82709808dc7a80b4f16
- raw / normalized rows: 18,934
- seasons: 2023-2026
- sportsbook inputs: 0
- game outcomes loaded: false
- predictive model fit: false

The prior red run 37494584450 was a workflow-expression/PYTHONPATH bug and
produced no scientific output.

## What the source does well

Raw nflverse injury data has strong role/identity coverage.

### 2024
- 6,215 injury rows
- 22 source weeks
- all 32 teams
- position complete: 99.81%
- GSIS identity complete: 100%
- broad defensive-front rows: 1,909
- defensive-front GSIS identity complete: 100%

### 2025
- 6,068 rows
- 22 source weeks
- all 32 teams
- position complete: 100%
- GSIS complete: 100%
- defensive-front rows: 1,861
- defensive-front GSIS complete: 100%

### 2026 through completed Week 4
- 1,052 rows
- Weeks 1-4 all present
- all 32 teams
- position complete: 100%
- GSIS complete: 100%
- defensive-front rows: 369
- defensive-front GSIS complete: 100%

Provider position vocabulary is stable and directly identifies LB/DT/DE groups
without fuzzy player-name inference.

## Frozen blocker

The preregistered source gate required >=95% report/game-status completeness on
defensive-front rows.

Observed:
- 2023: 45.07%
- 2024: **43.90%**
- 2025: **42.50%**
- 2026: **34.69%**

Therefore:
- 2024 gate: FAIL
- 2025 gate: FAIL
- 2026 gate: FAIL

Practice status is nearly complete:
- 2023 overall: 99.41%
- 2024: 99.42%
- 2025: 99.26%
- 2026: 99.81%

But the frozen V1 asked whether a role-specific opponent **game/report-status
availability** state could be constructed with historical/live parity. It
cannot.

Switching after this result to practice participation as the operative signal
would define a different scientific question and is not allowed as a rescue.

## Interpretation

The obstacle is not player identity or defensive position. It is provider
status semantics.

The nflverse injury table includes many practice-report rows for which a final
game-status designation is blank. Therefore an aggregate like "number of OUT
front-seven defenders" cannot be built over the whole injury population by
treating missing status as healthy, questionable or available.

Missing game status must remain missing.

The source also must not be reinterpreted as T-75 official-inactive authority.

## Disposition

SOURCE_PARITY_NOT_CLEARED

Do not:
- fit a defensive-injury coefficient;
- treat blank report status as ACTIVE;
- substitute practice status into this frozen V1;
- count all injury-report rows as unavailable;
- backfill official inactives;
- score 2026 outcomes against an invented proxy.

A genuinely new follow-up would require a separately justified source contract
with historical/live role-specific game-status or official-inactive coverage,
or a separately preregistered practice-participation hypothesis. Neither is
authorized by this result.
