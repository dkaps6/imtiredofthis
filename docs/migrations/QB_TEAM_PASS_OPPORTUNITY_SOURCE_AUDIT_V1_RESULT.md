# QB Team Pass Opportunity Source Audit V1 — Result

## Disposition

One family is source-eligible for predictive preregistration:

`SCHEDULE_REST_CONTEXT`

The other two V1 families remain ineligible under the exact frozen audit gate and are not carried into the next predictive experiment.

## Canonical lineage

- Branch: `research-qb-team-pass-opportunity-source-audit-v1`
- Frozen source-audit plan: `f7ec18db7cd122910647a3f9d190398cee1b5a57`
- Evaluator commit: `3e8f6af85976f0eb70513bf66da1746a10dda0b5`
- Tested/workflow head: `aef74b4565f76d0f888ee4c05a4ec31e759add01`
- GitHub Actions run: `34533408818`
- Job: `103059297754`
- Artifact: `10174433662` (`qb-team-pass-opportunity-source-audit-v1`)
- Artifact digest: `sha256:47b57a1398e184c0ca1da29e72fa3be1fbf64bbf8c98b8d17e1da539873e2515`
- Parent QB opportunity-chain result commit: `43bc0a0db245f401ec270bbe48b6d8812315139b`
- Parent disposition: `TEAM_PASS_OPPORTUNITY_PRIMARY_DIAGNOSTIC`

## Integrity / source boundary

- target M89 QB-game identifiers: `884`
- target outcomes or parent residuals loaded by the source auditor: `false`
- sportsbook inputs used: `false`
- model fitting used: `false`
- production changed: `false`
- PBP source: `nflreadpy.load_pbp`
- schedule source: `nflreadpy.load_schedules`

Regular-season source rows loaded:

| Source | 2023 | 2024 | 2025 |
|---|---:|---:|---:|
| nflreadpy.load_pbp | 47,399 | 47,274 | 46,452 |
| nflreadpy.load_schedules | 272 | 272 | 272 |

## Family results

### 1. SCHEDULE_REST_CONTEXT

Disposition:

`SOURCE_ELIGIBLE_FOR_PREDICTIVE_PREREGISTRATION`

All frozen source gates passed.

- required seasons loaded: PASS
- core schema/coverage >=95%: PASS
- no sportsbook/result field required: PASS
- target-game outcome not required: PASS
- strict-prior/pregame constructible: PASS
- 2024-2025 target coverage >=95%: PASS (`1.0`)
- unique team-game keys: PASS
- materially distinct from prior closed QB ledger: PASS
- source semantics explicit: PASS

Eligible target-game fields are limited to deterministic schedule context:

- `gameday`
- `weekday`
- `home_team`
- `away_team`
- `home_rest`
- `away_rest`

The schedule table also contains score/result/sportsbook fields, but they were explicitly prohibited and unused by the audit. This family therefore remains football-only/pregame-safe for a separately preregistered predictive screen.

### 2. PENALTY_DRIVE_EXTENSION

Disposition:

`SOURCE_INELIGIBLE_SCHEMA_OR_COVERAGE`

The family had complete strict-prior history availability (`1.0`) and explicit safe semantics for `first_down_penalty`, but failed the exact frozen V1 core raw-field coverage gate.

The result is preserved as-is. V1 does not reinterpret the gate after inspecting field-specific coverage.

Important semantic boundary retained:

- `first_down_penalty` is safe as the explicit nflfastR indicator that a penalty converted a first down;
- generic `penalty` establishes that a penalty occurred;
- generic accepted/declined penalty rate was **not** claimed without an explicit accepted/declined field contract.

This family is not authorized for the next predictive test.

### 3. FOURTH_DOWN_AGGRESSION

Disposition:

`SOURCE_INELIGIBLE_SCHEMA_OR_COVERAGE`

The family had complete strict-prior history availability (`1.0`) and explicit fourth-down attempt outcome semantics, but failed the exact frozen V1 core raw-field coverage gate.

The result is preserved as-is. The gate is not loosened post-result.

This family is not authorized for the next predictive test.

## Officiating crew

Disposition remains:

`QUARANTINED_NO_HISTORICAL_PREGAME_PROVENANCE`

No actual-game official identity is treated as a pregame feature by assumption.

## Scientific meaning

The parent diagnostic established that upstream team pass opportunity/dropback volume is the primary remaining attempt mechanism and carries nearly all of the shared QB/WR opportunity miss.

V1 source audit found exactly one new information family clean enough to test without reopening prior work: deterministic schedule/rest context.

This result says **nothing yet** about whether rest/schedule context predicts the team-pass-opportunity residual. No target residual was opened in this migration.

## Immediate next authorized step

Freeze one development-only predictive screen for `SCHEDULE_REST_CONTEXT` against the already-fixed M89 team-pass-opportunity baseline.

The predictive screen must:

- use only the source-eligible schedule/rest fields;
- use no sportsbook or result fields;
- preserve the M89/M90 production model unchanged;
- use a temporal development split with 2025 untouched during candidate selection;
- fit one fixed model architecture only;
- test the upstream team-pass-opportunity residual first, not direct passing-yard residuals;
- authorize a separate untouched-2025 confirmation only if the frozen development gates pass.
