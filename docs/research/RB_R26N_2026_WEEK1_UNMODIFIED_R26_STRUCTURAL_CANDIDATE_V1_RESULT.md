# RB R26N 2026 Week-1 Unmodified-R26 Structural Candidate V1 — Result

Status: COMPLETE — STRUCTURAL CANDIDATE PASSED
Date: 2026-09-09

## Canonical execution

- branch: `research-rb-r26n-2026-week1-unmodified-r26-structural-candidate-v1`
- successful workflow run: `34396075045`
- successful job: `102616001356`
- exact head SHA: `3b7a00e282cb925bd7a33175bc0b7d08d1467b2f`
- artifact: `10121598376`
- artifact name: `rb-r26n-2026-week1-unmodified-r26-structural-candidate-v1`
- artifact digest: `sha256:887929203053cb62904aaaeda9d995c9645163814da181972799f08fe4465c62`

## Frozen-study disposition

`R26N_2026_WEEK1_STRUCTURAL_CANDIDATE_PASS_READY_FOR_SHADOW_INTEGRATION_DESIGN`

All 28 frozen R26N structural gates passed.

This is a prospective structural-candidate result only. No 2026 game outcome was used and no live shadow or production promotion is authorized by R26N.

## Current 2026 Week-1 structural candidate

- production population: `468` players
- production teams: `32`
- production games: `16`
- RB/FB rows: `107`
- R26L vacancy teams: `31`
- exact non-vacancy team: `CIN`
- changed RB/FB rows: `104`
- R9 training season: `2025`
- R9 reliability: `1.0`
- R9 refit in R26N: `false`
- serialized R19/R9 inner model SHA-256: `9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba`

## Structural invariants

- maximum RB/FB room-pool gap: `0.0`
- maximum team-entitlement delta: `1.1102230246251565e-16`
- maximum non-RB/FB entitlement delta: `0.0`
- maximum non-vacancy RB/FB entitlement delta: `0.0`
- CIN remained exact baseline in the only non-vacancy room
- player universe remained exactly `468` before/after
- candidate entitlement remained finite and nonnegative
- strict-prior identity history stopped at time key `202518`

## Leakage / authority safeguards

- `2026_outcomes_used = 0`
- `sportsbook_football_inputs_used = 0`
- `same_week_depth_used = false`
- `r9_refit = false`
- `production_parameters_changed = false`
- `r22_changed = false`
- `receiving_yard_means_changed = false`
- `receiving_distribution_regenerated = false`
- `live_shadow_activation_authorized = false`
- `production_promotion_authorized = false`
- `shadow_integration_design_authorized = true`

## Exact parent authorities

R26M prospective qualification:
- run `34390505549`
- artifact `10119429741`
- digest `sha256:1306a3a2e58a0b129ac7e9fe34ad6407d87c491494e8fc27dd0284ba96996b76`
- required disposition `2026_WEEK1_UNMODIFIED_R26_SHADOW_CANDIDATE_DESIGN_QUALIFIED`

Current production Full Slate:
- run `34317211395`
- artifact `10090547415`
- digest `sha256:7eab77e41c5879d4f54d87497eee0d1186010784cfc4ef966178930e16fb7c3b`
- head `f8417f55b04ce0e19baf260e9d532765034c47f1`

R26L current Week-1 regime/vacancy state:
- run `34389455694`
- artifact `10119058769`
- digest `sha256:3351dfb5bbf6b571174a94ddf0a03179d70786edbb91d316e4fdf2c3cf005c46`

R19 prospective serialized R9 authority:
- run `34288244770`
- artifact `10080377483`
- digest `sha256:11432b9d7b7f2367935a862b63c30df9f40955e479806ab67d90921b63a907c7`
- inner model SHA-256 `9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba`

## Frozen plan / implementation hashes

From the successful workflow:
- plan SHA-256: `5033bbd7a8de619333f1a6c290d0eb574b830e3cff76dbb72fb6f062bebf4ff4`
- implementation lock SHA-256: `afa16ef8695088e83e0698a5656092a676d8c9de330220b7040e18f238a8f619`
- original frozen candidate builder SHA-256: `ddec40e2373611971572d7f11fd5e966660e4412256736c0a84fbab68c313e6a`

The original frozen candidate builder remained byte-identical through both mechanical repairs.

## Mechanical execution lineage

### Run-1 representation repair

First authoritative launch:
- run `34395291505`
- job `102613379938`
- head `3834b6765f60efc80566b7a28ecd081d6fe8fd00`

It passed the frozen contract, protected-production boundary, and parent checks, then failed before any scientific disposition because immutable PlayerForm used compact `player_clean_key` values while immutable model-context lacked that key and fell back to display names.

Repair intent was frozen first:
- `docs/research/RB_R26N_RUN1_MECHANICAL_REPAIR_V1.md`
- commit `8268327683e9d826c05d2d72a4467e353f4fa9f1`

Hash-tracked staging helper:
- `scripts/backtest/stage_r26n_production_identity_key_repair_v1.py`
- commit `0b4c2df7c14b16bd0e945b425aac7f35e015fa3d`

Successful-run repair audit proved:
- 468 source rows / 468 staged rows
- exact display-identity equality
- zero fuzzy matching
- zero normalization heuristics
- zero football-value changes
- zero players added/removed
- immutable downloaded parent untouched

### Second dtype-only compatibility repair

Repaired run `34395790276` passed the first repair, then failed before R9 scoring because pandas refused `string[python]` versus `object` as-of join-key dtypes.

Second repair intent was frozen first:
- `docs/research/RB_R26N_SECOND_MECHANICAL_REPAIR_V1.md`
- commit `dbfe863212dc5b2684ec1005d66102342bf87777`

Hash-tracked compatibility wrapper:
- `scripts/backtest/run_rb_r26n_with_identity_dtype_compat_v1.py`
- commit `9abdaa9a6262aa87ac601d7bc665209b94036b20`

It casts only identity join-key dtypes to plain Python object, verifies key values and all non-key values remain unchanged, and delegates to the original protected identity function and original frozen R26N builder.

Final successful launch commit:
`3b7a00e282cb925bd7a33175bc0b7d08d1467b2f`

## Scientific interpretation

R26N establishes that the already-qualified, unmodified R26 vacancy-gated R9 opportunity/reception mechanism can be materialized on the exact current 2026 Week-1 production football population while preserving every frozen structural conservation and leakage boundary.

It does **not** establish 2026 predictive accuracy, because no 2026 outcome exists in the study. It also does **not** authorize altering current R22 receiving-yard means/distributions or production.

## Authorized next step

R26N authorizes only a separately frozen downstream shadow-integration design study. That next study must prove how the R26N opportunity/reception overlay can coexist with the current production receiving-yard/R22 authority without silently moving protected receiving-yard means, distributions, markets, or production parameters.

No live shadow activation and no production promotion are authorized at this checkpoint.
