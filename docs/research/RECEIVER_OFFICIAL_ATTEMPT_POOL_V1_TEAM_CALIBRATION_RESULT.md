# Receiver Official-Attempt Pool V1 — Team Calibration Result

Date: 2026-09-25

Disposition: **RECEIVER_OFFICIAL_ATTEMPT_POOL_V1_TEAM_CALIBRATION_FAILED_CLOSED**

The exact frozen candidate failed its full gate. It is not eligible for
player-level integration and receives no rescue.

## Authority

- branch: `research-receiver-official-attempt-pool-v1-calibration`
- run: `36142802477`
- job: `108096474184`
- head: `bbb97d5705a91d44a7e0389b602764c07d5bd915`
- artifact: `10868246439`
- digest: `sha256:aa08bef81113d082f80b246eabe8bff091a916927c2ab7f8a681a39cb0fd3a4b`
- candidate variants: 1
- parameters fit: 0
- sportsbook inputs: 0
- target-game outcomes upstream: 0
- scored team-games: 1088
- strict-prior conversion provenance: 1088 / 1088

## Frozen candidate

Baseline:
`projected_dropbacks = projected_plays * 0.57`

Candidate:
`projected_official_attempts = projected_dropbacks * strict_prior_pass_attempts_per_dropback`

## Official-pass-attempt result — failed

2024:
- MAE `6.421701 -> 6.707473` — worse

2025:
- MAE `6.497062 -> 6.449047` — slightly better

Pooled:
- MAE `6.459381 -> 6.578260` — worse
- p90 AE `13.092315 -> 13.761984` — worse
- absolute bias `1.3662 -> 2.7785` — worse
- changed-row candidate closer rate: **47.98%**

The frozen official-attempt gates therefore failed.

## Team-target result — all target-side gates passed

Despite failing official-attempt calibration, the same conversion improved
realized **team receiver targets** in both seasons.

2024:
- target MAE `6.505208 -> 6.171176`

2025:
- target MAE `6.671869 -> 5.972131`

Pooled:
- MAE `6.588539 -> 6.071654`
- p90 AE `13.150607 -> 12.618877`
- absolute bias `2.86256 -> 1.28222`
- changed-row candidate closer rate: **56.53%**

Every preregistered team-target gate passed.

This does not rescue the failed candidate. It is discovery evidence for a
different semantic quantity: **receiver targetable dropbacks are not identical
to either all dropbacks or official pass attempts.**

## Important architecture discovery from the trace

Every historical team-game in this calibration received exactly:

`projected_dropback_rate = 0.57`

There was zero team/game variation in 2024 or 2025.

Pooled baseline-dropback forecast vs realized dropbacks:
- MAE: `7.01858`
- bias: approximately `-3.04093`
- correlation: approximately `0.0636`

Source audit confirms this is not merely a historical harness fallback.
Production `scripts/modeling/rules_v2.py::project_game_script()` currently
hardcodes:

`pass_share = 0.57`

The fixed 57% rule is intentional Migration-21 production behavior.

## Prior calibration archaeology

This rule was promoted after legitimate earlier research:

- Migration 18 showed simplified pass-tendency structures materially beat the
  former `55% + PROE + lead/trail` rule.
- Historical team-tendency variants improved receiving/reception MAE but carried
  pass-yard bias and rushing tradeoffs.
- Migration 20 compared fixed 53-57% plus small team-identity adjustments.
- On the 2025 walk-forward objective, fixed 57% produced the best pass-yards MAE
  (`62.333007`) and passed downstream guardrails.
- tiny team-identity variants were worse on pass-yards and receiving/receptions.
- Migration 21 therefore promoted fixed 57%.

Thus fixed 57% was a calibrated stability choice, not an accidental constant.

Later QB/shared-opportunity research independently established that pass-
opportunity-rate error remains a major upstream residual source. Do not
reinterpret Migration 21 as proving all teams truly have identical pass
tendency.

## Scientific interpretation

The exact official-attempt-pool candidate is closed.

The target-side result supports a **new semantic hypothesis**:

`receiver target pool = targetable dropbacks`

where targetable dropbacks exclude:
- sacks;
- QB scrambles;
- official attempts that create no player target, such as throwaways/spikes and
  other non-target attempts under the stats provider's target semantics.

A new targetable-dropback study must be separately frozen and validated on an
independent time period before 2024-2025 are revisited.

## Anti-reinvention

Do not rescue this candidate by:
- relaxing the official-attempt gates;
- position/player carveouts;
- mixing hierarchical reconciliation;
- changing the 57% pass share inside this candidate;
- choosing a new attempt conversion window;
- adding pressure/scramble thresholds;
- changing rushing simultaneously.

The next lane, if pursued, is a distinct targetable-dropback conversion.
