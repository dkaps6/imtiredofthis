# PRISM Player Participation-Anomaly State V1 — Source Result

Date: 2026-10-18

Status: **SOURCE READY / PLAYER-SPECIFIC STATE ONLY / NO PERFORMANCE SCORING**

Model: **PRISM — Player Role & Individual Situation Model**

Frozen source plan:
`docs/research/PRISM_PLAYER_PARTICIPATION_ANOMALY_STATE_V1_SOURCE_PLAN.md`

Freeze:
`fd787e681aa5c7b6b55c5f8d0dc1d67ec4651b32`

Canonical execution:
- run `39160554692` — **SUCCESS**;
- exact head `a9308535acf72a90f2b656f833f854cb372a34f8`;
- artifact `12147549966`;
- artifact name `prism-player-participation-anomaly-state-v1-source-39160554692`;
- digest `sha256:e7753e981f0ecaaf23289280830fd32b9d733402c02ab9aaa0116fc929450449`;
- strict repository audit: PASS.

Disposition:

`PRISM_PLAYER_PARTICIPATION_ANOMALY_STATE_V1_SOURCE_READY`

## What this source now knows

For a positive-participation player-game, PRISM can compare that named player
only to his own same-team, same-season strictly-prior snap role.

Frozen baseline:
- most recent up to 4 positive-participation games;
- minimum 3;
- median offensive snap percentage.

Frozen major participation collapse:
- current snap retention <= 0.65 of the player's own baseline;
- AND absolute snap-percentage drop >= 0.20.

The source then keeps injury causation separate:
- current-week injury-report corroboration;
- next-scheduled-game injury-report corroboration, with the week at which that
  information became available preserved;
- unresolved collapse when neither source corroborates.

Position is inventory only. It is not the rule.

## 2024

- positive-participation rows: **6,577**;
- player-baseline-supported rows: **4,806**;
- insufficient-baseline rows: **1,771**;
- injury-corroborated major participation collapses: **139**;
- unresolved major participation collapses: **182**;
- current-week injury corroboration: **84**;
- next-scheduled-game injury corroboration: **55**;
- median player prior snap percentage on supported rows: **0.515**;
- median current snap percentage among collapse rows: **0.210**;
- median retention ratio among collapse rows: **0.390**.

All frozen source gates passed.

## 2025

- positive-participation rows: **6,551**;
- player-baseline-supported rows: **4,750**;
- insufficient-baseline rows: **1,801**;
- injury-corroborated major participation collapses: **149**;
- unresolved major participation collapses: **195**;
- current-week injury corroboration: **85**;
- next-scheduled-game injury corroboration: **64**;
- median player prior snap percentage on supported rows: **0.505**;
- median current snap percentage among collapse rows: **0.220**;
- median retention ratio among collapse rows: **0.392**.

All frozen source gates passed.

## Meaning

A positive box-score observation is not automatically an ordinary role
observation.

PRISM can now source-validly distinguish:
- normal/noncollapsed positive participation;
- a major participation collapse with injury chronology supporting the
  interpretation that the row may be contaminated;
- a major participation collapse whose cause remains unresolved;
- insufficient personal history.

This is the state layer required to prevent an injury-shortened positive-usage
game from silently becoming ordinary evidence of role decline.

## Important as-of protection

When corroboration comes from the next scheduled team-game injury report, the
earlier game may be retrospectively treated as injury-corroborated only for a
future target whose week is strictly after the corroboration week.

No future information is backdated.

## What this does not authorize

This result does not:
- change PlayerForm;
- drop a historical game;
- change a target/carry/yard projection;
- call every injury-report game limited;
- call every unresolved collapse injury-caused;
- fit an injury penalty;
- change production.

## Exact next action

Audit the actual PlayerForm history-ingestion seam.

Question:

> Which exact named-player observations are currently admitted as ordinary
> PlayerForm historical evidence even though this source now marks them as an
> injury-corroborated participation collapse or an unresolved participation
> anomaly?

That bridge must remain source-only first.

## Integrity

- performance outcomes loaded for scoring: **0**;
- predictive relationships scored: **0**;
- projection errors scored: **0**;
- coefficients fit: **0**;
- thresholds searched after freeze: **0**;
- 2026 outcomes used: **0**;
- sportsbook/OddsAPI used: **0**;
- production changed: **false**.
