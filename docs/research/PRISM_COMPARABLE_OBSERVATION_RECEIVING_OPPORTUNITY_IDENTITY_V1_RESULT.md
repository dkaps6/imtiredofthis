# PRISM Comparable-Observation Receiving Opportunity Identity V1 — Result

Date: 2026-10-18

Status: **CLOSED / NAMED-PLAYER RULE DID NOT QUALIFY / NO PRODUCTION CHANGE**

Model: **PRISM — Player Role & Individual Situation Model**

Frozen contract:
`docs/research/PRISM_COMPARABLE_OBSERVATION_RECEIVING_OPPORTUNITY_IDENTITY_V1_FREEZE.md`

Freeze:
`43e1c81a177e47118415d02fc072758e22f50825`

Canonical execution:
- run `39161342922` — SUCCESS;
- exact head `bcbeffa8dbebb869eadb004cde4a2032c417bf0f`;
- artifact `12148837679`;
- artifact name `prism-comparable-observation-receiving-identity-v1-39161342922`;
- digest `sha256:ac6272b41632389d4a5d996cef53ff7aae539e4ada034287bfbed57171ba24be`;
- strict repository audit: PASS.

Disposition:

`PRISM_COMPARABLE_OBSERVATION_RECEIVING_OPPORTUNITY_IDENTITY_V1_CLOSED`

## Player-first question

For an affected named RB/WR/TE, should an injury-corroborated
partial-participation game be completely prevented from updating ordinary
current-season PlayerForm receiving-role identity?

The candidate changed only that named player's current-season target-share
history. It did not use a position correction.

## 2024 development

Directly affected:
- target games: **362**;
- named players: **64**.

Named-player result:
- equal-player average closeness change: **-0.000428** target-share units;
- median named-player change: **-0.000867**;
- improved: **35.94%**;
- tied: **1.56%**;
- worsened: **62.50%**;
- p10 / p25 / p75 / p90:
  **-0.005681 / -0.002660 / +0.000830 / +0.003541**.

The frozen player-level development gates failed.

## 2025 independent confirmation

Directly affected:
- target games: **293**;
- named players: **62**.

Named-player result:
- equal-player average closeness change: **-0.000546** target-share units;
- median named-player change: **-0.000483**;
- improved: **35.48%**;
- tied: **0.00%**;
- worsened: **64.52%**;
- p10 / p25 / p75 / p90:
  **-0.004275 / -0.002552 / +0.000670 / +0.003779**;
- player-cluster 95% interval:
  **[-0.001321, +0.000259]**.

The frozen independent player-level gates failed.

## Interpretation

The exact rule is closed:

> Completely deleting an injury-corroborated partial-participation game from
> current-season receiving-role identity is not a reliable general solution for
> affected named players.

This does **not** mean the partial game should be treated as an ordinary full
game.

The source evidence remains valid:
- the player's participation materially collapsed relative to his own prior role;
- injury chronology can corroborate many such collapses;
- current PlayerForm admits many of those games as ordinary historical rows.

The predictive result says the partial game still contains useful player
information, so erasing the entire observation loses too much information for
most affected players.

## Next distinct player-first frontier

Separate:

1. **participation exposure** — how much of the offense the named player was
   actually on the field for;
2. **opportunity earning conditional on that exposure** — how often the named
   player earned targets while participating.

A partial game can contain strong opportunity-earning evidence even when its
full-game target total is small.

This is distinct from:
- deleting the game;
- target-share trajectory;
- Stage-1 any-snap participation;
- Stage-2 finite active-set allocation;
- WR-R15 / TE-R5P snap features.

Before any predictive formula is scored, prove that exact named-player weekly
target evidence can be paired with source-valid offensive participation
exposure at sufficient coverage and without calling snaps routes.

## Anti-rescue

Do not:
- change the injury-collapse thresholds;
- change the history window;
- partially weight the excluded row as a rescue of this V1;
- subset WR1 / TE1 / lead backs;
- use position-specific coefficients;
- use sportsbook inputs;
- use 2026 outcomes.

Any future participation-conditioned opportunity mechanism must be frozen as a
new family.

## Integrity

- abnormal target games excluded from ordinary-role scoring: **596**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- sportsbook inputs: **0**;
- OddsAPI calls: **0**;
- production changed: **false**.
