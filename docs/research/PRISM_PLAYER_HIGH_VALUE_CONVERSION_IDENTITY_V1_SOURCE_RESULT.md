# PRISM Player High-Value Conversion Identity V1 — Source / Support Result

Date: 2026-10-17

Frozen plan:
`docs/research/PRISM_PLAYER_HIGH_VALUE_CONVERSION_IDENTITY_V1_SOURCE_PLAN.md`

Canonical run:
- run `39093918886` — **SUCCESS**
- exact head `5b517c1ac71a1f67d24cddf510c0f7ef5fe191b9`
- artifact `12123727812`
- digest `sha256:ee5f7f5641725a19a5e9c1f33d82a6544c637e8da9e57ad11b4f8bcfdfe36c7d`

Overall disposition:

`PRISM_PLAYER_HIGH_VALUE_CONVERSION_IDENTITY_SOURCE_READY_PARTIAL`

## Frozen support gate

The predeclared gate required, separately for 2024 and 2025:
- >=3 strictly-prior same-player high-value opportunities;
- >=350 supported target player-games;
- >=40 distinct supported named players.

No gate was changed after inspection.

## Channel adjudication

### RB rushing inside 10 — SOURCE READY

2024:
- supported player-games: **513**
- supported named players: **84**

2025:
- supported player-games: **509**
- supported named players: **86**

Native `rush_touchdown` semantics:
- finite coverage 100%;
- binary labels true;
- duplicate player-play contradictions 0.

Disposition:
`RB_RUSH_INSIDE10_PLAYER_CONVERSION_SOURCE_READY`.

### RB receiving red zone — SOURCE BLOCKED

2024:
- supported player-games: **250**
- supported named players: **65**

2025:
- supported player-games: **251**
- supported named players: **70**

The named-player breadth gate passed, but the frozen >=350 supported
player-game gate failed in both seasons.

Native `pass_touchdown` semantics themselves were clean:
- finite coverage 100%;
- binary labels true;
- receiving-TD-without-complete-pass contradictions 0.

Disposition:
`RB_RECEIVING_REDZONE_PLAYER_CONVERSION_SOURCE_BLOCKED`.

Do **not** lower the player-game support gate, lower the >=3 prior-opportunity
history floor, pool RB receiving with another position, widen the red-zone band
or use all RB targets after seeing this result.

### TE receiving red zone — SOURCE READY

2024:
- supported player-games: **372**
- supported named players: **76**

2025:
- supported player-games: **407**
- supported named players: **90**

Native `pass_touchdown` semantics:
- finite coverage 100%;
- binary labels true;
- receiving-TD-without-complete-pass contradictions 0.

Disposition:
`TE_RECEIVING_REDZONE_PLAYER_CONVERSION_SOURCE_READY`.

## Event inventory

2024:
- RB inside-10 rushes: 1,119; TDs 294;
- RB red-zone targets: 385; TDs 68;
- TE red-zone targets: 595; TDs 155.

2025:
- RB inside-10 rushes: 1,078; TDs 275;
- RB red-zone targets: 410; TDs 83;
- TE red-zone targets: 665; TDs 194.

## Scientific interpretation

This result does **not** say player conversion identity predicts future TD
conversion.

It says the source/support contract is strong enough to score exactly two
separate named-player conversion cells:
- RB rushing inside 10;
- TE receiving red zone.

RB receiving is source-blocked before predictive scoring.

The question remains downstream of opportunity entitlement:

`HIGH-VALUE OPPORTUNITY EXISTS -> NAMED-PLAYER CONVERSION IDENTITY`.

It is not a rescue of the closed recent high-value-share mechanism and it does
not certify ATD.

## Integrity

- predictive candidates scored: 0
- 2026 outcomes read: 0
- sportsbook inputs: 0
- OddsAPI calls: 0
- production mutation: false
