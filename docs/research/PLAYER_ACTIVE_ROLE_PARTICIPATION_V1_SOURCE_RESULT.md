# Player Active-Role Participation V1 — Source Audit Result

Date: 2026-10-16

Status: **SOURCE READY ALL / MECHANISM FREEZE AUTHORIZED / RESEARCH ONLY**

Version:
`PLAYER_ACTIVE_ROLE_PARTICIPATION_V1_SOURCE_AUDIT`

Canonical corrected run:
- run `38895205962` — SUCCESS
- job `116953914726`
- exact head `28ebf8d7159fdd1fec8c2e4a0fe1c9e2c691537c`
- artifact `12044155086`
- digest `sha256:a42322579aee682497223c7f53bce0b14a96dc6f466558e811770e681a035dad`

The corrected head explicitly excludes bye roster weeks before source adjudication.

Frozen source contract:
`docs/research/PLAYER_ACTIVE_ROLE_PARTICIPATION_V1_SOURCE_AUDIT.md`

## Overall disposition

`PLAYER_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY_ALL`

Per-position:
- QB: `QB_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY`
- RB: `RB_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY`
- WR: `WR_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY`
- TE: `TE_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY`

No predictive model has been scored yet.

## Source authority

Historical seasons:
- 2024
- 2025

Pregame universe:
- weekly roster authority;
- ACT status where available;
- stable GSIS identity;
- exact M78 official game-day inactives removed;
- player remains eligible even when target-game offensive snaps are zero.

Official inactive archive:
- 1,088 team-weeks total;
- 544 in each season;
- SHA-256:
  `d39aaf0feea101f3e0d2721ebd4118ef33fb1a4d3c76670e2a4f17734e37b609`.

Participation source:
- historical PFR offensive snap counts;
- absent player row is treated as zero only when the team-week snap source itself is valid.

## Exact source result

Every frozen integrity/support gate passed in every position and both seasons:
- roster team-week coverage 100%;
- snap team-week coverage 100%;
- stable roster identity 100%;
- positive-snap player -> roster bridge 100%;
- chronology violations 0;
- official-inactive -> roster identity bridge above the frozen 95% floor;
- all position-specific row/player/prior-history support floors passed.

### QB

2024:
- 1,097 pregame player-games;
- 90 distinct QBs;
- 681 positive-snap rows;
- 416 valid zero-snap rows;
- positive participation rate 62.08%;
- 73 QBs with >=4 prior same-team universe games;
- 65 with >=8.

2025:
- 1,107 player-games;
- 95 distinct QBs;
- 687 positive-snap rows;
- 420 valid zero-snap rows;
- positive participation rate 62.06%;
- 79 with >=4 prior same-team games;
- 71 with >=8.

### RB

2024:
- 1,845 pregame player-games;
- 166 distinct RB/FB/HB;
- 1,612 positive-snap rows;
- 233 valid zero-snap rows;
- positive participation rate 87.37%;
- 126 players with >=4 prior same-team games;
- 107 with >=8.

2025:
- 1,866 player-games;
- 170 distinct RB/FB/HB;
- 1,631 positive-snap rows;
- 235 valid zero-snap rows;
- positive participation rate 87.41%;
- 139 with >=4;
- 125 with >=8.

### WR

2024:
- 2,834 pregame player-games;
- 253 distinct WRs;
- 2,651 positive-snap rows;
- 183 valid zero-snap rows;
- positive participation rate 93.54%;
- 209 WRs with >=4 prior same-team games;
- 163 with >=8.

2025:
- 2,817 player-games;
- 253 distinct WRs;
- 2,627 positive-snap rows;
- 190 valid zero-snap rows;
- positive participation rate 93.26%;
- 217 with >=4;
- 198 with >=8.

### TE

2024:
- 1,730 pregame player-games;
- 143 distinct TEs;
- 1,664 positive-snap rows;
- 66 valid zero-snap rows;
- positive participation rate 96.18%;
- 123 TEs with >=4 prior same-team games;
- 102 with >=8.

2025:
- 1,715 player-games;
- 146 distinct TEs;
- 1,646 positive-snap rows;
- 69 valid zero-snap rows;
- positive participation rate 95.98%;
- 132 with >=4;
- 119 with >=8.

## Scientific interpretation

The source is rich enough to test a separate player-level **active-role / zero-state** mechanism.

This is important because the full-roster replay showed the model can assign material opportunity to players who are legitimately in the pregame roster universe but do not have a real offensive role in that game.

The next authorized question is exactly:

> Does this named player's own strictly-prior participation history forecast whether he records any offensive snap in the target game better than a strictly-prior position-level participation baseline?

That mechanism must be frozen before scoring.

Stage 1:
**active-role participation probability**.

Stage 2:
**conditional opportunity allocation given participation**.

Do not collapse the two.

## Anti-retest / no rescue

Do not relabel this as:
- Availability -> Opportunity rule-order repair;
- M78/M79 generic inactive QB identity;
- RB Vacancy;
- teammate-availability role elasticity V1;
- RB carry/snap allocation;
- RB receiving-room share;
- WR/TE target-share trajectory;
- WR-R15 / TE-R5P snap entitlement;
- route-participation/YPRR source work.

Do not use:
- target-game weekly-stat presence to define the pregame universe;
- target-game snap percentage as a predictor;
- sportsbook data;
- OddsAPI;
- 2026 outcomes.

## Boundaries

- sportsbook inputs: 0
- OddsAPI calls: 0
- 2026 outcomes: 0
- predictive models fit: 0
- projection values changed: 0
- production changed: false

Canonical model status remains:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.
