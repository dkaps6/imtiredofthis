# RB PLAYER ROLE MATCHUP — INDIVIDUAL RUN-LOCATION SOURCE V1 RESULT

Date: 2026-10-11

Status: **SOURCE QUALIFIED. INDIVIDUAL RB RUN-LOCATION x OPPONENT LOCATION-DEFENSE STATE IS HISTORICALLY RECONSTRUCTABLE. NO PREDICTIVE CLAIM YET.**

Frozen plan:
- `docs/research/RB_INDIVIDUAL_RUN_LOCATION_MATCHUP_SOURCE_V1_PLAN.md`

Canonical authority:
- workflow run: `38189983578`
- exact head: `3fc811bac17881e8bef651dfb6c7d35f348c0be7`
- artifact: `11720966759`
- artifact digest: `sha256:dec7d48d1a2c39ef255a6391b0014f75779ceb5939dfc7a90feca384074caad5`
- workflow conclusion: `SUCCESS`

Disposition:
`RB_INDIVIDUAL_RUN_LOCATION_MATCHUP_SOURCE_QUALIFIED`

## Coverage

Target universe:
- 2024 + 2025 regular seasons
- Weeks 2-18
- target-week weekly-roster RB/HB/FB identities

Observed:
- universe rows: **3,993**
- stable GSIS rows: **3,993**
- stable GSIS coverage: **100.0%**
- player-profile-ready rows: **2,654**
- defense-ready on player-profile rows: **2,654**
- source-ready matchup rows: **2,654**
- defense support on qualified player profiles: **100.0%**

By season:
- 2024: **1,343**
- 2025: **1,311**

Integrity:
- max player LEFT/MIDDLE/RIGHT mass gap:
  `2.220446049250313e-16`
- same/future player feature violations: **0**
- same/future defense feature violations: **0**
- sportsbook inputs used: **0**
- football outcomes / residuals scored: **0**
- candidate models fit: **0**
- production changed: **false**

All frozen source-readiness gates passed.

## What this proves

For an individual target-week RB, before the game, public data can reconstruct:

1. that player's own strictly-prior rushing-location mix:
   - LEFT
   - MIDDLE
   - RIGHT
2. the opponent defense's strictly-prior YPC and success rate allowed in the
   same three locations;
3. one player-specific interaction built from the RB's actual directional role.

This is not the closed M95B team-level directional interaction.

## Next legal move

Freeze a separate predictive diagnostic BEFORE reading RB rushing outcomes.

That test must:
- preserve the frozen RB carry/snap opportunity mechanism;
- change only the declared per-carry rushing-efficiency seam;
- score source-ready RB rushing yards in 2024 and 2025 independently;
- require replication in both seasons;
- use no fitted coefficient, location threshold, role carveout, or subgroup
  search;
- fail closed if the exact candidate does not replicate.

Source qualification alone authorizes no production change.
