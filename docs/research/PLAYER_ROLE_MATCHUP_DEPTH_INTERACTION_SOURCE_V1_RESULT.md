# PLAYER ROLE MATCHUP — DEPTH INTERACTION SOURCE V1 RESULT

Date: 2026-10-11

Status: **SOURCE QUALIFIED. PLAYER-SPECIFIC RECEIVER DEPTH x OPPONENT DEPTH-DEFENSE STATE IS HISTORICALLY RECONSTRUCTABLE. NO PREDICTIVE CLAIM YET.**

Frozen plan:
- `docs/research/PLAYER_ROLE_MATCHUP_DEPTH_INTERACTION_SOURCE_V1_PLAN.md`

Canonical authority:
- workflow run: `38189258833`
- exact head: `6eaaa793ec18cdd7b116c54db8d7943cebb40a2a`
- artifact: `11721146177`
- artifact digest: `sha256:8e1227f2b27ac1b42c1790e1666302dbfcecf59b643df8807aa56baa3a540fcc`
- workflow conclusion: `SUCCESS`

## Frozen disposition

`PLAYER_DEPTH_MATCHUP_SOURCE_QUALIFIED`

## Coverage

Target universe:
- 2024 and 2025 regular seasons
- Weeks 2-18
- target-week weekly-roster WR / TE identities

Observed:
- universe rows: **9,677**
- stable GSIS rows: **9,677**
- stable GSIS coverage: **100.0%**
- player-profile-ready rows: **7,184**
- defense-ready on player-profile rows: **7,184**
- player-specific matchup-state-ready rows: **7,184**
- defense support on qualified player profiles: **100.0%**

By season:
- 2024: **3,518**
- 2025: **3,666**

By position:
- WR: **4,666**
- TE: **2,518**

## Integrity

- max player depth-band mass gap:
  `2.220446049250313e-16`
- same/future player feature violations: **0**
- same/future defense feature violations: **0**
- sportsbook inputs used: **0**
- football outcomes / model residuals scored: **0**
- candidate models fit: **0**
- production changed: **false**

All frozen source-readiness gates passed.

## What this proves

For an individual target-week WR or TE, we can reconstruct before the game:

1. that player's own strictly-prior target-depth distribution across:
   - behind LOS,
   - short,
   - intermediate,
   - deep;
2. the opponent defense's strictly-prior completion rate and receiving YPT
   allowed in those same depth bands;
3. a player-specific interaction that weights the defense by how this particular
   player has actually been used.

This is not direct WR-CB assignment and does not claim shadow coverage.

## Anti-retest distinction

This does not reopen:
- generic defense-vs-position multipliers;
- Football Matchup Transmission V1 simple integrations;
- team Coverage-v2;
- historical direct WR-CB assignment;
- M72 aggregate receiving-weapons x aggregate defense for QB residuals.

The qualified state is individual-receiver-specific.

## Next legal move

Freeze a separate predictive diagnostic **before reading receiving outcomes**.

That diagnostic must:
- preserve the existing player opportunity / target entitlement baseline;
- score WR and TE separately;
- test incremental player receiving value rather than QB residuals;
- require replicated value in both 2024 and 2025;
- use no coefficient, depth-band, subgroup, or threshold search after scoring;
- fail closed if the exact frozen candidate does not replicate.

Source qualification alone authorizes no production change.
