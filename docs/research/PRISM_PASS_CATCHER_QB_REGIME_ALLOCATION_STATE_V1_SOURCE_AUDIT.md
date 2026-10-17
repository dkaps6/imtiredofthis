# PRISM Pass-Catcher × QB-Regime Allocation State V1 — Frozen Source Audit

Date: 2026-10-17

Status: **FROZEN SOURCE/LINEAGE AUDIT BEFORE ANY PREDICTIVE SCORING**

Model:
**PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_PASS_CATCHER_QB_REGIME_ALLOCATION_STATE_V1_SOURCE_AUDIT`

## 1. Player-first question

PRISM must understand the named pass catcher in his actual current offensive
situation.

This source audit asks only:

> Before kickoff, can PRISM identify a leakage-safe current QB regime and recover
> enough strictly-prior history to describe how this named WR/TE was allocated
> targets when that QB regime was active, separately from his generic recent
> target-share state?

This is **not**:
- pair-specific YPT/efficiency;
- a team receiver-room target-rate model;
- C1/C3 target-mass calibration;
- WR/TE target-share trajectory;
- a mean correction;
- a QB projection model.

No predictive candidate is scored in this audit.

## 2. Anti-retest reconciliation

Protected closed/occupied work remains protected:

- raw QB-receiver pair **YPT** is CLOSED;
- C1 group target-mass and C3 broad QB/receiver combination are CLOSED;
- Receiver Room Targets-per-Play V1 is CLOSED;
- Offensive Regime Boundary Room History V1 Stage A FAILED CLOSED;
- WR/TE target-share trajectory is protected and may not be renamed;
- WR-R15 / TE-R5P remain protected;
- generic QB mean research after M89/M90 remains frozen.

This V1 is distinct because its object is:
**named pass catcher × pregame-selected QB identity × target allocation**.

Any future test must control for the pass catcher's ordinary strictly-prior
target-share state so QB-regime history cannot win merely by restating recent
usage.

## 3. Historical source authority

Target seasons:
- 2024
- 2025.

Pregame roster / inactive authority:
- Player Active-Role Participation V1 source;
- run `38895205962`;
- artifact `12044155086`;
- digest `sha256:a42322579aee682497223c7f53bce0b14a96dc6f466558e811770e681a035dad`;
- exact pregame roster universe after official game-day inactives;
- stable GSIS player identity;
- 100% roster team-week coverage and snap team-week coverage.

Historical player statistics:
- nflverse weekly player stats;
- stable GSIS player identity;
- only completed games strictly before the target week may enter target-week
  state.

No sportsbook source is permitted.

## 4. Frozen pregame QB-regime identity

For each target team-week:

1. Take active QB rows from the exact target-week pregame universe.
2. For each active QB, sum **strictly-prior same-season same-team pass attempts**.
3. If at least one active QB has >0 such attempts, select the active QB with the
   largest strictly-prior pass-attempt total.
4. Otherwise, among those same active QB identities, sum **prior-season
   same-team pass attempts** and select the largest positive total.
5. Deterministic tie break: stable player ID ascending.
6. If no active QB has positive same-season or prior-season same-team passing
   history, mark the team-week `QB_REGIME_SOURCE_UNAVAILABLE`.

This is a pregame regime identity proxy.

It is **not** allowed to use:
- target-game pass attempts;
- target-game snaps;
- target-game starter outcome;
- postgame depth information;
- sportsbook QB listings.

## 5. Strictly-prior player × QB history

For each WR/TE target player-team-week with a source-valid QB regime:

For each completed same-season team game strictly before the target week:
- identify that past game's primary QB as the same-team QB with the most pass
  attempts in that **already-completed prior game**;
- compute the named player's target share from that past game using canonical
  player targets and team targets.

Build two histories:

### A. Overall current-season role
All valid same-team prior games.

Required descriptive fields:
- total prior games;
- mean target share;
- latest target share;
- latest-2 mean where supported.

### B. Same-QB-regime role
Only prior games whose completed-game primary QB identity equals the
target-week pregame-selected QB regime identity.

Required fields:
- same-QB prior games;
- mean target share;
- latest target share;
- latest-2 mean where supported;
- fraction of total prior games played in same-QB regime.

No target-game target/result may enter either history.

## 6. Source-readiness gates

Adjudicate WR and TE separately in 2024 and 2025.

Require ALL per family-season:

1. target pregame roster identity coverage = 100%;
2. QB-regime source-valid team-week coverage >= 90%;
3. target WR/TE stable identity >= 99%;
4. prior completed-game primary-QB identity coverage >= 99%;
5. player target/team-target identity coverage >= 99%;
6. at least 70% of eligible target player-games have >=2 same-QB prior games;
7. at least 50% have >=3 same-QB prior games;
8. support with >=2 same-QB prior games:
   - WR >= 1,000 target rows and >=150 distinct players;
   - TE >= 600 target rows and >=80 distinct players;
9. chronology violations = 0;
10. target-game outcomes used in predictor/source state = 0;
11. sportsbook inputs = 0;
12. OddsAPI calls = 0;
13. predictive candidates scored = 0;
14. production changes = 0.

Do not lower gates after running the audit.

## 7. If source-ready

Do **not** change projections.

Freeze a separate predictive allocation-state test before scoring.

That future test must ask whether same-QB-regime player target-share state adds
incremental predictive information beyond:
- ordinary strictly-prior player target-share state;
- protected WR/TE trajectory state;
- existing WR-R15 / TE-R5P entitlement;
- Stage-1 participation;
- existing finite allocation architecture.

The primary outcome must be player target allocation/share, not receiving yards,
so efficiency remains downstream and separate.

Any market translation requires another frozen bridge.

## 8. If source-blocked

Fail closed.

Do not:
- infer the target starter from target-game pass attempts;
- substitute sportsbook QB availability;
- lower same-QB history gates;
- pool WR/TE to rescue support;
- use pair YPT;
- turn QB-regime absence into a generic boost/penalty;
- fit 2026 Week-4/5 outcomes.

## 9. Integrity at freeze

- predictive candidates scored: 0
- 2026 outcomes used: 0
- Week-6 outcomes used: 0
- sportsbook inputs: 0
- OddsAPI calls: 0
- coefficients fit: 0
- thresholds searched: 0
- projection changes: 0
- production changes: false

Disposition:
`PRISM_PASS_CATCHER_QB_REGIME_ALLOCATION_STATE_V1_SOURCE_AUDIT_FROZEN`.
