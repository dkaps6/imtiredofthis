# PRISM Player Team-Context Transition V1 — Canonical Source Result

Date: 2026-10-18

Status: **SOURCE READY / STRICT NAMED-PLAYER TEAM-CONTEXT BOUNDARY / NO PREDICTIVE SCORING**

Model:
**PRISM — Player Role & Individual Situation Model**

Canonical frozen source plan:
`docs/research/PRISM_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_PLAN.md`

Freeze:
`bd254ec9435cd0a17e2c7d82d78532e67c2cd23e`

Canonical execution:
- run `39182076127` — **SUCCESS**;
- exact head `6c047e4df67702affde52493f7d1b90296751968`;
- job `117836769489` — **SUCCESS**;
- artifact `12156204035`;
- artifact name
  `prism-player-team-context-transition-v1-source-39182076127`;
- digest
  `sha256:abc2a5b9d3ea2a4d683e56626ea2d33495fa1e7cad2c36c56c01071ccc0339c8`;
- strict repository audit: **PASS**.

Disposition:

`PRISM_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_READY`

## Lineage

The earlier looser contract at
`fc63601c34760d042adcc00fff12976496622574`
and its SOURCE_READY run `39181783189` are diagnostic/noncanonical only.

The later freeze `bd254ec9...` is authoritative because it additionally
requires:
- zero unresolved multi-team stable-ID player-weeks;
- stronger total/family transition support;
- >=4 previous-team roster observations;
- current same-family roster-competition metadata.

## Player-first question

Can PRISM identify, before kickoff, that this exact named RB/WR/TE has moved
from one NFL team context to another, while preserving stable identity,
chronology, current-team competition metadata and 2026 semantic parity?

The answer is **yes**.

This is a source-state result only.

It does not say a team change should raise or lower any projection.

## 2024 historical support

- admitted player-weeks: **11,599**;
- distinct named players: **867**;
- scheduled team-week roster coverage: **100.00%**;
- stable-ID coverage: **100.00%**;
- transition events: **329**;
- transition named players: **287**;
- offseason / in-season transitions:
  **216 / 113**;
- prior-team >=4 roster observations:
  **268 / 329 = 81.46%**;
- prior-team >=8:
  **248 / 329 = 75.38%**;
- current same-family competitor metadata coverage:
  **100.00%**.

By family:
- RB: 90 events / 77 players;
  prior >=4 **81.11%**;
  prior >=8 **77.78%**;
- WR: 161 / 141;
  prior >=4 **82.61%**;
  prior >=8 **74.53%**;
- TE: 78 / 69;
  prior >=4 **79.49%**;
  prior >=8 **74.36%**.

All frozen 2024 source gates passed.

## 2025 independent source support

- admitted player-weeks: **11,732**;
- distinct named players: **839**;
- scheduled team-week roster coverage: **100.00%**;
- stable-ID coverage: **99.92%**;
- transition events: **351**;
- transition named players: **271**;
- offseason / in-season transitions:
  **202 / 149**;
- prior-team >=4:
  **281 / 351 = 80.06%**;
- prior-team >=8:
  **241 / 351 = 68.66%**;
- competitor metadata coverage:
  **100.00%**.

By family:
- RB: 105 events / 73 players;
  prior >=4 **73.33%**;
  prior >=8 **62.86%**;
- WR: 159 / 129;
  prior >=4 **85.53%**;
  prior >=8 **73.58%**;
- TE: 87 / 69;
  prior >=4 **78.16%**;
  prior >=8 **66.67%**.

All frozen 2025 source gates passed.

## 2026 prospective parity

- latest complete observed roster week: **6**;
- latest complete-week team coverage: **100.00%**;
- stable-ID coverage: **99.85%**;
- transition events: **257**;
- named players: **240**;
- offseason / in-season:
  **216 / 41**;
- failed parity gates: **NONE**.

No 2026 player outcomes were used.

## Integrity

- ambiguous multi-team player-weeks: **0**;
- chronology violations: **0**;
- skipped-prior-observation violations: **0**;
- competitor metadata coverage on historical transitions: **100%**;
- target-game snaps used: **0**;
- target-game targets/carries/receptions/yards/TDs used: **0**;
- predictive relationships scored: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- sportsbook/OddsAPI used: **0**;
- paid OddsAPI calls: **0**;
- production changed: **false**.

## Scientific meaning

PRISM may now treat a source-certified team change as a real named-player
current-context boundary.

That boundary answers only:

> this player's current team / competition environment is not the same
> environment that generated his immediately prior roster history.

It does not authorize:
- a universal new-team penalty;
- a universal new-team boost;
- position-average transfer;
- vacated-opportunity transfer;
- automatic entitlement reset;
- automatic persistence of the old-team role.

Those are predictive questions and must be frozen separately.

## Exact next boundary

Freeze exactly one market-specific predictive mechanism before scoring.

First authorized seam:

**RB rushing opportunity identity across a team boundary.**

Question:

> For this exact RB in his first source-certified new-team context, is his own
> strictly-prior old-team carry-share identity more predictive of his first
> new-team carry share than the already-canonical current-team RB rushing
> entitlement?

This is a direct identity-vs-current-context arbitration question.

Requirements:
- transition RB is primary unit;
- 2024 development / 2025 untouched confirmation;
- strictly-prior old-team player history only;
- minimum four valid prior-team carry-share observations;
- canonical current-team RB entitlement remains the baseline;
- finite team rushing opportunity and current-team competition preserved;
- no global recent-share replacement;
- no RB carry/snap 50/50 retest;
- no active-set bridge retest;
- no yards/efficiency/TD translation;
- no 2026 outcomes;
- no sportsbook/OddsAPI;
- no production mutation.
