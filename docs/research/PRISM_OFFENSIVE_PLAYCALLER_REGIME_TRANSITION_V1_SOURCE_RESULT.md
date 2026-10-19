# PRISM Offensive Playcaller Regime Transition V1 — Source Result

Date: 2026-10-18

Status: **SOURCE BLOCKED / 2024 NAMED-PLAYER PRIOR-ROLE SUPPORT INSUFFICIENT / NO PREDICTIVE SCORING**

Model:
**PRISM — Player Role & Individual Situation Model**

Frozen source plan:
`docs/research/PRISM_OFFENSIVE_PLAYCALLER_REGIME_TRANSITION_V1_SOURCE_PLAN.md`

Freeze:
`533e628c99359fbfddbb9f4d787a96d8dbfbc67c`

Canonical execution:
- run `39181153541` — **SUCCESS**;
- exact head `f62d72df3a8b9ca0ce2491d58ac800504006d053`;
- job `117833981762` — **SUCCESS**;
- artifact `12155618757`;
- artifact name
  `prism-offensive-playcaller-regime-transition-v1-source-39181153541`;
- digest
  `sha256:0bfcebc29d7ddeaf07c565afc6f1ac010d1974473a9475d966fb32ccac8ab5ec`;
- strict repository audit: **PASS**.

Disposition:

`PRISM_OFFENSIVE_PLAYCALLER_REGIME_TRANSITION_V1_SOURCE_BLOCKED`

## Player-first question

Can PRISM source-certify that a named RB/WR/TE entered the target game under a
different primary offensive playcaller than the team used in its immediately
previous game, with enough strictly-prior same-player role history to support a
later market-specific predictive test?

The playcaller-regime object itself is source-valid.

The full V1 is nevertheless **SOURCE BLOCKED** because the frozen historical
same-player support gate does not pass in 2024.

No player outcome was scored.

## Historical source identity passed

Historical primary-playcaller chronology:
- exact frozen M68 source blob verified;
- team/week caller identity coverage: **100%** in both seasons;
- effective-week chronology: **PASS**;
- duplicate caller identities: **0**;
- named-player attachment integrity: **PASS**.

The playcaller object is therefore a legitimate pregame football-state object.

## 2024 source support

- team-weeks: **544**;
- caller identity coverage: **100.00%**;
- offseason Week-1 caller changes: **15**;
- documented in-season changes: **4**;
- total caller-change events: **19**;
- transition player-weeks: **484**;
- distinct named players on transition rows: **445**.

Frozen same-player support:
- transition rows with >=3 strictly-prior same-team positive-participation rows:
  **41 / 484 = 8.47%**.

By family:
- RB: 125 transition rows / 114 players;
  prior-role support **8.00%**;
- TE: 126 / 116;
  prior-role support **11.11%**;
- WR: 233 / 215;
  prior-role support **7.30%**.

Failed frozen historical gate:

`prior_positive_support_share_ge_040`

Required:
**>= 40%**

Observed:
**8.47%**

Do not lower the gate.

## 2025 source support

- team-weeks: **544**;
- caller identity coverage: **100.00%**;
- offseason Week-1 caller changes: **14**;
- documented in-season changes: **4**;
- total caller-change events: **18**;
- transition player-weeks: **438**;
- distinct named players: **382**;
- >=3 same-team strictly-prior positive-participation rows:
  **185 / 438 = 42.24%**.

By family:
- RB: 120 transition rows / 105 players;
  prior-role support **40.83%**;
- TE: 111 / 95;
  prior-role support **43.24%**;
- WR: 207 / 183;
  prior-role support **42.51%**.

All frozen 2025 source gates passed.

## 2026 prospective semantic parity

The frozen 2026 playcaller snapshot passed all prospective parity gates:

- Week-1 primary callers: **32 / 32 teams**;
- duplicate team/effective-week rows: **0**;
- documented midseason overrides: **1**;
- maximum effective week represented: **5**;
- primary-caller + effective-week semantics compatible with historical source:
  **PASS**;
- 2026 player outcomes used: **0**.

The snapshot includes Denver's Week-5 primary-caller handoff from Davis Webb to
Sean Payton.

Prospective parity is therefore **not** the blocker.

## Why V1 still fails closed

The intended future predictive question requires comparing the named player's
current role under the new playcaller with his own established same-team role
identity.

In 2024, most caller changes occur at the season boundary.

That means the player often has little or no strictly-prior **same-team**
positive-participation history before the first game under the new caller.

The V1 source contract required >=40% support because without that player
history PRISM would be forced to substitute:
- position priors;
- broad team averages;
- raw post-change usage;
- or a different player's role.

That would violate the assignment.

Therefore the source is blocked rather than weakened.

## What this does NOT mean

This result does not mean:
- offensive playcaller changes are irrelevant;
- coaching regime cannot alter a player's role;
- the 2025 support is invalid;
- Denver's 2026 handoff is unimportant.

It means this exact historical V1 cannot support a clean two-season
player-first predictive experiment under its frozen same-team history
requirement.

## Anti-rescue

Do not rescue with:
- reducing the >=3 prior-row requirement;
- reducing the 40% support gate;
- using position averages;
- using team averages as player identity;
- fitting only 2025 after seeing 2024 fail;
- excluding offseason transitions to manufacture support;
- using target-game snaps/targets/carries to infer the role change;
- merging playcaller and QB identity;
- broad coaching grades;
- sportsbook information.

## Integrity

- predictive player outcomes scored: **0**;
- target-week player snaps used to define state: **0**;
- target-week carries/targets/receptions/yards/TDs used: **0**;
- 2026 outcomes used: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- sportsbook inputs: **0**;
- OddsAPI calls: **0**;
- paid OddsAPI calls: **0**;
- production changed: **false**.

## Exact next boundary

Do not extend or retune the playcaller family.

Return to:

`PRISM_CURRENT_ROLE_STATE_ARBITRATION_FRONTIER_OPEN__SOURCE_AUDIT_ONLY`

The next source object must be genuinely different and must still answer:

> Why should this specific player's current role identity differ from his
> longer historical identity before kickoff?

A clean next candidate is a **named-player team-context transition**:

> Did this exact player change NFL teams before the target game, such that his
> old-team role history and current-team competition environment are different
> football states?

That object is:
- player identity first;
- directly observable from stable weekly roster history;
- distinct from recent share;
- distinct from teammate vacancy;
- distinct from active-set renormalization;
- distinct from depth chart;
- distinct from QB-pair history;
- distinct from playcaller state.

It must be source-audited before any outcome relationship is scored.
