# PRISM Opportunity Quality V1 — Reconciled Source Result

Date: 2026-10-16

Status: **SOURCE READY ALL — RECONCILED / NO PREDICTIVE CANDIDATE SCORED / NO PRODUCTION CHANGE**

Model:
**PRISM — Player Role & Individual Situation Model**

Overall disposition:

`PRISM_OPPORTUNITY_QUALITY_SOURCE_READY_ALL_RECONCILED`

This result reconciles two independently frozen source sub-audits. Neither result
rescues the other; both are required before a predictive mechanism may be frozen.

## 1. Core team high-value topology authority — READY

Frozen plan:
`docs/research/PRISM_OPPORTUNITY_QUALITY_TEAM_VOLUME_V1_SOURCE_PLAN.md`

Pre-audit freeze:
`40a9370899abddaae5a985aa6d67a2a481e85f9e`

Pre-interpretation source-integrity reconciliation:
`docs/research/PRISM_OPPORTUNITY_QUALITY_V1_DUAL_SOURCE_RECONCILIATION.md`

Canonical hardened execution:
- run `38991605252` — SUCCESS;
- exact head `e590ce4754da2c7f7aa01beab2ee148aa5b613a0`;
- artifact `12085367606`;
- digest `sha256:68b680c745ae8b8479b3dbaf8f41dcc4f6cb728a0f39b0ac10b77dc3ac8e4b90`.

Disposition:
`PRISM_OPPORTUNITY_QUALITY_TEAM_VOLUME_SOURCE_READY`

The hardened run explicitly distinguishes:
- a source-present team-game with zero high-value events; from
- a scheduled team-game whose rush/target event source is missing.

No missing source is relabeled as a football zero.

Across 2022, 2023, 2024 and 2025:
- scheduled team-game coverage: 100%;
- rusher stable identity: 100%;
- targeted receiver stable identity: 100%;
- strict-prior offense-history attachment for Weeks 5-18: 100%;
- strict-prior opponent-history attachment: 100%;
- core source-ready: true.

Also source-ready:
- drive topology;
- optional end-zone-target inputs.

These optional source labels do not authorize their predictive use.

## 2. PRISM Stage-2 player compatibility authority — READY ALL

Frozen plan:
`docs/research/PRISM_OPPORTUNITY_QUALITY_TEAM_HIGH_VALUE_VOLUME_V1_SOURCE_PLAN.md`

Final pre-evaluator freeze:
`48d9331f86aa052c90bc48816b2f031b84900c4b`

Canonical execution:
- run `38991429245` — SUCCESS;
- exact head `d14ba07f7f0e79f2b3f16a0b018a50c30744be91`;
- artifact `12085262613`;
- digest `sha256:77739799f34237538d7ffcba9715939c59d6a5b50f811f06f2f5836ff9deb31d`.

Disposition:
`PRISM_OPPORTUNITY_QUALITY_SOURCE_READY_ALL`

### RB rush inside-10

2024:
- 1,400 player-games;
- 159 players;
- 338 positive inside-10 team-games;
- strictly-prior offense attach 100%;
- strictly-prior opponent attach 100%;
- failed frozen source gates: none.

2025:
- 1,431 player-games;
- 157 players;
- 337 positive inside-10 team-games;
- strictly-prior offense attach 100%;
- strictly-prior opponent attach 100%;
- failed frozen source gates: none.

Disposition:
`RB_RUSH_INSIDE10_PRISM_OPPORTUNITY_QUALITY_SOURCE_READY`

### RB receiving red-zone targets

2024:
- 1,400 player-games;
- 159 players;
- 391 positive red-zone target team-games;
- strictly-prior offense attach 100%;
- strictly-prior opponent attach 100%;
- failed frozen source gates: none.

2025:
- 1,431 player-games;
- 157 players;
- 393 positive red-zone target team-games;
- strictly-prior offense attach 100%;
- strictly-prior opponent attach 100%;
- failed frozen source gates: none.

Disposition:
`RB_RECEIVING_REDZONE_PRISM_OPPORTUNITY_QUALITY_SOURCE_READY`

### TE receiving red-zone targets

2024:
- 1,334 player-games;
- 138 players;
- 391 positive red-zone target team-games;
- strictly-prior offense attach 100%;
- strictly-prior opponent attach 100%;
- failed frozen source gates: none.

2025:
- 1,312 player-games;
- 142 players;
- 393 positive red-zone target team-games;
- strictly-prior offense attach 100%;
- strictly-prior opponent attach 100%;
- failed frozen source gates: none.

Disposition:
`TE_RECEIVING_REDZONE_PRISM_OPPORTUNITY_QUALITY_SOURCE_READY`

## 3. Anti-retest meaning

This does **not** reopen:

`PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_CLOSED`.

The closed question was whether recent named-player high-value share itself
directly predicts next-game high-value share better than ordinary usage.

That remains terminal.

The source-ready question is different:

`TEAM HIGH-VALUE OPPORTUNITY CREATION -> CURRENT CANONICAL PLAYER ENTITLEMENT -> HIGH-VALUE PLAYER COUNT`.

No player recent red-zone / inside-10 share has been selected as a candidate.

## 4. Why both source authorities are required

The raw-PBP authority proves the finite team quality environment exists and can
be reconstructed strictly prior to the target game.

The Stage-2 compatibility authority proves that environment can be joined to the
exact PRISM player identities and already-qualified canonical opportunity
entitlement in the two valid historical target seasons:
- 2024 development;
- 2025 independent confirmation.

A future mechanism may not use 2022/2023 as Stage-2 player confirmation years.

## 5. Integrity

Across both canonical source runs:
- predictive candidates scored: 0;
- target-game high-value counts used as predictors: 0;
- player high-value-share rescue attempts: 0;
- fitted predictive coefficients: 0;
- threshold searches: 0;
- sportsbook inputs: 0;
- OddsAPI calls: 0;
- 2026 outcomes: 0;
- production changed: false.

## 6. Exact next action

Do not score a quality mechanism yet.

First persist the **numeric canonical pregame parent opportunity denominator**
from the already-qualified Stage-2 historical source:
- RB rushing -> canonical pregame team rushing-opportunity mean;
- RB receiving / TE receiving -> canonical pregame team target-opportunity mean.

The qualified source audit already proved those denominators finite but its
published player join artifact retained only a boolean finite flag.

A mechanical persistence addendum must:
- change no entitlement;
- change no denominator;
- change no source gate;
- read no target-game outcome;
- simply retain the already-computed numeric denominator in a new immutable
  source artifact.

Only after that exact parent quantity is persisted may one opportunity-quality
predictive mechanism be frozen before scoring.

The reason is architectural: PRISM quality must remain attached to its existing
pregame opportunity quantity rather than inventing a detached scoring-area count
model.
