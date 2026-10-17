# PRISM Opportunity Quality V1 — Dual Source Reconciliation

Frozen: 2026-10-16

Status: **FROZEN BEFORE INTERPRETATION OF EITHER SOURCE AUDIT / NO PREDICTIVE SCORING**

Two opportunity-quality source plans were frozen independently within seconds on
the same research branch before either source audit had a result.

They are overlapping but not scientifically interchangeable.

This document reconciles their roles before either result is interpreted.

## A. Core team-pool / topology authority

Contract:
`docs/research/PRISM_OPPORTUNITY_QUALITY_TEAM_VOLUME_V1_SOURCE_PLAN.md`

Frozen commit:
`40a9370899abddaae5a985aa6d67a2a481e85f9e`

Role:
prove from raw public football PBP/schedule/roster sources that PRISM can
reconstruct the finite team high-value opportunity environment itself:
- inside-10 team rushing opportunity;
- red-zone team target opportunity;
- team/opponent strictly-prior high-value state;
- complements / conservation;
- optional drive and end-zone topology.

This is the **core team-volume source authority**.

It is not sufficient by itself to authorize a PRISM player mechanism because it
does not prove compatibility with the already-qualified Stage-2 historical
player entitlement authority.

## B. PRISM Stage-2 player-compatibility authority

Contract:
`docs/research/PRISM_OPPORTUNITY_QUALITY_TEAM_HIGH_VALUE_VOLUME_V1_SOURCE_PLAN.md`

Final pre-evaluator freeze commit:
`48d9331f86aa052c90bc48816b2f031b84900c4b`

Role:
use immutable previously-qualified parent artifacts to prove that the
high-value team pools can be joined to the exact PRISM Stage-2 historical
player rows / canonical entitlement in:
- 2024 development;
- 2025 independent confirmation.

Cells:
- RB rush inside-10;
- RB receiving red-zone targets;
- TE receiving red-zone targets.

This is the **PRISM player-compatibility authority**.

It is not sufficient by itself to authorize a mechanism if the raw core
team-volume source fails.

## Canonical source-readiness rule

A future predictive mechanism may be frozen only for a cell satisfying BOTH:

1. core raw team-volume/topology source is ready under Plan A; and
2. that cell is Stage-2-compatible in both 2024 and 2025 under Plan B.

Neither result may rescue the other.

If Plan A is blocked:
- no opportunity-quality mechanism advances, even if Plan B joins cleanly.

If Plan A is ready but one Plan-B cell is blocked:
- only cells ready in Plan B may advance;
- blocked cells remain blocked.

If Plan A and all three Plan-B cells are ready:
- overall source frontier becomes
  `PRISM_OPPORTUNITY_QUALITY_SOURCE_READY_ALL_RECONCILED`.

## No duplicate predictive experiments

These two audits are source sub-audits of one PRISM opportunity-quality frontier.

They must not produce two separately tuned predictive candidates.

After reconciled source readiness, write ONE separate pre-score mechanism
freeze.

That later mechanism must obey the anti-retest boundary:
- do not reuse recent player high-value share as the candidate;
- do not rescue `PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_CLOSED`;
- separate team high-value creation from named-player entitlement;
- keep finite-pool conservation;
- use 2024 development / 2025 confirmation where Stage-2 player authority is
  required.

## Source-integrity hardening for Plan A

Before Plan A is interpreted, its implementation must distinguish:

- a source-present team-game with a genuine zero high-value count; from
- a scheduled team-game whose rush or target event source is absent.

The raw-PBP evaluator currently merges event aggregates onto schedule and
zero-fills count columns. That is mechanically acceptable only after explicit
source-presence flags prove the corresponding parent channel exists.

Frozen repair:
- persist `rush_source_present`;
- persist `target_source_present`;
- preserve missing channel state as missing until coverage is adjudicated;
- require rush-channel scheduled team-game coverage >=99%;
- require target-channel scheduled team-game coverage >=99%;
- only source-present team-games may convert absent high-value sub-counts to
  genuine zero;
- do not change any existing identity, yardline, chronology, conservation or
  support threshold.

This is a source-integrity correction, not a scientific gate change.

## Integrity at reconciliation

At this freeze:
- neither source result has been used to choose this reconciliation;
- predictive candidates scored: 0;
- target-game outcomes used as predictors: 0;
- sportsbook inputs: 0;
- OddsAPI calls: 0;
- 2026 outcomes: 0;
- production changed: false.
