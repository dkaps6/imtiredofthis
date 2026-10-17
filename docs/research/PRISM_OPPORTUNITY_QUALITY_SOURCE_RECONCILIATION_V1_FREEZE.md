# PRISM Opportunity Quality Source Reconciliation V1 — Freeze

Frozen: 2026-10-16

Status: **FROZEN SOURCE-INTEGRITY RECONCILIATION / NO PREDICTIVE SCORING**

Purpose:
reconcile two independently frozen source audits before any PRISM opportunity-quality mechanism is designed.

## Parent A — Stage-2 compatibility / immutable-parent audit

- source-plan freeze commit: `48d9331f86aa052c90bc48816b2f031b84900c4b`;
- run `38991429245` — SUCCESS;
- exact head `d14ba07f7f0e79f2b3f16a0b018a50c30744be91`;
- artifact `12085262613`;
- digest `sha256:77739799f34237538d7ffcba9715939c59d6a5b50f811f06f2f5836ff9deb31d`;
- disposition `PRISM_OPPORTUNITY_QUALITY_SOURCE_READY_ALL`.

This parent proves 2024/2025 Stage-2 player identity / entitlement compatibility for:
- RB rush inside-10;
- RB receiving red zone;
- TE receiving red zone.

## Parent B — raw-PBP finite-pool audit

- source-plan freeze commit: `40a9370899abddaae5a985aa6d67a2a481e85f9e`;
- run `38991429809` — SUCCESS;
- exact head `e18266a791f85363630c3cd9d9b2e05092b4a48e`;
- artifact `12084992535`;
- digest `sha256:753dcc94db548b3be7e4da185d3c72cbc2c766bcf56520d6c99a213dfeff1ee8`;
- disposition `PRISM_OPPORTUNITY_QUALITY_TEAM_VOLUME_SOURCE_READY`.

This parent reconstructs full team finite pools directly from 2022-2025 PBP while preserving missing/unmodeled player identity as complement mass.

## Exact reconciliation population

Compare only common 2024 and 2025 regular-season Weeks 5-18 team-games.

Expected key:
`season + week + team`.

Required:
- identical key population;
- no duplicate key in either authority.

## Exact fields

Require exact equality, tolerance **0** after numeric normalization, for every common team-game:

1. total team rushing attempts;
2. team inside-10 rushing attempts;
3. total team targets;
4. team red-zone targets.

These are integer finite football counts. There is no scientific reason to allow drift.

Also require:
- 832 overlapping target team-games total (416 in 2024 + 416 in 2025);
- zero left-only keys;
- zero right-only keys.

## Reconciled scientific interpretation

If exact parity passes and both parents retain their source-ready dispositions:

- raw PBP is the canonical finite team-pool authority;
- Parent A remains the canonical Stage-2 named-player entitlement compatibility authority;
- the two may be used together by a later separately-frozen mechanism.

If any field/key differs:
- disposition is source reconciliation blocked/integrity failure;
- do not choose whichever authority is more favorable;
- do not score an opportunity-quality mechanism.

## No predictive inference

This reconciliation:
- fits no model;
- scores no candidate;
- reads no 2026 outcome;
- uses no sportsbook;
- calls no OddsAPI;
- changes no production code.

The closed player high-value-share persistence V1 remains terminal.

## Possible dispositions

- `PRISM_OPPORTUNITY_QUALITY_SOURCE_RECONCILED_READY_ALL`
- `PRISM_OPPORTUNITY_QUALITY_SOURCE_RECONCILIATION_BLOCKED`
- `PRISM_OPPORTUNITY_QUALITY_SOURCE_RECONCILIATION_INTEGRITY_FAILURE`
