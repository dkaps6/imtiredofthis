# PRISM PlayerForm Participation-State Bridge V1 — Source Result

Date: 2026-10-18

Status: **SOURCE BLOCKED / SOURCE-DENOMINATOR DESIGN FAILURE / NO PREDICTIVE SCIENCE**

Model: **PRISM — Player Role & Individual Situation Model**

Frozen plan:
`docs/research/PRISM_PLAYERFORM_PARTICIPATION_STATE_BRIDGE_V1_SOURCE_PLAN.md`

Freeze:
`3bf001af49ec103d3233b5299d743482f0505daf`

Execution:
- run `39160797272` — SUCCESS mechanically;
- exact head `ff092fdaf84d449f2c4c1fa28a11fbb29e33e63d`;
- artifact `12148811928`;
- digest `sha256:c4986ae1d5e2ad76b79fa3f6b1fe3b8a6f5f4f9216400e8a667d799460bc2d70`.

Disposition:

`PRISM_PLAYERFORM_PARTICIPATION_STATE_BRIDGE_V1_SOURCE_BLOCKED`

## What failed

V1 required >=98% identity reconciliation from participation-anomaly rows to the
raw `nflreadpy.load_player_stats(..., summary_level="week")` ledger.

Observed:
- 2024: **83.49%**;
- 2025: **78.20%**.

That gate failed in both seasons.

## Why this is a source-denominator problem

The raw weekly-stat provider is not a complete weekly roster/participation
identity ledger.

PlayerForm itself later applies a positive-usage admission gate, but the raw
weekly stats source can already omit rostered/snap-participating players who
record no weekly box-score usage.

Therefore absence from the raw stats ledger cannot be interpreted as failed
identity reconciliation for a participation-anomaly row.

The correct identity authority already exists upstream:
- the certified weekly-roster / snap / injury observation-state lineage;
- stable IDs on V1 anomaly rows were **100%** in both 2024 and 2025.

## Useful but non-dispositive observation

V1 mechanically found that many anomaly rows are in the exact PlayerForm
admitted ledger:

2024:
- injury-corroborated collapses: 139;
- admitted by PlayerForm: **118**;
- unresolved collapses: 182;
- admitted by PlayerForm: **116**.

2025:
- injury-corroborated collapses: 149;
- admitted by PlayerForm: **121**;
- unresolved collapses: 195;
- admitted by PlayerForm: **126**.

These counts are not promoted as certified bridge evidence because the frozen
source gate failed.

## Correct next source question

Do not weaken V1.

Freeze V2 with the correct lineage:

1. anomaly identity comes from the already-certified roster-based stable ID;
2. generate the exact PlayerForm **admitted** weekly ledger with the frozen
   current PlayerForm implementation;
3. exact membership is determined by
   `season + week + team + stable_id`;
4. no completeness claim is made about the raw weekly-stat provider.

Then the source question is exactly:

> Is this already-certified named-player anomaly row present in the exact
> PlayerForm history ledger or not?

No predictive outcome is required to answer that.

## Integrity

- performance outcomes scored: **0**;
- projections generated: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- sportsbook/OddsAPI used: **0**;
- production changed: **false**.
