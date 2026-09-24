# NFL HANDOFF — 2026-09-24 — RUSH POOL V1 INTEGRATION FAILED CLOSED / NEW SCIENCE NEXT

GitHub is canonical over chat memory.

## Production state

Current production main entering this docs-only closure:
`b5b1816e7d11cadca412faf841fbfe5336798c2e`

Production remains unchanged:
- RB Rush+Receiving Conservation V2 stays active/protected;
- Rush Pool Evidence Guard V1 is **not** in production;
- TE Width V2 remains failed closed.

## Rush Pool Evidence Guard V1

Upstream allocator-only science qualified:
`RUSH_POOL_EVIDENCE_GUARD_V1_QUALIFIED`

But the separate current-stack integration failed:
`RUSH_POOL_EVIDENCE_GUARD_V1_PRODUCTION_INTEGRATION_FAILED_CLOSED`

Authority:
- run `36049144898`
- job `107800031207`
- head `808072634603c03e6c86fd3c45ea311a39b13fa6`
- artifact `10830277740`
- digest `sha256:b6a0715d38984fa91b8815561b4e72679684affcc142ee773a8b7428342412a5`
- PR #633 closed/not merged
- Repo CI `36049328629` SUCCESS
- preserved paid-artifact replay `36049328727` SUCCESS

Mechanics all passed. Science failed.

Most important failures:
- 2025 ALL rush-att MAE `1.249158 -> 1.253482`
- 2025 RB rush-att MAE `3.442835 -> 3.456101`
- 2025 QB rush-att MAE `1.726906 -> 1.733759`
- 2025 ALL rush-yard MAE `7.845055 -> 7.855806`
- QB rush-yard MAE worsened in both 2024 and 2025

Positive but insufficient:
- RB rush-yard MAE improved in both years
- RB rush+receiving MAE/p90 improved in both years

No rescue is authorized.

Canonical result:
`docs/production/RUSH_POOL_EVIDENCE_GUARD_V1_INTEGRATION_RESULT.md`

## Exact next action

Return to genuinely different Week-3 model-improvement science.

Do not reopen:
- Rush Pool Evidence Guard V1 via any carveout/threshold/top-N/evidence-definition variant;
- TE Width V2;
- retrospective M96 router variants;
- generic QB pass-yard retuning;
- closed C1/C3 receiving formulations.

The next investigation should remain structural and read-only first. Prefer finding a contradiction with a different information source or mechanism, not another transform of the same rushing shares.

High-value audit questions:
- where current-season role/participation information is available but not reaching final player opportunity;
- where room/team opportunity mass is correct but player-level entitlement is stale;
- where injury/personnel transitions fail to propagate across multiple markets;
- where final production specialists create cross-market contradictions after individually reasonable component projections;
- where current 2026 usage state materially diverges from strict-prior hierarchy without requiring retrospective outcome fitting.

Prospective RB Vacancy V1 remains separate and valid; Week 2 had no qualifying preserved definitive-unavailable RB/FB event.

## Memory-efficient restart rule

Read only:
1. `AGENTS.md`
2. top of `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. this handoff
4. newest Issue #535 comments after the V1 integration closure
5. current main and any newly opened research branch

Do not load older handoffs unless explicitly needed.
