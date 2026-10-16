# Player Opportunity Difficulty -> Target Allocation Source Audit V1 — Result

Date: 2026-10-14

Status: **COMPLETE / PLAYER_OPPORTUNITY_DIFFICULTY_TARGET_ALLOCATION_SOURCE_READY / RESEARCH ONLY**

Frozen audit:
- `docs/research/PLAYER_OPPORTUNITY_DIFFICULTY_TARGET_ALLOCATION_SOURCE_V1.md`

Authority:
- branch: `research-individual-opportunity-roadmap-2026-10-08`
- exact scored head: `1cb9a582ef64305a5bfa7cdd47454b472c42e6fd`
- workflow: `Research Player Opportunity Difficulty Target Allocation Source V1`
- run: `38655896001` — SUCCESS
- job: `116165448363`
- artifact: `11937475072`
- digest: `sha256:4be927801792c646f859f9aa91918b8fa9817197f5bfab52769238f86cb541a2`

Recovered exact authority:
- run `37782189538`
- artifact `11552447166`
- digest `sha256:94a4bee7011877b1d7a24708d5ba3e86ccc5d29b427bea650af574fc1f09fa24`
- manifest: `WR_TE_FOLD_AUTHORITIES_RECOVERED_EXACT`
- provider rebuild used: false

## Frozen disposition

`PLAYER_OPPORTUNITY_DIFFICULTY_TARGET_ALLOCATION_SOURCE_READY`

Both WR and TE promoted target means can be mapped to exact pregame full-team target-allocation state without target-game team volume.

## WR

Authority rows:
- 4,193
- 273 named WRs

Pregame team target total:
`pred_team_targets = pred_targets / entitlement_tgt_share`

Support:
- recoverable rows: 4,193 / 4,193 (100%)
- team-games: 1,088
- maximum within-team-game implied total range: `1.3500311979441904e-13`
- pregame team-total lineage ready: true

Thus the promoted WR-R15 player target mean and player target share are algebraically consistent with one common pregame team target total.

## TE

Authority rows:
- 3,214
- 167 named TEs

Pregame team target total:
`pred_team_targets = b0_expected_targets / b0_target_probability`

The unchanged TE-R5P candidate target mean then maps to:
`candidate_full_team_share = candidate_targets_r5p / pred_team_targets`

Support:
- recoverable rows: 3,214 / 3,214 (100%)
- team-games: 1,594
- maximum within-team-game implied total range: `8.526512829121202e-14`
- candidate full-team share recoverable: 100%
- candidate mean reconstruction maximum gap: `8.881784197001252e-16`
- pregame team-total lineage ready: true

No realized target-game team volume is required to manufacture a pregame TE probability.

## Scientific consequence

The already-confirmed same-player WR/TE opportunity-difficulty signal can now be tested at the target-allocation/distribution layer while preserving:
- promoted player target means;
- pregame team target volume;
- team mass accounting;
- sportsbook separation.

The next experiment must explicitly isolate **allocation/share uncertainty** from team pass-volume uncertainty so the same-player difficulty state is not double-counted against existing team-volume simulation.

WR also requires an incremental guard against the already-qualified WR target-share excess-volatility / Dirichlet lineage. TE remains separate because the prior TE target-share volatility family did not satisfy its frozen three-season confirmation gate.

## Integrity

- sportsbook inputs: 0
- OddsAPI calls: 0
- 2026 outcomes: 0
- production changed: false

Mechanical failures before the canonical green:
- run `38655508713`: original September artifact expired; no audit score
- run `38655625044`: shell invocation formatting failure after recovered authority download; no audit score

Neither is a scientific result.
