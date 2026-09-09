# RB R26N Run-1 Mechanical Repair V1

Status: FROZEN MECHANICAL REPAIR BEFORE RERUN
Date: 2026-09-09

Study:
`RB_R26N_2026_WEEK1_UNMODIFIED_R26_STRUCTURAL_CANDIDATE_V1`

Frozen scientific plan:
`docs/research/RB_R26N_2026_WEEK1_UNMODIFIED_R26_STRUCTURAL_CANDIDATE_V1_FROZEN_PLAN.md`

Original implementation lock:
`docs/research/RB_R26N_2026_WEEK1_UNMODIFIED_R26_STRUCTURAL_CANDIDATE_V1_IMPLEMENTATION_LOCK.md`

First launch:
- workflow run `34395291505`
- job `102613379938`
- exact head `3834b6765f60efc80566b7a28ecd081d6fe8fd00`

## What passed before the failure

The first R26N attempt passed all pre-materialization governance checks:
- frozen plan/implementation lineage check;
- protected production boundary against `main@f8417f55b04ce0e19baf260e9d532765034c47f1`;
- exact R26M artifact digest;
- exact current Full Slate artifact digest and production head;
- exact R26L artifact digest;
- exact R19 artifact digest;
- exact inner serialized R19/R9 model SHA-256.

The workflow printed `R26N_PROTECTED_PRODUCTION_BOUNDARY_PASS` and `R26N_IMMUTABLE_PARENTS_PASS` before candidate materialization.

No R26N scientific disposition or candidate artifact was written.

## Exact mechanical failure

Candidate materialization failed while reconstructing the current promoted entitlement through protected production code:

`RuntimeError: football simulation universe != certified model-context roster`

The mismatch was representational, not population-level. Examples from the traceback:
- model-context side: `('ARI', 'Bam Knight')`, `('ARI', 'Carson Beck')`, ...
- PlayerForm side: `('ARI', 'bamknight')`, `('ARI', 'carsonbeck')`, ...

Direct inspection of immutable production artifact `10090547415` established:
- `player_form_consensus.csv`: 468 rows;
- `model_context_bridge.csv`: 468 rows;
- exact `(team, player display name)` set difference: `0` in both directions;
- duplicate `(team, player display name)` rows: `0` in both sources;
- PlayerForm contains the canonical compact `player_clean_key`;
- model-context contains no `player_clean_key` column, causing protected V1 `_identity_frame()` to fall back to display player names.

Therefore the first run did not expose a scientific R26N gate failure. It exposed a missing-key representation seam in the isolated reconstruction path.

## Frozen minimum repair

The repair is restricted to the R26N reconstruction staging layer.

The repaired builder must:
1. leave the downloaded immutable `parent_production` artifact tree untouched;
2. verify exact one-to-one `(team, player display name)` identity equality between the 468-row PlayerForm and model-context parents;
3. create an isolated temporary copy of the production artifact for reconstruction only;
4. add `player_clean_key` to the staged model-context copy by exact one-to-one join from the immutable PlayerForm `(team, player, player_clean_key)` mapping;
5. fail closed on any duplicate display identity, blank PlayerForm key, missing mapping, extra mapping, row-count change, or post-staging identity mismatch;
6. run the exact same protected promoted-entitlement reconstruction on that staged copy;
7. preserve the compact PlayerForm key as the reconstructed football-universe identity so strict-prior R9 attachment semantics remain unchanged.

The repair may not:
- alter the immutable parent artifact in place;
- normalize or fuzzy-match player names;
- add/remove a player or team;
- change any football feature, entitlement value, TE-R5P/WR-R15 parameter, R9 parameter, vacancy state, gate, threshold, disposition, receiving-yard/R22 authority, or production file;
- use sportsbook inputs, 2026 outcomes, or same-week depth.

## Rerun contract

The repaired builder must remain bound to the original 28 frozen R26N structural gates. The workflow must continue to verify all original parent digests and protected production boundaries before materialization.

This repair note is frozen before the repaired candidate builder is committed or rerun.