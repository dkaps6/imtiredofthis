# RB R26O Run 2 Mechanical Dtype Repair V1

## Scope

This note records the second mechanical execution failure of the frozen R26O 2026 Week-1 receptions-only shadow-integration compatibility study.

The frozen R26O scientific plan, evaluator, 38 gates, Monte Carlo tolerance, seed, iterations, parent artifacts, and authority ceiling remain unchanged.

## Failed execution

- Workflow run: `34398389376`
- Job: `102623855653`
- Head: `54ee8d0b67a342fc550ce6eee870e5741be909f9`
- Failure classification: **MECHANICAL / NO SCIENTIFIC DISPOSITION**

The run passed:

- frozen plan/evaluator/repair-lineage verification
- protected production boundary
- immutable R26N / Full Slate / R22 parent verification
- exact 468-row identity staging
- exact production-base QB C2 selector staging and SHA-256 verification
- exact V3 promoted entitlement reconstruction
- exact QB C2 production integration/state parity

It then failed before any R26O structural gate was evaluated, inside R22 strict-prior RB identity attachment.

## Exact blocker

`pandas.merge_asof` rejected otherwise identical identity join keys because the current RB frame carried `player_clean_key` as pandas `string[python]` while the strict-prior history frame carried it as plain `object`:

`pandas.errors.MergeError: incompatible merge keys [0] string[python] and dtype('O'), must be the same type`

The same staging/runtime dtype incompatibility was already isolated and repaired mechanically in R26N without changing any identity values or football features.

## Authorized repair

Do not modify the frozen R26O evaluator or protected R22 production code.

Add a hash-tracked R26O runtime wrapper that, only for the R22 `_attach_identity` compatibility seam:

1. copies the current RB frame, strict-prior states frame, and previous-state frame;
2. casts only `player_clean_key` and `team` columns (when present) to plain Python/object dtype;
3. verifies the cast changes no string identity value;
4. verifies row counts are unchanged;
5. verifies every non-key column remains exactly equal;
6. delegates to the original protected `_attach_identity` implementation;
7. runs the frozen R26O evaluator with its original CLI arguments.

No key string may be normalized, rewritten, fuzzy-matched, added, or removed. No football/R9/R22 values may change.

## Authority

This is a dtype-only execution compatibility repair. It is not a new candidate, a scientific retry, a production change, an R22 change, or permission to alter any R26O gate/threshold.
