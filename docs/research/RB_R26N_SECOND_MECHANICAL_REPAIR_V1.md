# RB R26N Second Mechanical Repair V1

Status: FROZEN MECHANICAL REPAIR BEFORE RERUN
Date: 2026-09-09

Study:
`RB_R26N_2026_WEEK1_UNMODIFIED_R26_STRUCTURAL_CANDIDATE_V1`

Frozen scientific plan and original candidate builder remain unchanged.

Authoritative repaired attempt exposing this blocker:
- workflow run `34395790276`
- job `102615051239`
- exact head `ca0848b9392d54eaa016de7918126b5b6e2e831f`

## What passed

Before the failure, this run passed:
- original frozen plan and original candidate-builder byte-integrity checks;
- first mechanical-repair note/helper byte-integrity checks;
- protected production boundary;
- all four immutable parent artifact/digest checks;
- current production artifact head check;
- exact inner serialized R19/R9 model SHA-256;
- first mechanical identity-key staging repair;
- proof that the immutable downloaded production parent remained untouched.

The first repair audit reported:
- 468 source rows;
- 468 staged rows;
- zero display-identity mismatches;
- zero duplicate display identities;
- zero blank PlayerForm keys;
- zero staged missing keys;
- 468 exact compact staged identities;
- no fuzzy matching;
- no normalization heuristic;
- no football-value changes;
- no players added or removed.

No R26N disposition or candidate artifact was written.

## Exact second mechanical failure

The unchanged frozen builder loaded strict-prior identity history for every season 2013 through 2025 successfully, then failed before R9 scoring at the protected runtime as-of join:

`pandas.errors.MergeError: incompatible merge keys [0] string[python] and dtype('O'), must be the same type`

Trace path:
- R26N builder calls `attach_identity(...)`;
- `scripts/modeling/rb_receiving_identity_runtime_v1.py::_snapshot_queries` calls `pd.merge_asof(...)` by `player_clean_key`;
- current reconstructed football rows carry pandas `string[python]` key dtype from the protected production identity frame;
- historical identity state rows carry Python-object dtype;
- the key values are not reported as different; pandas refuses the join solely because the dtypes differ.

This failure occurred before:
- manual serialized R9 scoring;
- vacancy softmax redistribution;
- candidate target/reception creation;
- any of the 28 frozen R26N gates;
- any scientific disposition.

## Frozen minimum repair

Keep the original R26N candidate builder byte-for-byte unchanged.

Add a hash-tracked compatibility wrapper that, immediately before delegating to the protected `attach_identity` function:
1. copies the current RB frame, historical states, and previous-season frame;
2. casts only identity join-key columns (`player_clean_key`, and `team` where present) to plain Python object dtype on all sides;
3. leaves every key string value exactly unchanged;
4. leaves row order/count and every non-key football/R9 feature value unchanged;
5. calls the original protected `attach_identity` implementation;
6. delegates all remaining R26N logic to the original frozen candidate builder.

The wrapper may not:
- normalize, lowercase, strip, fuzzy-match, or otherwise transform key values;
- add/remove rows;
- change historical state values;
- change R9 features, scaler, coefficients, intercept, clip, reliability, or training authority;
- alter R26L vacancy state;
- alter any frozen R26N gate/threshold/disposition;
- alter production code or artifacts;
- use sportsbook inputs, 2026 outcomes, or same-week depth.

This repair note is frozen before the dtype-compatibility wrapper is committed or rerun.