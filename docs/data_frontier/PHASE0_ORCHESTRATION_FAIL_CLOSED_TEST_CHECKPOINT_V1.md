# Phase-0 orchestration fail-closed test checkpoint V1

**Scope:** isolated data-frontier engineering only. No predictive experiment, production science, sportsbook logic, paid run, Issue #535 direction, or frozen position-plan change.

## Verified starting lineage

- branch: `data-frontier-phase0-bdb-contact-v1`
- starting head: `c84529314d2746815ee9072f70bd629a10f0225e`
- prior checkpoint: `docs/data_frontier/PHASE0_ORCHESTRATION_CHECKPOINT_V1.md`

## Work completed

Added `tests/test_bdb_2024_phase0_orchestration.py` to make the orchestration ordering an executable contract.

The tests assert:

1. a nonzero subprocess stage is converted into a fail-closed `RuntimeError`;
2. if structural integrity fails, the fidelity module is never invoked;
3. the failure status records only stages completed before the failure and preserves `contact_detector_changed=false`;
4. on a successful orchestration path, fidelity executes only after source/artifact QA, enrichment, and structural integrity.

This does not change the frozen contact detector, geometry definitions, thresholds, source labels, or any predictive behavior.

## Validation status

The test contract has been source-reviewed against `scripts/data_frontier/run_bdb_2024_phase0.py`. This checkpoint does not claim an executed pytest result because this automation environment does not provide a checked-out branch/runtime for executing repository tests directly.

## Next unfinished task

Add artifact provenance/hashing so every eventual real-corpus fidelity report is tied to the exact normalized source files and upstream generated artifacts. The provenance layer should fail closed on hash mismatch and should remain independent of predictive science.
