# Phase-0 Run Checkpoint — Contact Fidelity Report V1

**Scope:** isolated data-frontier engineering only. No predictive experiment, model retuning, sportsbook logic, production change, or Issue #535 direction.

**Parent checkpoint:** `b804bb0e0aad22358c164a8c2c002d9cc9f9fb8c` on `data-frontier-phase0-bdb-contact-v1`.

## Completed

Added `scripts/data_frontier/bdb_2024_contact_fidelity.py`, which consumes the existing versioned contact/enrichment artifacts and writes `contact_fidelity_report_v1.json`.

The report keeps five questions separate:

1. **Benchmark disposition:** attempted, scoreable, abstained and reason counts.
2. **Geometry resolution:** whether the already-selected first-contact defenders can be resolved at the selected frame.
3. **Source-label reconciliation:** separate overlap rates for primary tackle, assist, missed tackle and any source label.
4. **Distribution sanity:** finite-count/missingness plus min/P05/median/P95/max for closing-speed, pursuit-angle and sideline geometry.
5. **Source-window context:** carries forward the event-window diagnostics rather than interpreting absent events as absent football events.

The report explicitly records `contact_detector_changed=false`. It does not tune the frozen one-yard/two-consecutive-frame detector and does not compute projection, betting, or model metrics.

Added focused tests in `tests/test_bdb_2024_contact_fidelity.py` for separation of abstention/overlap/geometry accounting and preservation of missing geometry.

## Validation status

Source-level review completed. Real BDB 2024 files remain unavailable in this environment because authenticated Kaggle competition access/rule acceptance has not been performed. Therefore no real-corpus fidelity numbers are claimed in this checkpoint. The new tests are committed but should only be reported as passing after execution in a verified runtime.

## Next unfinished task

Add artifact-level invariant checks before a real-corpus run: key uniqueness, enriched-play parity with scoreable detector rows, defender-row referential integrity, finite/range checks for geometry, and manifest/report version consistency. Fail closed on structural corruption; continue to abstain rather than impute ambiguous football semantics.
