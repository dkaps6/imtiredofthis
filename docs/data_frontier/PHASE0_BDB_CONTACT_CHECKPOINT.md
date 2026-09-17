# Data Frontier Phase 0 — BDB 2024 Contact Checkpoint

**Branch:** `data-frontier-phase0-bdb-contact-v1`

**Branch base:** `62b2ba6c14c974bfa492c506e8ba473a1f760b69` (documentation-only inventory branch head).

**Scope:** isolated data-engineering only. No predictive experiment, no production-science change, no Issue #535 direction, no sportsbook input, no paid production run.

## Completed

1. Frozen normalized relational schema:
   - `docs/data_frontier/DATA_SCHEMA_V1.md`
2. Frozen BDB 2024 contact ground-truth protocol:
   - `docs/data_frontier/BDB_2024_CONTACT_GROUND_TRUTH_PROTOCOL_V1.md`
3. Implemented BDB 2024 ingestion/normalization/contact geometry module:
   - `scripts/data_frontier/bdb_2024_contact.py`
4. Added isolated module package marker:
   - `scripts/data_frontier/__init__.py`
5. Added synthetic unit tests:
   - `tests/test_bdb_2024_contact.py`

## Frozen V1 contact rule

- offense-normalized x coordinate;
- geometric carrier-defender distance;
- contact candidate threshold `<= 1.0 yard`;
- must persist for at least `2` consecutive available frames;
- earliest qualifying frame is first-contact candidate;
- simultaneous defenders at the earliest contact frame are all preserved;
- source tackle/assist/missed-tackle labels are retained separately and used for reconciliation, not overwritten by geometry.

The threshold is an engineering benchmark rule, not asserted football truth. It must not be repeatedly tuned after observing benchmark results. A materially different contact detector requires an explicit V2 protocol.

## Current output contract

A real BDB run will produce:

- `rb_contact_features.csv`
- `benchmark_dispositions.csv`
- `source_manifest.json`
- `qa_summary.json`

under a user-specified output directory.

QA explicitly records:

- attempted play count;
- scoreable play count;
- every benchmark disposition;
- first-contact/source-label overlap rate;
- source-file names, sizes and SHA-256 digests;
- `predictive_metrics_computed = false`;
- `sportsbook_inputs_used = false`;
- `production_changed = false`.

## Validation completed before commit

The module was syntax-checked and exercised with synthetic fixtures. The test suite covering the initial benchmark mechanics produced:

`7 passed`

Covered cases:

1. left/right coordinate mirroring equivalence;
2. two-consecutive-frame first-contact requirement;
3. left/right derived geometry equivalence;
4. simultaneous-contact preservation;
5. duplicate player-frame key fail-closed behavior;
6. visible no-contact abstention instead of silent dropping;
7. QA proof that no prediction, sportsbook input or production change is involved.

This is synthetic engineering validation only. **No real Big Data Bowl contact result has been run or inspected yet.**

## Required next tasks, in order

### A. Real-file contract compatibility audit

Before a full competition run, verify exact BDB 2024 column names/types and event labels against official files. Make compatibility changes only when they preserve the frozen scientific/engineering intent; document any source-contract amendment.

### B. Expand normalization outputs

Persist normalized player/football tracking and source tackles under the schema contract, rather than only the final convenience feature table.

### C. Add richer contact geometry

Implement and test:

- relative velocity / closing-speed proxy;
- pursuit-angle proxy;
- carrier-to-sideline distance;
- source tackle/missed-tackle overlap breakdown;
- unresolved and multi-contact diagnostics.

Do not introduce run concept, intended gap or block responsibility into V1.

### D. Real BDB 2024 run

Run only when the competition files are lawfully available in the execution environment. Raw competition data should not be committed to git.

The run is a **data-fidelity benchmark**, not a projection experiment.

### E. Post-run QA review

Inspect:

- source joins and missingness;
- scoreable/abstention rates;
- source-label overlap;
- event timing failure modes;
- geometry distributions;
- obvious coordinate/identity defects.

Do not tune contact thresholds opportunistically against favorable overlap. Any V2 must be prospectively documented.

## Stop conditions

Stop and request a user decision if execution would require:

- purchasing/licensing a source;
- accepting new legal/data-use terms;
- modifying active production/research science;
- using sportsbook data upstream;
- merging to `main`;
- posting a new scientific direction into Issue #535.

Otherwise continue autonomously within this branch.