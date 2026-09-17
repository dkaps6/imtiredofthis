# Phase 0 Run 03 — Contact Enrichment Checkpoint

**Lane:** isolated data engineering only. No predictive experiment, production science, Issue #535 direction, sportsbook logic, or paid run.

**Starting branch head verified:** `9d0323c70b5545db36e09d8eeff85d72ea8b93aa`.

## Completed

Added `scripts/data_frontier/bdb_2024_contact_enrichment.py`, a versioned enrichment layer over the already-frozen `BDB_2024_CONTACT_GEOMETRY_V1` contact detector.

The layer writes two auditable artifacts:

- `rb_contact_features_enriched_v1.csv` — one row per already-scoreable contact play;
- `rb_contact_defender_geometry_v1.csv` — one row per resolved first-contact defender.

New deterministic fields include:

- carrier distance to nearest sideline at first contact;
- max/mean relative closing-speed proxy among first-contact defenders;
- min/mean pursuit-angle error among first-contact defenders;
- number of first-contact defenders with resolvable motion geometry;
- separate overlap flags for primary tackle, assist, missed tackle, and any source semantic label.

The defender-level artifact preserves the exact defender ID and separate BDB source-label flags, allowing later QA without collapsing tackle/assist/missed-tackle semantics.

## Scientific / detector boundary

`contactDetectorChanged=False` is emitted explicitly. The frozen 1-yard / two-consecutive-frame detector is not retuned. No projection outcome, model metric, sportsbook value, or production feature is computed.

The enrichment uses original BDB `x/y/s/dir` for instantaneous motion geometry. It does not mix mirrored `x_offense` coordinates with unmirrored source direction angles.

## Tests added

`tests/test_bdb_2024_contact_enrichment.py` covers:

1. deterministic geometry + separate source-label reconciliation while retaining the original feature version;
2. abstention behavior when the selected contact frame cannot be resolved in tracking.

These tests are committed as source-level QA. A passing runtime result is not claimed in this checkpoint unless an execution environment actually runs them.

## Access boundary

No real BDB competition files were downloaded or competition terms accepted. Real-file benchmark execution remains blocked until data are lawfully available to the runtime/user.

## Next checkpoint

Build a benchmark-fidelity report over the enriched artifacts that measures data-engineering coverage only: geometry resolution rate, source-label overlap decomposition, event-window stratification, missingness/abstention reasons, and distribution sanity checks. Do not compute predictive accuracy or retune the contact threshold.
