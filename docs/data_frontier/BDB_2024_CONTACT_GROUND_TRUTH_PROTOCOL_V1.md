# Big Data Bowl 2024 Contact Ground-Truth Protocol V1

**Status:** frozen Phase-0 engineering benchmark. No predictive experiment. No production-science change.

## Objective

Determine whether the data-frontier pipeline can convert official Big Data Bowl 2024 tracking + tackle labels into a deterministic, auditable RB/ball-carrier contact dataset suitable as ground truth for later video/CV validation.

This protocol evaluates **data fidelity only**. It does not ask whether the resulting variables improve projections, betting performance, model MAE, CRPS or any production output.

## Source contract

Primary source: NFL Big Data Bowl 2024 competition data.

Verified public slice:

- Weeks 1-9 of the 2022 NFL season;
- player tracking for all 22 players plus football on competition-filtered plays;
- tracking fields including frame, time, team/club, x/y, speed, acceleration, distance, orientation, direction and event tags;
- tackle truth table including tackle, assist, forced fumble and PFF missed-tackle attribution.

The competition corpus is a bounded, cleaned public slice and is not treated as a complete NFL historical/live tracking contract.

## Expected raw files

The ingestion layer should accept the official competition naming pattern without requiring permanent storage in git:

- `games.csv`
- `players.csv`
- `plays.csv`
- `tackles.csv`
- `tracking_week_1.csv` ... `tracking_week_9.csv`

Raw competition files must remain outside git unless licensing and repository-size policy explicitly permit otherwise. The code must operate from a user-provided/local data directory.

## Scope population

Primary Phase-0 population:

1. regular competition plays with a resolvable ball carrier;
2. a valid tracking sequence containing the carrier and defending players;
3. a resolvable snap and/or handoff/possession phase sufficient to establish a pre-contact trajectory;
4. at least one tackle-table defender attribution OR a source event/end state sufficient to preserve the play for negative/abstention analysis.

Do not silently discard ambiguous plays. Record a `benchmark_disposition`.

Allowed dispositions:

- `SCOREABLE`
- `NO_BALL_CARRIER`
- `MISSING_TRACKING`
- `NO_CONTACT_EVENT_RESOLUTION`
- `AMBIGUOUS_CONTACT`
- `SOURCE_LABEL_INCOMPLETE`
- `UNSUPPORTED_PLAY_TYPE`

## Coordinate convention

Normalize every play into offense-left-to-right coordinates before deriving longitudinal yardage geometry.

Recommended normalized field:

- `x_offense` = distance in yards from the offense's own goal line toward the opponent goal line.

For source plays already moving right:

- `x_offense = x`.

For source plays moving left:

- mirror by official field length convention so forward offensive motion is positive.

The exact mirror formula must be unit-tested against source play direction and known yardline examples before contact-yard calculations are considered valid.

Do not alter source `x`/`y`; normalized coordinates are derived columns.

## Event resolution hierarchy

The pipeline may consume source event tags where available but should not assume every desired semantic event is directly labeled.

### Snap

Use source `ball_snap` event when present. Otherwise abstain from snap-dependent calculations.

### Handoff / possession start

Preferred hierarchy:

1. explicit source handoff event;
2. source event known to represent run possession start under the competition contract;
3. deterministic possession inference only if separately implemented and validated.

Phase-0 code must record `handoff_method` and confidence.

### First contact

BDB 2024 does not provide a universal explicit first-contact truth field for every play. V1 therefore defines **candidate first contact** geometrically, then validates the candidate against tackle/missed-tackle source labels and event/end timing.

A production-quality contact detector is **not** authorized merely because a geometric candidate exists.

## Frozen geometric first-contact candidate V1

For a scoreable ball-carrier play:

1. identify carrier track;
2. identify defending player tracks in each frame;
3. consider frames at/after possession start and before play end;
4. compute Euclidean carrier-defender center distance in field yards;
5. a defender becomes a contact candidate when distance is `<= 1.0 yard` for at least **2 consecutive available frames**;
6. first-contact candidate is the earliest qualifying frame across defenders;
7. if multiple defenders qualify within the same earliest-frame window, preserve all simultaneous candidates rather than selecting one arbitrarily.

Why 1.0 yard / 2 frames:

- this is a conservative engineering starting rule, not football truth;
- it is frozen before any benchmark result is inspected;
- it should be evaluated for fidelity, not tuned repeatedly against outcomes.

If the rule fails, a V2 requires a newly documented prospective rule rather than post-hoc threshold fishing.

## Frozen derived contact features V1

For every `SCOREABLE` play with a valid first-contact candidate:

### Longitudinal quantities

- `handoff_x_offense`
- `first_contact_x_offense`
- `play_end_x_offense`
- `yards_before_contact_geom = first_contact_x_offense - handoff_x_offense`
- `yards_after_first_contact_geom = play_end_x_offense - first_contact_x_offense`

These are geometry measurements and are not asserted to equal any vendor's proprietary YBC/YAC definition.

### Carrier state

At first contact:

- speed;
- acceleration;
- direction;
- orientation.

### Defender state

For every simultaneous first-contact candidate:

- defender ID;
- speed;
- acceleration;
- pre-contact distance;
- relative position vector;
- relative velocity / closing-speed proxy where frame-to-frame data support it;
- pursuit-angle proxy.

### Space state

At handoff and first contact:

- nearest defender distance;
- second-nearest defender distance;
- defender count within 1/2/3/5 yards;
- carrier lateral field coordinate;
- distance from nearest sideline.

Do not label a defender "unblocked" in V1. Block status requires blocker/rusher/run-block relationship information not supplied by this benchmark contract.

## Source-tackle reconciliation

`tackles.csv` remains authoritative for source attribution labels.

For each play emit:

- source primary tackle IDs (`tackle == 1`);
- assist IDs;
- forced-fumble IDs;
- missed-tackle IDs;
- whether geometric first-contact candidate overlaps any source tackle/missed-tackle defender;
- time/frame gap between first geometric contact and play-end/tackle event where available.

A geometric first-contact defender is not automatically the final tackler. The benchmark should explicitly measure this distinction.

## Data-quality gates

These are engineering gates, not scientific model gates.

### A. Input integrity

PASS only if:

- required input files exist;
- required key columns are present;
- `gameId + playId` keys join to plays without duplicate ambiguity;
- tracking player-frame keys are unique after documented normalization;
- all retained tracking rows have finite x/y;
- play direction is resolvable for all `SCOREABLE` plays.

### B. Provenance

PASS only if every output artifact records:

- source dataset name;
- source file manifest;
- row counts per raw file;
- deterministic feature version;
- generation timestamp;
- code/commit lineage if run inside CI or a checked-out repo.

### C. No forbidden inputs

PASS only if zero sportsbook/odds/model-projection columns or production-model assets are accessed.

### D. Determinism

Given identical input bytes and the same code version, row keys and numeric outputs must be identical within floating-point tolerance.

### E. Abstention visibility

Every source play in the attempted population receives a benchmark disposition. No unexplained row dropping.

### F. Source-label reconciliation

The evaluator must report, not hide:

- share of geometric first-contact defenders who appear among source tackle/assist/missed-tackle IDs;
- share of source tacklers that were not first geometric contacts;
- multiple-contact rate;
- unresolved-contact rate;
- distributions of frame gaps and geometry values.

There is **no predeclared accuracy threshold for promotion** in V1 because this is the first extraction benchmark and no video-derived model is being promoted. The purpose is to establish baseline fidelity and failure modes honestly.

## Output artifacts

Recommended local outputs under `data/data_frontier/bdb_2024_contact_v1/`:

- `games.parquet` or `.csv`
- `plays.parquet`
- `players.parquet`
- `player_tracks.parquet`
- `football_tracks.parquet`
- `tackles.parquet`
- `rb_contact_features.parquet`
- `benchmark_dispositions.csv`
- `source_manifest.json`
- `qa_summary.json`

Git should contain code/tests/docs, not the competition raw data or large generated tracking artifacts.

## Required tests before a real run

1. coordinate mirroring right/left equivalence;
2. nearest-defender geometry on a toy play;
3. consecutive-frame contact rule;
4. simultaneous-contact preservation;
5. no-contact abstention;
6. duplicate player-frame fail-closed behavior;
7. tackle-table normalization;
8. deterministic output ordering;
9. no sportsbook dependencies/imports in `scripts/data_frontier/`.

## Phase-0 success criterion

Phase 0 is considered implemented when:

- the schema and protocol are frozen;
- ingestion/normalization code passes unit tests on synthetic fixtures;
- the real BDB 2024 dataset can be supplied locally and processed without code changes;
- the QA report reconciles every attempted play and explicitly quantifies unresolved/ambiguous cases.

That milestone authorizes only a later **data-fidelity** experiment comparing video-derived tracks to BDB tracking truth. It does not authorize any predictive NFL-prop experiment.