# Defender Proximity Exposure Contract V1

**Status:** ENGINEERING CONTRACT — NO PRODUCTION SCIENCE CHANGE

## Purpose

Define a leakage-safe, semantically constrained surface for WR/TE defender-proximity exposure derived from approved tracking data. This is infrastructure and descriptive feature engineering only. It does not authorize projection changes, betting changes, coverage-assignment claims, or predictive-lift experiments.

## Semantic firewall

**Proximity is not coverage assignment.**

A nearby defender at any sampled instant may reflect zone structure, switch/release rules, help responsibility, motion, scramble response, bracket structure, or incidental geometry. V1 must never label a defender as the player's authoritative matchup, primary coverage defender, shadow assignment, or responsibility owner.

Permitted language:

- nearest-defender distance;
- local defender density;
- release-window proximity;
- throw-arrival proximity;
- route-window proximity exposure;
- player/route historical proximity profile.

Forbidden language without a separately qualified source:

- covered by;
- assigned defender;
- shadowed by;
- CB1 matchup;
- man-coverage responsibility.

## Canonical grain

Raw derived observations:

`season, week, game_id, play_id, player_id, frame_window, route_family`

Strict-prior aggregate surface:

`season, week, game_id, player_id, market_family`

Stable player/game/play identifiers are required whenever supplied by the source. Name-only joins are forbidden.

## Authorized geometry

When source fields support them, V1 may derive:

- nearest eligible defender distance at release-window samples;
- nearest eligible defender distance during route progression;
- nearest eligible defender distance at target/throw-arrival windows;
- count of eligible defenders within fixed distance bands;
- percentile/median proximity by player;
- percentile/median proximity by player × route family;
- dispersion and sample count;
- missing/invalid frame counts;
- source season/week coverage.

Distance bands and frame windows must be declared in a versioned manifest before descriptive outputs are inspected. They may not be selected by realized betting/model performance.

## Route handling

Route family may be used only when the underlying route label is source-qualified or produced by an independently frozen deterministic classifier. Unknown route is a valid state and must not be silently imputed.

Route-conditioned proximity metrics must always retain unconditioned sample counts so sparse route families are visible.

## Defender eligibility

Defender candidates must be players on the opposing defense present in the approved tracking frame. Eligibility logic must be deterministic and documented.

The algorithm may select the nearest eligible defender geometrically. It may not infer responsibility from that selection.

## Strict-prior historical surface

For target kickoff `T`, every historical proximity aggregate supplied to a target row must use only source observations from games completed before `T`.

Required fields include:

- `proximity_version`;
- `source_manifest_version`;
- `history_cutoff`;
- `games_n`;
- `plays_n`;
- `frames_n`;
- route-conditioned sample counts where applicable;
- missingness/coverage state.

No target-game tracking data may enter a pregame feature.

## Persistence/descriptive research boundary

Engineering QA may measure season/week coverage, deterministic reproducibility, and within-player descriptive stability to validate that a field is not random ingestion noise.

This contract does **not** authorize testing whether the field improves projections, residuals, calibration, hit rate, ROI, or bet selection. Such science requires a separately frozen plan and blind gate.

## Output surfaces

### Play/window exposure table

One row per player/play/window with:

- stable IDs;
- route family/state;
- frame-window definition;
- nearest-defender distance summaries;
- defender-density summaries;
- eligible defender count;
- valid frame count;
- missing frame count;
- source/version fields.

### Pregame historical proximity table

One row per target player/game containing strict-prior rolling summaries, sample counts, cutoff, and coverage state.

### QA audit

Must report:

- source coverage by season/week;
- stable-ID join rate;
- valid geometry rate;
- unknown-route rate;
- zero/implausible distance rate;
- sparse-history rate;
- deterministic hash/replay result;
- leakage checks.

## QA invariants

1. Target-game frames are excluded from target pregame aggregates.
2. Every defender candidate belongs to the opposing team for the tracked play.
3. Stable IDs drive joins; normalized names may be audit-only.
4. Missing geometry remains missing and never becomes zero.
5. Unknown route remains explicit.
6. Distance units are declared and consistent.
7. Frame-window definitions are versioned and deterministic.
8. Aggregates retain sample counts.
9. Repeated builds from identical inputs are byte-deterministic after canonical ordering.
10. No output field asserts coverage responsibility.
11. No betting/model outcome is used to choose thresholds, windows, route groupings, or weights.
12. No production artifact is modified by this layer.

## Integration hooks

Authorized consumers are engineering-only context stores, the historical analog descriptor registry, and future separately authorized research datasets.

The historical analog layer may consume certified strict-prior proximity descriptors under `HISTORICAL_ANALOG_INDEX_CONTRACT_V1`, but analog selection remains outcome-agnostic and no failed-closed analog family is reopened.

## Not authorized

- production projection changes;
- bet-selection changes;
- outcome-trained thresholds;
- outcome-trained proximity weights;
- authoritative WR-CB matchup labels;
- paid data acquisition;
- reopening any failed-closed research family.

## Disposition

`DEFENDER_PROXIMITY_EXPOSURE_CONTRACT_V1_FROZEN`
