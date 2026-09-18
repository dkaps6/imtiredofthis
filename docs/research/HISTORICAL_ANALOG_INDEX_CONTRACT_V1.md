# Historical Analog Index Contract V1

**Status:** ENGINEERING CONTRACT — NO PRODUCTION SCIENCE CHANGE

## Purpose

Define a leakage-safe historical-neighbor index that can retrieve football-context analogs from the repository's canonical historical base without outcome-trained neighbor selection. V1 is infrastructure only: it freezes descriptor construction, eligibility, distance accounting, provenance, and QA. It does not test whether analog outcomes improve projections or bets.

## Canonical target grain

`season, week, game_id, player_id, market_family`

Each target row must carry:

- `target_kickoff`
- `analog_index_version`
- `descriptor_version`
- `candidate_cutoff`
- `candidate_count`
- `eligible_candidate_count`
- `neighbor_count_requested`
- `neighbor_count_returned`
- `season_distribution`
- `coverage_state`
- `materialized_at`

## Strict-prior candidate boundary

For target kickoff `T`, a candidate game is eligible only if its game was completed before `T` and every descriptor used for that candidate was itself available under the candidate's pregame cutoff.

Forbidden in neighbor selection:

- target-game box score or result;
- target-game snaps/routes/touches/targets;
- target or candidate postgame role statements;
- projection residuals or realized betting outcomes;
- sportsbook result labels;
- future-season information;
- any descriptor fitted using the target outcome.

Historical rows must reuse the canonical historical base under `HISTORICAL_DATA_REUSE_POLICY_V1`; the analog layer must not create a separate QB/RB/WR/TE raw-history universe.

## Descriptor families authorized for indexing

Descriptors are pregame state only and must remain namespaced.

### Player usage/history

Examples when supported by canonical strict-prior history:

- prior games with current team;
- rolling pass/rush/target/reception opportunity;
- prior target/carry shares;
- prior route/snap shares only when source-qualified;
- rolling efficiency summaries with explicit sample counts;
- prior availability state.

### Role/environment regime

From the frozen role/environment ledger and personnel-continuity layer:

- same-team continuity;
- games since team/depth/availability transition;
- room additions/departures;
- prior observed departed opportunity;
- coaching/play-caller continuity;
- explicit source/coverage flags.

No field may claim that vacated opportunity will transfer to a specific player.

### Team/game environment

Leakage-safe pregame team context may include:

- pace/pass/rush tendency summaries;
- opponent historical defensive context;
- home/away;
- rest and schedule descriptors;
- weather when known pregame;
- injury/availability context.

Sportsbook descriptors are excluded from V1 unless separately authorized. No paid odds pull is required or permitted by this contract.

### Advanced geometry/exposure hooks

Versioned strict-prior BDB-derived descriptors may later join when certified at the frozen context surfaces, including receiver/route separation history, defender proximity exposure, and blocker/rusher interaction history.

Semantic firewalls remain mandatory:

- proximity is not coverage assignment;
- interaction is not authoritative blocking responsibility.

## Market-family separation

Neighbor indexes may share a canonical descriptor store, but each `market_family` must declare its allowed descriptor subset and scaling manifest. V1 may define descriptor availability for passing, rushing, receiving, receptions, and combined opportunity families without learning market-specific weights from outcomes.

## Distance construction

V1 permits deterministic, outcome-agnostic distance only.

Required behavior:

1. numeric descriptors are scaled using parameters fitted only on candidate rows eligible before target kickoff;
2. categorical mismatch penalties are fixed in the versioned descriptor manifest, not outcome-tuned;
3. missingness is represented explicitly and cannot silently become zero;
4. distance contributions are retained by descriptor family;
5. deterministic tie-breaking uses stable keys, never realized outcome.

Default engineering baseline: equal-weight normalized distance across available approved descriptors, with family contribution counts reported. This baseline is not a scientific claim and is not authorized for production.

## Neighbor freeze before outcome attachment

The index must materialize and hash the neighbor selection before any realized candidate outcomes are attached for descriptive research.

Required sequence:

1. construct target pregame descriptor;
2. construct eligible candidate pool;
3. compute deterministic distances;
4. freeze ordered neighbor IDs and selection hash;
5. only then, in a separate downstream step, attach candidate realized outcomes if an authorized research plan requires them.

The selection artifact itself must contain no target outcome.

## Output surfaces

### Analog selection table

One row per target-neighbor relation:

`season, week, game_id, player_id, market_family, neighbor_rank, analog_game_id, analog_player_id`

Plus:

- `distance_total`
- per-family distance contributions;
- descriptor overlap count;
- missing descriptor count;
- candidate game completion timestamp;
- selection hash;
- source manifest/version fields.

### Analog target audit

One row per target containing candidate/neighbor counts, season distribution, coverage, cutoff, and exclusion reason counts.

### Descriptor manifest

Versioned declaration of every descriptor's source family, grain, type, scaling rule, missingness rule, strict-prior eligibility, and semantic caveats.

## Sparse-history safeguards

- Do not force `k` neighbors when eligible history is insufficient.
- Report `neighbor_count_returned < neighbor_count_requested` explicitly.
- Preserve season distribution to expose concentration in a single era.
- New-team/new-role players retain transition state rather than being coerced into long-tenure analogs.
- Advanced-data fields unavailable in older seasons may reduce overlap but may not exclude all older history unless a later frozen research plan explicitly requires that field family.

## QA invariants

1. Every candidate game completed before target kickoff.
2. Every target descriptor is strict-prior at target kickoff.
3. Candidate descriptor state is pregame-relative to the candidate game, not reconstructed with hindsight.
4. Neighbor selection is byte-deterministic for identical source manifests.
5. Target outcome is absent from selection inputs and selection artifact.
6. Outcome attachment, when authorized later, occurs only after neighbor IDs are frozen.
7. Stable IDs are used for players/games whenever available; name-only canonical joins are forbidden.
8. Missingness remains distinguishable from zero.
9. Distance contribution totals reconcile to the reported total distance.
10. Candidate and exclusion counts reconcile.
11. No failed-closed research family is reopened by materializing this infrastructure.

## Authorized engineering work

Authorized now:

- descriptor registry/schema implementation;
- strict-prior candidate-pool materializer;
- deterministic scaling/distance utilities;
- neighbor selection and hashing;
- QA/coverage audits;
- synthetic/unit tests;
- manifests and sanitized artifacts.

Not authorized now:

- predictive-lift experiments;
- tuning descriptor weights against outcomes;
- choosing `k` by betting/model performance;
- projection/calibration changes;
- bet-selection changes;
- production integration;
- reopening failed-closed analog architectures.

## Relationship to prior failed QB conditional analog work

This contract does **not** rerun, rescue, or reinterpret the failed-closed QB conditional analog V1 architecture. It creates a general-purpose historical context retrieval surface for future separately authorized research. Any future scientific analog hypothesis requires its own frozen plan and blind gate.

## Disposition

`HISTORICAL_ANALOG_INDEX_CONTRACT_V1_FROZEN`
