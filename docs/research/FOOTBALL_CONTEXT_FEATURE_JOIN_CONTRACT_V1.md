# Football Context Feature Join Contract V1

**Status:** RESEARCH / ENGINEERING CONTRACT — NO PRODUCTION SCIENCE CHANGE

## Purpose

Define one leakage-safe join surface for new football-context datasets so BDB/tracking, role/environment, personnel continuity, matchup analog, defender-proximity, and OL/DL context can be engineered without rebuilding position-specific historical universes or silently changing production science.

## Canonical grains

### Player-game context
Primary key:

`season, week, game_id, player_id`

Fallback identity is permitted only through the repository's existing verified identity registry. Name-only joins are not an acceptable canonical key.

### Team-game context
Primary key:

`season, week, game_id, team`

### Player-opponent exposure context
Primary key:

`season, week, game_id, player_id, opponent_player_id, context_family`

This is the preferred surface for WR/TE defender proximity and blocker/rusher interaction summaries. It is an exposure table, not a declaration of exclusive matchup responsibility.

## Strict-prior contract

Every feature intended for a pregame research row must carry enough provenance to prove it was knowable before the target game's kickoff.

Required provenance fields for externally observed/current-state inputs:

- `source_family`
- `source_locator` or stable source identifier
- `observed_at_utc`
- `effective_at_utc` when distinct from observation time
- `target_game_start_utc`
- `strict_prior_eligible`
- `lineage_version`

For historical aggregates derived only from completed games, the latest contributing game must precede the target game's kickoff. Same-game outcome information is forbidden.

## Feature-family namespaces

New columns must be namespaced by family to prevent collisions and make ablation possible later without changing the base table.

Recommended prefixes:

- `role_` — player role/environment state
- `continuity_` — roster/team/personnel continuity
- `analog_` — historical matchup similarity descriptors
- `bdb_` — Big Data Bowl/tracking-derived geometry
- `coverage_` — defender proximity/exposure summaries
- `trench_` — blocker/rusher and OL/DL interaction summaries
- `coach_` — coaching/play-caller environment

## Base-history reuse

The join layer consumes the canonical historical player-game and team-week foundations defined by `HISTORICAL_DATA_REUSE_POLICY_V1.md`.

It must not independently download or reconstruct separate QB/RB/WR/TE base histories.

If an exact historical base artifact is retained, reuse it. If not retained, deterministic rehydration through the frozen canonical builder is allowed and is not new science.

## Missingness is evidence

No new context family may silently convert unavailable data into a neutral football statement.

Each family should expose explicit availability fields such as:

- `<family>_available`
- `<family>_coverage_count`
- `<family>_source_quality`

Unknown, not-applicable, and observed-zero must remain distinguishable.

## Tracking/BDB boundary

BDB/tracking-derived features are allowed as genuinely new information. They must remain versioned separately from ordinary game-log history.

Tracking features may summarize geometry, separation, spacing, engagement, motion, alignment, or exposure. They may not be backfilled into seasons for which the source does not exist using outcome-trained inference and then represented as observed history.

## WR/TE defender proximity boundary

Defender proximity should initially be represented as probabilistic/exposure context, for example:

- share of relevant frames/snaps near defender;
- average/minimum separation conditional on route/target state;
- repeated exposure counts;
- alignment-conditioned proximity.

Do not label a defender as the receiver's authoritative shadow/assignment solely from proximity unless an independently validated assignment rule/source exists.

## OL/DL boundary

Trench context should distinguish:

1. observed tracking interaction geometry;
2. inferred blocker-rusher exposure;
3. authoritative protection/assignment labels.

The first two are allowed research surfaces. The third requires a source that actually establishes responsibility; nearest-player geometry alone must not be promoted to an assignment label.

## Historical analog boundary

The analog layer is an index over pregame-known descriptors, not a new model family by default.

Each analog query must record:

- target row identity;
- feature set/version;
- eligible historical cutoff;
- distance/similarity definition;
- candidate count before filtering;
- returned neighbor count;
- season distribution of neighbors.

Outcome columns may be attached only after neighbors are selected from pregame descriptors. They must never participate in neighbor selection.

## Role/environment boundary

Role changes should be represented as timestamped state transitions where possible rather than overwritten current labels. Examples include team change, depth/room change, injury-driven opportunity change, coaching/play-caller change, and explicit role statements.

Qualitative statements require source provenance and observation time. Derived opportunity changes from canonical history should retain their formula/version.

## No-science-change rule

Creating, joining, validating, or profiling these context tables does not authorize predictive integration.

No feature from this contract enters production weights, projections, probability calibration, bet selection, or promotion gates without a separately frozen experiment and explicit disposition.

## QA gates before any later experiment

A context-family artifact must fail closed if any applicable gate fails:

- key uniqueness at declared grain;
- team/game/player identity resolution;
- strict-prior timestamp eligibility;
- source lineage present;
- missingness explicitly represented;
- no outcome columns used to construct pregame descriptors;
- coverage reported by season/week/position where applicable;
- deterministic rebuild or documented source snapshot/hash.

## Disposition

`FOOTBALL_CONTEXT_FEATURE_JOIN_CONTRACT_V1_FROZEN`
