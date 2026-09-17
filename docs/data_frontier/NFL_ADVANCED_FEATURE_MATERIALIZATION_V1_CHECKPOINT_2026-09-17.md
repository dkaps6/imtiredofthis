# NFL Advanced Feature Materialization V1 — Checkpoint — 2026-09-17

## Scope

This checkpoint records the first clean deterministic materialization of the frozen
`NFL_ADVANCED_FEATURE_DICTIONARY_V1` contract.

It is an engineering result only.

- predictive experiment: **NO**
- production-science change: **NO**
- sportsbook input: **NO**
- Issue #535 change: **NO**
- raw Kaggle competition files committed/uploaded: **NO**
- derived per-row feature tables committed/uploaded: **NO**
- sanitized QA/manifests uploaded: **YES**

Branch:
`data-frontier-advanced-feature-materializers-v1`

Parent feature-contract branch:
`data-frontier-advanced-feature-contract-v1`

Canonical Actions run:
`35287452116`

Run source SHA:
`98aaed4f64a71d81a4dba24a7a820a1da7a5674c`

Final disposition:

**`NFL_ADVANCED_FEATURE_MATERIALIZATION_V1_ALL_SOURCES_PASS`**

## What was built

Reusable source-specific materializers now exist for:

1. BDB 2021 / 2018 route and defender-proximity geometry;
2. BDB 2023 / 2021 blocking-interaction and protection geometry;
3. BDB 2026 Analytics / 2023 targeted-receiver throw-window geometry.

Each materializer:

1. downloads the official source ephemerally in GitHub Actions;
2. recomputes the frozen source fingerprint;
3. fails closed on source-hash drift;
4. reconstructs the exact frozen geometry;
5. creates the contracted raw/retrospective fields ephemerally;
6. creates strict-prior historical snapshots where the dictionary allows them;
7. validates temporal leakage rules;
8. hashes the complete ephemeral derivative tables;
9. uploads only sanitized QA JSON;
10. explicitly records that no predictive experiment or production change occurred.

## BDB 2021 route geometry result

Official source:
`nfl-big-data-bowl-2021`

Verified source hash:
`55de76561799514779f3fd64b57c02a52430596f4f104c2b6060f3df8fd9e1b4`

Contracted fields: **12 / 12 materialized**

Real-corpus results:

- route player-plays: **78,343**
- conflicting route-label player-plays: **0**
- wrong-side route labels: **0**
- strict-prior player x route history snapshots: **58,484**
- temporal violations: **0**
- target-game rows used in pregame history: **0**

Ephemeral derivative hashes:

- `bdb2021_route_playerplays_v1.csv`
  - rows: **78,343**
  - SHA-256:
    `3b97f8afbf26cd39cc4ba1a5f043677ce48c900c9350a285b2e7380d8aee0ea4`
- `bdb2021_route_history_snapshots_v1.csv`
  - rows: **58,484**
  - SHA-256:
    `e9e4cd626a0e0cacd8e8abc9e5266867ba3352ddaa6460cd4c5a9fe19184f2bf`

Sanitized QA artifact:

- artifact ID: `10525126488`
- digest:
  `sha256:6d0ffc264a6c49b0b8fd0257a5ec933c8bd55a23b53efd04a1a2311465174ed6`

## BDB 2023 protection result

Official source:
`nfl-big-data-bowl-2023`

Verified source hash:
`1c3e1eb6fcd0cf85807c649804fa0c72421c76e973e40a742523f1dd3a929182`

Contracted fields: **13 / 13 materialized**

Real-corpus results:

- resolvable source interactions: **46,524**
- reconstructed interactions: **46,396**
- pooled geometry coverage: **99.724873%**
- duplicate blocker-target pairs: **0**
- unresolved blocked-player references: **2**
- strict-prior blocker-history snapshots: **3,935**
- temporal violations: **0**
- target-game rows used in pregame history: **0**

The original semantic limitation remains intact:

`pff_nflIdBlockedPlayer` is a blocked-player / blocking-interaction identity.
It is not promoted to universal primary blocker-rusher responsibility.

Ephemeral derivative hashes:

- `bdb2023_protection_interactions_v1.csv`
  - rows: **46,396**
  - SHA-256:
    `521682a6c2e0e5ced6b4ab96accf793aae061c63a0234dca76ce4459e6dd206d`
- `bdb2023_blocker_history_snapshots_v1.csv`
  - rows: **3,935**
  - SHA-256:
    `d09077cdb81a4b53676070f213f3ee4662e0eb06f853b1dbf35f3c1a138bcb78`

Sanitized QA artifact:

- artifact ID: `10524658390`
- digest:
  `sha256:e0b4a53508869255706889a5e3202664641c63c451737af7e20df6cc80557400`

## BDB 2026 throw-window result

Official source:
`nfl-big-data-bowl-2026-analytics`

Verified source hash:
`228554c6600ac4e73529e5b6309193c8a3371acca7b4bc59792188acd814fb07`

Contracted fields: **22 / 22 materialized**

Real-corpus results:

- published plays: **14,108**
- valid release geometry: **14,107**
- release coverage: **99.992912%**
- terminal targeted-receiver geometry: **14,107 / 14,107**
- targeted-receiver role non-unique: **0**
- targeted receiver outside official prediction scope: **0**
- targeted receiver missing from output: **0**
- plays with no official predicted defender in post-release output: **1,141**
- strict-prior receiver-history snapshots: **7,161**
- strict-prior receiver x route snapshots: **35,417**
- route x man/zone benchmark rows: **24**
- route x detailed coverage-family benchmark rows: **93**
- temporal violations: **0**
- target-game rows used in pregame history: **0**
- landing/post-release fields used pregame: **0**

Nearest-defender semantics remain geometric proximity only.
No individual coverage responsibility is claimed.

Ephemeral derivative hashes:

- `bdb2026_targeted_receiver_throw_window_v1.csv`
  - rows: **14,107**
  - SHA-256:
    `dbdfc1cebb95d8c0cd31b2be71311fb7289c5d3db643c86bab3ed43e342340fd`
- `bdb2026_receiver_history_snapshots_v1.csv`
  - rows: **7,161**
  - SHA-256:
    `e60882727a03ba14ff9e43d5229d6d491d8833a76f91d9f6412f313c89993efe`
- `bdb2026_receiver_route_history_snapshots_v1.csv`
  - rows: **35,417**
  - SHA-256:
    `f75be8dd9907eb5396f7f050952c23271a128ff64ef8422fe1d35785a7b0a44f`
- `bdb2026_route_man_zone_benchmark_v1.csv`
  - rows: **24**
  - SHA-256:
    `68a8cb8351fdc830f92c52bd42ad5153e8855ce946286c62b24c1ea2538abb0c`
- `bdb2026_route_coverage_family_benchmark_v1.csv`
  - rows: **93**
  - SHA-256:
    `44b3c31390afce8d0e85ed5970d5160a79262fcecf048b7d1e9af69ec03b05ba`

Sanitized QA artifact:

- artifact ID: `10525021698`
- digest:
  `sha256:c4e7aa3f781bca0becfd97df6674eb63351459685a2d724e1806b6c37fefa4bf`

## Temporal-use result

The most important engineering result is that the materializers now make the
pregame/post-kickoff boundary executable rather than advisory.

Pregame-capable V1 features are generated only as strict-prior snapshots.

The run proved:

- same-week / future-history violations: **0**
- same-game partial-history usage: **0**
- target-game observations in pregame history: **0**
- BDB 2026 landing/post-release pregame usage: **0**

If a future experiment wants a raw target-game release/throw/arrival/post-release
value, it must fail the V1 temporal contract.

## Source-rights boundary

The implementation preserves the existing source-access design.

Full source data and full per-row derived feature tables exist only on the
ephemeral Actions runner. GitHub retains:

- materializer code;
- source hashes;
- contracts;
- QA counts/distributions;
- temporal audits;
- ephemeral derivative file hashes;
- sanitized artifacts/checkpoints.

Authenticated Kaggle research access is not interpreted as unrestricted
production/live-feed licensing.

## What this unlocks

The project now has reproducible historical feature generators for families that
were previously unavailable to the production/research stack, including:

- route-conditioned receiver proximity history;
- first/second defender spacing history;
- receiver crowding history;
- blocking-interaction distance history;
- protection timing history;
- chip/release interaction semantics;
- targeted-receiver release-space history;
- route x coverage-family geometry benchmarks.

These are now engineered data assets.

They are **not yet predictive winners**.

## Next research boundary

The next step may be a controlled predictive experiment, but only after selecting
one pregame-safe hypothesis and coordinating it with the active research lane.

The highest-value first candidates for the current receiving-yards weakness are
the BDB 2026 strict-prior targeted-receiver release features:

- `hist_receiver_release_nearest_defender_median_yards`
- `hist_receiver_release_second_defender_median_yards`
- `hist_receiver_release_crowding_2yd_rate`
- `hist_receiver_release_crowding_3yd_rate`
- `hist_receiver_release_geometry_sample_count`

A route-conditioned feature may also be considered:

- `hist_receiver_route_release_nearest_defender_median_yards`

but the target-game route itself is not known pregame and cannot be supplied from
retrospective source truth. Any route-conditioned deployment must use a separately
validated pregame route-tendency/scenario mechanism.

Protection features should remain a separate QB/pass-efficiency research family
rather than being mixed casually into the first WR candidate.

## Disposition

**`NFL_ADVANCED_FEATURE_MATERIALIZATION_V1_ALL_SOURCES_PASS`**

Feature contracts and deterministic materializers are now frozen enough to support
an explicitly authorized, leakage-safe research candidate without recreating the
underlying data-discovery work.
