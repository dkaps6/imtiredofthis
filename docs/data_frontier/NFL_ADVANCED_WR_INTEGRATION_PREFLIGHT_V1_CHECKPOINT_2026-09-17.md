# NFL Advanced Feature WR Integration Preflight V1 — Checkpoint — 2026-09-17

## Scope

Engineering/integration preflight only.

This checkpoint answers whether the BDB 2026 Analytics strict-prior targeted-receiver history can be joined cleanly to the project's GSIS/nflverse receiving-yards history without loose identity matching or target-week leakage.

It does **not** fit a model, calculate predictive metrics, retune production, use sportsbook inputs, or touch Issue #535.

Canonical successful run: `35288481584`  
Run head: `70b9dc80f3363920170a590bf75806b6feebefc7`  
Sanitized artifact ID: `10525711036`  
Artifact digest: `sha256:e9e8278096c77e728824b98de8686b971832afe272bbcb1a16310d835564dd20`

Final disposition:

**`NFL_ADVANCED_WR_INTEGRATION_PREFLIGHT_V1_COMPLETE`**

## Identity result

BDB targeted-receiver identities: **464**

Certified to GSIS/nflverse identity: **457 / 464 = 98.4914%**

Certified methods:

- `NAME_DOB_POSITION`: **455**
- `NAME_POSITION_PHYSICAL`: **2**

Excluded from certified joins:

- `NAME_POSITION_UNIQUE_REVIEW_ONLY`: **7**
- unresolved: **0**

Direct BDB `nfl_id` -> nflverse `nfl_id` match rate: **0.0%**

This is an important semantic result. The BDB tracking `nfl_id` must not be assumed to be the nflverse player-table `nfl_id`. The integration layer therefore uses corroborated player identity and promotes only certified GSIS mappings.

The 7 review-only identities remain excluded from research joins unless separately verified. Name-only uniqueness is not sufficient.

Ephemeral crosswalk:

- rows: **464**
- SHA-256: `4a2cc86fb09bf772025fa8b90d625c85e54a97b86493d1ca43dd4366f38c689c`
- uploaded: **NO**

nflverse player file used:

- SHA-256: `769f144ca86a0c3c2d7a59ad02fe370a206926ec71e16c9c1dc7f09dd1715c87`

## Receiving-outcome bridge

2023 nflverse regular-season weekly outcome table:

- rows: **17,806**
- unique GSIS identities: **1,942**
- weeks: **1–18**
- identity column: `player_id`
- receiving-yards column: `receiving_yards`

Ephemeral weekly-outcome derivative:

- SHA-256: `b381b15d22135e796b07ab971db2bd5214a973b5abd7267eae085fb507c4a8d3`
- uploaded: **NO**

## Advanced-history panel

Strict-prior receiver-history snapshots from Advanced Feature V1:

- rows: **7,161**
- unique BDB identities: **464**
- certified-identity history rows: **7,059**
- rows with all four core advanced receiver-history fields ready: **4,001**
- rows joined to receiving outcome: **4,432**

Core advanced fields:

- `hist_receiver_release_nearest_defender_median_yards`
- `hist_receiver_release_second_defender_median_yards`
- `hist_receiver_release_crowding_2yd_rate`
- `hist_receiver_release_crowding_3yd_rate`

WR candidate cohort satisfying all gates:

- **1,548 player-weeks**
- **152 unique WRs**
- target weeks **2–18**

Week-level WR candidate counts:

| Week | Rows |
|---:|---:|
| 2 | 15 |
| 3 | 54 |
| 4 | 81 |
| 5 | 76 |
| 6 | 91 |
| 7 | 79 |
| 8 | 109 |
| 9 | 93 |
| 10 | 91 |
| 11 | 98 |
| 12 | 106 |
| 13 | 89 |
| 14 | 109 |
| 15 | 117 |
| 16 | 118 |
| 17 | 110 |
| 18 | 112 |

Ephemeral integration panel:

- rows: **7,161**
- SHA-256: `b6264480b520cc2b6db152e932116d218eb9ec4c8545343c8c63ade7b45c27b3`
- uploaded: **NO**

## Temporal result

The panel reuses the Advanced Feature V1 strict-prior contract.

Verified:

- target-game observation used: **NO**
- same-game partial history used: **NO**
- strict-prior history only: **YES**

No target-game BDB release geometry, landing geometry, post-release geometry, receiving yards, targets, or receptions enter an advanced-history feature.

The receiving-yard value is outcome/label only for a future controlled research experiment.

## What this means

The advanced throw-window data is not merely a historical geometry curiosity.

There is now a deterministic, identity-certified, leakage-safe path from:

`BDB 2026 tracking -> strict-prior receiver advanced history -> GSIS identity -> 2023 weekly receiving-yard outcome`

with a non-trivial WR candidate cohort.

This establishes **research feasibility**, not predictive value.

No correlation, MAE, RMSE, feature importance, model fit, or promotion gate has yet been calculated.

## Cross-position implication

This WR panel is the first proof of the integration pattern, not a position boundary.

The same architectural principle is now tracked separately under:

`docs/data_frontier/NFL_CROSS_POSITION_CONTEXT_STATE_V1_ENGINEERING_PLAN.md`

That lane covers QB/RB/WR/TE directly and OL/defensive/coaching context as environment providers.

For target-receiver geometry specifically, future preflights may separately enumerate TE/RB/FB cohorts. The current WR cohort was selected only because receiving-yards research is the active lane.

## Next scientific boundary

The current dataset is large enough to justify designing **one pre-registered controlled research candidate**, subject to active-research coordination.

The first candidate should not be "advanced data kitchen sink."

It should isolate one mechanism, for example:

> Does strict-prior targeted-receiver release-space/crowding history add stable out-of-sample information to the frozen WR receiving-yards baseline after existing opportunity/entitlement features are accounted for?

The separate cross-position context-state layer should then allow later experiments to test whether advanced historical skill interacts with meaningful environment transitions such as new QB, new team, role promotion, or changed room competition.

## Disposition

**`NFL_ADVANCED_WR_INTEGRATION_PREFLIGHT_V1_COMPLETE_RESEARCH_FEASIBLE_NO_PREDICTIVE_TEST_YET`**
