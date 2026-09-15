# WR Phase 4B — Prior-Roster Multi-Alias GSIS Target Source Audit V2 — Result

## Status

**SOURCE-ONLY MECHANICAL AUDIT PASSED.**

This document does **not** contain a Phase 4B receiving-yard attribution result and does not authorize a production change. It preserves the source/identity evidence required before the Phase 4B evaluator can be amended.

## Authority

Frozen WR-R15 authority:

- run: `34238301577`
- artifact: `10061328722`
- name: `wr-r15-wr1-anchor-participation-v1`
- digest: `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`
- variant: `WR_R15_WR1_ANCHORED_PARTICIPATION`
- candidate rows: `4,193`

## Audit lineage

Branch: `research-wr-phase4-authority-exact-opportunity-attribution-v1`

Script:
- `scripts/research/audit_wr_phase4b_prior_roster_gsis_alias_v2.py`
- commit: `cafb93a3fb525bb7b0278f61c212a588c69d0de8`

Workflow:
- `.github/workflows/research-wr-phase4b-prior-roster-gsis-alias-v2.yml`
- head: `defb5c218a2a6481d2ce2ac3fd940f612bdc74aa`

Canonical run:
- run: `34922511782`
- job: `104233530633`
- conclusion: `SUCCESS`
- artifact: `10378757263`
- artifact name: `wr-phase4b-prior-roster-gsis-alias-v2`
- digest: `sha256:70772b5c358f921e31681f0c2cfbf064556542f68dd9f4706fa0d09b9520c7fb`

## Why this audit was necessary

The initial Phase 4B preflight used an exact `player_clean_key` merge against nflverse weekly player statistics. That left roughly 40% of the WR2+ domain unresolved. A subsequent source audit showed most of those rows were present on weekly rosters, but an independent PBP audit proved that weekly-stat absence could not safely be treated as zero because some stat-absent rows had real target volume.

Claude then independently identified concrete name-key drift inside the unresolved set, including:

- `Chris Godwin` vs `Chris Godwin Jr.`
- `Velus Jones` vs `Velus Jones Jr.`
- `Nathaniel Dell` vs `Tank Dell`

The literal R17/R19 resolver pattern still begins from an exact authority-name key, so this audit tested a stricter deterministic multi-alias roster-to-GSIS bridge while retaining the same fail-closed and temporal protections.

## Frozen source-only identity rule tested

Weekly roster rows provide **identity aliases only**. They provide no target counts, receiving yards, projection inputs, or model features.

Available roster aliases retained when present:

- `full_name`
- `football_name`
- `player_name`
- `player`
- `name`
- `short_name`

Each alias is tied to a stable GSIS player ID.

For an authority row `(season, week, team, player)`, only roster rows strictly before that target week may contribute identity evidence. Resolution order:

1. current-team exact full alias
2. current-team suffix-insensitive alias
3. globally unique exact full alias
4. globally unique suffix-insensitive alias

If a stronger criterion is ambiguous, resolution fails closed. There is no fuzzy matching and no target-week roster identity resolution.

After a stable GSIS ID is resolved, actual targets are counted only from nflverse PBP using:

`REG week1-18 AND pass_attempt==1 AND two_point_attempt!=1 AND no_play!=1 AND receiver_player_id nonnull`

A zero target count is considered observed only when:

1. the identity is resolved to a stable GSIS ID,
2. the corresponding PBP team-game exists, and
3. there is no qualifying target event for that GSIS ID.

Unresolved or ambiguous identities remain excluded. No zero imputation is performed.

## Results

### Layer 4 — WR2+ allocation domain

- rows: `5,321`
- resolved GSIS: `5,244`
- resolved coverage: `98.55290358955083%`
- unresolved: `75`
- ambiguous: `2`
- PBP team-game missing: `0`
- resolved positive-target rows: `3,208`
- resolved zero-target rows: `2,036`

Resolution methods:

- current-team exact full alias: `5,160`
- current-team suffix-insensitive alias: `4`
- globally unique exact full alias: `78`
- globally unique suffix-insensitive alias: `2`
- no prior roster alias: `75`
- ambiguous: `2`

### Layers 2/3 — canonical anchor-observable domain

- rows: `6,012`
- resolved GSIS: `5,933`
- resolved coverage: `98.68596141051231%`
- unresolved: `77`
- ambiguous: `2`
- PBP team-game missing: `0`
- resolved positive-target rows: `3,998`
- resolved zero-target rows: `1,935`

Resolution methods:

- current-team exact full alias: `5,837`
- current-team suffix-insensitive alias: `4`
- globally unique exact full alias: `90`
- globally unique suffix-insensitive alias: `2`
- no prior roster alias: `77`
- ambiguous: `2`

## Strongest validation — exact parity against R15 authority

The audit independently resolved `4,154 / 4,193` R15 candidate authority rows and compared the GSIS/PBP target count against frozen R15 `actual_targets`.

- resolved authority rows: `4,154`
- exact target parity rows: `4,154`
- parity failures: `0`
- maximum absolute target difference: `0.0`

This establishes exact target-count parity on every candidate authority row for which the prospective identity bridge resolves a stable GSIS ID.

## Concrete drift cases

### Chris Godwin

The authority name `Chris Godwin` resolves to GSIS `00-0033921` using strictly-prior roster alias evidence. For 2023 Week 9, the PBP count is `6` targets, matching the weekly-stat row that displays `Chris Godwin Jr.`.

### Nathaniel / Tank Dell

The authority name `Nathaniel Dell` resolves to GSIS `00-0038977` once prior alias evidence exists. For 2023 Week 11, the PBP count is `10` targets, matching the weekly-stat `Tank Dell` row.

2023 Week 1 remains unresolved because no prior 2023 roster alias exists. That is intentional under the prospective strict-prior identity rule.

### Velus Jones

The authority name `Velus Jones` resolves to GSIS `00-0037745`; suffix display drift to `Velus Jones Jr.` no longer causes exclusion once prior identity evidence exists.

## Boundaries

Canonical run asserted:

- `receiving_yard_fields_loaded=false`
- `target_week_roster_used_for_resolution=false`
- `fuzzy_matching=false`
- `zero_imputation_performed=false`
- sportsbook inputs: `0`
- challenger model authorized: `false`
- production change: `false`

## Current disposition

`PHASE4B_IDENTITY_TARGET_SOURCE_MECHANICALLY_CLEARED_PENDING_INDEPENDENT_CLAUDE_SOURCE_AMENDMENT_REVIEW`

The Phase 4B evaluator must **not** yet be patched or run for receiving-yard attribution until the independent source-amendment review is complete.

Issue #535 review request: comment `5673916678`.
