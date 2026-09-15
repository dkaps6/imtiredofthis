# WR Phase 4A Full-Room Target Source Reconciliation V1 — Frozen Source-Only Plan

## Purpose

Validate an independent historical source for **actual full WR-room target mass** before any Phase-4 opportunity/efficiency attribution is allowed.

This is source validation only. It must not compute receiving-yard residual decomposition, fit a challenger, inspect sportsbook information, or make any production change.

## Authority

Exact WR-R15 OOS authority:

- run `34238301577`
- artifact `10061328722`
- artifact name `wr-r15-wr1-anchor-participation-v1`
- digest `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`

Files used:

1. `wr_r15_confirmation_predictions.csv` — graded OOS prediction/outcome cohort.
2. `wr_r15_confirmation_features.csv` — canonical WR2+ modeled-room membership and entitlement features.
3. `wr_r15_conservation_audit.csv` — team-game conservation proof only; not player identity.

## Structural facts frozen before this audit

- prediction cohort: 4,193 rows per variant, 1,088 `(season, week, team)` team-games;
- candidate WR1 present in 1,026 team-games;
- feature file is WR2+-only;
- the graded prediction cohort is not the full modeled WR room;
- candidate `pred_targets` summed over the graded subset is not a conserved room-mass object;
- R15 runtime conservation applies to entitlement mass, not to the incomplete graded prediction subset.

## Canonical modeled-room membership

For a team-game, define expected canonical modeled WR identities as:

- the candidate prediction WR1 anchor (`wr_rank == 1`), when present; plus
- every WR2+ `player_clean_key` in `wr_r15_confirmation_features.csv` for the same `(season, week, team)`.

A team-game belongs to the **truly complete graded subset** only when the set of candidate prediction identities equals that expected canonical modeled-room set exactly.

Frozen pre-audit expected count: **71 team-games**.

If the implementation does not reproduce 71, fail closed.

## Independent actual-target source

Use nflverse weekly player statistics through `nflreadpy.load_player_stats(..., summary_level="week")` and the repository's PlayerForm normalization/canonical team/name conventions.

WR positions for room aggregation: `WR`, `LWR`, `RWR`, `SWR`.

The source is used only for target counts in this audit.

## Reconciliation tests

### A. Player-level parity on the complete subset

For every candidate prediction identity in the 71 complete team-games:

- match raw weekly target count by `(season, week, team, player_clean_key)`;
- require `raw_targets == actual_targets` exactly within floating tolerance;
- a missing raw row may be treated as zero only when the authority `actual_targets` is also zero;
- any positive-target authority row without an exact raw identity match fails the gate.

This prevents room-total parity from hiding offsetting player errors.

### B. Full raw WR-room target-total parity

For each of the 71 complete team-games:

- sum raw weekly targets across **all** WR-position rows for that team-game;
- compare to summed authority `actual_targets` across the complete canonical modeled room;
- require exact equality within floating tolerance for every team-game.

If raw weekly stats contain additional WR target-getters outside the canonical modeled room, the audit must surface them explicitly and the gate fails. We do not silently drop them.

## Outputs

Write only source-reconciliation evidence:

- complete-subset membership CSV;
- player-level parity CSV;
- team-game room-total parity CSV;
- extra-raw-WR contributor CSV;
- source audit JSON;
- result JSON.

## Pass/fail

`PHASE4A_FULL_ROOM_TARGET_SOURCE_RECONCILED` only if all of the following hold:

1. exact authority artifact lineage passes;
2. 1,088 authority team-games reproduced;
3. 1,026 WR1-present team-games reproduced;
4. 71 truly complete canonical room team-games reproduced;
5. player-level target parity passes for every complete-subset authority identity;
6. full raw WR-room target-total parity passes for all 71 team-games;
7. zero ambiguous/duplicate raw identity keys in the audited subset;
8. sportsbook inputs = 0.

Otherwise disposition is `PHASE4A_FULL_ROOM_TARGET_SOURCE_NOT_RECONCILED` and **Phase 4 opportunity/efficiency attribution remains blocked**.

## Stop rules

A failed source reconciliation does not authorize:

- dropping mismatch games after seeing them;
- switching to the 1,026 WR1-present subset and calling it full-room;
- ignoring extra raw WR target-getters;
- name-fuzzy rescue;
- receiving-yard attribution;
- target-model redesign;
- production changes.

Any source-contract correction after a fail must be separately explained and re-frozen before another attempt.
