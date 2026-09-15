# WR Phase 4A Full-Room Target Source Reconciliation V1 — Result

## Disposition

**`PHASE4A_FULL_ROOM_TARGET_SOURCE_RECONCILED`**

The independent nflverse weekly player-stat source exactly reconciles to the canonical WR-R15 authority on the truly complete modeled-room validation subset. Phase-4 diagnostic attribution is therefore authorized. Challenger modeling remains unauthorized.

## Canonical lineage

- branch: `research-wr-phase4-authority-exact-opportunity-attribution-v1`
- frozen plan: `1c3094aea72b533f4250f3c185113d49a6ed8970`
- source-audit implementation: `7b86b97043c3b353d18ee1ceb65a9335d6fb5e79`
- workflow head: `e6fa16cc3f128c4d8729c33333e069ba38ab2bad`
- workflow: `.github/workflows/research-wr-phase4a-target-source-v1.yml`
- run: `34919390334`
- job: `104223873835`
- artifact: `10376694364`
- artifact name: `wr-phase4a-target-source-reconciliation-v1`
- artifact digest: `sha256:069dfe4d935d58e9fd232dea9c08ae07a306ade811e7382ccf5f5b23d4c89548`

Exact WR-R15 authority:

- run `34238301577`
- artifact `10061328722`
- digest `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`

## Structural correction carried into the audit

The WR-R15 confirmation predictions are a graded OOS subset, not the full modeled WR room.

- 1,088 authority team-games
- WR1 present in the graded predictions for 1,026 team-games
- only 71 team-games have candidate prediction identities exactly equal to the WR1 anchor plus every canonical WR2+ identity in `wr_r15_confirmation_features.csv`

Therefore the source reconciliation was run on those 71 truly complete team-games, not on all 1,026 WR1-present games.

## Exact source-reconciliation result

- complete canonical team-games: **71**
- complete canonical player rows: **369**
- raw WR rows on those games: **369**
- player-level parity rows: **369 / 369**
- player-level parity failures: **0**
- positive-target authority rows missing raw identity: **0**
- max absolute player target delta: **0.0**
- room-total parity team-games: **71 / 71**
- room-total parity failures: **0**
- max absolute room target delta: **0.0**
- extra raw positive-target WR contributors outside canonical room: **0**
- extra raw WR targets: **0.0**
- receiving-yard fields loaded: **false**
- sportsbook inputs: **0**

## Interpretation

The weekly nflverse / PlayerForm-normalized target source reproduces the canonical authority target counts exactly at both player and full-room level wherever the canonical artifact exposes a truly complete modeled WR room.

This clears that source for the next diagnostic layer:

1. actual full WR-room target mass across all 1,088 authority team-games;
2. M38 WR1 target attribution where anchor identity is observable;
3. WR2+ pool and within-pool allocation attribution;
4. exact opportunity-vs-efficiency receiving-yard error decomposition on the canonical graded OOS rows.

This source result does **not** imply R15 is wrong, does not authorize a challenger, and does not change production science.

## Boundaries

- no production change
- no challenger model
- no sportsbook feature selection
- no paid Full Slate
- no RB work
