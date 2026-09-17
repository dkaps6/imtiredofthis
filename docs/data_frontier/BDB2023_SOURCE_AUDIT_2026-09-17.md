# BDB 2023 Official Source Audit — 2026-09-17

## Scope

Isolated NFL data-frontier source audit only. No predictive experiment, no production-science change, no Issue #535 change.

- Branch: `data-frontier-phase0-bdb-contact-v1`
- Workflow: `.github/workflows/data-frontier-bdb2023-source-audit.yml`
- Canonical run: `35264295731`
- Workflow commit: `3cc21316690f5e53bee80b339f9f809a67ac2767`
- Audit version: `BDB2023_SOURCE_AUDIT_V1`
- Official Kaggle competition: `nfl-big-data-bowl-2023`
- Raw competition files uploaded: **NO**
- Corpus stored only in GitHub runner temporary storage during the audit.

## Corpus identity

Canonical source-manifest SHA-256:

`1c3e1eb6fcd0cf85807c649804fa0c72421c76e973e40a742523f1dd3a929182`

All expected files were present. No expected file was missing.

| File | Rows | Columns |
|---|---:|---:|
| `games.csv` | 122 | 7 |
| `players.csv` | 1,679 | 7 |
| `plays.csv` | 8,557 | 32 |
| `pffScoutingData.csv` | 188,254 | 15 |
| `week1.csv` | 1,118,122 | 16 |
| `week2.csv` | 1,042,774 | 16 |
| `week3.csv` | 1,121,825 | 16 |
| `week4.csv` | 1,074,606 | 16 |
| `week5.csv` | 1,097,813 | 16 |
| `week6.csv` | 973,797 | 16 |
| `week7.csv` | 906,292 | 16 |
| `week8.csv` | 978,949 | 16 |

## Exact PFF scouting schema

`pffScoutingData.csv` contains:

- `gameId`
- `playId`
- `nflId`
- `pff_role`
- `pff_positionLinedUp`
- `pff_hit`
- `pff_hurry`
- `pff_sack`
- `pff_beatenByDefender`
- `pff_hitAllowed`
- `pff_hurryAllowed`
- `pff_sackAllowed`
- `pff_nflIdBlockedPlayer`
- `pff_blockType`
- `pff_backFieldBlock`

Observed `pff_role` values are exactly:

- `Pass`
- `Pass Route`
- `Pass Block`
- `Pass Rush`
- `Coverage`

Observed `pff_blockType` codes are:

`SW`, `PP`, `PT`, `CL`, `PA`, `PU`, `CH`, `NB`, `BH`, `UP`, `SR`, `PR`.

The audit does not infer code meanings from abbreviations; semantic definitions must come from official competition documentation or a separately verified contract before downstream interpretation.

## Responsibility / pressure fields actually present

The official corpus exposes blocker/rusher semantics substantially stronger than aggregate pressure data:

- defensive outcome: `pff_hit`, `pff_hurry`, `pff_sack`
- blocker loss: `pff_beatenByDefender`
- allowed outcome: `pff_hitAllowed`, `pff_hurryAllowed`, `pff_sackAllowed`
- blocker -> defender identity: `pff_nflIdBlockedPlayer`
- block type: `pff_blockType`
- backfield-block indicator: `pff_backFieldBlock`

Non-null counts from the source audit:

- `pff_hit` / `pff_hurry` / `pff_sack`: 94,127 each
- `pff_beatenByDefender` / allowed-pressure fields: 48,087 each
- `pff_nflIdBlockedPlayer`: 46,526
- `pff_blockType`: 47,904
- `pff_backFieldBlock`: 47,903

These counts strongly indicate role-scoped semantics rather than universally populated player-play fields. Downstream normalization must condition on `pff_role` and may not reinterpret nulls as zeros across unrelated roles.

## Join integrity

The source audit passed the following structural checks:

- PFF key: `(gameId, playId, nflId)`
- duplicate PFF key rows: **0**
- unique PFF game/play pairs: **8,557**
- PFF game/play pairs missing from `plays.csv`: **0**
- duplicate `(gameId, playId)` rows in `plays.csv`: **0**

This means every audited PFF play maps cleanly to the official play table.

## Tracking schema

All eight weekly tracking files use the same 16-column shape:

`gameId`, `playId`, `nflId`, `frameId`, `time`, `jerseyNumber`, `team`, `playDirection`, `x`, `y`, `s`, `a`, `dis`, `o`, `dir`, `event`.

Week-1 observed event labels include:

- `ball_snap` / `autoevent_ballsnap`
- `pass_forward` / `autoevent_passforward`
- `pass_arrived`
- `pass_outcome_caught`
- `pass_outcome_incomplete`
- `qb_sack`
- `qb_strip_sack`
- `first_contact`
- `handoff`
- `play_action`
- `man_in_motion`
- `line_set`
- `shift`
- `run`
- and additional event labels.

No downstream event aliasing is frozen by this audit; event normalization must be explicit and versioned.

## Current interpretation

BDB 2023 is a valid official public ground-truth laboratory for blocker-rusher / pass-protection work. Unlike the aggregate sources closed in M85, it contains direct blocker-to-defender identity plus blocker and rusher outcome labels.

The next required gate is not predictive modeling. It is a semantic-contract audit of the actual blocker -> rusher assignment graph:

1. assignment coverage over `Pass Block` rows;
2. same-play resolution of every `pff_nflIdBlockedPlayer` to a `Pass Rush` row;
3. duplicate/self-assignment checks;
4. double-team / multi-blocker structure;
5. block-type and backfield-block coverage;
6. consistency of blocker-allowed outcomes with matched-rusher pressure outcomes.

Only after that contract passes should geometry be attached to these labels.
