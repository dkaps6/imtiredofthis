# WR-R5 — Target / Catch / YPR Individual Mechanism Decomposition Result

## Canonical run

- Run: `34066489537`
- Job: `101576040241`
- Tested SHA: `3ce108a6b3b2f718bd621cf82fd4a1ebf67558a4`
- Artifact: `wr-r5-target-catch-ypr-individual-decomposition`
- Artifact ID: `9999098530`
- Artifact SHA256: `5582977976cf55d0a163f5fa7ce4dfcf5355d5b3f82822cefab3b1040f92755f`
- Conclusion: `success`

## Integrity

- ND5 rows: `2130`
- Merged rows: `2130`
- Scoreable rows: `2130`
- Unscoreable rows: `0`
- Qualifying players (>=8 games): `133`
- Projected factor reconciliation max abs error: approximately `1.42e-14`
- Actual factor reconciliation max abs error: approximately `1.42e-14`
- Shapley decomposition max abs error: approximately `2.84e-14`
- Sportsbook inputs used: `false`
- Model fitting used: `false`
- Production changed: `false`

## Frozen disposition

`WR_TARGET_CATCH_YPR_INDIVIDUAL_MECHANISMS_MAPPED`

## Dominant mechanism counts

- `TARGETS`: `72`
- `MIXED`: `44`
- `YPR`: `16`
- `CATCH`: `1`

## Scientific interpretation

WR-R4 had shown that reception-volume error was the dominant recurring receiving-yard mechanism for most qualifying WRs. WR-R5 resolves that further: the reception-volume problem is overwhelmingly a **target-opportunity problem**, not a target-to-catch conversion problem.

The result materially strengthens the case for player-centric target-entitlement / role / team-pass-environment research. Catch-rate tuning is not supported as the next broad WR direction.

Examples among high-MAE players classified `TARGETS` include Puka Nacua, Jaxon Smith-Njigba, Ricky Pearsall, George Pickens, Jameson Williams, Ja'Marr Chase, Alec Pierce, A.J. Brown, Drake London and Amon-Ra St. Brown. Examples classified `YPR` include Zay Flowers, DK Metcalf and DJ Moore. CeeDee Lamb and Justin Jefferson were among `MIXED` examples.

This classification is diagnostic, not permission for per-player constants. The next legitimate WR work should test leakage-safe pregame mechanisms that explain each player's target opportunity in the relevant role/team/game context, then require full-stack M38-vs-candidate walk-forward validation before any production change.