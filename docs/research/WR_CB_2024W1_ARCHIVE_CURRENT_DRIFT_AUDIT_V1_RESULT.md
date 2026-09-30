# WR-CB 2024 Week 1 Archived-vs-Current Pairing Drift Audit — 2026-09-30

**Disposition: NO FACTUAL PAIRING DRIFT DETECTED IN THE PERSISTED PRE-GAME SUBSET. SOURCE/MODEL GATE REMAINS CLOSED.**

## Inputs

Archived pregame authority:
- verification run `36768720380`, artifact `11122412408`;
- exact Wayback capture `2024-09-07T00:57:14Z`;
- digest-matched archived body SHA-256 `0854732b5af783b1d67fd64bcf6598a86c03a0f3669e70e4399b903c441dc33f`;
- archived JSON-LD articleBody SHA-256 `85998a169935efd81dece50d0d162d41d4f88c58bfdd18d1239b2afbe497680d`;
- 66 factual pairings parsed, 57 still pregame at capture, 46 strict two-sided exact-week roster-ready.

Later/current public-page source audit:
- integrated source-quality run `36742368919`, artifact `11110249961`;
- same exact FantasyAlarm 2024 Week-1 source URL;
- current parser emits 66 factual pairings.

## Exact comparison

Compared normalized factual keys only:

`(WR team, canonical WR key, opponent, canonical CB key, alignment bucket)`.

Results:
- archived persisted pregame factual rows: **57**;
- those exact keys found in current edited-page audit: **57/57**;
- archived strict exact-week identity-ready rows: **46**;
- those exact keys found in current edited-page audit: **46/46**;
- current-page factual row count: **66**, equal to the 66 total factual rows recovered from the archived body.

The last count equality is supportive but **not** row-by-row proof for the 9 archived rows that were already post-kickoff at the September 7 capture, because those 9 were intentionally not persisted in the sanitized pregame ledger.

## Interpretation

For this one source week, the page's later modification timestamp did **not** alter any of the 57 persisted pregame WR↔CB pairing identities/alignment rows we can independently verify. Therefore `MODIFIED_AFTER_GAME_KICKOFF_UNVERIFIED` is correctly a provenance **risk state**, not proof the factual matchup table changed.

Do **not** generalize this result to other weeks. Other modified pages still require independently captured bodies or equivalent point-in-time evidence. Missing historical slot rows remain missing, never zero. The source remains editor-projected pregame alignment, not route-by-route observed coverage.

No editorial grades, sportsbook inputs, game outcomes, parameter fitting, production changes or Week-3 work were used.
