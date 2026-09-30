# WR-CB 2022W5 Verified Snapshot Lock Correction V2 — 2026-09-30

## Disposition

**V1 LOCK SUPERSEDED; V2 IS THE AUTHORITATIVE SANITIZED 2022W5 SOURCE LOCK.**

The successful source verifier itself remains unchanged and authoritative:
- run `36773818193` = SUCCESS;
- artifact `11124612896`;
- 54 parsed factual pairings;
- 41 verified pregame factual rows;
- 36 strict exact-week two-sided identity rows;
- target outcomes not accessed;
- parameters fit = 0.

## Why a v2 lock exists

A post-freeze row-for-row integrity check compared the committed v1 CSV to the original successful artifact's `source_rows`. The verifier artifact is internally consistent, but the v1 CSV contained one manual transcription error:

- WR team: NYJ
- WR GSIS: `00-0033871`
- opponent: MIA
- alignment: `RWR_VS_LCB`
- erroneous v1 CB GSIS: `00-0034384`
- artifact-authoritative CB GSIS: `00-0037353`

No other scientific/source result changed. This was a lock-materialization error after the successful run, not a source-verification or identity-gate failure.

Per the immutable-lock contract, **v1 was not edited in place**. It remains in Git history as superseded evidence.

## Authoritative lock

`data/research/wr_cb_verified_snapshot_lock_2022w05_v2.csv`

The v2 content is generated directly from the exact 36 `source_rows` in artifact `11124612896`, preserving artifact row order.

V2 SHA-256:
`4eb261dac8b45f04b93ac427fce8666fb72625fbd1cf67c2f4351bfe0e501425`

The hash previously written in the first result note corresponds to the artifact-exact v2 content; it did **not** match the mistranscribed v1 bytes.

## Boundary

This correction accesses no outcomes, sportsbook data, editorial grades, or protected 2025 targets. It fits no parameters and does not clear the source/model gate.
