# WR-CB 2024 Week 1 Verified Archived Snapshot Result — 2026-09-30

**Disposition: 46 STRICT PRE-KICKOFF, EXACT-WEEK IDENTITY-VERIFIED FACTUAL PAIRINGS RECOVERED. SOURCE/MODEL GATE REMAINS CLOSED.**

## Provenance authority

- Exact public source article: FantasyAlarm 2024 Week 1 WR-CB report, article ID `163226`.
- Internet Archive exact capture timestamp: **2024-09-07 00:57:14 UTC**.
- CDX SHA-1/Base32 digest: `JE62ZZZYJ4TELVSMS7LVWRPWWTXE53HA`.
- Exact replay body SHA-256: `0854732b5af783b1d67fd64bcf6598a86c03a0f3669e70e4399b903c441dc33f`.
- Archived JSON-LD `articleBody` SHA-256: `85998a169935efd81dece50d0d162d41d4f88c58bfdd18d1239b2afbe497680d`.
- Source publication timestamp embedded in the captured article: **2024-09-06 23:59:46 UTC**.
- Verification run `36768720380` **SUCCESS**, job `110069346626`, artifact `11122412408`, artifact digest `sha256:19986b8ff1890ca0d507a029d854d400d69abd8c3edc7a2affa632def6bf8a8f`.
- No raw archived article page is committed. Only hashes, structural evidence and sanitized factual identity rows are preserved.

## Recovery

The outer archived HTML contains zero `<table>` and zero `<tr>` nodes, so the original parser initially returned zero pairings. A structure-only audit found one **172,709-character JSON-LD `articleBody`** with:
- 3 HTML tables;
- 149 rows;
- 1,794 HTML tags;
- WR, CB and matchup terminology;
- FantasyAlarm player links.

Parsing that exact in-memory, digest-matched `articleBody` recovered **66 factual WR-CB pairings**. Because the archive capture occurred after the Thursday opener, each row was checked against its own game kickoff. **57** rows were still genuinely pregame at the September 7 capture time.

## Strict weekly identity gate

The 57 pregame factual rows were then re-audited with **no provider-ID bridge and no fallback**:
- WR must resolve uniquely on the stated offense's exact 2024 Week-1 roster;
- CB must resolve uniquely on the stated opponent's exact 2024 Week-1 defensive roster;
- WR team/opponent schedule must match;
- alignment must be explicit outside/slot.

Result:
- 57/57 schedule-consistent;
- 54/57 WR exact-week identities;
- 48/57 CB exact-week opponent identities;
- **46/57 pass both identity gates**;
- **11/57 quarantined**;
- provider bridge used: **false**.

Quarantine counts overlap:
- WR not exact: 3;
- CB not exact on opponent: 9;
- schedule mismatch: 0.

Examples explain why fail-closed is correct: archived source typos/name variants such as `Dax Hll`, `Kristan Fulton`, short-name/suffix variants, plus one malformed slot row showing `Christian Kirk` on SEA against DEN with `Kader Kohou`. We do **not** manually rescue these rows.

## Meaning

This is the first lane result that supplies **independently captured historical pregame content**, not merely a current article whose publication/edit metadata happens to be compatible. It proves the archive-recovery method can work.

It does **not**:
- establish completeness of Week 1 or the historical archive;
- establish actual route-by-route man coverage responsibility;
- validate the site's editorial matchup grade;
- authorize a WR-CB model feature;
- use sportsbook inputs or game outcomes;
- fit parameters or modify production.

The recovered pairings remain **editor-projected pregame alignments**. Source/model gate stays CLOSED until enough historical snapshots are recovered, sampling/completeness is quantified, and a frozen strict-prior scientific design is established before any protected confirmation outcomes are examined.
