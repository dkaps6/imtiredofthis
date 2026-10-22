# WR-CB Public Archive Content-Version Timing Audit V1 — 2026-09-30

**Status: SOURCE GATE CLOSED.** No WR-CB model fitting/promotion; this audit reads public-source metadata and the existing research artifact only. No sportsbook calls, betting outcomes, or editorial matchup grades as features.

## Authority / no redundant acquisition

- Public-source implementation commit: `c7a66d6b42d03d0a530b622e9969482bd2693a11` on draft PR #665.
- Source-acquisition + enriched timing audit: GitHub Actions run `36721920444`, source-audit job `109908902673` SUCCESS, artifact `11101400208` (ZIP digest `sha256:0d6105b23ca45402f7b3d52847d6e0161acb66fb2b1640a12f3cecf673380a3f`); Repo CI run `36721920418` SUCCESS.
- Exact row-level cross-tab reproduced offline by reading the **existing** `fantasyalarm_wr_cb_source_quality_rows.csv` inside that ZIP; no second archive scrape and no nflverse outcomes.
- Parser distinguishes `published_at_utc` from machine-readable `modified_at_utc`; missing/conflicting/invalid modification metadata is explicitly unverified. Per-WR actual game kickoff is the comparison clock.
- Source status `METADATA_PRE_KICKOFF_COMPATIBLE_NOT_SNAPSHOT_PROOF` means only that the reported last edit precedes kickoff; it does **not** independently prove the table was present in the same form before kickoff. `MODIFIED_AFTER_GAME_KICKOFF_UNVERIFIED` likewise does **not** prove matchups were edited; it says the current page is not sufficient historical pregame proof.

## Observed impact

Current 71 season-week source pages contain **4,459 pairing rows**. Published pre-kickoff: **4,300**. The previous source-quality filter (publication + schedule + identity + alignment only) provisionally admits **4,063** rows; it was never model promotion.

- `METADATA_PRE_KICKOFF_COMPATIBLE_NOT_SNAPSHOT_PROOF`: **3,795** observed rows.
- `MODIFIED_AFTER_GAME_KICKOFF_UNVERIFIED`: **660** observed rows.
- `MISSING_KICKOFF`: **4** rows.
- **479 of 4,063** previously publication-eligible rows are from pages reporting post-kickoff modification and therefore lack pregame archived-content proof.
- **3,584** publication-eligible rows remain metadata-compatible after that exclusion, but **zero of 4,459 have independently verified historical pregame content snapshots**. Do not label 3,584 model-ready.

| Source season | observed rows | publication-eligible rows | post-kickoff-modification rows | affected publication-eligible rows | remaining provisionally compatible |
|---|---:|---:|---:|---:|---:|
| 2021 | 136 | 131 | 0 | 0 | 131 |
| 2022 | 917 | 825 | 28 | 9 | 816 |
| 2023 | 1,121 | 1,011 | 144 | 82 | 929 |
| 2024 | 907 | 818 | 322 | 259 | 559 |
| 2025 (intended untouched confirmation) | 1,124 | 1,045 | 160 | 123 | 922 |
| 2026 W1–3 (research/prospective inventory only) | 254 | 233 | 6 | 6 | 227 |
| **Total** | **4,459** | **4,063** | **660** | **479** | **3,584** |

Eight source season-weeks have postgame modification metadata for **every observed row**: 2023 W8; 2024 W1/W2/W4/W13/W15; 2025 W4/W14. Treat these as highest-priority historical snapshot-recovery candidates, not as known-to-be-corrupted data.

2025's full **18/18 article-week inventory does not imply a complete validated confirmation population**: 123/1,045 publication-eligible rows have this source-version risk; earlier evidence also found explicit incomplete slot coverage in some vintages. 2026 pages captured after W1–3 games cannot serve as genuine forward immutable captures.

## Distinct remaining source blockers

1. **Historical content provenance:** for any pair admitted to strict-prior retrospective fitting or confirmation, recover independently dated *pregame* snapshot bytes, or otherwise defensible contemporaneous preservation showing that exact factual pair. An article's own publish and edit timestamps alone are **insufficient**. Record snapshot source, capture timestamp, original URL, hash, provenance quality; fail closed on uncertain rows.
2. **Breadth:** 2021 source inventory 2/18; 2022 14/18; 2023 17/18; 2024 17/18; 2025 18/18; current 2026 3/3. Search has identified a 2024 W17 *Saturday-games-only* public article candidate; do not silently count it as full Week-17 coverage. Older selective slots are missing, not zero.
3. **Identity:** Claude independent provider-ID cross-audit `36713950808` found quarantined bridge anchors but zero current mapping delta; the PRE_KICKOFF-only anchor patch `71dc2b0` is now validated. Claude separately owns a bounded **unanchored collision census**; never treat anchored-only 7 WR/2 CB collisions as a global census.
4. **Semantics:** public archive rows are **editorial pregame projected WR↔CB pairings**, not verified play-by-play coverage assignments or quantified defender exposure. Independently assess whether this projection is a valid forward signal. Never fabricate true coverage responsibility or use the site's editorial grade as football features.
5. **Prospective capture:** a reproducible, time-stamped immutable *pre-kickoff* capture for future 2026 articles can sidestep historical edit ambiguity without paid inputs. The current archived HTML acquired after a game does not retroactively become a pregame capture.

## Policy

Keep the source-quality audit's existing 4,063 publication-eligible figure only as a **provisional source parser/identity diagnostic**. The code now emits a separate content-version timing status and explicit `historical_content_snapshot_verified_rows=0`. No `TOP_WEAPON_ESCAPE_HATCH` test, no production restoration of retired `coverage_penalty()`, no Week-3 rescore, no odds spend. First address independently verifiable point-in-time content and source/assignment semantics.