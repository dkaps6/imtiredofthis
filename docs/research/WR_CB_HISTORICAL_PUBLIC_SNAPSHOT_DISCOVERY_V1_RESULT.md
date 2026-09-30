# WR-CB Historical Public-Archive Snapshot Index Pilot V1 — Result

**Disposition: ONE PARTIAL-WEEK ARCHIVE INDEX CANDIDATE, ZERO CONTENT-VERIFIED ROWS.** Research-only; source/model gate CLOSED.

## Authority
- Branch `research-wr-cb-historical-snapshot-discovery-v1` based on current source-audit draft #665.
- Plan/script/testing commit `5fea667e3280a05f44036b872029c4ee0ac729e2`.
- GitHub Actions run `36739781377`, job `109970838872` **PASS** (including metadata parser tests), artifact `11110411467`, ZIP digest `sha256:df7e0b89c31eae10cf29f5fd23191b7308734ca17817704621250cce3a15fcf3`.
- Queried exact article URL metadata only at public Internet Archive CDX and up to three time-adjacent Common Crawl collections, four preregistered high-risk season-weeks. No public article body acquired, no FantasyAlarm archive re-scrape, no sportsbook, no outcomes.

## Four target findings
| Target | Public index findings | Disposition |
|---|---|---|
| 2023 Week 8 | Internet Archive timeout, Common Crawl 504 | ACCESS UNRESOLVED; absence **not** established |
| 2024 Week 1 | Internet Archive indexed `20240907005714`, exact source article, digest `JE62ZZZYJ4TELVSMS7LVWRPWWTXE53HA`. Also Oct 4/9 captures after all games. Common Crawl requests 404/504. | **PARTIAL-WEEK SNAPSHOT CANDIDATE** only |
| 2025 Week 4 | Internet Archive connect timeout, Common Crawl 502/504 | ACCESS UNRESOLVED; absence **not** established |
| 2025 Week 14 | Internet Archive connect timeout, Common Crawl 404/502 | ACCESS UNRESOLVED; absence **not** established |

2024 W1 exact source URL:
`https://www.fantasyalarm.com/articles/nfl/wide-receivers/2024-fantasy-football-wr-cb-matchup-report-week-1-drake-london-looks-to-take-off/163226`.

Its frozen source index record says capture **September 7, 2024 00:57:14 UTC**. Week-1 kickoff span from source schedule: September 6 00:20 through September 10 00:15 UTC. This snapshot was **after** the Thursday opener and before later games. Never treat it as full-week pregame provenance; a real exact-content recovery would need to verify the archived body and apply EACH WR's game kickoff, excluding Thursday/already-played cases.

## Next bounded action
Freeze a separate exact archival replay audit for **that one indexed 2024 W1 capture only**. Require source URL identity, archived timestamp, returned body/Wayback archive digest match where possible, parsing of explicit factual pairings, real per-WR kickoff, and exact archived-content hash. An archived index hit is not an archived page-content proof. If Wayback replay is inaccessible, record `BODY_UNAVAILABLE` and close rather than reusing the current edited article.

Index-side 2025 lookups are currently infrastructure-inconclusive; do not label untouched 2025 confirmation as historically verified. Independently captured prospective 2026+ pages remain the reliable fallback.

No model fitting, no editorial matchup grades as inputs, no source gate clear, no production change.