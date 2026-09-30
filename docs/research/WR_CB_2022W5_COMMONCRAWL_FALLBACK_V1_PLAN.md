# 2022 Week 5 Common Crawl Independent Archive Fallback V1

Wayback availability metadata reports pregame capture `20221005195142`, but exact Wayback replay/CDX digest access failed closed. Do not weaken that gate.

Common Crawl's `CC-MAIN-2022-40` independently spans September 24–October 8, 2022, overlapping source publication and the October 7 first Week-5 kickoff. Query only the exact FantasyAlarm Week-5 URL.

If an exact Common Crawl capture exists before the first kickoff:
1. require index timestamp, WARC locator and SHA-1 digest;
2. fetch only that one WARC byte range;
3. require WARC target URL and WARC payload digest;
4. recompute SHA-1 from archived payload and require index == WARC == bytes;
5. parse explicit factual WR↔CB rows;
6. apply per-row schedule/kickoff;
7. require exact Week-5 WR and opponent defensive roster identities with no bridge/fallback.

No raw page committed, no current page reacquisition, no outcomes, sportsbook inputs, grades or fitting.
