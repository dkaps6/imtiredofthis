# 2022 Week 5 Exact Wayback Replay-Mode Probe V1

The availability API reports snapshot `20221005195142` before the first Week-5 kickoff, but exact `id_` replay returned 404 and CDX digest lookup was unavailable. Common Crawl exact-index fallback returned HTTP 504.

This bounded probe asks only whether Wayback can independently serve that exact timestamp through normal/`if_` replay while preserving the timestamp in the final URL and exposing WR/CB article structure. It stores only status, headers, sizes and hashes—not article text and not football pairings.

A successful replay mode is **not** enough to source-lock rows. It merely permits a separate exact-memento parse/identity audit. If every exact mode fails, close 2022 W5 as archive-access unresolved and do not weaken the source standard.
