# RB-ND2B — Leakage-Safe Backfield Source As-Of Audit Results

## Purpose

Determine whether historical pregame depth-chart and lagged snap information exists with enough timestamp integrity and coverage to build an explicit backfield-allocation layer without leaking target-game outcomes.

## Authoritative run

- Workflow: `RB-ND2B Backfield Source AsOf Audit`
- Run: `33509092341`
- Job: `99860131990`
- Tested SHA: `cafb52854b4476eb27cd12d5064d5e6f52247b73`
- Artifact: `9800848065`
- Artifact SHA256: `e8f5d3dd60da7464a5000485061087a3b2ef7bdcb66ec9254025e829c1f4be77`
- Execution: success
- Model fit/search: 0
- Sportsbook inputs: 0
- Production change: 0

## 2025 timestamped depth source

- total records: `554,215`
- timestamp range: `2025-08-03 10:09:07 UTC` through `2026-03-14 07:32:09 UTC`
- unique snapshot dates: `221`
- RB rows: `37,242`

For every 2025 regular-season team-game, the audit selected the latest depth snapshot **strictly before kickoff**.

- team-games: `544`
- team-game pregame depth coverage: **`100.0%`**
- median snapshot age at kickoff: **`10.77 hours`**
- p90 snapshot age: `17.90 hours`

Against the exact 1,393 M94C 2025 RB/FB rows:

- pregame depth-rank coverage: **`94.9749%`**
- Week 1 depth-rank coverage: **`96.4706%`**

This directly repairs the prior M94C research limitation in which `role`/`rules_role` were unpopulated across the historical RB trace because unsafe/non-week-tagged depth snapshots were intentionally excluded.

## Historical snap source

PFR snap data contained `53,227` rows with offensive snap counts and percentages.

Against M94C RB rows:

- prior-week offensive-snap coverage, all weeks: `78.6073%`
- prior-week coverage, Weeks 2-18: **`83.7156%`**

Target-game snap information remains forbidden. Only lagged/prior-game snap and participation information may be used pregame.

## Important depth-rank nuance

The 2025 depth source's `pos_rank` is not guaranteed to be a simple one-player RB1/RB2/RB3 ordinal. Multiple rank-1 entries can exist across position/depth slots. Therefore rank is a current-role signal, not a deterministic carry-share rule. The allocation engine must combine depth state with lagged snap/carry share, competitor information, availability, continuity, and other football context.

## Durable conclusion

The missing current-role layer is **historically reconstructable with strong timestamp integrity**. This is no longer merely a conceptual limitation.

The next allocation test should preserve the existing team/RB rushing-opportunity pool and ask whether explicit pregame role information can improve how that pool is assigned among RB1/RB2/RB3/FB players. It should be treated as a correction/add-on to the existing RB architecture, not presumed to replace M94C wholesale.