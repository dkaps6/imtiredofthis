# QB TEAM PASS INTENT — FIRST-PARTY ARCHIVE TRANSPORT V1C PLAN

Frozen: 2026-10-08  
Branch: `research-individual-opportunity-roadmap-2026-10-08`  
Parent diagnostic: `QB_M89_M90_OPPORTUNITY_EFFICIENCY_DECOMPOSITION_V1_RESULT.md`  
Parent source failure: `QB_FIRST_DOWN_PUBLIC_INTENT_SOURCE_V1B_RESULT.md`

## Purpose

Test one materially different retrieval transport for the still-open QB team-pass-opportunity seam:

**direct first-party NFL club archive discovery via official robots/sitemaps and first-party media indexes, without public search-engine HTML.**

This is a source/transport audit only. It cannot inspect target-game football outcomes, model residuals, sportsbook information, or fit a predictive model.

The unresolved scientific question remains whether genuinely new, pregame, target-week team pass-volume / game-plan / starter-intent information can be acquired with reliable as-of provenance. The QB mean itself is not being reopened. M89/M90 and QB C2 remain protected.

## Why this is materially different from V1B

V1B failed operationally because unauthenticated DuckDuckGo/Bing HTML search transport returned empty, blocked, redirect-wrapper, or junk results. Its frozen result was `RETRIEVAL_AUTOMATION_NOT_QUALIFIED`; this was explicitly not a scientific rejection of public pregame intent.

V1C does not query a search engine.

It discovers material only from:
1. each club's official `robots.txt`;
2. sitemap URLs declared there, with conservative same-club recursion;
3. fixed first-party sitemap fallbacks (`/sitemap-index.xml`, `/sitemap.xml`) when robots omits a sitemap declaration;
4. first-party URLs enumerated by those sources.

No team-specific search query strings are allowed. No Google/Bing/DDG fallback is allowed.

## Pre-run public feasibility evidence

Before freezing this plan, a bounded manual feasibility check established that the transport family exists:
- Baltimore exposes a first-party timestamped transcript archive at `baltimoreravens.com/news/transcripts`;
- Philadelphia exposes a first-party media transcript database with date and season/week filters on `media.philadelphiaeagles.com`;
- Miami exposes a first-party transcript index at `miamidolphins.com/news/transcripts`;
- New England publishes timestamped first-party coach/QB transcripts;
- NFL club robots files inspected for New England and Philadelphia explicitly declare `sitemap-index.xml`.

These examples justify a league-wide transport audit. They do **not** count as predictive evidence and do not pre-qualify the source family.

## Frozen 32-team universe

All current NFL clubs, exactly once:
`ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LV LAC LAR MIA MIN NE NO NYG NYJ PHI PIT SF SEA TB TEN WAS`.

Only official club domains in the committed manifest are eligible. Subdomains of an official club domain are first-party; unrelated hosts are not.

## Historical discovery window

Candidate historical years:
- 2023
- 2024
- 2025

The audit may use sitemap `lastmod` or an explicit year in a canonical URL as **discovery evidence only**. This does not certify publication time and may not later be treated as a timestamp-safe target-week observation.

A later semantic/source audit must independently verify publication time before any row can become model-eligible.

## Candidate page vocabulary

A first-party URL is a retrieval candidate only if its canonical URL path contains at least one frozen source-family token:

- `transcript`
- `press-conference`
- `press-conferences`
- `media-availability`
- `media_availability`
- `what-they-said`
- `quotes`

This is a transport discovery vocabulary, not a football semantic label. A page can be retrieved and still contain no eligible target-week intent.

## Frozen transport qualification gates

V1C is `FIRST_PARTY_ARCHIVE_TRANSPORT_QUALIFIED_FOR_SEMANTIC_AUDIT` only if all pass:

1. all 32 clubs are attempted from the frozen domain manifest;
2. at least 24/32 clubs expose a reachable robots-declared or fixed-fallback sitemap transport;
3. at least 24/32 clubs enumerate at least one first-party candidate URL from the frozen vocabulary;
4. at least 20/32 clubs show candidate discovery evidence in at least two of the three historical years 2023-2025;
5. at least 12/32 clubs show candidate discovery evidence in all three years;
6. no external search-engine HTML endpoint is contacted;
7. no football outcomes, model residuals, sportsbook fields, target labels, or 2026 Week-5 outcomes are read;
8. predictive models fit = 0;
9. production changes = false.

If any gate fails, V1C is `FIRST_PARTY_ARCHIVE_TRANSPORT_NOT_QUALIFIED`. Thresholds may not be lowered after seeing the result.

A mechanical parser/HTTP repair is allowed only if the first run demonstrates an implementation defect while preserving all source and qualification gates.

## What a pass means

A pass authorizes exactly one separately frozen **semantic/timestamp coverage audit** using the direct first-party transport. That later audit must preserve the original V1 source-family standards, including:
- target-week, pregame information only;
- official sources first;
- unambiguous pre-kickoff provenance;
- no postgame explanations;
- no football outcomes or residuals during source qualification;
- no predictive fit before source/schema qualification;
- no relaxation of the four-season/position protections elsewhere in the model.

A V1C pass does **not** establish predictive value and does not authorize a QB attempts model.

## What a fail means

If V1C fails without a genuine mechanical implementation defect, the public-intent retrieval route is formally source-blocked under the current free/public constraints. Do not return to search-engine scraping, manual 500+ row crawling, or same-data pass-rate models.

The QB opportunity seam may reopen only for another materially new, scalable, timestamp-safe pregame information source.

## Protected science

- M89/M90 unchanged.
- QB C2 unchanged.
- no generic YPA/mean reopening.
- no schedule/rest D1 retry.
- no PBP D2 retry.
- no same-data attempts repackaging.
- no sportsbook upstream.
- no OddsAPI calls.
- no production workflow changes.
- Week-5 prospective locks remain ungraded.
