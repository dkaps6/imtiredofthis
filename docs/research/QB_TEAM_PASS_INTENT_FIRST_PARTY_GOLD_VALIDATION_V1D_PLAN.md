# QB TEAM PASS INTENT — FIRST-PARTY GOLD VALIDATION V1D PLAN

Frozen: 2026-10-08  
Branch: `research-individual-opportunity-roadmap-2026-10-08`

Parent source-transport authority:
- V1C run: `37868152267`
- head: `ef5a76526f7f14ad1928132d0666150eb39f29ac`
- artifact: `11589296706`
- digest: `sha256:2821ed40cd27c0454bf7eb8d100145c02b75121e52524bc044d4a056178fd880`
- disposition: `FIRST_PARTY_ARCHIVE_TRANSPORT_QUALIFIED_FOR_SEMANTIC_AUDIT`

Immutable operational gold fixture:
- `docs/research/QB_TEAM_PASS_INTENT_V1D_GOLD_PREFIX.csv`
- exactly 13 completed rows from the pre-existing V1 manual prefix;
- 2023 Week 2, ARI through HOU in frozen manifest order;
- 12 rows have an eligible OFFICIAL canonical source;
- ATL has an eligible LOCAL_ATTRIBUTABLE canonical source after official review found no sufficiently explicit qualifying intent.

## Purpose

Validate that the materially new V1C direct first-party archive transport can recover and timestamp the already-reviewed official gold sources without public search-engine HTML, team-specific search queries, target-game outcomes, model residuals, sportsbook data, or predictive fitting.

This is still a source/schema audit. It does not test whether the source content predicts pass attempts or passing yards.

## What the gold fixture may and may not do

The gold canonical URLs and timestamps are validation labels only.

They may be used after deterministic first-party archive enumeration to score:
- whether the canonical source was independently enumerated;
- whether the canonical page is directly fetchable;
- whether its publication metadata is safely pre-kickoff;
- whether the host is correctly classified as first-party;
- whether the one known local-fallback case is routed correctly.

They may NOT:
- seed sitemap traversal;
- be inserted into the discovered URL set;
- alter team-specific query logic;
- teach football direction;
- change semantic vocabulary;
- create a numeric football feature;
- select rows based on outcomes.

## Frozen transport

For each of the 12 OFFICIAL gold rows:
1. begin only from the frozen official domain manifest;
2. read official `robots.txt`;
3. follow same-domain declared sitemap URLs, with only the fixed `/sitemap-index.xml` and `/sitemap.xml` fallbacks;
4. recursively enumerate same-domain sitemap URLs under the same bounded V1C transport;
5. normalize URLs only for scheme/host/trailing-slash/query/fragment equivalence;
6. score canonical-source reacquisition only after enumeration is complete;
7. separately fetch the gold canonical URL and extract publication metadata.

The ATL LOCAL_ATTRIBUTABLE row is never fetched from AJC in V1D. It must be classified `LOCAL_FALLBACK_REQUIRED`.

No Google, Bing, DuckDuckGo, site-search endpoint, or external search API is allowed.

## Frozen publication timestamp extraction

Allowed methods, in order:
1. JSON-LD `datePublished`;
2. OpenGraph/article metadata such as `article:published_time`;
3. explicit publication meta fields containing `datePublished` / `publishDate` equivalents;
4. `<time datetime=...>` publication field.

A URL year, sitemap `lastmod`, JavaScript build timestamp, page modification time, or undifferentiated visible date is not enough for V1D timestamp certification.

An extracted timestamp must be parseable to an absolute datetime and strictly before the frozen target-game kickoff.

V1D scores timestamp-safety correctness against the gold row's `timestamp_safe=true`; exact string equality to the historical manually recorded timestamp is not required because equivalent metadata may be represented in UTC.

## Frozen gold gates

The V1D disposition is `FIRST_PARTY_GOLD_VALIDATION_QUALIFIED_FOR_SEMANTIC_COVERAGE_AUDIT` only if all pass:

1. exact gold fixture row count = 13;
2. exact source-class split = 12 OFFICIAL / 1 LOCAL_ATTRIBUTABLE;
3. official canonical sitemap reacquisition >= 11/12;
4. official canonical direct fetch success = 12/12;
5. timestamp-safe correctness >= 12/12;
6. official source-class correctness = 12/12;
7. ATL routes exactly to `LOCAL_FALLBACK_REQUIRED`;
8. no public search-engine HTML or external search API contacted;
9. no football outcomes read;
10. no model residuals read;
11. no sportsbook/game-market data used;
12. no 2026 Week-5 outcomes read;
13. predictive models fit = 0;
14. production changed = false.

The canonical-reacquisition threshold is deliberately not 12/12 because an official site's present sitemap inventory can omit an old page while the stable canonical page remains directly fetchable. This does not relax the old V1B canonical-source gate (75%); 11/12 = 91.7%.

The 12/12 direct-fetch and timestamp-safety gates are stricter than the inherited >=95% timestamp-correctness gate because 11/12 would be only 91.7%.

## Mechanical repair rule

One generic implementation/parser repair is allowed if the first run fails before a valid source verdict or demonstrates a parser defect affecting multiple sites.

Forbidden repairs:
- team-specific URL insertion;
- hard-coded canonical-source discovery;
- accepting sitemap `lastmod` as publication timestamp;
- changing the gold labels;
- weakening any frozen gate after seeing results.

## Advancement rule

If V1D passes:
- authorize one separately frozen semantic/source-coverage audit using V1C/V1D transport;
- preserve the original V1 target-week, pre-kickoff intent vocabulary and source-qualification gates;
- still do not fit a QB predictive candidate.

If V1D fails after one legitimate generic parser repair:
- preserve the failure;
- do not return to manual 500+ row historical crawling;
- source-gate this direct first-party route unless a materially new scalable source becomes available.

## Protected science

Unchanged:
- M89/M90;
- QB C2;
- no generic QB YPA/mean reopening;
- no same-data attempts model;
- no schedule/rest D1 retry;
- no PBP D2 retry;
- sportsbook remains downstream only;
- no OddsAPI call;
- Week-5 prospective research remains ungraded.
