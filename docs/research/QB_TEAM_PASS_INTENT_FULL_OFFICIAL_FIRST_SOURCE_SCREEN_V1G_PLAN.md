# QB TEAM PASS INTENT — FULL OFFICIAL-FIRST SOURCE SCREEN V1G PLAN

Frozen: 2026-10-08  
Branch: `research-individual-opportunity-roadmap-2026-10-08`

Parent authorities:
- V1C first-party archive transport: QUALIFIED.
- V1D gold reacquisition/provenance: QUALIFIED.
- V1E deterministic lexical router: FAILED CLOSED at 10/12 recall; do not retune.
- V1F frozen pretrained semantic router: QUALIFIED at 11/12 positive recall, 1.000 gold precision.
- V1F canonical run: `37869329499`
- V1F artifact: `11589428615`
- V1F digest: `sha256:5518821c4d92ac2ec9f50a24e2059c87ff25d486890e6bec0b3af7f25eab6154`

## Purpose

Run one full deterministic historical **official-first source screen** over the original public-intent source-audit universe.

This is still source/retrieval research. It does not read football outcomes, model residuals, player performance, sportsbook/game-market data, or fit a QB predictive model.

The only question is whether the already-qualified direct first-party transport plus the already-qualified frozen V1F semantic router finds enough timestamp-safe target-week official material to justify completing the original source-qualification audit.

## Frozen universe

Exactly:
- seasons: 2023, 2024, 2025;
- regular-season sampled weeks: 2, 5, 8, 11, 14, 17;
- every scheduled team in those sampled weeks;
- expected team-week universe: **536**.

The universe is built only from historical schedule identity and kickoff time via the repository historical schedule helper. No game result fields may be read or retained.

## Official discovery transport

For each official club domain:
1. read official `robots.txt`;
2. follow same-domain sitemap URLs declared there;
3. allow only fixed same-domain fallbacks `/sitemap-index.xml` and `/sitemap.xml`;
4. recursively enumerate same-domain sitemap URL records under the same V1C bounds;
5. retain each URL plus sitemap `lastmod` only as discovery metadata.

No Google, Bing, DuckDuckGo, public search-engine HTML, site-search endpoint, or external search API.

## Target-week candidate window

A page can enter a team-week candidate set through either generic route:

### A. Date-window route
Its sitemap `lastmod` parses to a date/time from **kickoff minus 9 days through kickoff**.

### B. Strong URL route
Regardless of sitemap `lastmod`, its decoded canonical path contains at least one target-specific generic token:
- an opponent city/nickname synonym from the frozen league-wide synonym table;
- `week-<N>`, `week<N>`, `week_<N>`, or equivalent separated `week <N>`;
- one of the frozen source-family tokens:
  `transcript`, `press-conference`, `press-conferences`, `media-availability`, `media_availability`, `what-they-said`, `quotes`, `game-preview`.

Strong URL route still requires either the target season year or the target opponent/week token to avoid opening unrelated archive pages.

Sitemap `lastmod` is **never** publication-time certification.

## Candidate ranking and cap

Before any page fetch, assign a fixed discovery score:
- +4 opponent synonym in URL path;
- +3 exact week token in URL path;
- +2 frozen source-family token in path;
- +1 sitemap `lastmod` inside the target-week window.

Sort by:
1. discovery score descending;
2. absolute sitemap-lastmod distance to kickoff ascending when available;
3. normalized URL ascending.

Retain at most **20 official candidate URLs per team-week**.

The cap is operational only. A row with no accepted official page after the cap remains `OFFICIAL_UNRESOLVED_REQUIRES_FALLBACK_OR_REVIEW`; it is never finalized as no-source.

## Page qualification

For each retained candidate, in ranked order:

1. fetch page from the first-party club host;
2. require final redirect to remain first-party;
3. extract publication time only by the frozen V1D allowed methods;
4. require absolute publication timestamp strictly before kickoff;
5. require publication timestamp >= kickoff minus 9 days;
6. require target-game relevance using the frozen V1E league-wide opponent synonym method;
7. construct fragments using the frozen V1F fragment pool:
   - 5-65 words;
   - contains frozen offensive-context term;
   - excludes frozen postgame/result language;
8. score fragments with the exact frozen V1F model, revision, prototypes and thresholds;
9. accept the first page that becomes `AUTO_REVIEW_READY`.

Stop official searching for that team-week after first accepted official candidate, matching the original V1 collection protocol.

## Frozen semantic router

Unchanged:
- model: `sentence-transformers/all-MiniLM-L6-v2`
- revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- package: `sentence-transformers==3.4.1`
- positive threshold: **0.38**
- semantic margin: **0.05**
- exact frozen positive/negative prototypes from V1F
- no fine-tuning or threshold/prototype search.

## V1G output state

Each team-week receives exactly one V1G screen state:

- `OFFICIAL_AUTO_REVIEW_READY`
- `OFFICIAL_UNRESOLVED_REQUIRES_FALLBACK_OR_REVIEW`

V1G does **not** emit:
- `ELIGIBLE_INTENT_SOURCE_FOUND`;
- `PUBLIC_PREGAME_SOURCE_FOUND_NO_INTENT_CONTENT`;
- `TIMESTAMP_UNSAFE_ONLY`;
- `NO_RECONSTRUCTABLE_SOURCE`.

Those remain final V1 source-audit dispositions and require the later source-qualification stage.

For AUTO_REVIEW_READY rows preserve:
- official locator;
- publication timestamp/method;
- opponent;
- frozen semantic tag;
- positive/negative scores and margin;
- <=25-word evidence;
- candidate rank.

## Frozen lower-bound density readout

Report AUTO_REVIEW_READY lower-bound rates:
- pooled;
- by season;
- by sampled week;
- by franchise.

Also report how many franchises have official AUTO_REVIEW_READY coverage >=50%.

Compare these **descriptively** to the parent V1 density gates:
- pooled >=70%;
- each season >=60%;
- each sampled week >=50%;
- >=24 franchises with >=50%.

Timestamp and stable-source quality are mechanically 100% for accepted V1G official rows by construction and must be reported, but this does not automatically promote them to final V1 eligible dispositions.

## V1G disposition

`OFFICIAL_AUTO_READY_LOWER_BOUND_CLEARS_PARENT_DENSITY_GATES`
only if:
- exact universe = 536;
- pooled official AUTO_REVIEW_READY >=70%;
- every season >=60%;
- every sampled week >=50%;
- >=24 franchises have >=50% official AUTO_REVIEW_READY coverage;
- every accepted page is first-party and timestamp-safe;
- evidence <=25 words;
- only frozen V1 tags emitted;
- no contamination or predictive fitting.

Otherwise:
`OFFICIAL_AUTO_READY_LOWER_BOUND_DOES_NOT_CLEAR_PARENT_DENSITY_GATES`.

This disposition is a routing result only, not `PUBLIC_INTENT_SOURCE_QUALIFIED`.

## Advancement rule

If the lower bound clears:
- freeze the final V1 source-qualification stage using accepted official rows plus deterministic fallback/review handling for unresolved rows;
- do not refit or tune V1F;
- still do not fit a QB predictive candidate until the original V1 source audit is formally qualified.

If the lower bound does not clear:
- inspect only aggregate unresolved mechanics/source classes;
- do not change V1F thresholds, candidate cap, window or prototypes;
- local fallback/review may still be required under the original V1 protocol;
- unresolved rows are not negative evidence.

## Integrity / anti-retest

- football outcomes read = 0
- model residuals read = 0
- sportsbook/game-market inputs = 0
- paid OddsAPI calls = 0
- 2026 Week-5 outcomes read = 0
- football predictive models fit = 0
- production changes = 0
- M89/M90 and QB C2 remain protected
- same-data attempts repackaging remains closed
- schedule/rest D1 remains closed
- PBP D2 remains closed
- no generic QB mean/YPA reopening
