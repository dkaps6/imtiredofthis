# QB TEAM PASS INTENT — COMMON CRAWL ARCHIVE TRANSPORT V1I PLAN

Frozen: 2026-10-11  
Branch: `research-individual-opportunity-roadmap-2026-10-08`

Parent authorities:
- V1G official-first screen: `OFFICIAL_AUTO_READY_LOWER_BOUND_CLEARS_PARENT_DENSITY_GATES`
- canonical V1G run/artifact: `37875606789` / `11593130438`
- repaired V1H direct-local transport: `LOCAL_DIRECT_TRANSPORT_PARTIAL`
- canonical repaired V1H run: `38172568766`
- canonical repaired V1H artifact: `11714987323`
- canonical repaired V1H digest: `sha256:860dda964846625a3a2188acd5951c3092b28ca36aa882a1ab72a32554a1d469`
- repaired V1H baseline: 123 / 158 unresolved rows with reachable/enumerable direct local transport = 77.8481%
- unchanged V1H transport gate: >=80%, therefore >=127 / 158 rows

## Purpose

Test one materially different, free historical archive connector after the exact
V1H direct-publisher transport design stopped below its frozen gate.

V1I is **archive transport/discovery only**.

It does not:
- fetch or score football outcomes;
- read model residuals;
- use sportsbook/game-market information;
- fit a football predictive model;
- promote a source to final `ELIGIBLE_INTENT_SOURCE_FOUND`;
- assign a final negative V1 disposition;
- change the frozen local publisher map.

The connector is Common Crawl's public CDXJ URL index / WARC archive.

## Why this is a legal restart

The preserved V1B result explicitly closed repeated unauthenticated public
search-engine HTML scraping and allowed a future restart only with a materially
different stable retrieval source/connector.

Common Crawl is materially different:
- archived web captures rather than live search-engine HTML;
- public URL index with stable archive record pointers;
- no sportsbook or outcome information;
- no need to bypass current publisher access controls.

V1I must not attempt to defeat current publisher robots, client challenges,
paywalls, or anti-bot systems.

## Immutable input

V1I consumes the exact repaired V1H artifact from run `38172568766`.

Before use it must verify:
- artifact id `11714987323`;
- artifact digest `sha256:860dda964846625a3a2188acd5951c3092b28ca36aa882a1ab72a32554a1d469`;
- exact unresolved row count = 158;
- exact direct-reachable row count = 123;
- exact direct-unreachable row count = 35.

The exact direct-unreachable rows are mechanically inherited from V1H. V1I
does not redefine that set.

## Frozen publisher identity

Use the exact V1H `LOCAL_SOURCES` map.

No publisher may be added, removed, or substituted.

For an archive-needed row, Common Crawl may be queried only for the two frozen
publisher domains already assigned to that franchise.

## Common Crawl index authority

At runtime:
1. read `https://index.commoncrawl.org/collinfo.json`;
2. use only collections whose advertised crawl interval overlaps the frozen
   archive-capture window for the target row;
3. query the collection's HTTPS CDX API endpoint;
4. use serial requests only;
5. sleep at least 0.60 seconds between index calls;
6. use a descriptive research User-Agent;
7. retry only 429/500/502/503/504 with bounded backoff.

No proxies and no parallel Common Crawl index requests.

## Frozen archive-capture window

For each archive-needed team-week:
- lower bound: target kickoff minus 9 days;
- upper bound: target kickoff plus 90 days.

Why the post-kickoff archive allowance exists:
- V1I is locating an archived copy, not using archive-capture time as source
  publication time;
- a later qualification stage must still recover the embedded publication time
  and require that publication time to be pre-kickoff under the original V1
  rules;
- any page published after kickoff will therefore fail later even if Common
  Crawl captured it.

The +90-day bound is frozen before V1I results are inspected.

## Frozen CDX query

For each archive-needed row x mapped publisher x overlapping crawl collection:

- `matchType=domain`
- `output=json`
- `from=<kickoff-9d>`
- `to=<kickoff+90d>`
- exact HTTP status filter = 200
- URL regex filter built only from the frozen existing team/opponent synonym
  vocabulary
- limit = 200 records
- collapse by URL key when supported

No result-driven query terms, player names, scores, outcomes, props, or model
errors.

## Frozen candidate rule

A CDX record is `ARCHIVE_CAPTURE_READY` for a row only if all are true:
1. returned URL is on one of that row's two frozen publisher domains;
2. capture timestamp is inside the frozen -9d/+90d archive window;
3. HTTP status is 200;
4. MIME is HTML/XHTML or missing while detected MIME is HTML/XHTML;
5. URL path matches the target team or opponent using the existing frozen V1
   synonym method;
6. archive record has non-empty `filename`, positive integer `offset`, and
   positive integer `length`.

For each ready record preserve:
- Common Crawl collection id;
- capture timestamp;
- original URL;
- MIME/status;
- WARC filename;
- offset;
- length.

V1I does not fetch the WARC payload.

## Frozen combined transport readout

A V1H unresolved row is transport-ready if either:
- V1H direct local transport already marked it reachable/enumerable; or
- V1I finds >=1 `ARCHIVE_CAPTURE_READY` record on a frozen mapped publisher.

Report:
- baseline direct-ready rows;
- archive-needed rows;
- archive-rescued rows;
- combined transport-ready rows;
- combined rate;
- rescue counts by franchise/publisher;
- Common Crawl collection/API errors.

## Frozen V1I gate

Disposition is
`COMMON_CRAWL_ARCHIVE_TRANSPORT_RESCUES_V1H_GATE`
only if all pass:

1. exact unresolved universe = 158;
2. exact inherited V1H direct-ready baseline = 123;
3. exact inherited archive-needed rows = 35;
4. combined direct-or-archive transport-ready rows >=127 / 158 (>=80%);
5. frozen publisher map unchanged;
6. every archive-ready record belongs to a frozen mapped domain;
7. no current publisher anti-bot/access-control bypass attempted;
8. football outcomes read = 0;
9. model residuals read = 0;
10. sportsbook/game-market inputs used = false;
11. paid OddsAPI calls = 0;
12. 2026 Week-5 outcomes read = 0;
13. football predictive models fit = 0;
14. production changed = false.

Otherwise:
`COMMON_CRAWL_ARCHIVE_TRANSPORT_DOES_NOT_RESCUE_V1H_GATE`.

The 80% requirement is unchanged from V1H. V1I is not allowed to lower it.

## Advancement rule

If V1I rescues the transport gate:
- freeze V1J final source-qualification bridge;
- V1J may fetch only the specific preserved WARC records selected under V1I;
- V1J must apply the original V1 publication-time/source/semantic rules;
- V1J must independently validate promotion of V1G AUTO_REVIEW_READY rows before
  final source qualification;
- no predictive QB model may be fit until final source qualification passes.

If V1I does not rescue the transport gate:
- close historical local-fallback reconstruction under the current free
  transport family;
- do not cycle through more search engines or publisher bypasses;
- preserve official-first coverage as a separate source result and evaluate a
  newly frozen prospective-only path if scientifically justified.

## Protected science

Unchanged:
- M89/M90;
- QB C2;
- generic QB mean/YPA reopening CLOSED;
- same-data attempts repackaging CLOSED;
- schedule/rest D1 CLOSED;
- PBP D2 CLOSED;
- sportsbook downstream only;
- no Week-5 prospective grading;
- production unchanged.
