# QB TEAM PASS INTENT — LOCAL FALLBACK DIRECT-PUBLISHER TRANSPORT V1H PLAN

Frozen: 2026-10-11  
Branch: `research-individual-opportunity-roadmap-2026-10-08`

Parent authorities:
- original V1 source-audit plan: `docs/migrations/QB_FIRST_DOWN_PUBLIC_INTENT_SOURCE_V1_PLAN.md` on the preserved public-intent branch;
- original frozen collection protocol: `docs/migrations/QB_FIRST_DOWN_PUBLIC_INTENT_SOURCE_V1_COLLECTION_PROTOCOL.md`;
- V1B search-engine HTML retrieval: `RETRIEVAL_AUTOMATION_NOT_QUALIFIED`; any restart requires materially different stable transport;
- V1G official-first full screen: `OFFICIAL_AUTO_READY_LOWER_BOUND_CLEARS_PARENT_DENSITY_GATES`;
- canonical V1G run: `37875606789`;
- canonical V1G artifact: `11593130438`;
- canonical artifact digest: `sha256:385810c787da55f1da6678c4722757f0c20147ef4cf7db32398e50c3d8a0257c`;
- V1G rows: 536 total, 378 official AUTO_REVIEW_READY, 158 unresolved requiring fallback/review.

## Purpose

Test whether the **original V1 local-fallback step** can be executed at practical historical scale without reopening the failed V1B public-search-engine transport.

V1H is a transport/discovery audit only. It does not promote any V1G row to final `ELIGIBLE_INTENT_SOURCE_FOUND`, does not assign final negative dispositions, does not fit a football model, and does not read football outcomes.

The exact question is:

> Can a fixed, attributable, local-publisher map plus direct publisher robots/sitemap archives provide stable deterministic transport for the unresolved V1G team-weeks?

## Immutable V1G input

V1H must consume the exact preserved V1G artifact from run `37875606789`.

It must verify before use:
- artifact id `11593130438`;
- artifact digest `sha256:385810c787da55f1da6678c4722757f0c20147ef4cf7db32398e50c3d8a0257c`;
- V1G version identity;
- exact universe 536;
- exactly 378 `OFFICIAL_AUTO_REVIEW_READY`;
- exactly 158 `OFFICIAL_UNRESOLVED_REQUIRES_FALLBACK_OR_REVIEW`.

Do **not** rerun V1G to recreate this input.

## Frozen local-publisher map

The map is declared for **all 32 franchises before V1H retrieval results are inspected**. It is not chosen only for low-coverage V1G teams.

Each team receives two attributable local/regional publishers where practical. These are discovery transports only; later semantic qualification still must satisfy the parent V1 rules.

- ARI: azcentral.com; arizonasports.com
- ATL: ajc.com; 11alive.com
- BAL: baltimoresun.com; wbal.com
- BUF: buffalonews.com; wgrz.com
- CAR: charlotteobserver.com; wcnc.com
- CHI: chicagotribune.com; suntimes.com
- CIN: cincinnati.com; wcpo.com
- CLE: cleveland.com; wkyc.com
- DAL: dallasnews.com; star-telegram.com
- DEN: denverpost.com; 9news.com
- DET: freep.com; detroitnews.com
- GB: jsonline.com; greenbaypressgazette.com
- HOU: houstonchronicle.com; click2houston.com
- IND: indystar.com; wthr.com
- JAX: jacksonville.com; news4jax.com
- KC: kansascity.com; kshb.com
- LAC: latimes.com; ocregister.com
- LAR: latimes.com; ocregister.com
- LV: reviewjournal.com; lasvegassun.com
- MIA: miamiherald.com; sun-sentinel.com
- MIN: startribune.com; twincities.com
- NE: boston.com; masslive.com
- NO: nola.com; theadvocate.com
- NYG: nj.com; nydailynews.com
- NYJ: nj.com; nydailynews.com
- PHI: inquirer.com; delcotimes.com
- PIT: post-gazette.com; triblive.com
- SEA: seattletimes.com; thenewstribune.com
- SF: sfchronicle.com; mercurynews.com
- TB: tampabay.com; wtsp.com
- TEN: tennessean.com; wkrn.com
- WAS: washingtonpost.com; wtop.com

No publisher may be added, removed, or swapped after transport results are inspected in V1H.

## Direct transport only

For each frozen publisher domain:
1. fetch `robots.txt`;
2. follow same-domain sitemap declarations;
3. allow same-domain `/sitemap.xml`, `/sitemap-index.xml`, and `/sitemap_index.xml` fallbacks;
4. recursively enumerate same-domain sitemap records under fixed safety bounds;
5. preserve URL and sitemap `lastmod` only as discovery metadata.

Prohibited:
- Google;
- Bing;
- DuckDuckGo;
- public search-engine HTML;
- external search APIs;
- sportsbook/game-market search or pages.

A publisher may redirect between bare/www subdomains or same registrable domain. Cross-publisher redirects are not accepted.

## Frozen unresolved universe

Only the 158 V1G unresolved rows are evaluated for local fallback transport.

V1H does not rescore the 378 official AUTO_REVIEW_READY rows.

## Frozen discovery window

For each unresolved team-week:
- target publication/discovery window = kickoff minus 9 days through kickoff;
- use the same frozen target-team and opponent synonym vocabulary already present in V1E/V1G;
- a local URL is a candidate only when it has at least one target-team or target-opponent URL token **and** either:
  - sitemap `lastmod` is inside the 9-day window; or
  - the URL contains the target season or exact week token.

This is deliberately stricter than “all local news in a 9-day window” and mechanically mirrors the original V1 fixed-query concepts.

No sitemap `lastmod` is treated as publication-time certification.

## Frozen bounds

Per publisher:
- maximum sitemap documents parsed: 250;
- maximum enumerated URLs: 100,000.

Per unresolved team-week/publisher:
- retain at most 60 ranked candidate URLs;
- ranking uses only URL/date discovery metadata:
  - +4 opponent token;
  - +4 target-team token;
  - +3 exact week token;
  - +2 target season token;
  - +1 sitemap `lastmod` inside target window;
  - tie-break by lastmod distance to kickoff, then normalized URL.

These bounds are transport constraints only and cannot generate a final V1 negative disposition in V1H.

## Frozen outputs

Report:
- transport status for every frozen publisher;
- reachable publisher count by franchise;
- unresolved team-weeks with >=1 transport-complete mapped publisher;
- unresolved team-weeks with >=1 local candidate URL;
- candidate-count distribution;
- zero-candidate rows;
- transport errors by publisher;
- direct-search integrity flags.

V1H emits only:
- `LOCAL_DIRECT_TRANSPORT_READY`
- `LOCAL_DIRECT_TRANSPORT_PARTIAL`

It does not emit any original final V1 availability disposition.

## Frozen transport gates

V1H is `LOCAL_DIRECT_TRANSPORT_READY` only if all pass:
1. exact unresolved input = 158;
2. frozen publisher map covers all 32 franchises with exactly two declared publishers each;
3. >=24/32 franchises have at least one reachable/enumerable mapped local publisher;
4. >=80% of the 158 unresolved rows have at least one reachable/enumerable mapped local publisher;
5. public search-engine HTML contacted = false;
6. external search API used = false;
7. football outcomes read = 0;
8. model residuals read = 0;
9. sportsbook/game-market inputs used = false;
10. paid OddsAPI calls = 0;
11. 2026 Week-5 outcomes read = 0;
12. football predictive models fit = 0;
13. production changed = false.

Local candidate yield is reported descriptively and is **not** a transport qualification gate because a complete local search is allowed to legitimately find no eligible source.

## Advancement rule

If `LOCAL_DIRECT_TRANSPORT_READY`:
- freeze V1I final source-qualification bridge;
- V1I may use the exact V1G 378 official rows plus direct local fallback retrieval on the 158 unresolved rows;
- V1I must independently validate promotion precision for V1G AUTO_REVIEW_READY before claiming final V1 eligibility;
- V1I must not lower V1F thresholds, alter prototypes, extend the 9-day window, or alter the original V1 density gates.

If `LOCAL_DIRECT_TRANSPORT_PARTIAL`:
- inspect only aggregate transport failure classes;
- do not change semantic thresholds or density gates;
- a later repair may change transport implementation only, not source science.

## Protected science

Unchanged:
- M89/M90;
- QB C2;
- generic QB mean/YPA reopening CLOSED;
- same-data attempts repackaging CLOSED;
- schedule/rest D1 CLOSED;
- PBP D2 CLOSED;
- sportsbook remains downstream only;
- no Week-5 prospective grading;
- no production changes.
