# QB TEAM PASS INTENT — COMMON CRAWL ARCHIVE TRANSPORT V1I-R1 RECERTIFICATION PLAN

Frozen: 2026-10-11

Parent:
- V1I run `38173788868`
- V1I artifact `11716151694`
- V1I digest `sha256:4464dafff9928d0bd04774df44b56efef02fe876e91595538a801edbdff17470`

## Reason for recertification

The first V1I readout used the existing broad team/opponent synonym vocabulary in
archive URL relevance. That vocabulary includes geographic names such as
`dallas`, `seattle`, and `green bay`.

Artifact inspection showed this can accept ordinary local-news URLs merely
because the publisher city appears in the URL, even when the page is unrelated
to the NFL team or target game.

Therefore the original V1I scientific count is superseded and cannot authorize
advancement.

This repair is source-transport only. It does not re-query Common Crawl and does
not change the frozen publisher map, archive window, 80% transport gate, or any
football/model science.

## Immutable input

Consume the exact preserved V1I artifact:
- run `38173788868`
- artifact `11716151694`
- digest `sha256:4464dafff9928d0bd04774df44b56efef02fe876e91595538a801edbdff17470`

Expected:
- 158 unresolved rows
- 123 direct-ready rows inherited from repaired V1H
- 35 archive-needed rows

## Frozen franchise-identity vocabulary

Archive rescue may use only franchise-identity tokens, never geography-only
tokens:

- ARI: cardinals
- ATL: falcons
- BAL: ravens
- BUF: bills
- CAR: panthers
- CHI: bears
- CIN: bengals
- CLE: browns
- DAL: cowboys
- DEN: broncos
- DET: lions
- GB: packers
- HOU: texans
- IND: colts
- JAX: jaguars, jags
- KC: chiefs
- LV: raiders
- LAC: chargers
- LAR: rams
- MIA: dolphins
- MIN: vikings
- NE: patriots
- NO: saints
- NYG: giants
- NYJ: jets
- PHI: eagles
- PIT: steelers
- SF: 49ers, niners
- SEA: seahawks
- TB: buccaneers, bucs
- TEN: titans
- WAS: commanders

## Frozen strict archive-rescue rule

For an archive-needed row, a preserved V1I capture counts toward transport rescue
only if:
1. it already satisfied the original V1I frozen domain/capture/status/MIME/WARC
   pointer rules;
2. its URL path contains at least one frozen franchise-identity token for either
   the target team or target opponent;
3. geography-only tokens do not satisfy this rule.

This is still a transport result, not semantic/source qualification. Old,
non-intent, or non-target-week football pages can still be rejected later by
the original publication-time and semantic qualification rules.

## Gate

Keep the original unchanged transport gate:
- direct-ready baseline = 123;
- combined strict direct-or-archive ready >=127 / 158 (>=80%).

Disposition:
- `STRICT_ARCHIVE_IDENTITY_RECERTIFICATION_PASSES_V1H_GATE`
- or `STRICT_ARCHIVE_IDENTITY_RECERTIFICATION_FAILS_V1H_GATE`

## Integrity

- no new network retrieval;
- no publisher changes;
- no football outcomes;
- no model residuals;
- no sportsbook/game-market data;
- no paid OddsAPI;
- no Week-5 outcome grading;
- no predictive football fitting;
- no production changes.
