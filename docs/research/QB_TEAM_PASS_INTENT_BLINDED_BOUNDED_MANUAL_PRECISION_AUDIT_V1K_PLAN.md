# QB TEAM PASS INTENT — BLINDED BOUNDED MANUAL PRECISION AUDIT V1K PLAN

Frozen: 2026-10-11

Parents:
- V1G official-first screen: 378/536 AUTO_REVIEW_READY, not final dispositions.
- V1J independent NLI semantic validator: NOT QUALIFIED because the ATL official negative was incorrectly accepted.
- V1J stopping rule: no NLI retuning/model cascade; bounded human/manual semantic adjudication is the only legal promotion-validation fallback.
- V1I-R1 corrected transport: 141/158 = 89.2405%, PASS.

## Purpose

Determine whether V1G `OFFICIAL_AUTO_REVIEW_READY` is semantically precise
enough to justify blanket promotion to final eligible-source status without
performing a prohibited hundreds-row manual crawl.

This is a bounded source-semantic audit only.

## Universe

Exact V1G AUTO_REVIEW_READY rows:
- 378 rows
- original V1G canonical artifact `11593130438`
- original V1G run `37875606789`

## Frozen sample size

Exactly **32** rows.

## Deterministic sample selection

Define the row hash as SHA256 of:

`season|week|team|locator`

Define V1F semantic-margin quartiles empirically across the 378 AUTO_REVIEW_READY
rows by rank.

Select 32 rows greedily. At each step choose the remaining row maximizing:

- +1000 if its V1F candidate tag is not yet represented;
- +300 if its (season, sampled-week) cell is not yet represented;
- +100 if its franchise is not yet represented;
- +50 if its semantic-margin quartile is not yet represented.

Break ties by ascending row hash.

The frozen sample must cover:
- all 7 V1G-emitted semantic tags;
- all 18 season x sampled-week cells;
- all 4 margin quartiles;
- at least 24 distinct franchises.

V1F tag/margin may be used only for deterministic sample construction. They are
hidden from semantic adjudication.

## Blinded review packet

For each selected row expose only:
- blind row id;
- season/week/team/opponent/kickoff;
- official locator;
- publication time/method;
- page title;
- source text excerpts.

Do **not** expose in the review packet:
- V1F candidate tag;
- V1F positive/negative scores;
- V1F semantic margin;
- V1G evidence fragment;
- football outcomes;
- model residuals;
- sportsbook/game-market data;
- Week-5 2026 outcomes.

## Review-context extraction

Fetch the selected official page and preserve first-party/timestamp constraints.

Split visible text with the frozen V1E fragmenter.

Eligible review fragments:
- 5-65 words;
- contain frozen V1E offensive-context vocabulary;
- do not match frozen postgame/result exclusions.

Rank excerpts without V1F scores:
1. fragments matching frozen V1E ACTION_RE first;
2. fragments containing target-opponent synonym next;
3. original page order.

Preserve at most 18 excerpts per page, each unmodified except whitespace
normalization.

The reviewer may not use outside outcome knowledge.

## Frozen adjudication labels

Each row receives exactly one blinded semantic decision:
- `ELIGIBLE_EXPLICIT_TARGET_GAME_INTENT`
- `NOT_ELIGIBLE_GENERIC_OR_DESCRIPTIVE`
- `INSUFFICIENT_CONTEXT_TO_PROMOTE`

Original V1 eligibility definition controls.

Eligible requires a timestamp-safe attributable pregame source with explicit
target-week/upcoming-game offensive plan, intended adjustment, or intended
offensive player usage.

Examples of non-eligible material:
- generic football description;
- historical recap;
- opponent defensive description without an offensive response plan;
- defensive-team commentary;
- injury/availability facts without stated offensive usage/plan;
- generic preparation language;
- unrelated pass/run terminology.

Ambiguous/insufficient rows do not count as eligible for blanket promotion.

## Frozen blanket-promotion validation gate

V1K disposition is
`BOUNDED_MANUAL_PRECISION_SUPPORTS_BLANKET_V1G_PROMOTION`
only if all pass:

1. exact sample = 32;
2. all 7 semantic tags represented in hidden selection metadata;
3. all 18 season-week cells represented;
4. all 4 margin quartiles represented;
5. >=24 franchises represented;
6. **>=30 / 32** rows adjudicated `ELIGIBLE_EXPLICIT_TARGET_GAME_INTENT`;
7. no football outcomes/residuals/sportsbook data used;
8. production unchanged.

Otherwise:
`BOUNDED_MANUAL_PRECISION_REJECTS_BLANKET_V1G_PROMOTION`.

The >=30/32 bar is intentionally stronger than the old >=0.85 routing precision
threshold because V1K is deciding whether hundreds of rows may be promoted
without individual manual review.

## Advancement rule

If V1K supports blanket promotion:
- freeze the final 536-row source-ledger bridge;
- V1G official AUTO_REVIEW_READY rows may be promoted subject to their already
  validated provenance/source/timestamp constraints;
- unresolved rows still require original local/final-negative handling.

If V1K rejects blanket promotion:
- **do not** manually adjudicate all 378 V1G rows;
- close blanket historical semantic automation under the current free/public
  design;
- no predictive QB intent candidate is authorized from this historical design;
- preserve transport results separately;
- move to a newly frozen prospective structured-intent collection path or the
  next legal opportunity mechanism.

## Integrity

- outcomes read = 0
- residuals read = 0
- sportsbook inputs = false
- paid OddsAPI = 0
- Week-5 outcomes = 0
- predictive football fits = 0
- production changes = false
