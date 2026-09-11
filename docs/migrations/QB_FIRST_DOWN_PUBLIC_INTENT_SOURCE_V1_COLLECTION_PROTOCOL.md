# QB First-Down Public Intent Source Audit V1 — Frozen Collection Protocol

Parent plan: `docs/migrations/QB_FIRST_DOWN_PUBLIC_INTENT_SOURCE_V1_PLAN.md`

This protocol is frozen before source-density results are inspected. It operationalizes collection only; it does not change the parent source classes, labels, semantic tags, qualification gates, or interpretation rules.

## Unit of work

Process every row in the deterministic manifest produced by `scripts/backtest/build_qb_public_intent_sample_v1.py` in ascending `(season, week, team)` order. Do not skip or reprioritize a team-week because of model error, player performance, betting information, or remembered game outcomes.

## Search window

For each team-week, only material publicly posted before kickoff may qualify. Search target-week material first. A source may refer to an earlier target-week personnel development if its publication date still clearly precedes kickoff and its content explicitly concerns the upcoming target game/circumstances.

## Fixed source order

1. Official team website/transcript/video-description/article archive.
2. Official head-coach/offensive-coordinator/team press-conference material discoverable through stable team/NFL pages.
3. Official team game-preview or matchup article carrying direct attributed offensive-plan statements.
4. Only if official material does not yield eligible intent content: attributable local beat reporting from a stable publisher.

Do not use national prediction pieces, fantasy advice, betting previews, sportsbook content, anonymous aggregation, social reposts without stable original attribution, or postgame recaps.

## Fixed query concepts

Use the same concept families for every team-week; team/opponent/date tokens vary mechanically.

Official-source concepts:
- `<team> <opponent> coach press conference week <week> <season>`
- `<team> <opponent> offensive coordinator press conference week <week> <season>`
- `<team> game preview <opponent> week <week> <season>`
- `<team> run pass game plan <opponent> <season>`
- `<team> offense approach <opponent> <season>`

Local-source fallback concepts:
- `<team> <opponent> coach said offense game plan week <week> <season>`
- `<team> <opponent> run game passing game plan <season>`
- `<team> offensive coordinator <opponent> plan <season>`

Search wording can be adapted only to resolve team-name ambiguity or site indexing (for example abbreviation versus full franchise name), not to pursue an outcome-shaped theory.

## Evidence stopping rule

For a team-week:
- stop official searching once one timestamp-safe eligible intent source with stable locator is found, except that a second official item may be retained if it contains genuinely additional semantic content;
- if official pregame material is found but contains no eligible intent content, record that fact before local fallback;
- use local fallback only when official searching under the fixed concepts is exhausted without eligible intent content;
- stop local searching once one timestamp-safe attributable eligible intent source is found;
- if no qualifying item is found after the fixed official and fallback concepts are exhausted, assign the appropriate frozen non-eligible disposition; do not invent further bespoke searches.

## Timestamp handling

Preserve the displayed publication date/time and locator. If only a date is shown, it can count only when that date unambiguously predates kickoff under the parent rule. Dynamically overwritten dates, undated pages that cannot be bounded before kickoff, or post-kickoff timestamps are `TIMESTAMP_UNSAFE` and cannot count as eligible coverage.

## Tagging boundary

Assign only the parent frozen tag vocabulary. Tags describe the quoted/paraphrased public statement; they do not encode expected direction, confidence, strength, or football outcome. Do not add tags after viewing aggregate coverage.

## Negative-row evidence

A non-eligible disposition is allowed only after the fixed search sequence is actually completed. `PENDING_REVIEW` manifest rows must never be converted to `NO_RECONSTRUCTABLE_SOURCE` merely because an automated query returned nothing once.

## Audit trail

For every team-week preserve:
- whether official search was completed;
- whether local fallback was completed when required;
- candidate and canonical locators reviewed;
- final frozen disposition;
- <=25-word evidence only for the canonical ledger row;
- notes for duplicate/syndicated material without counting duplicates as coverage.

## Contamination prohibitions

During collection do not open or use model residuals, QB/WR outcomes, final scores, player props, spreads, totals, moneylines, sportsbook pages, or postgame performance summaries as a search guide or qualification signal.

NE-SEA grading remains parked. Production changes remain prohibited.
