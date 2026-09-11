# QB First-Down Public Intent Source V1B — Frozen Automation/Retrieval Plan

## Status

This document freezes the automation-first extension of the already-frozen V1 public-intent source audit **before V1B retrieval-quality results are inspected**.

Parent branch/state:
- parent branch: `research-qb-first-down-public-intent-source-v1`
- parent head at V1B branch point: `13d1f3b78a25351d537aca31124ecc6c67de4ee7`
- parent frozen plan: `docs/migrations/QB_FIRST_DOWN_PUBLIC_INTENT_SOURCE_V1_PLAN.md`
- parent frozen collection protocol: `docs/migrations/QB_FIRST_DOWN_PUBLIC_INTENT_SOURCE_V1_COLLECTION_PROTOCOL.md`
- parent collection prefix: `docs/migrations/QB_FIRST_DOWN_PUBLIC_INTENT_SOURCE_V1_COLLECTION_PREFIX.csv`

The V1 manual crawl is paused. V1B exists only to test whether the same frozen source family can be collected and source-validated at practical scale. It does not rewrite the V1 scientific question, source hierarchy, source labels, semantic vocabulary, source-qualification gates, or production model.

## Non-negotiable inherited boundaries

V1B inherits all V1 contamination prohibitions and scientific boundaries:
- no football outcomes;
- no parent residuals or model errors;
- no QB/WR target-game performance;
- no final scores or postgame summaries as search guides;
- no player props, spreads, totals, moneylines, sportsbook/game-market pages, or odds fields;
- no predictive model fitting;
- no numeric football-effect score;
- no production integration or retuning;
- NE-SEA grading remains parked.

Sportsbook information remains downstream only and cannot teach this retrieval system or the upstream football projections.

## Purpose

Test whether deterministic automation can reduce the V1 historical source archaeology to a practical workflow while preserving source integrity.

V1B may automate:
1. deterministic team-week/query generation;
2. candidate discovery under the frozen V1 source/query hierarchy;
3. URL normalization and deduplication;
4. page retrieval and text extraction;
5. publication-date/time extraction;
6. pre-kickoff timestamp validation;
7. official-source classification;
8. target-game/opponent relevance checks;
9. conservative intent/evidence candidate extraction using the already-frozen V1 semantic vocabulary;
10. routing only ambiguous cases to manual review.

V1B may **not** use retrieval failure as proof of source absence. A failed automated query leaves a row unresolved unless the frozen search sequence is demonstrably exhausted.

## Frozen universe

Reuse the exact V1 deterministic universe builder and ordering. V1B does not change seasons, sampled weeks, teams, or ordering.

The V1 universe remains:
- 2023, 2024, 2025 regular seasons;
- Weeks 2, 5, 8, 11, 14, 17;
- every scheduled team in each sampled week;
- ascending `(season, week, team)` processing order.

## Gold/reference sample

The existing contiguous completed V1 prefix is the only initial gold/reference set. At the V1B branch point it contains 13 completed rows, 2023 Week 2 ARI through HOU.

Those rows may be used only to measure operational retrieval/source-validation quality. They may not be used to add bespoke team-week queries, hard-code canonical URLs, change the semantic vocabulary, or infer predictive football value.

The gold set is not evidence that the V1 family has passed source qualification.

## Frozen query/source hierarchy

V1B must generate the same V1 concept families mechanically.

Official-source concepts:
- `<team> <opponent> coach press conference week <week> <season>`
- `<team> <opponent> offensive coordinator press conference week <week> <season>`
- `<team> game preview <opponent> week <week> <season>`
- `<team> run pass game plan <opponent> <season>`
- `<team> offense approach <opponent> <season>`

Local fallback concepts:
- `<team> <opponent> coach said offense game plan week <week> <season>`
- `<team> <opponent> run game passing game plan <season>`
- `<team> offensive coordinator <opponent> plan <season>`

Allowed mechanical adaptations:
- full franchise name versus common abbreviation to resolve indexing ambiguity;
- `site:<official-team-domain>` restriction during the official phase;
- punctuation/whitespace normalization.

Forbidden adaptations:
- queries based on remembered outcomes, residuals, player performance, or known model misses;
- team-week-specific theory terms not present in the frozen concept family;
- bespoke extra searching for a row because expected evidence was not found.

## Candidate-discovery contract

The retrieval layer must be provider-neutral and auditable. Every discovered candidate must preserve:
- team-week key;
- source phase (`OFFICIAL` or `LOCAL_FALLBACK`);
- exact generated query;
- discovery provider/transport name;
- result rank;
- original URL;
- normalized URL;
- discovery timestamp.

Default V1B implementation may use a free/public search transport suitable for research automation. No paid sportsbook/game-market service may be used. A transport failure must be distinguishable from a true zero-result query.

For each generated query, retain at most the top 5 unique candidate URLs after normalization. This is an operational cap, not a source-qualification threshold.

Official-phase candidates are restricted to the predeclared official team domain for the target team. Local fallback is attempted only when the official phase has not produced an automation-qualified eligible-intent candidate or when official material has been mechanically exhausted without sufficient intent evidence.

## Page-retrieval and timestamp contract

For every candidate URL, V1B should attempt to preserve:
- HTTP status/fetch status;
- final URL after redirects;
- page title;
- publisher/host;
- publication timestamp candidates and extraction method;
- canonical URL if declared;
- extracted visible article/transcript text;
- evidence of target opponent/week relevance.

Timestamp extraction priority:
1. structured `datePublished`/equivalent JSON-LD;
2. article publication metadata such as `article:published_time`;
3. explicit `<time datetime>` publication field;
4. clearly labeled visible publication date/time.

A date inferred only from an unverified URL pattern is not timestamp-safe.

An eligible timestamp must unambiguously predate target-game kickoff. Ambiguous, dynamic, missing, or post-kickoff dates remain unresolved/unsafe under the V1 rule.

## Source-class contract

Mechanical source classification may auto-identify:
- official team host => official candidate;
- other host => non-official candidate requiring local-attribution validation before it can qualify.

V1B must not automatically treat an arbitrary non-official search result as attributable local beat reporting merely because it ranks highly.

## Intent/evidence extraction boundary

The frozen V1 semantic vocabulary remains unchanged:
- `RUN_EMPHASIS`
- `PASS_EMPHASIS`
- `EARLY_DOWN_AGGRESSION`
- `TEMPO_CHANGE`
- `PROTECTION_DRIVEN_PLAN`
- `DEFENSIVE_MATCHUP_PLAN`
- `PERSONNEL_AVAILABILITY_PLAN`
- `OTHER_EXPLICIT_OFFENSIVE_INTENT`

V1B may generate **candidate** semantic tags and an evidence span using deterministic text rules. Candidate tags are not final scientific labels until automation quality clears the validation gates below.

The extractor must prefer direct/attributed language referring to the upcoming target game or target-week circumstance. It must not score sentiment, predict pass rate, assign confidence about football direction, or create a numeric football feature.

Evidence output remains <=25 words for the final ledger. Longer machine snippets may be retained only in a separate retrieval-debug artifact that is never passed to predictive grading.

## Auto-accept versus manual-review boundary

A candidate can be provisionally `AUTO_REVIEW_READY` only when all are true:
- fetch succeeded;
- stable locator is preserved;
- source class is mechanically valid;
- publication timestamp/date is extracted by an allowed method and is safely pre-kickoff;
- target opponent/week relevance is detected;
- an attributed offensive-intent evidence candidate is present;
- no contamination/forbidden fields are present.

Anything else routes to `MANUAL_REVIEW_REQUIRED` or remains unresolved.

V1B automation must never auto-finalize `NO_RECONSTRUCTABLE_SOURCE` from a zero-result or failed fetch. Negative V1 dispositions still require evidence that the frozen official and, when required, local fallback search sequence was actually completed.

## Frozen operational validation gates

Before scaling beyond the gold/reference sample, V1B must evaluate automation against the existing completed prefix.

Primary operational gates:
1. **eligible-candidate recall >= 90%**: for at least 90% of gold rows, the automation retrieves within its capped candidate set either the preserved canonical source or another source that independently satisfies the same frozen V1 eligibility rules;
2. **canonical-source retrieval >= 75%**: the preserved canonical locator itself appears in the candidate set for at least 75% of gold rows;
3. **timestamp-safety correctness >= 95%** versus manual gold review;
4. **official/non-official source-class correctness = 100%** on the gold sample;
5. **target-game relevance precision >= 90%** among candidates marked relevant;
6. **candidate intent-evidence precision >= 85%** among candidates proposed as `AUTO_REVIEW_READY`;
7. zero forbidden outcome/model/market fields read or emitted;
8. zero production changes.

These are automation gates only. They do not replace or relax the parent V1 source-qualification gates.

## Frozen early-stop rules

The project has a one-working-day maximum for this hypothesis family.

Stop V1B and preserve a `RETRIEVAL_AUTOMATION_NOT_QUALIFIED` disposition if, after one generic implementation/repair pass:
- eligible-candidate recall on the gold sample remains below 80%; or
- timestamp-safety correctness remains below 90%; or
- the transport is materially rate-limited/unstable enough that deterministic completion is not practical; or
- substantial team-specific exceptions/manual archaeology are still required.

Do not lower gates, expand queries, or add bespoke team exceptions to rescue V1B.

If primary operational gates pass, V1B may scale to a representative deterministic validation slice before full-universe collection. Full V1 source qualification is still required before any predictive screen.

## Representative validation slice after gold pass

If and only if the gold gates pass, validate the automation on the first deterministic non-gold slice produced by the frozen manifest order, large enough to expose multiple franchises/source hosts without outcome-based selection.

Freeze the slice mechanically as the next 32 manifest rows after the completed gold prefix. Do not replace difficult rows.

The same operational quality checks apply. Ambiguous rows are manually reviewed; no football outcomes may be opened.

## Outputs

V1B should produce separate artifacts for:
1. deterministic query manifest;
2. raw candidate discovery ledger;
3. normalized page/source-validation ledger;
4. gold validation report;
5. manual-review queue;
6. contamination/integrity report;
7. if gates pass, representative-slice validation report.

No V1B retrieval/debug artifact may contain final scores, target-game performance, model residuals, player props, spreads, totals, moneylines, sportsbook data, or predictive grading.

## Advancement rule

- If V1B operational gates fail: preserve the failure and close/move on. Do not return to a 500+ row manual crawl.
- If V1B operational gates pass: scalable source collection is authorized under the unchanged V1 source rules.
- Only after the parent V1 source-qualification gates are satisfied may a separately frozen predictive study be designed.
- Source qualification alone never authorizes production use.

## Production authority

Unchanged. V1B is research-only and cannot alter any production projection authority or distribution.
