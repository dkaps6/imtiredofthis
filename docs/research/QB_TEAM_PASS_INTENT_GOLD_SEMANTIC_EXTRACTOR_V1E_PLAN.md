# QB TEAM PASS INTENT — GOLD SEMANTIC EXTRACTOR V1E PLAN

Frozen: 2026-10-08  
Branch: `research-individual-opportunity-roadmap-2026-10-08`

Parents:
- V1C direct first-party archive transport: run `37868152267`, artifact `11589296706`, QUALIFIED.
- V1D first-party gold validation: run `37868631780`, artifact `11589452212`, QUALIFIED.
- immutable gold fixture: `docs/research/QB_TEAM_PASS_INTENT_V1D_GOLD_PREFIX.csv`.

## Purpose

Validate a generic, deterministic **candidate semantic extractor** before it is allowed to scale across the frozen 2023-2025 source-qualification universe.

V1E does not produce a predictive football feature. It only asks whether already-retrieved, timestamp-safe first-party pages can be mechanically routed into:
- target-game relevant;
- candidate offensive-intent evidence present;
- candidate semantic tags from the already-frozen V1 vocabulary;
- or manual review required.

No outcome, residual, market, or predictive model information is permitted.

## Frozen gold cases

Positive cases:
- the 12 OFFICIAL canonical eligible-intent rows in the immutable gold prefix.

Negative safety case:
- ATL's frozen official candidate page from the gold prefix. The manual V1 review found official material but **no sufficiently explicit qualifying intent**, and therefore required local fallback.
- V1E must not auto-accept this official page as eligible intent.

The local AJC page itself is not fetched in V1E.

## Generic extractor boundary

The extractor may use only:
1. visible first-party page text;
2. title/URL text;
3. already-validated publication metadata;
4. frozen target opponent identity from the gold row;
5. a league-wide static team-name/nickname synonym table;
6. a league-wide static semantic lexicon derived only from the frozen V1 tag definitions.

It may not use:
- team-specific phrase exceptions;
- gold evidence text as a search key;
- known outcomes;
- model residuals;
- player stats;
- betting data;
- manual per-row regexes;
- LLM sentiment or numeric football direction scores.

## Frozen semantic vocabulary

Unchanged:
- `RUN_EMPHASIS`
- `PASS_EMPHASIS`
- `EARLY_DOWN_AGGRESSION`
- `TEMPO_CHANGE`
- `PROTECTION_DRIVEN_PLAN`
- `DEFENSIVE_MATCHUP_PLAN`
- `PERSONNEL_AVAILABILITY_PLAN`
- `OTHER_EXPLICIT_OFFENSIVE_INTENT`

The generic lexicon may propose zero or more candidate tags. Candidate tags are routing metadata, not model features.

## Target-game relevance

A page is mechanically target-game relevant only if at least one frozen opponent synonym occurs in:
- canonical URL/title; or
- visible page text.

No opponent inference from outcomes or schedule results is allowed.

## Candidate intent evidence

Visible text is split into bounded sentences/paragraph fragments.

A fragment becomes a candidate only when:
- it contains a frozen offensive-intent/action cue such as plan/approach/want/need/going to/will/emphasis/focus/opportunity/attack/establish/stick with/more touches;
- and it contains an offensive object or a specific frozen semantic cue from the V1 vocabulary;
- and it is not a postgame/result phrase under the frozen exclusion lexicon.

The extractor returns at most one best evidence fragment per page, capped to 25 words in the audit output.

This is intentionally conservative. Missing a gold positive routes to manual review; it cannot create a negative source disposition.

## AUTO_REVIEW_READY

A page is only `AUTO_REVIEW_READY` when all are true:
- first-party official host;
- direct fetch succeeds;
- publication timestamp is absolute and pre-kickoff;
- target-game relevance is detected;
- candidate offensive-intent evidence exists.

Otherwise it is `MANUAL_REVIEW_REQUIRED`.

No V1E page is automatically given a final V1 availability disposition.

## Frozen gold gates

V1E is `GOLD_SEMANTIC_EXTRACTOR_QUALIFIED_FOR_FULL_SOURCE_SCREEN` only if all pass:

1. exact positive official gold rows = 12;
2. exact negative official safety rows = 1 (ATL);
3. positive `AUTO_REVIEW_READY` recall >= 11/12;
4. target-game relevance detected on >= 11/12 positive rows;
5. candidate evidence detected on >= 11/12 positive rows;
6. ATL official negative safety case is **not** `AUTO_REVIEW_READY`;
7. pooled gold AUTO_REVIEW_READY precision >= 0.85, where the 12 positives are true positives and ATL is the only frozen official negative;
8. all emitted evidence fragments <=25 words;
9. only frozen semantic tags emitted;
10. public search-engine HTML contacted = false;
11. football outcomes read = 0;
12. model residuals read = 0;
13. sportsbook/game-market inputs = 0;
14. Week-5 2026 outcomes read = 0;
15. predictive models fit = 0;
16. production changed = false.

The inherited V1B operational gates were >=90% eligible-candidate recall, >=90% target relevance precision, and >=85% candidate intent-evidence precision. With 12 positive rows, >=11/12 is the smallest discrete threshold that preserves the >=90% intent.

## Mechanical repair rule

One generic parser/lexicon-mechanics repair is allowed only for a demonstrable implementation defect or a league-wide source-format issue.

Forbidden:
- adding a phrase because a specific team/page missed;
- team/opponent-specific semantic patterns;
- changing positive/negative gold labels;
- weakening thresholds after seeing results.

## Advancement rule

If V1E passes:
- authorize a separately frozen full historical **official-first source screen** over the original deterministic 2023-2025 Weeks 2/5/8/11/14/17 universe;
- the full screen may auto-route only `AUTO_REVIEW_READY` candidates;
- unresolved/negative official rows must remain review/fallback work and may not be silently finalized.

If V1E fails after one legitimate generic repair:
- do not scale the deterministic extractor;
- preserve V1C/V1D transport as qualified, but source-gate automated semantic classification pending a materially different text-extraction method.

## Protected science

M89/M90, QB C2, all Week-5 locks, and all previously closed QB mean/attempt families remain unchanged. No OddsAPI call is permitted.
