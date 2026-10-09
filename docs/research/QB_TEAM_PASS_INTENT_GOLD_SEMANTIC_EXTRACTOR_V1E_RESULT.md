# QB TEAM PASS INTENT — GOLD SEMANTIC EXTRACTOR V1E RESULT

Date: 2026-10-08

Status: **NOT QUALIFIED. DO NOT SCALE THE DETERMINISTIC LEXICON.**

Frozen plan:
- `docs/research/QB_TEAM_PASS_INTENT_GOLD_SEMANTIC_EXTRACTOR_V1E_PLAN.md`

Canonical run:
- workflow run: `37869035844`
- exact head: `027f362e6704998d7bbcb2caedd7f28db6233346`
- artifact: `11589562359`
- artifact digest: `sha256:ba26da4739a3a4e4539cb3f9ce4215986e95b40c1d9768dd2eff0aa3c322ee3a`
- workflow conclusion: `SUCCESS` (the experiment ran and certified cleanly)
- scientific disposition: `GOLD_SEMANTIC_EXTRACTOR_NOT_QUALIFIED`
- strict repository audit: `PASS`

## Frozen result

Gold positives: **12**  
Frozen official negative safety case: **1** (ATL)

Observed:
- positive target-game relevance: **12 / 12**
- positive candidate evidence found: **10 / 12**
- positive AUTO_REVIEW_READY: **10 / 12**
- ATL negative AUTO_REVIEW_READY: **0 / 1**
- AUTO_REVIEW_READY precision on frozen gold: **1.000**
- all evidence <=25 words: **PASS**
- only frozen tags emitted: **PASS**

Failed gates:
- candidate evidence recall required >=11/12; observed **10/12**
- AUTO_REVIEW_READY recall required >=11/12; observed **10/12**

Missed positive rows:
- **BUF**
- **CHI**

For both BUF and CHI:
- page fetch succeeded;
- host classified first-party;
- JSON-LD publication timestamp was safely pre-kickoff;
- target opponent relevance was detected;
- errors = 0;
- the deterministic generic lexicon simply failed to identify qualifying evidence.

ATL:
- fetch succeeded;
- first-party = true;
- timestamp-safe = true;
- target relevant = true;
- evidence not found;
- AUTO_REVIEW_READY = false.

## Interpretation

This is not a source-transport failure and not a provenance failure.

V1C and V1D remain strongly qualified:
- first-party archive/sitemap transport is scalable;
- known official canonical sources are reacquirable;
- publication provenance is timestamp-safe.

V1E shows that the **simple deterministic lexical semantic router is not reliable enough** to automate the full source audit under the inherited >=90% recall requirement.

Because BUF and CHI had no fetch, provenance, or target-relevance error, there is no evidence of a mechanical parser defect that would justify the one allowed repair. Adding missed-page wording after observing these rows would teach to the gold test and is forbidden.

Therefore no V1E repair is attempted and the failed 10/12 result is preserved.

## Next legal frontier

Do not scale V1E over the 536-row historical universe.

The QB source family itself is **not** scientifically rejected: V1C/V1D proved a viable first-party source transport. What is source-gated is automated semantic classification using the deterministic lexicon.

The lane may proceed only with a materially different, generic text-understanding method that:
- is frozen before gold scoring;
- uses no team-specific phrases or outcomes;
- remains source/retrieval research rather than a football predictive fit;
- preserves the same 12-positive + ATL-negative gold gates;
- receives no post-hoc threshold tuning.

If no such method clears the gold gate, close/source-gate this QB public-intent seam under current automation constraints rather than reverting to the prohibited 500+ row manual crawl.

## Integrity

- public search-engine HTML contacted: false
- football outcomes read: 0
- model residuals read: 0
- sportsbook/game-market inputs: 0
- paid OddsAPI calls: 0
- Week-5 2026 outcomes read: 0
- predictive football models fit: 0
- production changed: false
- predictive candidate authorized: false

## Protected science

M89/M90, QB C2, all Week-5 locks, and all previously closed same-data QB mean/attempt mechanisms remain unchanged.
