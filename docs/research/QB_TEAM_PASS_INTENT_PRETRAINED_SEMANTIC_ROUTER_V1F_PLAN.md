# QB TEAM PASS INTENT — PRETRAINED SEMANTIC ROUTER V1F PLAN

Frozen: 2026-10-08  
Branch: `research-individual-opportunity-roadmap-2026-10-08`

Parent state:
- V1C first-party archive transport: QUALIFIED.
- V1D first-party gold provenance/reacquisition: QUALIFIED.
- V1E deterministic lexical semantic router: `GOLD_SEMANTIC_EXTRACTOR_NOT_QUALIFIED` at 10/12 positive recall with 1.000 gold precision.
- V1E is frozen failed; do not retune its lexicon.

## Purpose

Test one materially different, generic text-understanding method for source routing:

**fixed pretrained sentence embeddings + fixed semantic prototypes**, with no football outcomes, model residuals, sportsbook inputs, team-specific phrases, or supervised fitting on the gold rows.

This is a source-review router, not a football predictive model.

## Frozen pretrained model

Hugging Face model:
- id: `sentence-transformers/all-MiniLM-L6-v2`
- pinned revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- library: `sentence-transformers`
- package pin for the audit workflow: `sentence-transformers==3.4.1`

The model weights are frozen before V1F gold scoring. No fine-tuning is allowed.

## Frozen page candidates

Reuse the exact V1E gold cases:
- 12 OFFICIAL eligible-intent canonical pages = positives;
- ATL official candidate page from the immutable gold row = negative safety case;
- local AJC page is not fetched.

Reuse V1D first-party, direct-fetch, publication-timestamp, and target-opponent relevance requirements.

## Fragment pool

Visible page text is split by the already-frozen V1E generic fragmenter.

A fragment may enter semantic scoring only if:
- 5-65 words;
- it contains at least one generic offensive-context term from V1E's frozen `OFFENSE_RE`;
- it does not match V1E's frozen postgame/result exclusion lexicon.

V1F deliberately does **not** require V1E's failed action-cue regex.

No gold evidence span or team-specific wording is used to create the fragment pool.

## Frozen positive semantic prototypes

Exactly one fixed English prototype per frozen V1 tag:

- RUN_EMPHASIS: "The speaker describes a specific plan to emphasize the running game or stay with the run in the upcoming game."
- PASS_EMPHASIS: "The speaker describes a specific plan to emphasize the passing game, throw more, or create more passing opportunities in the upcoming game."
- EARLY_DOWN_AGGRESSION: "The speaker describes a specific plan to be more aggressive on early downs in the upcoming game."
- TEMPO_CHANGE: "The speaker describes a specific plan to change offensive tempo, pace, or no-huddle usage in the upcoming game."
- PROTECTION_DRIVEN_PLAN: "The speaker describes a specific offensive protection plan or adjustment for pressure in the upcoming game."
- DEFENSIVE_MATCHUP_PLAN: "The speaker describes a specific offensive plan to respond to or exploit the opponent's defensive matchup, coverage, front, or pressure."
- PERSONNEL_AVAILABILITY_PLAN: "The speaker describes a specific planned change in offensive personnel usage, role, workload, touches, packages, or injury replacement for the upcoming game."
- OTHER_EXPLICIT_OFFENSIVE_INTENT: "The speaker describes a specific offensive game plan, approach, intended adjustment, or intended opportunity for the upcoming game."

## Frozen negative semantic prototypes

- "This is general evaluation or preparation commentary without a specific offensive plan for the upcoming game."
- "This is retrospective or descriptive football commentary rather than an explicit intended offensive action for the upcoming game."

## Frozen scoring

All embeddings are L2-normalized.

For every candidate fragment:
- `positive_score` = maximum cosine similarity to the eight positive prototypes;
- `candidate_tag` = tag owning that maximum;
- `negative_score` = maximum cosine similarity to the two negative prototypes;
- `semantic_margin = positive_score - negative_score`.

A fragment is a semantic intent candidate only when:
- `positive_score >= 0.38`; and
- `semantic_margin >= 0.05`.

These thresholds are frozen before gold scoring and may not be changed after seeing V1F results.

Choose the qualifying fragment with greatest semantic margin, then greatest positive score. Output <=25 words.

## AUTO_REVIEW_READY

A page is `AUTO_REVIEW_READY` only when:
- first-party official;
- fetch success;
- absolute pre-kickoff publication timestamp;
- target-opponent relevance;
- one V1F semantic intent candidate clears both frozen thresholds.

No final V1 source disposition is assigned automatically.

## Frozen gates

V1F is `PRETRAINED_SEMANTIC_ROUTER_QUALIFIED_FOR_FULL_SOURCE_SCREEN` only if all pass:

1. exact positives = 12;
2. exact negative = 1 ATL;
3. positive AUTO_REVIEW_READY >=11/12;
4. positive target relevance >=11/12;
5. positive semantic evidence >=11/12;
6. ATL is not AUTO_REVIEW_READY;
7. pooled AUTO_REVIEW_READY precision >=0.85;
8. evidence <=25 words;
9. only frozen V1 tags emitted;
10. exact model id and revision match the frozen values;
11. no supervised fitting/fine-tuning;
12. no search-engine HTML;
13. football outcomes read = 0;
14. model residuals read = 0;
15. sportsbook/game-market inputs = 0;
16. Week-5 2026 outcomes read = 0;
17. football predictive models fit = 0;
18. production changed = false.

## Stopping rule

**No V1F threshold or prototype retuning is allowed after gold scoring.**

If V1F fails:
- preserve the failure;
- do not try another embedding threshold/model/prototype cascade in this hypothesis family;
- formally automation-source-gate the QB public-intent seam under current free/public constraints;
- do not return to the 500+ row manual crawl.

If V1F passes:
- authorize one separately frozen full historical official-first source screen over the original deterministic 536 team-week universe;
- unresolved official rows remain review/local-fallback work; they are not auto-negative;
- still no predictive QB feature or model fit is authorized.

## Protected science

M89/M90, QB C2, Week-5 locks, generic YPA/mean, same-data attempts, schedule/rest D1, and PBP D2 remain unchanged. OddsAPI calls remain prohibited.
