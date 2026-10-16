# QB TEAM PASS INTENT — INDEPENDENT NLI SEMANTIC VALIDATOR V1J PLAN

Frozen: 2026-10-11  
Branch: `research-individual-opportunity-roadmap-2026-10-08`

Parents:
- V1F pretrained semantic router: qualified for retrieval routing, not final source dispositions.
- V1G official-first screen: 378/536 AUTO_REVIEW_READY; final promotion explicitly withheld pending precision validation.
- V1I-R1 strict archive identity recertification: transport gate passes conservatively at 141/158; no final source qualification claimed.

## Purpose

Validate one independent, frozen semantic adjudicator before it is allowed to
promote V1G AUTO_REVIEW_READY official pages toward final V1
`ELIGIBLE_INTENT_SOURCE_FOUND`.

This is source semantics only. It is not a football predictive model.

The adjudicator must distinguish:
- an explicit offensive plan/intended adjustment for the upcoming target game;
from:
- generic football description, historical recap, opponent description,
  defensive commentary, roster/injury facts without an offensive plan, or
  unrelated use of passing/running terminology.

## Frozen model

Hugging Face:
- model: `cross-encoder/nli-deberta-v3-small`
- revision: `fa2804872c3b4bd748f38c0185cc85775361e735`
- architecture: pretrained NLI cross-encoder
- no fine-tuning
- no supervised learning on this NFL gold set
- no threshold search after results

The model was trained for natural-language inference, not football prediction.

## Frozen gold universe

Exactly the immutable V1D gold fixture:
- 12 official canonical positive pages whose original manual source audit
  disposition was `ELIGIBLE_INTENT_SOURCE_FOUND`;
- ATL's official Week-2 candidate page as the one frozen official negative
  safety case; ATL's local AJC positive is not scored.

No new gold labels are created.

## Frozen page prerequisites

Reuse V1D/V1E source mechanics:
- first-party official host;
- fetch succeeds;
- publication timestamp parseable and unambiguously pre-kickoff;
- target opponent relevance detected;
- no postgame/result material.

## Frozen fragment pool

Use the already-frozen V1E fragmenter.

A fragment may enter NLI scoring only if:
- 5-65 words;
- it contains at least one frozen V1E offensive-context term;
- it does not match the frozen postgame/result exclusion regex.

V1J does not require the failed V1E action-cue regex.

## Frozen NLI hypothesis

Exact hypothesis:

> "The speaker states a specific offensive plan, intended adjustment, or intended player usage for the upcoming game."

No tag-specific hypotheses and no team-specific wording.

## Frozen scoring

For each eligible fragment:
- obtain NLI probabilities from the pinned model;
- identify the model's entailment label from its frozen config;
- `entailment_probability` = probability of entailment;
- `semantic_margin` = entailment probability minus the larger of neutral or contradiction.

A fragment qualifies only if:
- entailment probability >= **0.70**;
- semantic margin >= **0.20**.

Choose the qualifying fragment with:
1. highest semantic margin;
2. then highest entailment probability.

Output evidence is capped to 25 words.

These thresholds are frozen before gold scoring and may not be changed after
seeing V1J results.

## AUTO_PROMOTION_READY

A gold page is `AUTO_PROMOTION_READY` only when:
- all page prerequisites pass;
- at least one frozen NLI fragment clears both thresholds.

This remains source classification, not predictive direction or weighting.

## Frozen V1J gold gates

V1J is
`INDEPENDENT_NLI_VALIDATOR_QUALIFIED_FOR_OFFICIAL_PROMOTION_SCREEN`
only if all pass:

1. exact positive official gold rows = 12;
2. exact official negative rows = 1;
3. positive AUTO_PROMOTION_READY >= 11/12;
4. positive target relevance >= 11/12;
5. ATL official negative is not AUTO_PROMOTION_READY;
6. pooled AUTO_PROMOTION_READY precision >= 0.85;
7. evidence <=25 words;
8. only source text/title/URL/publication metadata/opponent identity are used;
9. exact model/revision/hypothesis/thresholds match this plan;
10. no supervised fitting/fine-tuning;
11. football outcomes read = 0;
12. model residuals read = 0;
13. sportsbook/game-market inputs used = false;
14. paid OddsAPI calls = 0;
15. 2026 Week-5 outcomes read = 0;
16. football predictive models fit = 0;
17. production changed = false.

Otherwise:
`INDEPENDENT_NLI_VALIDATOR_NOT_QUALIFIED`.

## Stopping rule

No threshold, hypothesis, model, or fragment-pool retuning is allowed after the
gold result.

If V1J passes:
- authorize one full official promotion screen over the exact 378 V1G
  AUTO_REVIEW_READY rows with the same frozen V1J validator;
- V1G rows rejected by V1J return to unresolved/fallback handling and are not
  automatically final negatives.

If V1J fails:
- do not run another NLI-model/threshold cascade;
- do not blanket-promote V1G AUTO_REVIEW_READY rows;
- preserve the public-intent lane as requiring bounded human/manual semantic
  adjudication or close it under current automation constraints.

## Protected science

Unchanged:
- M89/M90;
- QB C2;
- generic QB mean/YPA CLOSED;
- same-data attempts CLOSED;
- schedule/rest D1 CLOSED;
- PBP D2 CLOSED;
- sportsbook downstream only;
- no Week-5 prospective grading;
- production unchanged.
