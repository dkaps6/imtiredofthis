# QB TEAM PASS INTENT — INDEPENDENT NLI SEMANTIC VALIDATOR V1J RESULT

Date: 2026-10-11

Status: **NOT QUALIFIED. DO NOT RUN THE FULL 378-ROW NLI PROMOTION SCREEN.**

Frozen plan:
- `docs/research/QB_TEAM_PASS_INTENT_INDEPENDENT_NLI_SEMANTIC_VALIDATOR_V1J_PLAN.md`

Canonical authority:
- workflow run: `38186677190`
- exact head: `966c5a74c62e011224ae95cb1ad2f22a91455565`
- artifact: `11720102023`
- artifact digest: `sha256:8db2868c6a10813372daf1b8905f342349c07822abc90ef600effd9ffb1a6228`
- workflow conclusion: `SUCCESS`
- strict repository audit: PASS

## Frozen disposition

`INDEPENDENT_NLI_VALIDATOR_NOT_QUALIFIED`

## Gold result

- positive official gold rows: **12**
- frozen official negative: **1 (ATL)**
- positive AUTO_PROMOTION_READY: **12 / 12**
- positive target relevance: **12 / 12**
- ATL negative AUTO_PROMOTION_READY: **1 / 1** — FAIL
- pooled AUTO_PROMOTION_READY precision: **12 / 13 = 92.3077%**
- positive recall gate: PASS
- pooled precision >=0.85 gate: PASS
- ATL negative safety gate: **FAIL**

All contamination/integrity gates passed.

## Interpretation

The independent NLI validator is highly sensitive but not sufficiently specific
for automatic final source promotion.

It reproduced the exact failure mode V1J was designed to guard against:
general preparation/football language on ATL's official page was classified as
an explicit offensive-plan statement even though the original manual gold audit
correctly required local fallback.

Therefore:
- no threshold retuning;
- no hypothesis rewrite;
- no alternate NLI model cascade;
- no full 378-row automated NLI promotion screen;
- no blanket promotion of V1G AUTO_REVIEW_READY rows;
- final V1 source qualification remains unclaimed;
- predictive QB intent candidate remains unauthorized.

## Next legal frontier

Use the V1J stopping-rule fallback: a **bounded, blinded manual semantic precision
audit** of a deterministic sample of V1G AUTO_REVIEW_READY rows.

The audit must:
- freeze sample selection before adjudication;
- be stratified across season/week/franchise/tag/score-margin ranges;
- expose only source identity, publication metadata, target opponent, and source
  text needed for semantic judgment;
- hide football outcomes, residuals, sportsbook data, V1F score/tag during the
  actual semantic decision where practical;
- use the original V1 eligible-intent definition;
- measure whether V1G AUTO_REVIEW_READY precision is high enough to justify
  blanket promotion or whether row-level manual adjudication would be required.

If bounded precision is inadequate, close blanket historical automation rather
than returning to a prohibited 500+ row manual crawl.

## Integrity

- football outcomes read = 0
- model residuals read = 0
- sportsbook/game-market inputs used = false
- paid OddsAPI calls = 0
- 2026 Week-5 outcomes read = 0
- football predictive models fit = 0
- production changed = false
- final source qualification claimed = false
- predictive candidate authorized = false

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
