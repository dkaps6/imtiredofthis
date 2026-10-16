# QB TEAM PASS INTENT — FULL OFFICIAL-FIRST SOURCE SCREEN V1G RESULT

Date: 2026-10-08

Status: **OFFICIAL LOWER BOUND CLEARS THE PARENT SOURCE-DENSITY GATES. RESEARCH ONLY. FINAL V1 SOURCE QUALIFICATION NOT YET CLAIMED. NO QB PREDICTIVE CANDIDATE AUTHORIZED.**

Frozen plan:
- `docs/research/QB_TEAM_PASS_INTENT_FULL_OFFICIAL_FIRST_SOURCE_SCREEN_V1G_PLAN.md`

Canonical successful authority:
- workflow run: `37875606789`
- exact head: `2d2cfa2c8cf77aa6fccf3829bbfc4308af7c3bab`
- artifact: `11593130438`
- artifact digest: `sha256:385810c787da55f1da6678c4722757f0c20147ef4cf7db32398e50c3d8a0257c`
- workflow conclusion: `SUCCESS`
- exact-head Repo CI: `37875611856` SUCCESS
- exact-head preserved-paid replay: `37875611865` SUCCESS
- strict repository audit inside V1G: PASS

## Frozen disposition

`OFFICIAL_AUTO_READY_LOWER_BOUND_CLEARS_PARENT_DENSITY_GATES`

Exact frozen universe:
- team-weeks: **536**
- seasons: 2023, 2024, 2025
- sampled weeks: 2, 5, 8, 11, 14, 17

Official first-party AUTO_REVIEW_READY:
- **378 / 536**
- pooled lower-bound rate: **70.5224%**
- unresolved requiring fallback/review: **158**
- unresolved finalized as no-source: **0**
- team-weeks with zero ranked candidates: **17**
- team with zero enumerated official sitemap URLs: **TEN**

## Parent density-gate comparison

### Pooled
- parent gate: >=70%
- V1G official-only lower bound: **70.5224%**
- result: PASS

### By season
- 2023: **71.4286%**
- 2024: **70.2247%**
- 2025: **69.8864%**
- parent gate: each >=60%
- result: PASS

### By sampled week
- Week 2: **72.9167%**
- Week 5: **73.8095%**
- Week 8: **71.1111%**
- Week 11: **61.6279%**
- Week 14: **73.8095%**
- Week 17: **69.7917%**
- parent gate: each >=50%
- result: PASS

### Franchise breadth
- franchises with official AUTO_REVIEW_READY coverage >=50%: **25 / 32**
- parent gate: >=24
- result: PASS

All V1G lower-bound density gates passed.

## Franchise lower-bound notes

High official coverage:
- DET 100%
- LAR 100%
- LV 100%
- NYG 100%
- TB 100%
- CHI 94.12%
- CIN 94.44%
- DEN 94.12%
- JAX 94.12%

Weak official discovery/coverage:
- DAL 0%
- TEN 0%
- KC 5.56%
- GB 17.65%
- ARI 33.33%
- NE 46.67%
- CLE 47.06%

These are routing/source observations only. Do not infer football meaning from franchise coverage.

## Integrity

- direct first-party official transport only
- public search-engine HTML contacted: false
- external search API used: false
- football outcomes read: 0
- model residuals read: 0
- sportsbook/game-market inputs used: false
- paid OddsAPI calls: 0
- 2026 Week-5 outcomes read: 0
- supervised fit/fine-tuning: none
- football predictive models fit: 0
- production changed: false
- model: `sentence-transformers/all-MiniLM-L6-v2`
- revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- positive threshold: 0.38
- semantic-margin threshold: 0.05
- accepted pages are first-party and timestamp-safe by construction
- evidence >25 words: 0
- invalid semantic tags: 0

## Mechanical failure lineage

No failed run changed the frozen science.

1. First V1G run failed in a unit test before screening because the test incorrectly expected an old opponent-only URL to outrank a week+source+date-window URL. The implementation followed the frozen score correctly; only the test assertion was repaired.
2. Next run failed before screening because the legacy historical schedule helper's old nflverse URLs returned 404 and fallback kickoff data was incomplete. V1G was moved to the repo's maintained `scripts.build._schedule_utils.get_nfl_schedule` source.
3. Next run reached the universe but failed before screening on nflverse alias `LA`; the repo's existing `canon_team` mapper was applied, producing canonical `LAR`.
4. Canonical run `37875606789` then completed the exact 536-row screen, certification, strict audit, and artifact upload successfully.

No source thresholds, semantic thresholds, candidate cap, date window, model identity, prototype, or density gate changed.

## Interpretation

This is the strongest source result in the QB opportunity lane so far.

The prior public-intent lane is no longer reasonably described as:
- inaccessible;
- too sparse at the official-source level;
- blocked by search transport;
- blocked by timestamp provenance;
- blocked by semantic automation.

Using only official first-party sources and without resolving local fallback, the automated lower bound already exceeds every parent density threshold that can be evaluated from coverage.

This is **not yet** the formal parent V1 disposition `PUBLIC_INTENT_SOURCE_QUALIFIED`.

Why:
- V1F/V1G intentionally route accepted pages as AUTO_REVIEW_READY rather than final V1 `ELIGIBLE_INTENT_SOURCE_FOUND`;
- 158 rows remain unresolved and the original collection protocol forbids converting those rows to final negative dispositions without fallback/review.

Therefore:
- final source qualification claimed: **false**
- predictive QB candidate authorized: **false**
- final source-qualification stage authorized: **true**

## Next legal frontier

Do not build another attempts model yet.

The next step is to freeze a final source-qualification bridge that preserves the original V1 semantics and resolves the distinction between:
1. V1G's already timestamp-safe official AUTO_REVIEW_READY lower-bound rows; and
2. the 158 unresolved rows requiring fallback/review.

No V1F threshold/prototype/candidate-cap/window retuning is allowed.

Because the official-only lower bound already clears all density gates, fallback work cannot be used to rescue a failing density hypothesis. It is now required only to satisfy the original complete-ledger/final-disposition governance contract and to validate that AUTO_REVIEW_READY can be promoted to final eligible-source status without introducing unacceptable false positives.

Only after that formal source qualification may a separately frozen predictive QB intent candidate be designed.

## Protected science

Unchanged:
- M89/M90
- QB C2
- generic QB YPA/mean reopening CLOSED
- same-data attempts repackaging CLOSED
- schedule/rest D1 CLOSED
- PBP D2 CLOSED
- sportsbook remains downstream only
- no Week-5 prospective research grading
