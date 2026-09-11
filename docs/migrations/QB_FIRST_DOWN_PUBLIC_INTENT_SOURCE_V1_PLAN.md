# QB First-Down Public Intent Source Audit V1 — Frozen Plan

## Purpose

Audit whether a genuinely new **pregame target-game intent** information family is historically reconstructable, timestamp-safe, sufficiently dense, and operationally maintainable before any predictive use is considered.

Immediate scientific parent:
- branch: `research-qb-first-down-score-state-decomp-v1`
- preserved result commit: `d819296f24f459170747040951177ae113704fbb`
- canonical run: `34548863668`
- job: `103107307133`
- artifact: `10180050013`
- disposition: `FIRST_DOWN_WITHIN_SCORE_STATE_PROPENSITY_PRIMARY_DIAGNOSTIC`

The parent established that the dominant shared first-down pass/run-choice miss persists after holding down, field-position zone, and realized score state constant. Its frozen stopping rule requires a source audit for genuinely new target-game intent information rather than another transform of historical PBP tendencies.

This V1 is a **source-availability and timestamp-integrity audit only**. It cannot inspect football outcomes, fit a model, score a candidate, or alter production.

## New information family

Audit public, pregame, target-week natural-language statements that can reveal a team's intended offensive emphasis for the upcoming game and that are not reducible to prior-game tendency transforms.

Eligible source classes, in priority order:
1. official team/head-coach/offensive-coordinator press-conference transcripts or official article transcripts;
2. official team pregame/game-preview articles containing direct attributed coach/coordinator/player statements about offensive approach;
3. attributable local beat-reporting quotations from the same target week when an official transcript is unavailable.

Examples of eligible semantic content include explicit upcoming-game statements about:
- establishing or leaning on the run;
- throwing early / being aggressive through the air;
- protecting a changed/inexperienced offensive line or quarterback;
- deliberately changing tempo or early-down approach;
- exploiting a named defensive front/coverage tendency through pass/run selection;
- workload/game-plan changes caused by known target-week personnel availability.

These are examples of source semantics, **not** predeclared predictive directions and not outcome labels.

## Anti-reinvention boundary

This family is distinct from and must not rediscover:
- M67 broad prior-game offensive intent, aggregate injuries, formation/personnel, and continuity transforms;
- M68 prior-game opening-script DBR, first-drive/first-two-drive/Q1 tendencies, verified-playcaller history/change variables, and playoff leverage;
- the closed first-down choice-economics family based on strictly-prior pass-vs-run EPA/success;
- generic score-state or possession/dropback architectures from M64/M65;
- M89 catastrophic-completion/tail mechanisms.

A public statement is not eligible merely because it describes one of those historical variables. It must contain target-week, pregame information attributable to a dated public source and available before kickoff.

## Frozen audit seasons and sampling

Coverage audit universe:
- 2023, 2024, and 2025 NFL regular seasons.
- All 32 teams.
- No postseason games.

To keep the source audit reproducible without outcome-driven selection, use the following deterministic week sample for the first pass:
- Weeks 2, 5, 8, 11, 14, and 17 in each season.
- Every team with a scheduled regular-season game in each sampled week.

No game or team may be dropped based on eventual QB/WR performance, first-down DBR, betting line, residual size, or model error.

## Frozen timestamp rule

A source item is eligible only if all are true:
1. publication/transcript date-time is recoverable or the hosting page provides a date that unambiguously predates kickoff;
2. the statement refers to the upcoming target game or target-week offensive circumstances;
3. it was publicly available before target-game kickoff;
4. source attribution identifies the speaker or reporting outlet;
5. the URL or stable archive locator is preserved.

If a page date is ambiguous, post-kickoff, dynamically overwritten, or cannot be independently bounded before kickoff, mark it `TIMESTAMP_UNSAFE` and do not count it as eligible coverage.

## Frozen source hierarchy and duplicate handling

For each team-week:
- search official team sources first;
- if multiple official items repeat the same quotation, retain one canonical item and record duplicates separately;
- use attributable beat reporting only when official eligible material is absent or when it supplies genuinely additional target-week intent content;
- syndicated duplicates count once;
- retrospective/postgame explanations are prohibited.

## Frozen labels for source audit only

Each team-week receives exactly one availability disposition:
- `ELIGIBLE_INTENT_SOURCE_FOUND`
- `PUBLIC_PREGAME_SOURCE_FOUND_NO_INTENT_CONTENT`
- `TIMESTAMP_UNSAFE_ONLY`
- `NO_RECONSTRUCTABLE_SOURCE`

For eligible items, record zero or more semantic tags from this fixed vocabulary:
- `RUN_EMPHASIS`
- `PASS_EMPHASIS`
- `EARLY_DOWN_AGGRESSION`
- `TEMPO_CHANGE`
- `PROTECTION_DRIVEN_PLAN`
- `DEFENSIVE_MATCHUP_PLAN`
- `PERSONNEL_AVAILABILITY_PLAN`
- `OTHER_EXPLICIT_OFFENSIVE_INTENT`

Tagging is descriptive only. Do not assign numeric weights, signs, confidence scores, or predicted DBR effects in V1.

## Frozen outputs

Produce:
1. team-week source ledger with season, week, team, opponent, kickoff, source class, publisher, speaker, publication timestamp/date, URL/archive locator, availability disposition, semantic tags, and a <=25-word evidence paraphrase/quote compliant with source limits;
2. coverage summary by season, sampled week, team, and source class;
3. timestamp-safety failure summary;
4. duplicate/syndication summary;
5. semantic-tag frequency summary;
6. operational-maintainability notes identifying whether the source can realistically be refreshed during an in-season weekly pipeline without paid sportsbook/game-market data.

No football target, model residual, market result, or postgame performance metric may appear in V1 outputs.

## Frozen source-qualification gates

The family is `PUBLIC_INTENT_SOURCE_QUALIFIED` only if all pass:
1. deterministic sample universe is complete and auditable;
2. >=70% pooled team-week coverage with `ELIGIBLE_INTENT_SOURCE_FOUND`;
3. >=60% eligible coverage in each individual season;
4. >=50% eligible coverage in every sampled week pooled across seasons;
5. >=90% of counted eligible items have an unambiguous pre-kickoff timestamp/date;
6. >=80% of eligible items originate from official team sources or stable attributable local reporting with preserved locators;
7. at least 24 of 32 franchises have >=50% eligible coverage across their sampled team-weeks;
8. no outcome, parent residual, QB/WR target, future result, player-prop line, spread, total, or sportsbook/game-market field is read or used;
9. no model fitting or numeric predictive score is constructed;
10. zero production changes.

If any gate fails, disposition is `PUBLIC_INTENT_SOURCE_NOT_QUALIFIED` and no predictive screen is authorized from this exact source design.

## Frozen interpretation

- If qualified: authorize exactly one separately frozen development design that must prospectively define a reproducible text-to-feature representation before it is exposed to target outcomes. Qualification alone does not imply predictive value or production use.
- If not qualified because of coverage/timestamp instability: close this exact public-intent source design. Do not lower coverage gates or backfill with postgame material.
- If official sources are sparse but attributable local sources independently clear the full frozen gates, preserve that distinction explicitly before any next design.

## Prohibited actions

- No 2023/2024/2025 football outcome inspection during source qualification.
- No searching more aggressively for teams known to have large model misses.
- No post-hoc semantic-tag additions after inspecting source density.
- No LLM or manual sentiment score tied to outcomes.
- No sportsbook/game-market information as teacher, feature, filter, or validation signal.
- No production integration.
- NE-SEA grading remains parked.
