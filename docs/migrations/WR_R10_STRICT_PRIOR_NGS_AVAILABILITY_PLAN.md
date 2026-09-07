# WR-R10 Strict-Prior NGS Feature Availability — Frozen Plan

## Status
Source/feature-availability audit only. Frozen before results. No model fitting and no production change.

## Lineage
- WR production anchor: M38 hierarchy.
- WR-R9 source audit run `34073245454`, artifact `10001172792`, disposition `NGS_RECEIVING_SOURCE_PARTIAL_ONLY`.
- WR-R9 showed same-game WR-R1 join coverage was only 45.7%, but 83.39% of WR-R1 player-games had at least one earlier same-player NGS observation. Because same-week NGS is forbidden anyway, R10 tests the narrower strict-prior use case authorized by R9.

## Frozen scope
- Reference: exact WR-R1 paired casebook from run `34058453941`.
- Target seasons: **2021-2025** only. 2020 is excluded from the scientific-availability scope because the frozen NGS pull begins in 2020 and therefore cannot provide prior-season NGS history for 2020 Week 1. This scope restriction is frozen before R10 results and is based only on source chronology, not performance.
- NGS source: `nflreadpy.load_nextgen_stats(seasons=[2020,2021,2022,2023,2024,2025], stat_type="receiving")`, REG only.
- No fuzzy matching and no same-game NGS values.

## Predeclared fields
- `avg_separation`
- `avg_cushion`
- `avg_intended_air_yards`
- `percent_share_of_intended_air_yards`
- `avg_yac`
- `avg_expected_yac`
- `avg_yac_above_expectation`
- `catch_percentage`

## Strict-prior feature semantics
For every WR-R1 target player-game, NGS observations are eligible only when `(season, week)` is chronologically earlier than the target game. Cross-season history is allowed. Compute, for every field:
1. latest strictly-prior value (`prior1`);
2. mean of the latest 3 strictly-prior observations (`prior3_mean`) when at least 3 observations exist;
3. count of strictly-prior observations.

No target-game outcome or same-week NGS row is ever eligible.

## Outputs
Report pooled and by target season:
- target player-games;
- fraction with >=1 prior NGS observation;
- fraction with >=3 prior NGS observations;
- non-null availability for each `prior1` and `prior3_mean` field;
- median prior-observation count;
- identity/unmatched samples;
- explicit zero same-game leakage count.

## Frozen eligibility gate
`STRICT_PRIOR_NGS_FEATURES_ELIGIBLE` requires all:
1. all target seasons 2021-2025 represented;
2. duplicate normalized NGS player-game rate <=1%;
3. zero same-game/future observations used;
4. >=70% pooled WR-R1 target rows have >=1 prior NGS observation;
5. >=60% in every target season have >=1 prior NGS observation;
6. >=60% pooled target rows have >=3 prior NGS observations;
7. at least 4 of 8 fields have pooled `prior1` non-null availability >=70%;
8. at least 4 of 8 fields have pooled `prior3_mean` non-null availability >=60%.

If the gate passes, a separately frozen WR predictive experiment may use only these strict-prior features. If it fails: `STRICT_PRIOR_NGS_FEATURES_INELIGIBLE` and no NGS science is run this cycle. No coverage threshold changes after results.
