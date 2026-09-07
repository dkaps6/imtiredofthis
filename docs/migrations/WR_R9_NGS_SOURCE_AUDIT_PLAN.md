# WR-R9 Next Gen Stats Receiving Source Audit — Frozen Plan

## Purpose
WR-R8 showed that the already-recovered snap/depth variables are not sufficient to explain individual target-allocation errors, even inside the later-discovered TARGETS-dominant WR class. WR-R7 likewise showed that simple player/defense explosive history is insufficient for YPR-dominant WRs.

The next legitimate information family is richer route/coverage/ball-flight/after-catch context. Before testing any hypothesis, WR-R9 audits whether NFL Next Gen Stats receiving data can support leakage-safe multi-season work.

**WR-R9 is source audit only. It performs no predictive test, no model fitting, no parameter tuning, and no production change.**

## Frozen source
Use `nflreadpy.load_nextgen_stats(seasons=[2020,2021,2022,2023,2024,2025], stat_type="receiving")`.

The source is treated as realized weekly/postgame information. It is **never eligible for the same target game**. A future scientific experiment may use only chronologically prior completed games (for example `shift(1)` rolling histories).

## Frozen reference cohort
Use exact WR-R1 multi-season artifact from run `34058453941`, artifact `wr-r1-multiseason-2020-2025`.

Reference file: `wr_r1_paired_wr_casebook.csv`.

Because the casebook contains both `rec_yards` and `receptions`, source coverage is audited on the unique player-game keys:
- `season`
- `week`
- `team`
- normalized `player_clean_key`

Target seasons are exactly 2020-2025.

## Technical identity resolution
No fuzzy matching. NGS player identity is normalized from the first available exact field in:
- `player_display_name`
- `player_name`
- `player_short_name`

Team is resolved from the first available field in:
- `team_abbr`
- `team`
- `club`

Only canonical string normalization and known NFL team abbreviation normalization are allowed. Unmatched rows remain unmatched evidence.

## Candidate field availability audit
Report exact schema and non-null coverage for these predeclared semantic families when the corresponding exact NGS field exists:
- separation: `avg_separation`
- cushion: `avg_cushion`
- intended air yards: `avg_intended_air_yards`
- intended-air-yard share: `percent_share_of_intended_air_yards`
- YAC: `avg_yac`
- expected YAC: `avg_expected_yac`
- YAC above expectation: `avg_yac_above_expectation`
- catch percentage: `catch_percentage`

No alternate performance field is substituted after results. Missing fields are reported missing.

## Audit outputs
Report:
1. full NGS schema;
2. NGS rows by season and week;
3. duplicate rate on normalized `season/week/team/player` keys;
4. WR-R1 unique player-game coverage overall and by season;
5. NGS field non-null coverage overall and by season;
6. first-game/prior-game eligibility counts demonstrating how many WR-R1 rows could have at least one earlier same-player NGS observation if used with a strict shift;
7. unmatched WR-R1 and unmatched NGS samples for identity review.

## Frozen source dispositions
- `NGS_RECEIVING_SOURCE_MULTISEASON_ELIGIBLE` only if all are true:
  1. all six seasons 2020-2025 are present;
  2. a week field exists and has regular-season weekly variation;
  3. duplicate normalized player-game rate <= 1%;
  4. WR-R1 player-game join coverage >= 60% pooled;
  5. join coverage >= 50% in every individual season;
  6. at least four of the eight predeclared semantic fields exist with pooled non-null coverage >= 80% among matched NGS rows;
  7. at least 50% of WR-R1 player-games have a leakage-safe earlier same-player NGS observation.

- `NGS_RECEIVING_SOURCE_PARTIAL_ONLY` if seasons/week integrity pass but one or more coverage gates fail.

- `NGS_RECEIVING_SOURCE_INELIGIBLE` if season/week integrity fails or duplicate player-game rate exceeds 1%.

No coverage threshold will be changed after the audit.

## Authorized next step
Only a `MULTISEASON_ELIGIBLE` disposition authorizes a separately frozen 2020-2025 WR tracking/efficiency experiment. `PARTIAL_ONLY` requires a predeclared narrower source scope or another source family before science. `INELIGIBLE` closes this source for the current cycle.
