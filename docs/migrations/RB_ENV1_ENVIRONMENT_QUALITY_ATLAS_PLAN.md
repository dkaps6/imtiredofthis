# RB-ENV1 — Pregame Environment × Runner-Quality Outcome Atlas

## Question

Across frozen temporal RB test games, how strongly do pregame rushing environment and pregame runner quality relate to actual rushing production? Which RBs excel despite bad environments, and which fail despite good environments?

## Scientific role

Diagnostic only. This migration fits no model, searches no thresholds, changes no football projection, and uses no sportsbook data.

The purpose is to separate four concepts that have often been conflated:

1. pregame environment quality,
2. pregame runner quality,
3. workload/opportunity realized in the game,
4. rushing outcome.

Postgame outcomes and actual carries are used only to describe correlation and exceptions. They never define the pregame grades.

## Frozen sources

- M95B authoritative run `33357785600`, artifact `migration-95b-rb-offense-defense-matchup`.
- M95C authoritative run `33358467022`, artifact `migration-95c-rb-quality-environment-decomposition`.

## Pregame environment grade

Primary environment score is the frozen M95C out-of-sample incremental environment contribution for rushing yards:

`env_delta_yards = role_plus_environment prediction - role_baseline prediction`

This was chosen before this audit because M95C already established environment-only as the most stable mean-efficiency signal. It avoids inventing a new weighted feature score after observing outcomes.

Within each test season, rank `env_delta_yards` and define:

- bottom quartile: `BAD_SPOT`
- middle 50%: `NEUTRAL_SPOT`
- top quartile: `GOOD_SPOT`

## Pregame runner-quality grade

Use the frozen M95B pregame composite scores:

- `off_player_efficiency_score`
- `off_player_explosive_score`

Rank each within test season and average their percentile ranks. Define:

- bottom quartile: `WEAK_RB`
- middle 50%: `MID_RB`
- top quartile: `STRONG_RB`

No target outcome enters the grade.

## Frozen evaluation panel

- 2024 M95C OOF rows: expected `1,290`
- 2025 M95C OOF rows: expected `1,290`
- pooled: `2,580`

## Outputs

Report by season and pooled:

- Pearson/Spearman correlation of environment with actual rushing yards.
- Correlation with residual versus frozen role baseline.
- Correlation with actual YPC among 8+ carry games, postgame diagnostic only.
- Average carries/yards/YPC, 75+ and 100+ rates for bad/neutral/good spots.
- Full environment × runner-quality 3×3 table.
- Postgame actual-carry-bucket environment table to see whether environment survives similar realized opportunity.
- Exception casebook:
  - bad spot + 75+ rushing yards,
  - bad spot + >=30-yard positive residual,
  - good spot + <=-30-yard residual,
  - good spot + <=40 rushing yards.
- Repeat player profiles in good/bad spots.

## Interpretation rule

This audit may establish that environment shifts outcome probability without implying deterministic causality. A strong raw-yards relationship coupled with weak within-workload/YPC correlation would indicate that environment is entangled with role/opportunity and should not be used as a blind scalar YPC modifier.

Any future model change inspired by this atlas requires a separate precommitted temporal test.
