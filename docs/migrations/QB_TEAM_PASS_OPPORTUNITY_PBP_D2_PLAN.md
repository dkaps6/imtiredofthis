# QB Team Pass Opportunity PBP D2 — Frozen Development Plan

## Purpose

Test the two V1B source-eligible, genuinely new football information families as predictors of the physical M89-corrected `TEAM_PASS_OPPORTUNITY` residual:

1. `PENALTY_DRIVE_EXTENSION`
2. `FOURTH_DOWN_AGGRESSION`

This is a development-only screen. 2025 target outcomes remain untouched during candidate selection. Production cannot change in D2.

## Canonical lineage

- V1B result commit: `db4b0979657440ed4c9303d3888504abbfd353b7`
- V1B run: `34534419071`
- V1B job: `103062583287`
- V1B artifact: `10174823861`
- V1B digest: `sha256:b21a2d588e25a4866882aebb09b239b8c2305d09a8515c37ab5af519f09fbedf`
- Opportunity-chain run: `34523313743`
- Opportunity-chain artifact: `10170531084`
- Shared QB/WR run: `34066549394`
- Shared QB/WR artifact: `9999119623`
- M89 source run for pregame opponent identity only: `33331073376`

## Strict-prior feature construction

Use `pbp_team_game_candidate_counts.csv` from the immutable V1B artifact as the only event-count source.

For each target QB/team/week, target opponent identity may be read only from the M89 common trace key/context column `opponent`. No M89 target outcome column may enter feature construction.

Historical rows must satisfy strictly before target `(season, week)`. Use the most recent **8 team games** for each offense or defense view. No window search is permitted.

### PENALTY_DRIVE_EXTENSION — exactly 2 features

1. `team_first_down_penalty_rate_l8`
   - numerator: sum `first_down_penalties` over target offense's last 8 prior games
   - denominator: sum `offensive_pbp_rows` over those games

2. `oppdef_first_down_penalty_allowed_rate_l8`
   - last 8 prior games in which the target opponent appeared as `opponent`
   - numerator: sum first-down-by-penalty events committed by offenses facing that defense
   - denominator: sum corresponding offensive PBP rows

No generic penalty rate, penalty yards, accepted/declined inference, interaction, difference, sum, or threshold feature is allowed.

### FOURTH_DOWN_AGGRESSION — exactly 6 features

Offense last-8:

1. `team_fourth_go_rate_l8 = sum(go_attempts) / sum(fourth_down_decisions)`
2. `team_fourth_conversion_rate_l8 = sum(conversions) / sum(go_attempts)`
3. `team_fourth_pass_share_l8 = sum(go_passes) / sum(go_attempts)`

Opponent-defense allowed last-8, using prior games where target opponent is `opponent`:

4. `oppdef_fourth_go_allowed_rate_l8`
5. `oppdef_fourth_conversion_allowed_rate_l8`
6. `oppdef_fourth_pass_allowed_share_l8`

If a rate denominator is zero, use the strictly-prior league rate for that same quantity computed only from source games before the target `(season, week)`. No other imputation or shrinkage is authorized.

## Target and baseline

From the immutable opportunity-chain casebook:

- baseline team pass opportunity: `pred_D`
- actual diagnostic target: `actual_D`
- residual target: `actual_D - pred_D`
- baseline attempts: `pred_attempts`
- actual attempts: `actual_attempts`
- existing promoted QB mean: `football_synthesis`
- fixed existing `pred_C`, `pred_S`, and `pred_ypa`

Target outcomes are labels only.

## Frozen model architecture

Each family independently receives the same architecture:

- `StandardScaler`
- `Ridge(alpha=20.0, fit_intercept=False)`
- target: `actual_D - pred_D`

The no-intercept rule is frozen because D1 showed that a generic positive intercept can duplicate M89/M90's already-promoted downstream mean correction. D2 is explicitly an **incremental differential-information** test.

No alpha search, nonlinear model, ensemble, interaction terms, caps, clipping, or alternate target.

## Development split

2024 only:

- fit: Weeks 1-9
- holdout: Weeks 10-18

2025 target outcomes may not be loaded into fitting, candidate selection, or development scoring.

## Propagation

For each family candidate:

`candidate_D = pred_D + predicted_D_residual`

`candidate_attempts = candidate_D * pred_C * pred_S`

`candidate_pass_yards = football_synthesis + (candidate_attempts - pred_attempts) * pred_ypa`

All M89/M90 downstream mean logic remains otherwise unchanged. D2 changes only the upstream team-pass-opportunity factor in research propagation.

## Shared receiving test

Use only the 2024 rows from immutable `qb_wr_shared_pass_volume_secondary_2024_2025.csv`.

On 2024 Weeks 10-18 holdout rows, correlate the family-predicted D correction with `wr_reception_mass_residual`:

- Pearson
- Spearman
- same-sign

2025 WR target/reception outcomes remain untouched.

## Frozen metrics per independent family

### Team pass opportunity
- MAE
- RMSE
- bias
- correlation
- p90 absolute error

### QB attempts
- MAE
- RMSE
- bias
- correlation
- 8+ miss rate
- 10+ miss rate

### QB passing yards
- MAE
- RMSE
- bias
- correlation
- p90 absolute error
- 75+ miss rate
- 100+ miss rate

### Information transfer
- predicted D correction vs actual D residual: Pearson/Spearman/same-sign
- predicted D correction vs 2024 WR reception-mass residual: Pearson/Spearman/same-sign
- mean, mean-absolute, p90-absolute correction
- standardized Ridge coefficients
- paired 10,000-resample bootstrap probability that passing-yard MAE improves, seed `5621`

## Integrity gates

Scientific interpretation stops unless all pass:

1. exact immutable V1B source artifact used;
2. exact immutable opportunity-chain artifact used;
3. M89 trace used only for target opponent identity plus keys;
4. exact immutable 2024 shared-reception rows used for cross-position holdout test;
5. zero sportsbook/result features;
6. zero 2025 target outcome use in development;
7. exact last-8 window only;
8. exact 2 penalty features and exact 6 fourth-down features;
9. one Ridge alpha `20.0`, `fit_intercept=False` only;
10. no production changes;
11. baseline attempt identity reconciles within `1e-6`;
12. candidate changes only team pass opportunity before propagation;
13. fit/holdout keys unique and non-empty;
14. all target feature histories are strictly prior.

## Independent family survivor gates

A family is `D2_INDEPENDENT_SURVIVOR` only if all pass on 2024 Weeks 10-18:

1. team-pass-opportunity MAE improves by >= `0.15` opportunities;
2. team-pass-opportunity RMSE is non-worse;
3. QB attempt MAE improves by >= `0.10` attempts;
4. QB passing-yard MAE improves by >= `0.25` yards;
5. QB passing-yard RMSE is non-worse;
6. QB passing-yard correlation is non-worse;
7. QB p90 absolute passing-yard error is non-worse;
8. QB 100+ yard miss rate does not increase;
9. QB 10+ attempt miss rate does not increase;
10. predicted D correction Spearman vs actual D residual >= `0.10`;
11. predicted D correction Spearman vs 2024 WR reception residual >= `0.10`;
12. paired bootstrap `P(pass-yard MAE gain > 0) >= 0.70`;
13. all integrity gates pass.

Failing a gate preserves the family as `D2_INDEPENDENT_FAIL` with no retune.

## Combined candidate rule

A combined `PENALTY_PLUS_FOURTH_DOWN` candidate may be fit **only if both independent families survive**.

If authorized, it uses the union of the exact 8 frozen features and the identical `StandardScaler + Ridge(alpha=20, fit_intercept=False)` architecture. No interaction terms.

The combined candidate must satisfy the same survivor gates. The single candidate with the lowest holdout QB passing-yard MAE among surviving independent candidates and an eligible surviving combined candidate becomes the sole frozen confirmation candidate. Tie-breakers, in order:

1. lower team-pass-opportunity MAE;
2. lower QB attempt MAE;
3. higher WR-reception residual Spearman;
4. lower QB passing-yard RMSE.

## If a candidate survives

Without opening 2025 outcomes:

- refit that exact candidate architecture on all 2024 target rows;
- freeze scaler statistics, coefficients, intercept (=0), feature order, last-8 construction and fallback semantics into the D2 artifact;
- create a separate 2025 confirmation plan/branch;
- only then score untouched 2025.

D2 itself cannot change production.

## Stopping rule

Run each eligible independent family once. Run combined only if both survive. No feature/window/alpha/cap/model search. No 2025 rescue. No schedule/rest retest. No sportsbook inputs. No production mutation.
