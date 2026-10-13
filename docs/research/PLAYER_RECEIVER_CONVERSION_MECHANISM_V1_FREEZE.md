# PLAYER RECEIVER CONVERSION MECHANISM V1 — FROZEN HISTORICAL CONTRACT

Date: 2026-10-13

Status: **FROZEN BEFORE PREDICTIVE SCORING — RESEARCH ONLY**

Source authority:
- `docs/research/PLAYER_RECEIVER_CONVERSION_STATE_V1_SOURCE_RESULT.md`
- source run `38409898663`
- artifact `11820921870`
- digest `sha256:8967a609e4fe08de5affd936c01d7ee65e60ccf350a2fc4f48f83f945cc4a07a`

## Scientific question

Does strictly-prior **receiver-level charted conversion state** provide
incremental pregame information about a named WR/TE's next-game catch conversion
beyond the canonical football model's existing catch-rate mean?

This is materially distinct from:
- raw historical catch-rate persistence;
- raw YPT/YPR volatility;
- QB/offense-level FTN aggregation from M81;
- target-depth distribution widening;
- generic defense-v-position or coverage multipliers.

## Frozen positions

Two independent cells:
1. WR
2. TE

RB is excluded because its receiving mean/tail lineages have separate protected
authority and conventional historical receiving-efficiency rescue is closed.

QB is excluded.

## Seasons and chronology

Development season:
- **2023 only**

Untouched confirmation seasons:
- **2024**
- **2025**

Target weeks:
- regular-season Weeks 5-18.

Pregame charting history for every target game:
- same receiver identity;
- same team;
- same season;
- strictly earlier weeks only;
- latest up to 8 receiver-attributed games;
- minimum 4 prior receiver-attributed games.

No cross-season player-history carryover is allowed.

Target-game FTN charting may exist in the downloaded source but is prohibited
from feature construction.

## Canonical target

For every scoreable WR/TE player-game:

`canonical_catch_rate`

is the exact historical reconstruction used by the canonical rules stack:

`rules_catch_rate -> bayes_receptions_per_target -> receptions_per_target`

in that priority order.

Actual catch rate:

`actual_receptions / actual_targets`

Target residual:

`actual_catch_rate - canonical_catch_rate`

A target game is scoreable only when actual targets > 0. Actual target count is
used only to define the realized conversion outcome and the oracle-target
reception translation. It is never a pregame feature.

## Frozen charted features

Exactly two receiver-level pregame features are allowed.

### 1. Prior receiver drop rate conditional on catchable targets

Across the latest up to 8 strictly-prior same-team receiver-attributed games:

`prior_drop_per_catchable = sum(is_drop) / sum(is_catchable_ball)`

A row is ineligible if the prior history has zero charted catchable targets.

### 2. Prior relative catchability

Player catchability:

`player_catchable_rate = sum(is_catchable_ball) / receiver_target_rows`

Same-team offense catchability is computed over the latest up to 8 strictly
prior team games:

`team_catchable_rate = sum(is_catchable_ball) / receiver_target_rows`

Feature:

`prior_relative_catchable_rate = player_catchable_rate - team_catchable_rate`

This measures whether throws directed to this receiver have historically been
more or less catchable than the same offense's general target environment.

No completion rate, raw historical player catch rate, YPT, YPR, YAC, target
depth, CPOE, sportsbook value, defense-v-position average, or target-game
charting may enter the candidate.

## Frozen model

One model per position cell.

Development fit:
- 2023 scoreable rows only.
- `StandardScaler`
- `Ridge(alpha=20, fit_intercept=False)`

Target:
- canonical catch-rate residual.

No alpha search.
No feature selection.
No nonlinear fallback.
No interaction search.
No coefficient sign forcing.
No post-result clipping search.

The fitted 2023 scaler and coefficients are frozen and applied unchanged to both
2024 and 2025.

Candidate catch-rate mean:

`clip(canonical_catch_rate + predicted_residual, 0.05, 0.95)`

The [0.05, 0.95] probability bound is frozen only as a physical safety bound.

## Frozen support floors

WR:
- 2023 development: >=300 rows and >=75 players;
- each confirmation season: >=300 rows and >=75 players.

TE:
- 2023 development: >=150 rows and >=35 players;
- each confirmation season: >=150 rows and >=35 players.

Support failure is fail-closed for that position.

## Frozen confirmation metrics

The 2023 development fit does not count as confirmation.

For 2024 and 2025 independently:

### A. Catch-rate absolute error

Per row:
- baseline error = |actual catch rate - canonical catch rate|
- candidate error = |actual catch rate - candidate catch rate|
- improvement = baseline - candidate

Require:
- mean improvement > 0;
- game-cluster bootstrap 95% CI lower bound > 0;
- player-cluster bootstrap 95% CI lower bound > 0.

### B. Oracle-target reception absolute error

This isolates conversion from opportunity by using the realized target count
only as the evaluation denominator:

- baseline receptions = actual targets × canonical catch rate
- candidate receptions = actual targets × candidate catch rate
- compare each against actual receptions.

Require:
- mean absolute-error improvement > 0;
- game-cluster bootstrap 95% CI lower bound > 0;
- player-cluster bootstrap 95% CI lower bound > 0.

### C. Mean-shift guard

Require:

`abs(mean(candidate_catch_rate - canonical_catch_rate)) <= 0.02`

in each confirmation season.

This blocks a generic league/position catch-rate offset from masquerading as
receiver-level mechanism value.

Bootstrap:
- 2,000 reps;
- fixed seeds;
- resample clusters with replacement;
- no threshold search.

## Position-level PASS

A position receives:

`PLAYER_RECEIVER_CONVERSION_MECHANISM_CONFIRMED`

only if:
- development support passes;
- all 2024 support and A/B/C gates pass;
- all 2025 support and A/B/C gates pass.

No pooled-season rescue.
No majority-season rescue.
No subgroup rescue.

If a position fails:

`PLAYER_RECEIVER_CONVERSION_MECHANISM_NOT_CONFIRMED`

## Overall disposition

If at least one position passes unchanged in both confirmation seasons:

`PLAYER_RECEIVER_CONVERSION_MECHANISM_EXISTS`

Otherwise:

`PLAYER_RECEIVER_CONVERSION_MECHANISM_NOT_CONFIRMED`

An integrity failure supersedes either disposition.

## Integrity gates

Require:
- development season exactly 2023;
- confirmation seasons exactly 2024/2025;
- target weeks within 5-18;
- strictly-prior feature max week < target week for every row;
- same-team player history only;
- target-game FTN feature use = 0;
- sportsbook inputs = 0;
- OddsAPI calls = 0;
- 2026 outcomes = 0;
- model architecture count <=2 position models;
- threshold searches = 0;
- production changes = 0.

## Promotion boundary

A PASS does **not** promote production.

It authorizes only a separately frozen full-stack integration experiment that
must preserve:
- target entitlement;
- WR-R15 / TE-R5P authority;
- canonical team target conservation;
- QB M89/M90 and C2;
- all closed matchup/efficiency families.

A FAIL closes this exact receiver-drop / relative-catchability mechanism. Do not
rescue it with alternate windows, alpha values, subgroups, extra charted FTN
fields, or pooled confirmation seasons.
