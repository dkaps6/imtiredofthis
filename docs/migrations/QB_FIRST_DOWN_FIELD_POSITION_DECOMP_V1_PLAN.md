# QB First-Down Field-Position Decomposition V1 — Frozen Plan

## Purpose

Decompose the newly established shared `FIRST_DOWN` pass-propensity mechanism one physical layer deeper before any further predictive source search.

Parent work established:

`TEAM PASS OPPORTUNITY -> PASS-OPPORTUNITY RATE -> WITHIN-STATE PASS PROPENSITY -> FIRST-DOWN PLAY SELECTION`

A clean 2023 D1 predictive test then rejected simple first-down pass-vs-run efficiency economics as an explanation.

The remaining question is:

> Is the apparent first-down pass-choice miss actually driven by **where on the field first downs occur**, or does the miss remain primarily **within the same field-position zone**?

This is a deterministic postgame mechanism diagnostic only. It cannot fit a predictive model or change production.

## Parent lineage

Immediate parent:

- branch: `research-qb-first-down-choice-economics-d1`
- result commit: `ac9e7706d02d11dbcb3cfe9cef02f2bc03d95058`
- run: `34545121616`
- artifact: `10178725142`
- disposition: `FIRST_DOWN_CHOICE_ECONOMICS_D1_FAIL_NO_CONFIRMATION`

First-down mechanism authority:

- state-shared attribution run: `34544457497`
- artifact: `10178499888`
- digest: `sha256:db401ff605820441605c985bf2262cb6114e1b06c7468605bc6b0a0708c84f4d`
- disposition: `FIRST_DOWN_SHARED_PRIMARY_DIAGNOSTIC`

Down/distance parent:

- run: `34543801321`
- artifact: `10178268917`
- disposition: `WITHIN_STATE_PASS_PROPENSITY_PRIMARY_DIAGNOSTIC`

Shared receiver authority:

- run: `34066549394`
- artifact: `9999119623`
- digest: `sha256:4e148f982d2f8db8a8e19cd2bbe2dbeb775dee296efc01aaf37193d98db2ddc6`

## Anti-reinvention boundary

Repository audit found `yardline_100` in scoring/red-zone utilities but not as a QB pass-rate research mechanism. This diagnostic is not a replay of:

- M42 generic pass-rate history;
- M56 defense/pass-funnel context;
- M64/M65 pace, possession, or score-state occupancy;
- M67 situational DBR/formation/no-huddle/shotgun;
- M68 opening-script/playcaller/leverage;
- M77-M79 personnel discontinuity;
- M81 tactical call structure;
- M83 defensive adaptive gameplan;
- M87/M88 pass-funnel regimes;
- first-down choice-economics D1.

No sportsbook or game-market variable may enter.

## Frozen cohort

- Exact 884 M89 QB-games from the immutable state-shared attribution parent.
- Exact season split: 444 in 2024, 440 in 2025.
- Exact shared receiver cohorts: 440 2025 WR-target rows and 884 pooled WR-reception rows.
- Historical nflverse regular-season PBP from 2023-2025 may be used to construct strictly-prior field-position reference distributions and target-game diagnostic labels.
- Target-game PBP is diagnostic only and never a pregame feature.

## Frozen first-down opportunity semantics

Eligible first-down opportunity plays:

- regular season;
- valid possession team and defense team;
- `down == 1`;
- `qb_dropback == 1` OR `rush_attempt == 1`;
- exclude `two_point_attempt == 1` when present;
- exclude `no_play == 1` when present;
- require finite `yardline_100` for field-position decomposition.

`PASS_ORIGIN = qb_dropback == 1`, including sacks and scrambles under corrected M89 semantics.

`DESIGNED_RUN = rush_attempt == 1 AND qb_dropback != 1`; kneels are excluded when available.

## Frozen field-position zones

Every decomposable first-down play is assigned to exactly one of four mutually exclusive zones using `yardline_100`:

1. `BACKED_UP` — `yardline_100 >= 80`;
2. `OWN_OPEN_FIELD` — `50 < yardline_100 < 80`;
3. `PLUS_TERRITORY` — `20 < yardline_100 <= 50`;
4. `RED_ZONE` — `yardline_100 <= 20`.

The thresholds above may not change after results are visible.

## Frozen strictly-prior field-position reference

For each target `(season, week, team, opponent)` and each zone `z`, construct:

- offense first-down zone occupancy;
- offense first-down DBR within zone;
- opponent-defense allowed zone occupancy;
- opponent-defense allowed DBR within zone.

History window: last 8 completed regular-season games strictly before target week.

League prior: all eligible first-down plays strictly before target week.

Shrinkage: 4 league-equivalent games using the same game-count shrinkage form as the parent down/distance reference.

For each zone/quantity:

- shrink offense history toward strictly-prior league value;
- shrink opponent-defense history toward strictly-prior league value;
- reference value = equal-weight mean of the two finite shrunk values; if one is unavailable use the finite one; if neither is available use league prior.

Normalize the four reference zone occupancies to sum exactly to 1.0.

Clip reference within-zone DBR to `[0.05, 0.95]` only as a mechanical probability bound.

Define:

`R_zone = sum_z(P_z * Q_z)`

where `P_z` is reference zone occupancy and `Q_z` is reference within-zone DBR.

## Frozen reconciliation to parent first-down contribution

From the immutable parent row:

- `P_D1_parent` = parent reference occupancy of D1;
- `A_D1_parent` = target realized occupancy of D1;
- `Q_D1_parent` = parent reference D1 DBR;
- `B_D1_parent` = target realized D1 DBR;
- `C_D1_parent` = parent additive D1 contribution to `WITHIN_STATE_RATE`.

Define parent weight:

`W = 0.5 * (P_D1_parent + A_D1_parent)`

The field-position target values are:

- `A_z` = target-game first-down occupancy of zone z;
- `B_z` = target-game first-down DBR within zone z.

For a target zone with `A_z == 0`, set `B_z = Q_z` for decomposition bookkeeping.

Target first-down aggregate:

`B_field = sum_z(A_z * B_z)`

must reconcile `B_D1_parent` within `1e-10`.

The three frozen components are:

### 1. `ZONE_REFERENCE_LEVEL`

`L = R_zone - Q_D1_parent`

This captures any difference between the richer strictly-prior field-position reference and the parent D1 reference level.

### 2. `FIELD_POSITION_OCCUPANCY`

Exact two-factor Shapley occupancy term:

`O = 0.5 * [sum((A_z-P_z)*Q_z) + sum((A_z-P_z)*B_z)]`

### 3. `WITHIN_ZONE_PASS_PROPENSITY`

Exact two-factor Shapley rate term:

`R = 0.5 * [sum(P_z*(B_z-Q_z)) + sum(A_z*(B_z-Q_z))]`

They must satisfy:

`L + O + R = B_D1_parent - Q_D1_parent`

within `1e-10`.

For direct reconciliation to the parent additive first-down contribution, define:

- `C_LEVEL = W * L`
- `C_FIELD_POSITION_OCCUPANCY = W * O`
- `C_WITHIN_ZONE_PASS_PROPENSITY = W * R`

and require:

`C_LEVEL + C_FIELD_POSITION_OCCUPANCY + C_WITHIN_ZONE_PASS_PROPENSITY = C_D1_parent`

within `1e-10` for every row.

## Frozen outputs

For 2024, 2025, and pooled 2024-2025 report for each of the three parent-scaled components:

- N;
- mean contribution;
- mean absolute contribution;
- share of summed mean-absolute component mass;
- sign agreement with `C_D1_parent`;
- dominant component rate by row;
- p50 / p75 / p90 absolute contribution.

Repeat in two parent tail cohorts:

- absolute fixed-0.57 pass-rate miss >= `0.08`;
- absolute fixed-0.57 pass-rate miss >= `0.12`.

For each of the four zones also report by season and pooled:

- actual zone occupancy;
- reference zone occupancy;
- actual within-zone DBR;
- reference within-zone DBR;
- mean occupancy delta;
- mean within-zone DBR delta;
- mean and mean-absolute contribution to the unscaled occupancy/rate terms.

## Shared receiver attribution

Using the immutable receiver cohorts, report Pearson, Spearman, and same-sign rate between each parent-scaled component and:

- 2025 WR target-mass residual;
- pooled 2024-2025 WR reception-mass residual;
- 2024 WR reception-mass residual;
- 2025 WR reception-mass residual.

Also report `C_D1_parent` as the fixed reference signal.

No receiver outcome is a predictor.

## Frozen integrity gates

Scientific interpretation stops unless all pass:

1. exact 884 parent rows;
2. exact 444/440 season split;
3. exact 440/884 shared receiver cohorts;
4. no duplicate canonical keys;
5. decomposable first-down `yardline_100` coverage >= `99%` pooled and >= `98.5%` in each target season;
6. four zones mutually exclusive and exhaustive among decomposable first-down plays;
7. reference zone occupancy sums to 1 within `1e-10` every row;
8. actual zone occupancy sums to 1 within `1e-10` every row;
9. target field-position aggregate DBR reconciles parent `B_D1` within `1e-10` every row;
10. `L+O+R` identity max error <= `1e-10`;
11. parent-scaled three-component sum reconciles `C_D1_parent` within `1e-10` every row;
12. every reference input is strictly prior to the target week;
13. zero sportsbook/game-market inputs;
14. zero model fitting;
15. zero production changes;
16. target-game PBP used only after reference values are fixed and only for diagnostic labels;
17. shared receiver joins preserve exact cohort size and uniqueness.

## Frozen routing rule

Exactly one disposition:

- `FIRST_DOWN_FIELD_POSITION_OCCUPANCY_PRIMARY_DIAGNOSTIC`
- `FIRST_DOWN_WITHIN_FIELD_POSITION_PROPENSITY_PRIMARY_DIAGNOSTIC`
- `FIRST_DOWN_ZONE_REFERENCE_LEVEL_PRIMARY_DIAGNOSTIC`
- `MIXED_FIRST_DOWN_FIELD_POSITION_MECHANISM`
- `MECHANICAL_OR_INTEGRITY_FAIL_NO_SCIENCE`

A component is `PRIMARY` only if all are true:

1. it has the largest pooled mean-absolute parent-scaled contribution;
2. it is largest in both 2024 and 2025, or largest in one and within 10% of the largest in the other;
3. pooled mean-absolute contribution is at least 20% larger than the second-largest component;
4. 2025 WR-target absolute Spearman >= `0.25`;
5. pooled WR-reception absolute Spearman >= `0.15`.

If exactly one component clears all five, return its primary disposition. Otherwise return `MIXED_FIRST_DOWN_FIELD_POSITION_MECHANISM`.

No gate may change after results.

## Stopping rule

- No predictive model in V1.
- No alternate field-position thresholds after results.
- If field-position occupancy is primary, subsequent work may target prediction of first-down field-position distribution only.
- If within-zone propensity is primary, do not retest field position as a predictor; the evidence would indicate play-selection uncertainty persists even after holding field position constant.
- If zone-reference level is primary, audit the reference construction before any correction.
- If mixed, do not fit a generic first-down residual model.
- Sportsbook remains downstream only.
