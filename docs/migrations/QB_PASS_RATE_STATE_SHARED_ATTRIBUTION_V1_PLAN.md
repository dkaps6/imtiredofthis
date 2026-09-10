# QB Pass-Rate State Shared Attribution V1 — Frozen Plan

## Purpose

Route the newly established `WITHIN_STATE_PASS_PROPENSITY_PRIMARY_DIAGNOSTIC` one level deeper before any new predictive/source family is opened.

The parent down/distance decomposition proved that the remaining corrected M89 team pass-opportunity-rate error is driven primarily by **within-state pass propensity**, not down/distance occupancy or league-level centering. The same within-state mechanism carries strong shared receiver error.

The unanswered question is:

> Is the shared QB/receiver within-state pass-propensity miss concentrated on **first down**, **second down**, or **late downs (third/fourth)**, or is it genuinely distributed across states?

This migration is deterministic diagnostic attribution only. It may not fit a model, change production, inspect sportsbook inputs, or authorize a correction.

## Parent lineage

- Parent branch: `research-qb-pass-rate-down-distance-decomp-v1`
- Parent result commit: `3469e9b9bca9c847226f248a751227cea03e916b`
- Parent frozen plan: `5ac5cfca31096dfc53745948846691a27f5fae89`
- Parent run: `34543801321`
- Parent job: `103091966418`
- Parent artifact: `10178268917`
- Parent artifact name: `qb-pass-rate-down-distance-decomp-v1`
- Parent digest: `sha256:8772c9a1203027162b27eee2cf4d9e1fa32acc69af2337163d801e1ba553c8da`
- Parent disposition: `WITHIN_STATE_PASS_PROPENSITY_PRIMARY_DIAGNOSTIC`

Shared receiver lineage:

- Run: `34066549394`
- Artifact: `9999119623`
- Digest: `sha256:4e148f982d2f8db8a8e19cd2bbe2dbeb775dee296efc01aaf37193d98db2ddc6`
- 2025 WR-target cohort: 440 exact QB-games
- 2024-2025 WR-reception cohort: 884 exact QB-games

## Anti-reinvention boundary

This is not a new pass-rate predictor and does not reopen any prior family.

Do not use:

- M42 generic team pass-rate history;
- M56 static defensive/pass-funnel context;
- M64/M65 possession, score-state, pace, or state-occupancy models;
- M67 team situational DBR / formation / no-huddle / shotgun;
- M68 opening-script / playcaller / leverage;
- M77-M79 generic personnel/inactive corrections;
- M81 FTN tactical call structure;
- M83 defensive adaptive gameplan;
- M87/M88 pass-funnel threshold regimes;
- the failed 0.59 anchor as a production baseline;
- sportsbook/game-market fields.

The only inputs are immutable parent diagnostic rows and immutable shared receiver residual rows.

## Frozen cohort

- Exact 884 parent M89 QB-games.
- Exact canonical keys: `season, week, team, player_clean_key`.
- Exact shared receiver cohorts from Run `34066549394`.
- No new PBP download is required; target-game diagnostic state values already exist in the immutable parent artifact.

## Frozen per-state contribution

For each parent state `s`, retain the parent definitions:

- `P_s` = strictly-prior reference occupancy;
- `Q_s` = strictly-prior reference within-state DBR;
- `A_s` = target-game realized occupancy;
- `B_s` = target-game realized within-state DBR, with the parent zero-occupancy bookkeeping rule.

The exact additive state contribution to the parent `WITHIN_STATE_RATE` term is:

`C_s = 0.5 * (P_s + A_s) * (B_s - Q_s)`

for the eight frozen states:

1. `D1`
2. `D2_SHORT`
3. `D2_MEDIUM`
4. `D2_LONG`
5. `D3_SHORT`
6. `D3_MEDIUM`
7. `D3_LONG`
8. `D4`

The identity must hold for every row:

`sum_s(C_s) = parent WITHIN_STATE_RATE`

within `1e-10`.

No state threshold may be changed.

## Frozen state groups

To avoid post-result cherry-picking among eight states, routing occurs on three football-defined groups frozen now:

### `FIRST_DOWN`

`G_FIRST = C_D1`

### `SECOND_DOWN`

`G_SECOND = C_D2_SHORT + C_D2_MEDIUM + C_D2_LONG`

### `LATE_DOWN`

`G_LATE = C_D3_SHORT + C_D3_MEDIUM + C_D3_LONG + C_D4`

The identity must hold for every row:

`G_FIRST + G_SECOND + G_LATE = parent WITHIN_STATE_RATE`

within `1e-10`.

No alternative state grouping is advancement-eligible in V1.

## Frozen outputs

For each of the eight individual states and each of the three routing groups, report for 2024, 2025, and pooled 2024-2025:

- N;
- mean contribution;
- mean absolute contribution;
- p50 / p75 / p90 absolute contribution;
- sign agreement with parent `WITHIN_STATE_RATE`;
- sign agreement with full fixed-0.57 pass-rate residual;
- dominant absolute group rate by row for the three groups;
- share of summed mean-absolute group mass for the three groups.

Repeat group metrics in parent tail cohorts:

- absolute fixed-0.57 rate miss >= `0.08`;
- absolute fixed-0.57 rate miss >= `0.12`.

## Shared receiver attribution

Join the immutable shared receiver cohorts and report Pearson, Spearman, and same-sign rate between every state/group contribution and:

1. 2025 WR target-mass residual (exact n=440);
2. pooled 2024-2025 WR reception-mass residual (exact n=884);
3. 2024 WR reception-mass residual;
4. 2025 WR reception-mass residual.

Also report the parent total `WITHIN_STATE_RATE` correlation as the fixed reference.

### Frozen leave-one-group-out diagnostic

For each group `g`, define:

`WITHOUT_g = parent WITHIN_STATE_RATE - G_g`

For the 2025 WR-target and pooled WR-reception views, report:

`Spearman_drop_g = abs(Spearman(parent WITHIN_STATE_RATE, receiver residual)) - abs(Spearman(WITHOUT_g, receiver residual))`

This is descriptive attribution only. No regression, partial-correlation fit, feature selection, or coefficient optimization is allowed.

## Integrity gates

Scientific interpretation stops unless all pass:

1. exact 884 parent rows;
2. exact 444/440 season split;
3. exact 440/884 shared receiver cohorts;
4. no duplicate canonical keys;
5. all eight state contributions finite;
6. per-state sum reconciles parent `WITHIN_STATE_RATE` within `1e-10` every row;
7. three-group sum reconciles parent `WITHIN_STATE_RATE` within `1e-10` every row;
8. shared receiver joins preserve exact cohort size and uniqueness;
9. zero sportsbook inputs;
10. zero model fitting;
11. zero new target-game PBP loading;
12. zero production changes.

## Frozen routing rule

V1 returns exactly one of:

- `FIRST_DOWN_SHARED_PRIMARY_DIAGNOSTIC`
- `SECOND_DOWN_SHARED_PRIMARY_DIAGNOSTIC`
- `LATE_DOWN_SHARED_PRIMARY_DIAGNOSTIC`
- `DISTRIBUTED_WITHIN_STATE_SHARED_MECHANISM`
- `MECHANICAL_OR_INTEGRITY_FAIL_NO_SCIENCE`

A routing group is `SHARED_PRIMARY` only if all are true:

1. it has the largest pooled mean-absolute group contribution;
2. it is largest in both 2024 and 2025, OR largest in one season and within 10% of the largest in the other;
3. its pooled mean-absolute contribution is at least 20% larger than the second-largest group;
4. its 2025 WR-target absolute Spearman is at least `0.30`;
5. its pooled WR-reception absolute Spearman is at least `0.20`;
6. removing that group reduces absolute 2025 WR-target Spearman by at least `0.05` **or** reduces pooled WR-reception absolute Spearman by at least `0.05`.

If exactly one group clears all six, route to that group's primary disposition. Otherwise return `DISTRIBUTED_WITHIN_STATE_SHARED_MECHANISM`.

No threshold may change after results are visible.

## Stopping rule / next work

- No predictive correction is allowed in this migration.
- No production change is allowed.
- If `FIRST_DOWN` is primary, subsequent source research must target genuinely new pregame first-down play-selection intent/incentive information and may not recycle M67/M68.
- If `SECOND_DOWN` is primary, subsequent research must target second-down choice conditional on distance and may not merely repackage prior generic DBR history.
- If `LATE_DOWN` is primary, audit whether conversion/pressure/late-down tactical information is genuinely new versus M56/M67/M81 before any predictive work.
- If `DISTRIBUTED`, do not fit eight state-specific residual models; the next task is to identify a common latent week-specific play-selection driver or accept a larger irreducible game-plan component.
