# Player Target Share Trajectory Historical OOS-Fold Integration V1 — Frozen Plan

Date: 2026-10-08  
Status: **FROZEN BEFORE INTEGRATION SCORING — RESEARCH ONLY**

## Question

Does the already-frozen Target Share Trajectory V1 transformation improve
individual WR/TE opportunity allocation when applied on top of exact historical
out-of-sample specialist parents, without changing team/room target mass,
M38 WR1, receiving efficiency, or any sportsbook-facing logic?

This is not a new formula search. It reuses the exact prospective Week-5 rule:

`weight_i = baseline_i * exp(trajectory_delta_i)`

followed by exact protected-room renormalization.

## Immutable authorities

Trajectory state:
- run `37638269235`
- artifact `11490707699`
- name `player-target-share-trajectory-v1-37638269235`
- digest `sha256:2f032cb5e68f5e097e30ca56a2ebe30e505f90233bc60774e4318158fad3fafd`
- row source: `player_target_share_trajectory_rows.csv`
- only `season/week/team/player_clean_key/position_group/trajectory_delta/feature_max_week`
  may enter candidate construction;
- the artifact's outcome-derived `opportunity_error` is explicitly forbidden
  from candidate construction.

Recovered fold-safe parent authority:
- recovery run `37782189538`
- artifact `11552447166`
- name `wr-te-fold-authorities-recovered-v1`
- digest `sha256:94a4bee7011877b1d7a24708d5ba3e86ccc5d29b427bea650af574fc1f09fa24`
- status `WR_TE_FOLD_AUTHORITIES_RECOVERED_EXACT`
- no fresh provider rebuild;
- no sportsbook input;
- no paid OddsAPI.

Parent lineage:
- WR-R15: original OOS run `34238301577`, candidate fold rows only.
- TE-R5P: exact mechanical recovery of original run `34152797603`,
  source SHA `2194668c5af49ef15ee904c7d583cc61df0c58b4`, with published
  metrics reproduced exactly from pinned TE-R3 / TE-R4 parents.

## Frozen population

### Primary qualification population

**2024 only**, Weeks 5+:
- WR parent = `WR_R15_WR1_ANCHORED_PARTICIPATION`, train 2023 -> test 2024.
- TE parent = TE-R5P test-2024 OOS fold.
- all parent player rows are retained;
- if exact trajectory state is unavailable, `trajectory_delta = 0` and the row
  remains unchanged;
- no row is dropped because its realized workload was low or zero.

2024 is the only primary joint WR+TE season because it has exact authorized OOS
parents for both positions under the previously frozen historical specialist
lineage.

### Secondary replication

TE 2025 Weeks 5+ may be scored with the exact TE-R5P test-2025 fold.
It is **secondary only** because WR-R15's original contract forbids 2025
confirmation. It cannot rescue a failed 2024 joint result.

No 2023 integration score is authorized by V1.

## Frozen transformation

### WR

Baseline:
- use only WR-R15 candidate-variant parent rows;
- baseline opportunity = `pred_targets`;
- baseline receiving mean = `mc_rec_yards`;
- M38 WR1 = row with `wr_rank == 1`.

For each event/team:
- WR1 target opportunity is frozen exactly;
- only WR2+ rows are reweighted by `exp(trajectory_delta)`;
- normalize the WR2+ weights to preserve the exact baseline WR2+ target pool;
- exact total WR target pool therefore remains unchanged.

### TE

Baseline:
- opportunity = `candidate_targets_r5p`;
- receiving mean = `candidate_rec_yards_r5p`.

For each event/team:
- reweight all TE rows by `exp(trajectory_delta)`;
- normalize to preserve the exact baseline TE target pool.

### Efficiency translation

No efficiency is fit or changed.

For rows with positive baseline targets:
- baseline yards-per-target =
  baseline receiving-yards mean / baseline target mean;
- shadow receiving-yards mean =
  shadow target mean * unchanged baseline yards-per-target.

Rows with zero baseline target mean must remain zero receiving-yard mean and may
not be given new opportunity by an ad-hoc fallback.

## Chronology / source gates

Hard fail if:
- any `feature_max_week >= target week`;
- any trajectory row used outside exact season/team/player/position identity;
- duplicate parent or trajectory identities exist;
- a parent test season/train season contract is violated;
- an outcome-derived trajectory field enters candidate construction;
- any fresh historical provider rebuild is invoked;
- any sportsbook/player-prop field is used upstream;
- any 2026 outcome is read.

## Conservation gates

Hard fail above `1e-12`:
- WR1 absolute change;
- WR2+ target-pool change;
- total WR target-pool change;
- TE target-pool change.

No team/room opportunity mass is created or destroyed.

## Frozen evaluation

Primary player-output metrics on the 2024 joint population:
1. target-count MAE — baseline vs shadow;
2. receiving-yards MAE — baseline vs shadow.

Mechanism diagnostic:
- within-room target-share MAE on rooms with positive realized room targets.

Secondary:
- RMSE;
- signed bias;
- candidate-closer / baseline-closer / tie counts;
- weekly target-MAE deltas;
- 2025 TE-only replication.

The historical fold parents do not contain a frozen target-game whole-team target
denominator for every row. Therefore V1 does **not** manufacture an "actual team
target share" using a fresh provider rebuild. Target-count MAE is the primary
player-output metric and within-room target-share MAE is the allocation metric.
The prospective Week-5+ contract remains the authority for its exact team-share
grader.

## Frozen bootstrap

Primary uncertainty test:
- unit = player cluster, keyed `position|player_clean_key`;
- statistic = mean per-row improvement in absolute target error
  (`baseline_abs_error - shadow_abs_error`);
- reps = 5000;
- seed = `20261008`;
- report 95% percentile interval and
  `P(mean target-AE improvement > 0)`.

No alternative bootstrap or subgroup is selected after results.

## Disposition

`HISTORICAL_OOS_FOLD_TRAJECTORY_INTEGRATION_SUPPORTED` requires ALL:

1. 2024 pooled WR+TE target MAE improves;
2. 2024 WR target MAE improves;
3. 2024 TE target MAE improves;
4. 2024 pooled WR+TE receiving-yards MAE is non-worse;
5. 2024 WR receiving-yards MAE is non-worse;
6. 2024 TE receiving-yards MAE is non-worse;
7. player-cluster bootstrap
   `P(mean target-AE improvement > 0) >= 0.80`;
8. pooled within-room target-share MAE is non-worse;
9. every conservation and chronology gate passes.

Otherwise:
`HISTORICAL_OOS_FOLD_TRAJECTORY_INTEGRATION_NOT_CONFIRMED`.

2025 TE-only is disclosure/replication evidence and cannot change the primary
disposition.

## Governance

- zero fitted parameters;
- zero threshold search;
- zero formula/cap/window search;
- zero sportsbook inputs;
- zero paid OddsAPI;
- zero production mutation;
- no Week-5 2026 outcome grading;
- no relaxation of the four-prior-same-season-game trajectory rule;
- no WR-only or TE-only rescue after seeing results.

Even a supported historical disposition does not replace the immutable
prospective acceptance requirement:
- >=4 future locked weeks;
- >=400 scoreable WR/TE player-games;
- >=120 distinct identities;
- >=80 scoreable team-position rooms.

