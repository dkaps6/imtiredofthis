# QB First-Down Score-State Decomposition V1 — Frozen Plan

## Purpose

Decompose the surviving shared first-down `WITHIN_ZONE_PASS_PROPENSITY` mechanism one physical layer deeper before any further predictive source search.

Authoritative upstream path:

`TEAM PASS OPPORTUNITY -> PASS-OPPORTUNITY RATE -> WITHIN-STATE PASS PROPENSITY -> FIRST-DOWN PLAY SELECTION -> WITHIN-FIELD-POSITION PASS PROPENSITY`

The immediate parent established that field-position occupancy is not the explanation: pass-choice uncertainty remains dominant after holding first-down field-position zone constant.

This V1 asks:

> Is that remaining within-zone first-down pass-choice miss mainly caused by **realized score-state occupancy**, or does it persist as **within-score-state pass propensity** after both field position and score state are held constant?

This is a deterministic postgame mechanism diagnostic only. It cannot fit a predictive model or change production.

## Parent lineage

Immediate parent:
- branch: `research-qb-first-down-field-position-decomp-v1`
- result commit: `55d71b28d038d102c54b25a132c0cca79040b561`
- run: `34545401969`
- job: `103096807810`
- artifact ID: `10178825973`
- artifact digest: `sha256:734b1603b69f45a39f33e6d4770690bcc81b42f8a3b3dae33b74dce6c2cbeb37`
- disposition: `FIRST_DOWN_WITHIN_FIELD_POSITION_PROPENSITY_PRIMARY_DIAGNOSTIC`

Shared receiver authority remains run `34066549394`, artifact `9999119623`, digest `sha256:4e148f982d2f8db8a8e19cd2bbe2dbeb775dee296efc01aaf37193d98db2ddc6`.

## Anti-reinvention boundary

M64/M65 already tested richer possession/dropback and score-state occupancy architectures prospectively and did not promote. This diagnostic does **not** reopen those predictive families. It uses realized target-game score state only after strictly-prior references are fixed, solely to attribute the already-established post-M89 first-down within-zone error.

Do not fit score-state features. Do not change the ±7 / 8+ state boundaries. Do not use sportsbook/game-market variables.

## Frozen cohort

- Exact 884 parent QB-games: 444 in 2024, 440 in 2025.
- Exact 440 2025 WR-target and 884 pooled WR-reception shared cohorts.
- Parent first-down field-position casebook is immutable authority.
- Historical regular-season nflverse PBP from 2023-2025 may be used only to construct strictly-prior reference distributions and target-game diagnostic labels.

## Frozen first-down and field-position semantics

Use the immediate parent's exact eligible first-down semantics and exact four zones:

1. `BACKED_UP`: `yardline_100 >= 80`
2. `OWN_OPEN_FIELD`: `50 < yardline_100 < 80`
3. `PLUS_TERRITORY`: `20 < yardline_100 <= 50`
4. `RED_ZONE`: `yardline_100 <= 20`

`PASS_ORIGIN = qb_dropback == 1`.

## Frozen score-state semantics

Use the established repository convention based on pre-play `score_differential` from the possession team's perspective:

- `TRAILING_8_PLUS`: `score_differential <= -8`
- `NEUTRAL`: `abs(score_differential) <= 7`
- `LEADING_8_PLUS`: `score_differential >= 8`

Every decomposable first-down play with finite score differential belongs to exactly one state.

## Frozen strictly-prior reference

For every target `(season, week, team, opponent)`, field-position zone `z`, and score state `s`, construct:

- offense conditional score-state occupancy within zone `z`;
- offense DBR within `(z,s)`;
- opponent-defense allowed conditional score-state occupancy within zone `z`;
- opponent-defense allowed DBR within `(z,s)`.

History window: last 8 completed regular-season games strictly before the target week.

League prior: all eligible first-down plays strictly before target week.

Shrinkage: 4 league-equivalent games, matching the parent game-count shrinkage convention.

For each zone/state quantity:
- shrink offense toward strictly-prior league value;
- shrink opponent defense toward strictly-prior league value;
- reference = equal-weight mean of finite shrunk offense/defense values; if only one finite value exists, use it; if neither exists, use league prior.

Within each field-position zone, normalize the three reference score-state occupancies to sum exactly to 1.0.

Clip reference `(zone,state)` DBR to `[0.05, 0.95]` only as a mechanical probability bound.

## Frozen decomposition

From the immutable parent casebook, for each zone `z`:
- `P_z` = parent reference field-position occupancy
- `A_z` = target field-position occupancy
- `Q_z` = parent reference within-zone DBR
- `B_z` = target realized within-zone DBR

Define zone weight:

`W_z = 0.5 * (P_z + A_z)`

For each score state `s` within zone `z`:
- `P_zs` = reference conditional score-state occupancy in zone z
- `A_zs` = target conditional score-state occupancy in zone z
- `Q_zs` = reference DBR in `(z,s)`
- `B_zs` = target realized DBR in `(z,s)`

If `A_zs == 0`, set `B_zs = Q_zs` for decomposition bookkeeping.

Require target reconciliation:

`sum_s(A_zs * B_zs) = B_z`

for every zone with target zone occupancy > 0, within `1e-10`.

Define richer score-state reference within zone:

`R_z = sum_s(P_zs * Q_zs)`

Then define exact three-part decomposition of `B_z - Q_z`:

### 1. `SCORE_STATE_REFERENCE_LEVEL`

`L_z = R_z - Q_z`

### 2. `SCORE_STATE_OCCUPANCY`

`O_z = 0.5 * [sum_s((A_zs-P_zs)*Q_zs) + sum_s((A_zs-P_zs)*B_zs)]`

### 3. `WITHIN_SCORE_STATE_PASS_PROPENSITY`

`RPROP_z = 0.5 * [sum_s(P_zs*(B_zs-Q_zs)) + sum_s(A_zs*(B_zs-Q_zs))]`

Require:

`L_z + O_z + RPROP_z = B_z - Q_z`

within `1e-10` for every zone.

Aggregate back to the immediate parent's unscaled within-zone pass-propensity term:

- `LEVEL = sum_z(W_z * L_z)`
- `SCORE_OCCUPANCY = sum_z(W_z * O_z)`
- `WITHIN_SCORE_PROPENSITY = sum_z(W_z * RPROP_z)`

Require:

`LEVEL + SCORE_OCCUPANCY + WITHIN_SCORE_PROPENSITY = parent rate_unscaled`

within `1e-10` every row.

Finally multiply by the immediate parent's `parent_weight` to reconcile the exact parent-scaled component:

- `C_SCORE_STATE_REFERENCE_LEVEL = parent_weight * LEVEL`
- `C_SCORE_STATE_OCCUPANCY = parent_weight * SCORE_OCCUPANCY`
- `C_WITHIN_SCORE_STATE_PASS_PROPENSITY = parent_weight * WITHIN_SCORE_PROPENSITY`

Require their sum to equal parent `c_within_zone_pass_propensity` within `1e-10` every row.

## Frozen outputs

For 2024, 2025, and pooled, report for each of the three parent-scaled components:
- N
- mean
- mean absolute contribution
- share of summed mean-absolute mass
- sign agreement with parent `c_within_zone_pass_propensity`
- dominant component rate
- p50/p75/p90 absolute contribution

Repeat in parent fixed-0.57 pass-rate miss cohorts `abs >= 0.08` and `abs >= 0.12`.

For each field-position zone × score-state cell, report actual/reference occupancy, actual/reference DBR, mean deltas, and mean/mean-absolute contribution.

## Shared receiver attribution

Using immutable shared cohorts, report Pearson, Spearman, and same-sign relationship for each parent-scaled component versus:
- 2025 WR target-mass residual
- pooled WR reception-mass residual
- 2024 WR reception-mass residual
- 2025 WR reception-mass residual

Parent `c_within_zone_pass_propensity` remains the fixed reference signal.

No receiver outcome is a predictor.

## Frozen integrity gates

Scientific interpretation stops unless all pass:
1. exact 884 parent rows;
2. exact 444/440 split;
3. exact 440/884 shared receiver cohorts;
4. no duplicate canonical keys;
5. score differential coverage >=99% pooled and >=98.5% in each target season among decomposable first-down plays;
6. three score states mutually exclusive/exhaustive;
7. reference conditional score-state occupancy sums to 1 within `1e-10` for every zone/row;
8. actual conditional score-state occupancy sums to 1 within `1e-10` for every target-populated zone;
9. every target zone/state aggregate reconciles parent target zone DBR within `1e-10`;
10. every zone three-part identity max error <= `1e-10`;
11. aggregated unscaled sum reconciles parent `rate_unscaled` within `1e-10`;
12. aggregated parent-scaled sum reconciles parent `c_within_zone_pass_propensity` within `1e-10`;
13. all reference inputs strictly prior;
14. zero sportsbook/game-market inputs;
15. zero model fitting;
16. zero production changes;
17. target-game PBP diagnostic only;
18. shared joins exact and unique.

## Frozen routing rule

Exactly one disposition:
- `FIRST_DOWN_SCORE_STATE_OCCUPANCY_PRIMARY_DIAGNOSTIC`
- `FIRST_DOWN_WITHIN_SCORE_STATE_PROPENSITY_PRIMARY_DIAGNOSTIC`
- `FIRST_DOWN_SCORE_STATE_REFERENCE_LEVEL_PRIMARY_DIAGNOSTIC`
- `MIXED_FIRST_DOWN_SCORE_STATE_MECHANISM`
- `MECHANICAL_OR_INTEGRITY_FAIL_NO_SCIENCE`

A component is PRIMARY only if all are true:
1. largest pooled mean-absolute parent-scaled contribution;
2. largest in both seasons, or largest in one and within 10% of largest in the other;
3. pooled mean-absolute contribution at least 20% larger than second-largest;
4. 2025 WR-target absolute Spearman >=0.25;
5. pooled WR-reception absolute Spearman >=0.15.

If exactly one clears all five, route to it; otherwise route MIXED.

## Stopping rule

- No predictive model in V1.
- Do not retest generic M64/M65 score-state models.
- If score-state occupancy is primary, classify a meaningful portion of the remaining first-down miss as realized game-script uncertainty and only pursue genuinely new pregame game-state transition information.
- If within-score-state propensity is primary, treat the evidence as strong that first-down play-choice uncertainty persists after down, field position, and score state are held constant; the next step must be a source audit for genuinely new target-game intent information, not more transforms of historical PBP tendencies.
- If reference level is primary, audit reference construction before correction.
- If mixed, do not fit a generic first-down residual model.
- Sportsbook remains downstream only.
