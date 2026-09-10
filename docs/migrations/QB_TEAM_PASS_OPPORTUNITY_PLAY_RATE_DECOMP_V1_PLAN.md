# QB Team Pass Opportunity Play/Rate Decomposition V1 — Frozen Diagnostic Plan

## Purpose

Decompose the already-established M89-corrected `TEAM_PASS_OPPORTUNITY` error into its two physical internal factors:

1. `TOTAL_OFFENSIVE_PLAYS`
2. `PASS_OPPORTUNITY_RATE`

This is a no-fit diagnostic. It changes no production logic and uses no sportsbook data.

The parent QB opportunity-chain diagnostic established `TEAM_PASS_OPPORTUNITY` as the dominant remaining attempt-error mechanism and the primary carrier of the shared QB/receiver opportunity miss. D1 and D2 then rejected schedule/rest, penalty-created drive extensions and fourth-down aggression as sufficiently predictive corrections.

Before any additional source hunt, this diagnostic determines which internal factor of team pass opportunity actually owns the remaining error under the corrected M89 semantics.

## Canonical lineage

- D2 result commit: `0b965ec9c9373bf37401cc119ffecdac6e4f8aee`
- D2 run: `34534830257`
- D2 artifact: `10174978673`
- Opportunity-chain run: `34523313743`
- Opportunity-chain artifact: `10170531084`
- Opportunity-chain digest: `sha256:75cd32198caf7d9cbf193d5769e5762acce53352f2b2626cd92a9b117e75dafd`
- M89 source run: `33331073376`
- M89 artifact: `9737913528`
- Shared QB/WR run: `34066549394`
- Shared QB/WR artifact: `9999119623`

## Why this is not M73 again

M73 decomposed opportunity mechanisms on the pre-M89 frontier. M89 subsequently corrected official attempt semantics and explicitly changed the pass-opportunity accounting denominator to:

`official pass attempts + PBP sacks + PBP QB scrambles`.

This diagnostic is performed on the exact 884-row M89/M90 aligned cohort and uses the exact promoted M89 component trace. No pre-M89 M73 ranking is imported as a result.

## Frozen predicted factors

From the exact M89 pass-yards component row aligned to each common QB-game:

- `P_PLAYS = mc_projected_plays`
- `P_RATE = mc_dropback_rate`
- `P_D = mc_team_expected_dropbacks`

Required identity:

`P_PLAYS * P_RATE = P_D`

within `1e-6` opportunities on every row.

The diagnostic must also report the empirical distribution of `P_RATE`, including min/max/standard deviation and number of distinct rounded values, so the current model's pass-rate adaptivity is explicit.

## Frozen actual factors

Use nflverse/nflfastR regular-season PBP for 2024 and 2025 only as a postgame diagnostic label source.

Reproduce the M89 offensive-play universe exactly:

`ACTUAL_OFFENSIVE_PLAY = qb_dropback == 1 OR rush_attempt == 1`

For each team-game:

- `A_PLAYS = count(ACTUAL_OFFENSIVE_PLAY)`
- `A_D = corrected M89 pass_opportunities` from the immutable opportunity-chain casebook
- `A_RATE = A_D / A_PLAYS`

Required identity:

`A_PLAYS * A_RATE = A_D`

within `1e-6` opportunities.

Target-game PBP and actual outcomes are diagnostic labels only and can never become pregame features in this migration.

## Exact two-factor Shapley decomposition

For `f(PLAYS, RATE) = PLAYS * RATE`, decompose:

`A_D - P_D`

into exactly two order-neutral Shapley contributions:

### TOTAL_OFFENSIVE_PLAYS

`PLAY_CONTRIB = (A_PLAYS - P_PLAYS) * (P_RATE + A_RATE) / 2`

### PASS_OPPORTUNITY_RATE

`RATE_CONTRIB = (A_RATE - P_RATE) * (P_PLAYS + A_PLAYS) / 2`

Required identity:

`PLAY_CONTRIB + RATE_CONTRIB = A_D - P_D`

within `1e-6` opportunities per row.

No alternate ordering or decomposition is allowed after results are visible.

## Frozen oracle counterfactuals

For diagnostic recoverability only:

- `PERFECT_PLAYS_D = A_PLAYS * P_RATE`
- `PERFECT_RATE_D = P_PLAYS * A_RATE`

Compare each with `A_D` using MAE/RMSE/bias/correlation. These are postgame oracles, not deployable candidates.

## Cohorts

Primary:
- exact 884 M89/M90 QB-games, 2024-2025.
- report 2024, 2025 and pooled.

Shared receiver attribution:
- exact 440-row 2025 WR target-mass cohort;
- exact 884-row 2024-2025 WR reception-mass replication cohort.

Also report separately:
- all games;
- parent `ATTEMPTS_DOMINANT` games;
- absolute team-pass-opportunity miss >= 8;
- absolute team-pass-opportunity miss >= 10;
- D underprojection and D overprojection.

## Frozen metrics

For each component/cohort/year:

- mean contribution;
- mean absolute contribution;
- share of total absolute two-factor mass;
- sign agreement with total D residual;
- dominant-component row rate;
- p50/p75/p90 absolute contribution.

For direct factor prediction error, report:

### Offensive plays
- baseline play MAE/RMSE/bias/correlation (`P_PLAYS` vs `A_PLAYS`).

### Pass-opportunity rate
- baseline rate MAE/RMSE/bias/correlation (`P_RATE` vs `A_RATE`).

### Team pass opportunity
- baseline vs PERFECT_PLAYS_D vs PERFECT_RATE_D MAE/RMSE/bias/correlation.

## Shared receiver attribution

On the exact 2025 WR target-mass cohort, correlate each D-unit Shapley contribution with `wr_target_mass_residual`:

- Pearson;
- Spearman;
- same-sign rate;
- signed Q4-minus-Q1 WR residual gap.

Repeat on the exact 2024-2025 WR reception-mass cohort pooled and by season.

The previously established total `TEAM_PASS_OPPORTUNITY` residual relationship remains a reference baseline; this diagnostic does not redefine it.

## Integrity gates

Scientific interpretation stops unless all pass:

1. exactly 884 M89 common QB rows;
2. exactly 440 primary and 884 secondary shared receiver rows;
3. 2024 and 2025 nflverse PBP load successfully;
4. zero sportsbook inputs;
5. zero model fitting;
6. no production changes;
7. exact M89 predicted identity `P_PLAYS*P_RATE=P_D` max error <= `1e-6`;
8. exact actual identity `A_PLAYS*A_RATE=A_D` max error <= `1e-6`;
9. exact two-factor Shapley identity max error <= `1e-6`;
10. exact one-row-per team/week PBP play-count alignment for all 884 rows;
11. shared receiver source keys align without row loss;
12. target-game PBP/outcomes are used only as postgame labels;
13. no pre-M89 component ranking determines the result.

## Frozen routing rule

Only these scientific dispositions are allowed:

- `TOTAL_OFFENSIVE_PLAYS_PRIMARY_DIAGNOSTIC`
- `PASS_OPPORTUNITY_RATE_PRIMARY_DIAGNOSTIC`
- `MIXED_PLAY_RATE_NO_SINGLE_PRIMARY`
- `MECHANICAL_OR_INTEGRITY_FAIL_NO_SCIENCE`

A component is `PRIMARY` only if all are true:

1. largest pooled mean-absolute Shapley contribution;
2. largest in both 2024 and 2025, OR largest in one and within 10% of largest in the other;
3. absolute 2025 Spearman vs WR target-mass residual >= `0.30`;
4. absolute 2025 WR-target Spearman is at least `0.10` greater than the other component;
5. absolute pooled 2024-2025 Spearman vs WR reception-mass residual >= `0.25`.

If neither clears all five, disposition is `MIXED_PLAY_RATE_NO_SINGLE_PRIMARY`.

This diagnostic cannot promote a model. It only routes the next source/predictability audit.

## Stopping rule

Run this exact two-factor decomposition once. Do not fit a pass-rate model, do not modify the 0.57 rate, do not tune projected plays, and do not introduce a new source until this diagnostic is complete and recorded.
