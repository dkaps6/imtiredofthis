# QB Pass-Rate Down/Distance Decomposition V1 — Frozen Plan

## Purpose

Diagnose the remaining game-specific `TEAM_PASS_OPPORTUNITY_RATE` error under corrected M89 semantics without fitting a predictive model and without reopening closed generic pass-rate research.

The parent work established:

- team pass opportunity is the primary remaining QB attempt-error mechanism;
- within team pass opportunity, pass-opportunity/dropback **rate** is more important than total offensive play volume;
- the production rate is fixed at `0.57`;
- a corrected-semantics constant-anchor screen selected `0.59` on 2024 and materially reduced bias/opportunity/attempt error, but failed the preregistered minimum pass-rate MAE gain and therefore did not advance.

The new question is narrower:

> When a target game's corrected dropback rate departs from the league-centered pregame expectation, is the deviation driven primarily by **down-and-distance occupancy** (how often the offense reaches pass-heavy states) or by **within-state pass propensity** (how often it drops back once in those states)?

This is postgame mechanism attribution only. It cannot promote a model.

## Parent lineage

- Parent branch: `research-qb-pass-rate-anchor-semantic-recalibration-a1`
- Parent result commit: `629fddb2137469c7e53118ad43b8a411131e7669`
- Parent run: `34539132735`
- Parent job: `103077537276`
- Parent artifact: `10176570118`
- Parent digest: `sha256:cacf810f653d2f4c3bcb24a19359c03ab9e85ced2775b2a2c4fdb20e6a693ffc`
- Parent disposition: `QB_PASS_RATE_ANCHOR_SEMANTIC_A1_FAIL_NO_CONFIRMATION`

Other required lineage:

- M89 source run: `33331073376`
- M89 artifact: `9737913528`
- M89 corrected cohort: 884 QB-games, 2024-2025.
- Opportunity-chain diagnostic run: `34523313743`, artifact `10170531084`, disposition `TEAM_PASS_OPPORTUNITY_PRIMARY_DIAGNOSTIC`.
- Play/rate decomposition run: `34535405829`, artifact `10175200512`; pass-opportunity rate was the primary submechanism versus total plays.

## Anti-reinvention boundary

This is not M65, M67, or M73 again.

- M65 decomposed **score-state** occupancy only: neutral/trailing/leading shares and rates.
- M67 tested historical DBR conditional on situations such as third-and-long, early-down neutral, opening script, shotgun, no-huddle and playcaller context.
- M73 tracked first-down rate, third-down conversion, early-down success, sacks and turnovers as survival/mechanism diagnostics.

None of those studies decomposed corrected M89 target-game dropback-rate deviation into **down/distance occupancy versus within-down/distance DBR**.

No M42 historical pass-rate weighting, M67/M68 opening/playcaller feature, PROE, score-state model, sportsbook signal, or model-zoo candidate may be introduced here.

## Frozen cohort

- Exact 884 M89 corrected QB-games from 2024-2025.
- Exact canonical keys: `season, week, team, player_clean_key`.
- Raw nflverse regular-season PBP may be used only to construct strictly-prior reference distributions and target-game diagnostic state labels.
- Target-game PBP is never a pregame feature.

## Frozen opportunity-play semantics

An eligible offensive opportunity play is a regular-season play with:

- possession team matching the target team;
- `qb_dropback == 1` OR `rush_attempt == 1`;
- `two_point_attempt != 1`;
- `no_play != 1`.

A dropback is `qb_dropback == 1`. This preserves sacks and QB scrambles as pass-origin opportunities under corrected M89 semantics.

Rows with missing/invalid `down` are excluded from the down/distance state decomposition but must be reported as coverage loss. The study stops if decomposable opportunity-play coverage is below 98% pooled or below 97% in either target season.

## Frozen down/distance states

Every decomposable opportunity play is assigned to exactly one of eight mutually exclusive states:

1. `D1` — first down, any distance;
2. `D2_SHORT` — second down, `ydstogo <= 3`;
3. `D2_MEDIUM` — second down, `4 <= ydstogo <= 7`;
4. `D2_LONG` — second down, `ydstogo >= 8`;
5. `D3_SHORT` — third down, `ydstogo <= 3`;
6. `D3_MEDIUM` — third down, `4 <= ydstogo <= 6`;
7. `D3_LONG` — third down, `ydstogo >= 7`;
8. `D4` — fourth down, any distance.

No threshold or state definition may change after results are visible.

## Strict-prior reference distribution

For each target `(season, week, team, opponent)` and each state, construct two reference quantities using only games strictly before the target week:

- offense state occupancy / within-state DBR;
- opponent-defense allowed state occupancy / within-state DBR.

History window: last 8 eligible team-games.

Shrinkage: 4 league-equivalent games toward the strictly-prior league mean for the same state/quantity.

Pregame reference for each state = equal-weight mean of the shrunk offense and opponent-defense values when both are finite; use the single finite value when only one exists; otherwise use the strictly-prior league value.

Normalize the eight reference occupancy shares to sum exactly to 1.0. Clip reference within-state DBR to `[0.05, 0.95]` only as a mechanical probability bound.

The reference aggregate DBR is:

`R = sum_s(reference_occupancy_s * reference_dbr_s)`.

This reference is diagnostic only and is not a production candidate.

## Exact two-factor Shapley decomposition

For each target game, define:

- `P_s` = strictly-prior reference occupancy of state `s`;
- `Q_s` = strictly-prior reference DBR in state `s`;
- `A_s` = target-game realized occupancy of state `s`;
- `B_s` = target-game realized DBR in state `s`.

For a state with `A_s == 0`, set `B_s = Q_s` for decomposition bookkeeping; this leaves the actual aggregate unchanged and assigns no unsupported within-state rate effect to an unvisited state.

Reference rate:

`R = sum(P_s * Q_s)`

Actual decomposed rate:

`A = sum(A_s * B_s)`

Two-factor Shapley contributions:

`OCCUPANCY = 0.5 * [sum((A_s-P_s)*Q_s) + sum((A_s-P_s)*B_s)]`

`WITHIN_STATE_RATE = 0.5 * [sum(P_s*(B_s-Q_s)) + sum(A_s*(B_s-Q_s))]`

They must satisfy:

`OCCUPANCY + WITHIN_STATE_RATE = A - R`

within `1e-10` for every row.

Also retain the separate level term:

`LEVEL_VS_057 = R - 0.57`

so that:

`LEVEL_VS_057 + OCCUPANCY + WITHIN_STATE_RATE = A - 0.57`

within `1e-10` for every row.

No alternative ordering is permitted.

## Frozen outputs

Report for 2024, 2025 and pooled 2024-2025:

- actual aggregate corrected DBR;
- reference aggregate DBR;
- fixed-0.57 residual;
- mean and mean-absolute `LEVEL_VS_057`, `OCCUPANCY`, and `WITHIN_STATE_RATE` contributions;
- share of total mean-absolute three-part contribution mass;
- sign agreement with `actual_rate - 0.57`;
- dominant component rate by row;
- p50/p75/p90 absolute contribution;
- same metrics in absolute fixed-0.57 rate misses >= 0.08 and >= 0.12;
- season stability of component ranking.

For the eight states, also report:

- pooled and season-specific actual occupancy;
- reference occupancy;
- actual within-state DBR;
- reference within-state DBR;
- mean occupancy delta;
- mean within-state DBR delta;
- contribution to aggregate occupancy/rate error.

## Shared receiver attribution

Using the immutable shared-pass-volume cohorts from Run `34066549394` / Artifact `9999119623`, join the exact target keys and correlate, for 2025 WR target-mass residual and 2024-2025 WR reception-mass residual:

- `actual_rate - 0.57`;
- `LEVEL_VS_057`;
- `OCCUPANCY`;
- `WITHIN_STATE_RATE`.

Report Pearson, Spearman and same-sign rate. This is diagnostic attribution only. Receiver target/reception outcomes are never predictors.

## Integrity gates

Scientific interpretation stops unless all pass:

1. exact 884 M89 target rows;
2. exact one target team-game per M89 row after PBP alignment;
3. decomposable opportunity-play coverage >=98% pooled and >=97% in each season;
4. eight states mutually exclusive and exhaustive among decomposable plays;
5. reference occupancy sums to 1 within `1e-10` for every row;
6. actual occupancy sums to 1 within `1e-10` for every row;
7. target aggregate `sum(A_s*B_s)` reconciles target-game PBP `dropbacks/opportunity_plays` within `1e-10`;
8. two-factor Shapley identity max error <= `1e-10`;
9. fixed-0.57 three-part identity max error <= `1e-10`;
10. all reference inputs are strictly prior to target `(season, week)`;
11. zero sportsbook inputs;
12. zero model fitting;
13. zero production changes;
14. target-game PBP used only after reference values are fixed and only for diagnostic labels;
15. shared receiver cohort keys align without duplication.

## Frozen routing rule

This diagnostic may return exactly one of:

- `DOWN_DISTANCE_OCCUPANCY_PRIMARY_DIAGNOSTIC`
- `WITHIN_STATE_PASS_PROPENSITY_PRIMARY_DIAGNOSTIC`
- `LEVEL_CENTERING_PRIMARY_DIAGNOSTIC`
- `MIXED_DOWN_DISTANCE_MECHANISM_NO_SINGLE_PRIMARY`
- `MECHANICAL_OR_INTEGRITY_FAIL_NO_SCIENCE`

A component is `PRIMARY` only if all are true:

1. it has the largest pooled mean-absolute contribution among `LEVEL_VS_057`, `OCCUPANCY`, `WITHIN_STATE_RATE`;
2. it is largest in both 2024 and 2025, OR largest in one season and within 10% of the largest in the other;
3. its mean-absolute contribution is at least 20% larger than the second-largest pooled component;
4. in the 2025 WR-target attribution, absolute Spearman is at least `0.25`.

If no component clears all four, disposition is `MIXED_DOWN_DISTANCE_MECHANISM_NO_SINGLE_PRIMARY`.

## Stopping rule

- No predictive model in this migration.
- No alternative state thresholds after results.
- No feature search.
- No 2026 target outcomes.
- No sportsbook/game-market inputs.
- If occupancy is primary, next work may audit strictly-prior predictors of down/distance state occupancy only.
- If within-state rate is primary, next work may audit genuinely new pregame play-selection information only; do not recycle M42/M67/M68 families.
- If level centering is primary, do not promote 0.59 from A1; a new confirmation design would still be required.
- If mixed, do not fit a generic residual model; identify the unresolved physical layer first.
