# QB Opportunity Chain Decomposition V1 — Frozen Plan

## Purpose

Identify the physical source of the remaining M89/M90 QB **attempt-opportunity** error without reopening the closed generic QB mean-feature hunt and without changing production.

This is a diagnostic decomposition only. It follows the already-completed post-M89 `QB_INDIVIDUAL_MECHANISM_DECOMPOSITION`, which proved that remaining passing-yard error is materially split between attempt-volume and YPA/efficiency mechanisms, and `QB_R1_PLAYER_CONTEXT_MECHANISM_ROUTER`, which showed that recent player mechanism history plus the existing M89 context feature set does not reliably predict which mechanism will dominate the next game.

The new question is narrower and was not answered by those studies:

> Within the M89-corrected attempt term itself, how much error comes from team pass opportunity/dropbacks, pass-attempt conversion (sacks + QB scrambles), and primary-QB attempt share, and which of those subcomponents explains the already-established shared QB/WR opportunity miss?

No sportsbook, player-prop, or game-market field may be used.

## Canonical lineage

### QB mean / corrected semantics
- M89 scientific run: `33331073376`
- M89 artifact: `9737913528` (`m89-qb-data-integrity-casebook-synthesis`)
- M89 artifact digest: `sha256:9dd85efeb2ccf323e22b942af160f85622a1419473be985035709c6f4882888f`
- M89 corrected common cohort: 884 QB-games, 2024-2025.
- M90 confirmation remains authoritative for promotion; no M89/M90 coefficient or mean change is authorized here.

### Existing post-M89 mechanism decomposition
- Run: `34065369449`
- Tested SHA: `f8b2f6578560c66764d2102a0c765dfe67995015`
- Artifact: `9998765968`
- Disposition: `QB_INDIVIDUAL_MECHANISMS_MAPPED`
- Existing aggregate evidence: final synthesis MAE `55.060`; mean-absolute attempt contribution `50.517` yards; mean-absolute YPA contribution `43.057` yards.

### Failed mechanism router — do not repeat
- Run: `34072312396`
- Job: `101591795688`
- Tested SHA: `2a4d38d8b743f6b21b2ab95d112767adc3f697a0`
- Artifact: `10000858730`
- Disposition: `NO_ACTIONABLE_QB_PLAYER_CONTEXT_MECHANISM_ROUTER`
- The existing M89 context family and recent player mechanism history may not be repackaged as a new predictive candidate here.

### Shared QB/WR opportunity evidence
- Run: `34066549394`
- Job: `101576202002`
- Tested SHA: `ba8fce70caaf147ae8ac49003473a3aa6142fbae`
- Artifact: `9999119623`
- Artifact digest: `sha256:4e148f982d2f8db8a8e19cd2bbe2dbeb775dee296efc01aaf37193d98db2ddc6`
- 2025 QB-attempt residual vs WR target-mass residual: Pearson `0.6973048915`, Spearman `0.6706484209`, same-sign `0.7318181818`.
- 2024-2025 QB-attempt residual vs WR reception-mass residual: Pearson `0.5238926875`, Spearman `0.4994040246`, same-sign `0.7092760181`.

## Why this is not M73 again

M73 was a diagnostic on the pre-M89 canonical frontier. It decomposed large attempt misses using the then-current opportunity semantics. The repository later explicitly required attempt-dependent conclusions from the M86-M88 lineage to be reinterpreted under M89's corrected official-attempt definition.

This study uses the **post-M89 corrected 884-game trace and the exact promoted attempt construction**. It also adds the subsequently discovered QB/WR shared-opportunity evidence. No M73 mechanism ranking is carried forward as a result.

## Frozen cohort

Primary QB opportunity decomposition:
- Exact 884 M89 corrected QB-games from 2024-2025.
- Exact one-row-per QB/team/week alignment with `m89_corrected_qb_common_trace.csv` and M89 season `component_predictions.csv`.

Cross-position primary:
- Exact 440-row 2025 cohort from `qb_wr_shared_pass_volume_primary_2025.csv`.

Cross-position replication:
- Exact 884-row 2024-2025 cohort from `qb_wr_shared_pass_volume_secondary_2024_2025.csv`.

No row may be added or dropped after results are visible except for a documented pre-science mechanical/source-contract failure.

## Frozen opportunity semantics

For each aligned QB-game, define the pregame predicted factors from the M89 component trace:

- `P_D = mc_team_expected_dropbacks`
- `P_C = mc_pass_attempts_per_dropback`
- `P_S = mc_qb_pass_att_share`

Predicted QB attempts must reconcile:

`P_A = P_D * P_C * P_S = mc_expected_pass_attempts = M89 pred_attempts`

within `1e-6` attempts, subject only to documented floating-point precision.

Define realized factors using M89-corrected target-game observations strictly for postgame diagnostic attribution:

- `A_D = pass_opportunities = official_pass_attempts + PBP sacks + PBP QB scrambles`
- `A_C = official_pass_attempts / pass_opportunities`
- `A_S = actual_qb_attempt_share`

Realized QB attempts must reconcile:

`A_A = A_D * A_C * A_S = actual_attempts`

within `1e-6` attempts.

Target-game outcomes are diagnostic labels only and can never become pregame predictors in this study.

## Exact three-factor Shapley decomposition

For the multiplicative attempt function `f(D,C,S)=D*C*S`, decompose:

`A_A - P_A`

into exactly three Shapley contributions:

1. `TEAM_PASS_OPPORTUNITY` — change in team pass opportunities/dropbacks (`D`)
2. `ATTEMPT_CONVERSION` — change in attempts per pass opportunity (`C`), physically driven by sack/scramble conversion
3. `QB_SHARE` — change in the primary QB's share of official team pass attempts (`S`)

For each factor, average its marginal contribution across all 6 permutations of the three factors. The three contributions must sum to `actual_attempts - pred_attempts` within `1e-6` attempts for every row.

No one-at-a-time ordering may be selected after results are visible.

## Yard-equivalent reconciliation

The already-frozen post-M89 attempt contribution is:

`attempt_component_yards = (actual_attempts - pred_attempts) * (actual_ypa + pred_ypa) / 2`

Define `avg_ypa = (actual_ypa + pred_ypa)/2` and convert each Shapley attempt contribution to yards by multiplying by `avg_ypa`.

The three opportunity-chain yard contributions must sum exactly to the existing attempt contribution within `1e-6` yards.

The complete final synthesis residual identity must also reconcile:

`actual_pass_yards - football_synthesis`

=

`team_pass_opportunity_yards + attempt_conversion_yards + qb_share_yards + ypa_component - stack_adjustment - synthesis_adjustment`

within `1e-6` yards.

## Frozen outputs

Report for 2024, 2025, and pooled 2024-2025:
- mean contribution and mean-absolute contribution for each of the three attempt subcomponents;
- share of total mean-absolute opportunity-chain mass;
- sign agreement of each subcomponent with the total attempt residual;
- dominant subcomponent rate by row;
- p50 / p75 / p90 absolute subcomponent contribution;
- contribution behavior in all games and in the existing `ATTEMPTS_DOMINANT` games;
- underprojection vs overprojection separately;
- 8+ and 10+ absolute attempt-miss cohorts separately;
- stability of the aggregate ranking across 2024 and 2025.

Also report the already-frozen YPA, stack, and synthesis components only for reconciliation/context; do not refit or redefine them.

## Frozen shared-pass-volume attribution

On the exact 440-row 2025 WR target-mass cohort, correlate each **attempt-unit** opportunity subcomponent with `wr_target_mass_residual`:
- Pearson
- Spearman
- same-sign rate
- signed-subcomponent Q4-minus-Q1 WR residual gap

On the exact 884-row 2024-2025 WR reception-mass replication cohort, repeat Pearson, Spearman, same-sign rate, and season splits.

The existing total QB-attempt-residual correlations are retained as reference baselines and must be reproduced within `1e-9` where the exact source rows are reused.

This attribution is postgame diagnostic evidence only. Realized WR targets/receptions can never be fed into a QB pregame projection.

## Integrity gates

Scientific interpretation stops unless all pass:
1. exactly 884 M89 QB rows;
2. exactly 440 primary cross-position rows and 884 replication rows;
3. zero sportsbook inputs;
4. no production code/model/artifact changes;
5. M89 predicted-attempt product reconciliation max error <= `1e-6`;
6. realized-attempt product reconciliation max error <= `1e-6`;
7. three-factor Shapley attempt identity max error <= `1e-6`;
8. opportunity-chain yard sum vs frozen attempt-component max error <= `1e-6`;
9. full final synthesis residual identity max error <= `1e-6`;
10. exact key uniqueness and full alignment;
11. target-game PBP/official outcomes used only after predictions are fixed and only for diagnostic labels/attribution;
12. total QB-attempt residual vs WR target/reception correlations reproduce the canonical shared-volume evidence within `1e-9` where exact cohorts are reused.

## Interpretation / routing rule

This migration cannot promote a model. Its only authorized disposition is one of:
- `TEAM_PASS_OPPORTUNITY_PRIMARY_DIAGNOSTIC`
- `ATTEMPT_CONVERSION_PRIMARY_DIAGNOSTIC`
- `QB_SHARE_PRIMARY_DIAGNOSTIC`
- `MIXED_OPPORTUNITY_CHAIN_NO_SINGLE_PRIMARY`
- `MECHANICAL_OR_INTEGRITY_FAIL_NO_SCIENCE`

A subcomponent may be called `PRIMARY` only if all are true:
1. it has the largest pooled mean-absolute attempt contribution;
2. it is also largest in both 2024 and 2025 OR is within 10% of the largest in one season and largest in the other;
3. in the 2025 WR target-mass attribution, its absolute Spearman correlation is at least `0.30`;
4. its 2025 WR-target absolute Spearman is at least `0.10` greater than each other opportunity subcomponent.

If no subcomponent clears all four, disposition is `MIXED_OPPORTUNITY_CHAIN_NO_SINGLE_PRIMARY`.

The routing disposition does **not** authorize a predictive correction. It only determines where a subsequent source/predictability audit is allowed to focus.

## Stopping rule

- One decomposition specification only.
- No alternate factor definitions, windows, thresholds, player exclusions, model classes, or interaction forms after results are visible.
- Do not tune M89/M90.
- Do not retest the QB-R1 existing-context router.
- If one mechanism is primary, next step is a strict-prior source/predictability audit for that mechanism.
- If mixed, next step is source discovery for genuinely new week-specific team pass-state information rather than a generic residual model.
