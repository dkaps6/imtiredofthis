# PLAYER OUTPUT COMPONENT DECOMPOSITION V1 — FROZEN CONTRACT

Date: 2026-10-07
Branch: `research-player-output-component-decomposition-v1`

## Purpose

Decompose completed 2026 Weeks 1-4 **individual final player projection error**
into:

1. opportunity / workload error
2. effective per-opportunity output error

This diagnostic determines which player-landscape layer deserves the next
research lane.

It is not a fitted model and cannot promote any feature.

## Frozen population

Use the exact **baseline opportunity rows** from the historical-availability
parity artifact:
- run `37687979574`
- file `baseline_player_opportunity_rows.csv`

Pair them to the exact all-player point-scoreboard authority from:
- run `37683439543`

These two parents represent the same baseline replay state.

Do **not** pair the ACT-only opportunity rows to the baseline point scoreboard.
The availability-parity diagnostic changed six QB identities and reallocated
active-player opportunity mass, so that would mix two different model variants.

The ACT-only availability result remains a separate completed diagnostic. A
future ACT-only final-point decomposition would require rebuilding final point
projections on that exact ACT-only universe first.

Weeks:
- 2026 W1-W4

Required individual markets:
- QB pass_yards
- RB rush_yards
- RB rec_yards
- RB receptions
- WR rec_yards
- WR receptions
- TE rec_yards
- TE receptions

Rush+receiving yards is excluded from the primary decomposition because the
final production authority is not guaranteed to equal the algebraic sum of the
separate final point means. It may be reported secondarily only with an explicit
parity audit.

## Opportunity mapping

- QB pass_yards -> pass_attempts
- RB rush_yards -> carries
- RB rec_yards -> targets
- RB receptions -> targets
- WR rec_yards -> targets
- WR receptions -> targets
- TE rec_yards -> targets
- TE receptions -> targets

Use the already-frozen replay-matched baseline opportunity rows:
- `predicted_opportunities`
- `actual_opportunities`

Do not rebuild or refit opportunity science in this diagnostic.

## Effective final-model efficiency

For every paired row with positive predicted opportunity:

`model_effective_efficiency = final_point_projection / predicted_opportunities`

This is an algebraic decomposition of the **final model point prediction**,
including ensemble/specialist effects. It is not claimed to be the simulator's
raw YPT/YPC/YPA/catch-rate parameter.

Actual efficiency:

`actual_efficiency = actual_output / actual_opportunities`

For rows with actual opportunities = 0:
- actual output must also equal 0 within tolerance;
- define actual_efficiency = 0 only for the identity check;
- efficiency-oracle interpretation is marked zero-opportunity and excluded from
  efficiency-only summary slices.

## Counterfactual diagnostics

### Baseline

`baseline = predicted_opportunities * model_effective_efficiency`

Must reproduce the exact final point projection within `1e-10`.

### Opportunity oracle

`opportunity_oracle = actual_opportunities * model_effective_efficiency`

Interpretation:
what would the final player mean have looked like if workload were known
perfectly but the model's effective per-opportunity output remained unchanged?

### Efficiency oracle

For rows with actual opportunities > 0:

`efficiency_oracle = predicted_opportunities * actual_efficiency`

Interpretation:
what would the final player mean have looked like if per-opportunity conversion
were known perfectly but projected workload remained unchanged?

### Full identity

For rows with actual opportunities > 0:

`full_oracle = actual_opportunities * actual_efficiency`

Must reproduce actual output within `1e-10`.

## Required outputs

1. `player_output_component_rows.csv`
2. `player_output_component_market_summary.csv`
3. `player_output_component_week_summary.csv`
4. `player_output_component_workload_summary.csv`
5. `player_output_component_summary.json`

## Required diagnostics

By position + market:

- paired rows
- zero-actual-opportunity rate
- baseline MAE / bias / RMSE
- opportunity-oracle MAE / bias / RMSE
- opportunity-oracle MAE improvement and fraction removed
- efficiency-oracle MAE / bias / RMSE on efficiency-eligible rows
- baseline MAE on the same efficiency-eligible rows
- efficiency-oracle MAE improvement and fraction removed on the same rows
- correlation of opportunity error with final output error
- correlation of effective-efficiency error with final output error
- candidate-closer / baseline-closer / ties for each oracle
- per-week metrics

Also report fixed descriptive workload bins:

QB pass attempts:
- 0
- 1-20
- 21-30
- 31-40
- 41+

RB carries:
- 0
- 1-3
- 4-8
- 9-14
- 15+

Targets:
- 0
- 1-2
- 3-5
- 6-8
- 9+

These bins are descriptive only and may not be used to select a model rule.

## Interpretation

The result must distinguish:

- opportunity-dominant residual
- efficiency-dominant residual
- mixed residual

The descriptive label may follow the larger oracle MAE reduction, but the raw
continuous metrics are authoritative.

No threshold or promotion is implied by the label.

## Integrity gates

Fail closed if:

- replay-matched baseline opportunity identity population cannot be paired
  exactly to final point rows;
- Weeks outside 1-4 appear;
- sportsbook fields are used upstream;
- baseline algebraic reconstruction gap exceeds `1e-10`;
- full actual identity gap exceeds `1e-10`;
- actual output is nonzero when actual opportunity is zero;
- a market maps to the wrong opportunity family;
- any parameter is fit;
- any target-game outcome is used to construct baseline predictions.

## Protected boundaries

This diagnostic may not:
- reopen WR-R3;
- fabricate historical route participation;
- reopen Coverage-v2 team man/zone as the same feature;
- infer WR-CB assignments;
- revive the failed FMT V1 formulas;
- alter M38 / TE-R5P / WR-R15 / QB M89-M90 / RB authorities;
- promote any production change.

## Next-step rule

Only after the component decomposition is frozen may a new player-landscape
mechanism be opened.

If opportunity remains dominant:
continue only with individual role/share allocation.

If efficiency dominates:
audit player + opponent conversion mechanics for that exact market.

If mixed:
preserve the mechanisms as separate lanes.
