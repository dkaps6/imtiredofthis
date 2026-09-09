# RB R26K Week-1 Allocation Mechanism Atlas V1 — Result

Date: 2026-09-09
Branch: `research-rb-r26k-week1-allocation-mechanism-atlas-v1`
Head: `01be58dbd04613cb0ebf61c1c91bfb9dd7903bc7`
Run: `34376961740`
Job: `102552086927`
Artifact: `10114261724`
Digest: `sha256:74a3d9ac58fca360f6d2d23e19b28e1254f38e49e3be23878ee1b37cebe22c88`
Disposition: `2020_SPECIFIC_MECHANISM_NO_REPLICATED_ROUTER`

## Integrity

All integrity protections passed:
- exact immutable R26 parent digest verified;
- exact immutable R26J parent digest verified;
- production-protected files clean against `main@f8417f55b04ce0e19baf260e9d532765034c47f1`;
- R26 predictions not regenerated;
- R9 not refit;
- sportsbook inputs 0;
- future-outcome features 0;
- same-week historical depth not added;
- receiving-yard means unchanged;
- R22 unchanged;
- production parameters unchanged;
- R26J room-state join coverage 1.0;
- 2020-2025 required seasons all present.

R26K authorized no shadow and no production change.

## Frozen primary states

K1 `LARGE_COMPLEX_ROOM`:
`current_room_n >= 5 AND continuing_n >= 2 AND entrants_n >= 2`

K2 `VETERAN_PRESSURE_FLAT_HIERARCHY`:
`veteran_entry_n >= 1 AND baseline_top_room_share <= 1/3`

K3 `HIGH_VACATED_LOAD_FLAT_HIERARCHY`:
`sum_exit_last8_targets_pg >= 3.0 AND baseline_top_room_share <= 1/3`

K4 `MANY_CLAIMANTS_HIGH_VACATED_LOAD`:
`continuing_n >= 2 AND entrants_n >= 2 AND sum_exit_last8_targets_pg >= 3.0`

K5 `VETERAN_PRESSURE_HIGH_VACATED_LOAD`:
`veteran_entry_n >= 1 AND sum_exit_last8_targets_pg >= 3.0`

No threshold was moved after outcomes became visible.

## Result summary

Total 2020 Week-1 vacancy-incumbent signed reception absolute-error worsening from R26:
`5.377149318960339`

Materially harmful 2020 states:
- K1
- K2
- K4
- K5

Qualified replicated harmful states:
- none.

Therefore no R26L-style state router is authorized from R26K.

## State detail

### K1 — Large complex room

All seasons:
- n 51
- reception MAE `1.330860 -> 1.290115` (~3.06% better)
- RMSE `1.692966 -> 1.619003`
- bias `-0.749901 -> -0.275601`
- p90 `2.934033 -> 2.359892`

2020:
- n 14
- reception MAE ~14.60% worse
- 39.14% of total 2020 net worsening
- allocation-dominant failure.

2021-2025 pooled:
- n 37
- reception MAE ~7.82% better
- only 2024 individually harmful
- player allocation nonharmful pooled.

Conclusion: strong 2020 harm but opposite pooled modern-period direction.

### K2 — Veteran pressure + flat hierarchy

All seasons:
- n 50
- reception MAE `1.594652 -> 1.459725` (~8.46% better)
- RMSE `2.025870 -> 1.800680`
- bias `-1.095418 -> -0.527364`
- p90 `3.379437 -> 2.842010`
- target MAE `1.989767 -> 1.928497`

2020:
- n 17
- reception MAE ~18.73% worse
- 65.26% of total 2020 net worsening
- allocation-dominant failure.

2021-2025 pooled:
- n 33
- reception MAE ~16.81% better
- zero later seasons with harmful mean effect
- player allocation strongly nonharmful.

This is the clearest evidence that a 2020-specific state cannot safely become a modern Week-1 guard.

### K3 — High vacated load + flat hierarchy

All seasons:
- n 38
- reception MAE `1.499855 -> 1.366478` (~8.89% better)
- target MAE `1.750277 -> 1.699322`

2020:
- n only 6, below frozen support gate
- reception MAE ~13.89% worse.

2021-2025:
- n 32
- reception MAE ~12.99% better
- no later harmful season.

No 2020 material qualification due support, and modern-period direction is strongly beneficial.

### K4 — Many claimants + high vacated load

All seasons:
- n 39
- reception MAE `1.318903 -> 1.374299` (~4.20% worse)

2020:
- n 8
- reception MAE ~47.17% worse
- 68.39% of total 2020 net worsening
- allocation-dominant failure.

2021-2025 pooled:
- n 31
- reception MAE ~3.48% better
- harmful mean effect appears in 2024 and 2025, but pooled direction is beneficial
- player allocation nonharmful pooled.

This state came closest to temporal replication, but it failed the frozen outside-2020 pooled-harm gate. It therefore cannot authorize a router.

### K5 — Veteran pressure + high vacated load

All seasons:
- n 55
- reception MAE `1.527602 -> 1.415469` (~7.34% better)
- RMSE `2.006217 -> 1.778000`
- bias `-1.138076 -> -0.700066`
- p90 `3.206582 -> 2.786371`
- target MAE `1.782014 -> 1.706726`

2020:
- n 8
- reception MAE ~32.47% worse
- 53.75% of total 2020 net worsening
- allocation-dominant failure.

2021-2025 pooled:
- n 47
- reception MAE ~12.06% better
- only 2025 has harmful mean effect
- player allocation strongly nonharmful pooled.

Again, the state is harmful in 2020 but beneficial in the later period.

## Scientific interpretation

R26K confirms that the R26 2020 failure can be localized to football-coherent source states, but those same states do not behave as harmful regimes in 2021-2025.

The central evidence is therefore NOT:
`find a better 2020 guard`.

It is:
`the 2020 Week-1 relationship between roster transition state and receiving allocation is not transportable to the later period in the same way`.

Several states that explain a large share of 2020 harm are materially beneficial after 2020. A state-based fallback built from the 2020 failure would likely erase genuine 2021-2025 Week-1 gains, repeating the R26G/R26I failure mode.

This strengthens the component-preservation interpretation:
- preserve R26 Week-1 vacancy/R9 logic;
- preserve its strong 2021-2025 gains;
- do not create another blanket or compound guard from 2020;
- do not exclude 2020 yet;
- next test should address temporal transportability/relevance of the 2020 source regime to the actual 2026 Week-1 environment using source-only pregame information.

## Authority

- child candidate design authorized: false
- prospective shadow authorized: false
- production promotion authorized: false
- exclude 2020 authorized: false
- production changes: none.
