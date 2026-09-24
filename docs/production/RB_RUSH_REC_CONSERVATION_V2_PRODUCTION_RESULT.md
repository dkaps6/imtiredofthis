# RB Rush + Receiving Conservation V2 — Production Certification Result

Date: 2026-09-24

Status: `RB_RUSH_REC_CONSERVATION_V2_PRODUCTION_CERTIFIED`

## Scientific authority

Historical no-fit mean qualification:
- run: `36005675177`
- artifact: `10809812602`
- disposition: `RB_RUSH_REC_CONSERVATION_V2_MEAN_QUALIFIED`

Initial draw-level integration:
- run: `36006624473`
- artifact: `10810946362`
- disposition: `RB_RUSH_REC_CONSERVATION_V2_INTEGRATION_PASS`

Production-scope correction:
- `docs/production/RB_RUSH_REC_CONSERVATION_V2_SCOPE_AMENDMENT.md`
- V2 is **RB-only**, matching the historical qualification population.
- Week-2 qualified/integration sample contained 32 RB and 0 FB.
- FB is an explicit no-op.

Post-amendment re-integration:
- head: `ab9a74a5649030aa980be2c277106592229548fd`
- run: `36009227527`
- artifact: `10812095478`
- digest: `sha256:30fcf3205e18eadb9159e674ccdf41f7495d43ba23e15146e67065c5adb21f57`
- result: PASS

Post-amendment production certification:
- head: `28de84bd2f4965b9678fd6ce6391d65ae3562790`
- run: `36009239578`
- artifact: `10812360131`
- digest: `sha256:985384697b22754c5a38aee115d6956caf495540ef6738ac6a62fc6c964ce0c1`
- result: PASS

## Production formula

For non-Week-1 RB `rush_rec_yards` only:

`final rush_rec draw[i] = final-mean-aligned rush draw[i] + final-mean-aligned receiving draw[i]`

Therefore:

`final rush_rec model_proj = final standalone rush model_proj + final standalone receiving model_proj`

No fitted coefficient, blend, cap, router or sportsbook input is used.

## Historical evidence

2,787 complete historical RB player-games (2024-2025):

- MAE: **27.6853 -> 25.5718** (**2.1135 yd improvement**)
- RMSE: **39.8437 -> 36.4030**
- signed bias: **+14.3388 -> +9.3891**
- p90 absolute error: **64.7742 -> 57.2604**
- 30+ yard misses: **841 -> 768**
- candidate closer rate: **59.13%**

Independent season confirmation:
- 2024 MAE: **28.2918 -> 25.9442**
- 2025 MAE: **27.0783 -> 25.1991**

All frozen historical gates passed.

## Exact Week-2 production A/B

Preserved paid-origin authority:
- run `35282021679`
- artifact `10523345092`

V5 baseline vs V6 candidate:
- baseline priced rows: **3,178**
- candidate priced rows: **3,178**
- candidate-applied offer rows: **104**
- candidate-applied player-games: **32 RBs**
- max non-combo `model_proj` gap: **0.0**
- max non-combo `fair_prob` gap: **0.0**
- max combo final-vs-component-sum gap: **4.263256414560601e-14**
- max adapter-vs-component-sum gap: **2.842170943040401e-14**
- max pathwise identity gap: **0.0**
- sportsbook inputs added to V2 football: **0**
- Week-1 rows changed: **0**
- duplicate offer rows introduced: **0**

Every existing downstream Full Slate certification check passed under the same explicit current-role authority used by canonical production.

## Week-2 observational confirmation

Week-2 outcomes were not used to define, fit or gate the candidate.

Matched settled RB player-games: 32.

- baseline MAE: **30.3620**
- candidate MAE: **28.9014**
- improvement: **1.4606 yd**
- baseline signed underprojection bias: **+20.1339**
- candidate signed bias: **+9.8670**
- candidate closer: **16/32**

This is supportive only; historical 2024/2025 evidence remains the scientific authority.

## Scope/protections

V2 changes only:
- position RB;
- non-Week-1;
- rush+receiving yards.

Explicitly unchanged:
- FB;
- Week 1 P3/R22/R26;
- standalone RB rushing;
- standalone RB receiving yards;
- RB receptions/rush attempts;
- QB/WR/TE outputs;
- player universe;
- availability;
- sportsbook lines/odds/offer eligibility;
- M96.

## Disposition

`RB_RUSH_REC_CONSERVATION_V2_PRODUCTION_CERTIFIED`

The RB-only V6 production wrapper is authorized for stable Full Slate promotion.
