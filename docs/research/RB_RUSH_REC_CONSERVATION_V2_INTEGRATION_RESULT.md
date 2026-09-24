# RB Rush + Receiving Conservation V2 — Integration Result

Date: 2026-09-24

Status: `RB_RUSH_REC_CONSERVATION_V2_INTEGRATION_PASS`

## Authority

- parent mean result: `RB_RUSH_REC_CONSERVATION_V2_MEAN_QUALIFIED`
- frozen integration plan commit: `be52c7cc30b9bc4b71a26443702d4b22b39c6e25`
- integration head: `262ca4a379828e3b2b450462ec6dfa32ea83525f`
- run: `36006624473`
- job: `107656319860`
- artifact: `10810946362`
- artifact digest: `sha256:eefb91122eb98c1eccf517d8e501b3e1d3c5fcb678a660cbf3f7a4b92b8277a8`
- preserved Week-2 input authority: run `35282021679`, artifact `10523345092`

## Mechanical A/B result

Baseline and candidate were repriced from the same preserved Week-2 input artifact with the same current branch/code. The only switch was the pre-frozen V2 research flag.

- baseline priced rows: **3,178**
- candidate priced rows: **3,178**
- same offer identity set: **PASS**
- candidate-applied offer rows: **104**
- candidate-applied player-games: **32**

Unrelated-output protection:
- max non-combo `model_proj` gap: **0.0**
- max non-combo `fair_prob` gap: **0.0**
- candidate applied only to rush+receiving: **PASS**
- duplicate offers introduced: **0**

Conservation:
- max final combo vs priced standalone component-sum gap: **4.263256414560601e-14**
- max adapter target vs priced standalone component-sum gap: **2.842170943040401e-14**
- max pathwise draw identity gap: **0.0**
- finite/nonnegative candidate draws: **PASS**
- sportsbook inputs added to football: **0**
- Week-1 rows changed: **0**

All frozen integration gates passed.

## Week-2 observational confirmation

Week-2 outcomes were not used to define, fit, select, or gate the candidate.

On the 32 matched settled RB player-games:
- baseline MAE: **30.3620**
- candidate MAE: **28.9014**
- improvement: **1.4606 yd**
- baseline signed underprojection bias (actual - projection): **+20.1339**
- candidate bias: **+9.8670**
- candidate closer: **16/32**

This is supportive confirmation only. Historical 2024/2025 qualification remains the scientific authority.

## Disposition

`RB_RUSH_REC_CONSERVATION_V2_INTEGRATION_PASS`

The candidate is now eligible for a separately frozen production-promotion/certification step for non-Week-1 RB/FB rush+receiving yards.

Do not change the formula, add weights, or expand scope during promotion.
