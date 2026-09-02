# RB STACK6Q — Pregame Designed-Run-Call Model Results

## Status

Authoritative execution completed successfully. The frozen predictive family did **not** qualify. No production change and no P3 recomposition are authorized.

## Authoritative run

- Workflow: `RB STACK6Q Designed Run Call Model`
- Run: `33653547460`
- Job: `100326322852`
- Branch: `research-rb-stack6q-designed-run-call-model`
- Tested SHA: `db0cc80abc245c0af349e52d2aeabf5a6ea53ce5`
- Artifact: `rb-stack6q-designed-run-call-model`
- Artifact ID: `9855777371`
- Artifact SHA256: `60bff542e8495b208958a21b891f02e3d5b9806c6b03ffb465ff271e27a5ad76`
- Disposition: **`STACK6Q_DESIGNED_RUN_MODEL_NOT_QUALIFIED`**

## Integrity

- frozen 2024 M94C holdout rows: `186`
- 2025 M94C rows: `544`
- STACK6H rows: `544`
- joined 2025 rows: `544`
- W6-18 rows: `388`
- M94C vs STACK6H `T_hat` max abs difference: `3.55e-15`
- reconstructed PBP vs M94C rush truth max abs difference: `0.0`
- strict-prior construction: PASS
- feature count: `20`
- Ridge alpha: `10.0`
- M94C blend alpha: `0.75`
- feature / hyperparameter / model-family / threshold search: `0`
- sportsbook inputs: `0`
- target-week participation or injury inputs: `0`

State-specific eligible training rows from the leakage-safe 2024 W13-18 M94C holdout were:

- lead: `112`
- neutral: `176`
- trail: `127`

## Frozen W6-18 scores

| Arm | MAE | RMSE | Bias | Corr |
|---|---:|---:|---:|---:|
| M94C | 6.203455 | 7.741320 | +0.179588 | 0.250307 |
| STACK6Q | 6.135474 | 7.762518 | -0.217646 | 0.247525 |

Overall MAE improved only `0.067981` attempts, while RMSE worsened `0.021198` and correlation worsened `0.002782`.

## Frozen error populations

### P3 pool overprojection >=5 (`n=95`)

- M94C MAE: `8.892955`
- STACK6Q MAE: `8.251252`
- gain: **`+0.641702`**

### P3 pool underprojection >=5 (`n=96`)

- M94C MAE: `9.108912`
- STACK6Q MAE: `9.252212`
- gain: **`-0.143300`**

### W13-18 (`n=188`)

- M94C MAE: `5.992169`
- STACK6Q MAE: `5.932759`
- gain: `+0.059410`

## Frozen gates

| Gate | Value | Pass? |
|---|---:|---:|
| overall MAE gain >=0.20 | +0.067981 | No |
| RMSE gain >0 | -0.021198 | No |
| corr gain >=+0.02 | -0.002782 | No |
| abs bias worsening <=0.25 | +0.038058 | Yes |
| POOL_OVER_5 MAE gain >0 | +0.641702 | Yes |
| POOL_UNDER_5 MAE gain >0 | -0.143300 | No |
| W13-18 MAE gain >0 | +0.059410 | Yes |

Three of seven scientific gates passed. No gate is waived.

## Durable conclusion

STACK6P remains important: designed-run calls contain the overwhelming majority of the remaining within-state rushing-tendency oracle headroom. STACK6Q shows that this headroom is **not recoverable by one compact state-level team-week Ridge using recent team/opponent designed-run rates plus frozen M94C environment**.

Do not retune STACK6Q on 2025 and do not route its team-rush predictions into P3.

The asymmetry is informative: the model materially contracts false-high team rushing but makes false-low games worse. The next investigation should therefore change representation rather than tune this scalar family. A no-fit decomposition of designed-run behavior into football-natural within-state situational contexts (down/distance and associated context occupancy versus conditional run-call tendency) is the preferred next step. This should determine whether the missing information is primarily which situations an offense reaches or how it calls runs once in those situations.