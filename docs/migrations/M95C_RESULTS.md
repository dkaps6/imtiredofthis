# M95C Authoritative Results — RB Quality vs Environment Decomposition

## Run identity

- Workflow: `M95C RB Quality-Environment Decomposition`
- Authoritative run: `33358467022`
- Job: `99385041970`
- Tested head SHA: `708a896e06bfa1d6e0323c5b72e732b43799629b`
- Branch: `research-rb-m95c-quality-environment-decomposition`
- Artifact: `migration-95c-rb-quality-environment-decomposition`
- Artifact ID: `9745900457`
- Artifact SHA256: `6d6e8cf9728a6910274176bd257fc0082aedad31a621b5424c31ef60d75c7694`
- Result: workflow success; research gate failed.
- Formal disposition: `RETAIN_M95B_OFFENSE_PROFILE`
- Production change: `0`

## Primary gate result

The pre-specified final family (`role_plus_decomposition_and_raw`) did not earn advancement.

| Metric | Raw efficiency | Decomposition + raw | Result |
|---|---:|---:|---|
| 2024 rush-yard MAE | 22.562432 | 22.537715 | +0.024717 improvement |
| 2025 rush-yard MAE | 22.162674 | 22.201749 | -0.039075 regression |
| 2025 YPC MAE, 8+ carries | 1.392560 | 1.426755 | regression |
| 2025 carry MAE | 3.616081 | 3.613505 | essentially flat/slightly better |
| 2025 100+ rush AUC | 0.777209 | 0.790249 | +0.013040 |
| 2025 20+ explosive-run AUC | 0.649677 | 0.652865 | +0.003188 |

The gate failed because the aggregate rushing-yard gain did not replicate into 2025 and 2025 YPC worsened.

## Important secondary finding: environment is the stable mean signal

The environment-only family (`role_plus_environment`) was more stable than raw player rushing efficiency.

### 2024

- role baseline rush-yard MAE: `22.400210`
- raw efficiency: `22.562432` (worse than role by `0.162223`)
- environment only: `22.240046` (better than role by `0.160163`; better than raw by `0.322386`)

### 2025

- role baseline rush-yard MAE: `22.017752`
- raw efficiency: `22.162674` (worse than role by `0.144922`)
- environment only: `21.987084` (better than role by `0.030669`; better than raw by `0.175590`)

Thus yards-before-contact / expected-yard / stacked-box / time-to-LOS / team-environment information produced a small but directionally stable forward mean signal in both test seasons, while naive recent efficiency outcomes did not.

## Tail results

M95C's advanced information was more useful for high-end discrimination than for broad mean YPC.

### 2025 100+ rushing yards AUC

- role baseline: `0.779567`
- raw efficiency: `0.777209`
- environment: `0.787110`
- runner-created: `0.785419`
- decomposition: `0.790383`
- decomposition + raw: `0.790249`

### 2025 20+ explosive-run AUC

- role baseline: `0.642402`
- raw efficiency: `0.649677`
- environment: `0.662649`
- runner-created: `0.640752`
- decomposition: `0.655699`
- decomposition + raw: `0.652865`

The environment family was the best 2025 explosive-run discriminator. The combined decomposition was strongest for 100+ rushing-yard discrimination.

### 2024 runner-created tail signal

Runner-created metrics were especially useful for high-yardage classification in 2024:

- 75+ AUC: role `0.807142`, runner-created `0.817212`
- 100+ AUC: role `0.811564`, runner-created `0.826626`

That tail signal remained directionally positive for 100+ yards in 2025 (`0.785419` vs role `0.779567`) but did not improve broad rushing-yard MAE or YPC consistently.

## Coverage

Feature coverage was high and does not explain the failure:

- environment features: mean coverage ~`92.6%`, minimum ~`84.5%`
- runner-created features: mean coverage ~`91.8%`, minimum ~`84.5%`
- raw efficiency: ~`96.7%`
- role controls: mean ~`95.9%`

## Scientific interpretation

1. Raw recent rushing efficiency is noisy and should not be treated as a clean RB talent measurement.
2. Pregame rushing-environment information is a small but repeatable mean signal across both forward seasons.
3. Runner-created statistics (YAC, broken tackles, RYOE, rush-over-expectation) look more useful for tail/upside discrimination than as a universal mean adjustment.
4. Combining every advanced statistic into one large linear mean model overfits / dilutes the stable environment signal.
5. M95C therefore does **not** justify replacing M95B's offensive profile with the full decomposition.
6. The environment-only findings should be carried into the next scheme/personnel experiment rather than discarded.

## Recommended next research

M95D should focus on **predicting and explaining the rushing environment itself** rather than simply adding more RB efficiency variables.

Candidate football-only sources / structure:

- FTN charting available through nflverse (box count, backfield, formation, motion, RPO and related structure);
- participation/personnel context where historically available without leakage;
- offensive formation / shotgun / directional tendencies already present in PBP;
- defensive front / box and tackling vulnerability;
- weekly defensive missed-tackle information if a certified historical source can be recovered;
- team YBC and NGS expected-yard context as the stable M95C environment targets/features.

The known 25+ carry underprojection remains an opportunity-volume/allocation problem and is not solved by M95C.
