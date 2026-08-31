# M95D Authoritative Results — RB Rushing Environment / Scheme / Personnel

## Run identity

- Workflow: `M95D RB Rushing Environment Scheme`
- Authoritative run: `33359898917`
- Job: `99389092246`
- Tested head SHA: `706d2f2ae1a6bb56c28935aa9ad04d129e3aef61`
- Branch: `research-rb-m95d-rushing-environment`
- Artifact: `migration-95d-rb-rushing-environment-scheme`
- Artifact ID: `9746342025`
- Artifact size: `4,289,382` bytes
- Artifact SHA256: `7ed09cc60b14f4cddcb9f00bfd249bd3be0fcda14ee609fca4be89789fa75628`
- Result: workflow success; advancement gate failed.
- Formal disposition: `RETAIN_M95C_ENVIRONMENT_ONLY`
- Production change: `0`

## Control integrity

The authoritative run reproduces the exact M95C `role_plus_environment` control after two control-alignment fixes:

| Forward test | M95C original rush-yard MAE | M95D reproduced control |
|---|---:|---:|
| train 2023 → test 2024 | 22.240046 | 22.240046 |
| train 2023+2024 → test 2025 | 21.987084 | 21.987084 |

The same control also reproduces 2025 100+ rushing-yard AUC (`0.787110`) and explosive-20 AUC (`0.662649`). This makes the final M95D comparison apples-to-apples with M95C.

Runs #1–#3 are superseded for scientific interpretation. Their reruns corrected implementation/reporting and control-alignment issues before the authoritative comparison; they were not used to tune the M95D feature families or gates.

## New structural data recovered and certified for research

M95D successfully recovered and joined the following football-only historical sources:

- FTN charting: `185,215` rows, PBP join rate `0.965883`
- nflverse participation: `187,421` rows, PBP join rate `0.981651`
- PFR weekly advanced defense: `31,698` player-week rows

The 2025 source schemas confirmed useful fields including:

- FTN: QB location, offensive backfield count, defenders in box, motion, RPO, blitzers and pass rushers
- participation: offensive formation, offensive personnel, defenders in box, defensive personnel, players on play and pressure/coverage context
- PFR defense: combined tackles, missed tackles and missed-tackle percentage

M95D used leakage-safe rolling pregame versions of rushing structure / box / tackling information. No sportsbook or game-market input was used.

## Model families

1. `role_baseline`
2. `role_plus_m95c_environment`
3. `role_plus_environment_scheme`
4. `full_environment_matchup`

The added structural layer included rushing motion/RPO/shotgun/under-center structure, offensive backfield count, box exposure, 11/12 personnel usage, formation, defensive box tendencies and defensive missed-tackle tendencies. The full family added compact football interactions rather than a raw feature dump.

## 2024 forward test — train 2023 → test 2024

| Metric | M95C environment | Full M95D | Gain from M95D |
|---|---:|---:|---:|
| Carry MAE | 3.799904 | 3.805907 | -0.006002 |
| Rush-yard MAE | 22.240046 | 22.501985 | -0.261939 |
| YPC MAE, 8+ carries | 1.352909 | 1.367343 | -0.014434 |
| YBC/att MAE, 5+ carries | 1.071267 | 1.119319 | -0.048052 |
| 75+ rush AUC | 0.804435 | 0.806719 | +0.002285 |
| 100+ rush AUC | 0.806525 | 0.817781 | **+0.011256** |
| 20+ explosive-run AUC | 0.698571 | 0.693974 | -0.004596 |

The structural layer clearly failed as a universal mean/YBC improvement in 2024, but improved high-end 100+ yard discrimination.

## 2025 forward test — train 2023+2024 → test 2025

| Metric | M95C environment | Full M95D | Gain from M95D |
|---|---:|---:|---:|
| Carry MAE | 3.605489 | **3.585343** | **+0.020147** |
| Rush-yard MAE | **21.987084** | 22.026847 | -0.039763 |
| YPC MAE, 8+ carries | **1.396990** | 1.433504 | -0.036514 |
| YBC/att MAE, 5+ carries | **1.092654** | 1.098613 | -0.005960 |
| 75+ rush AUC | 0.781697 | 0.781266 | -0.000431 |
| 100+ rush AUC | 0.787110 | **0.796965** | **+0.009856** |
| 20+ explosive-run AUC | 0.662649 | 0.666126 | +0.003478 |

The simpler scheme family (without all interactions) reached `0.668843` explosive-20 AUC in 2025, but that explosive improvement did not replicate cleanly into the 2024 full test. The repeated 100+ rushing-yard signal is more credible.

## Gate result

- stable rush-yard mean gain in both years: `0`
- stable YBC mechanism gain in both years: `0`
- carry guard: `1`
- tail support: `1`

Formal disposition: **`RETAIN_M95C_ENVIRONMENT_ONLY`**.

## Scientific interpretation

1. The new scheme/personnel/box/tackling sources are real, high-coverage, and usable with leakage-safe historical timing.
2. Simply adding those structural variables to the universal rushing-yard/YPC mean model does **not** improve forward prediction. It makes rush-yard MAE worse in both forward seasons and also fails the YBC mechanism target.
3. The structural information contains a repeated **upper-tail signal**: 100+ rushing-yard AUC improves by about `0.0113` in 2024 and `0.0099` in 2025.
4. Therefore these variables should be retained for a future distribution/upside layer, not promoted as a universal mean correction.
5. The small 2025 carry improvement is not evidence that M95D solved workload. The known 20+/25+ carry compression remains an opportunity-volume/allocation problem.
6. M95C's simpler environment family remains the best validated mean-side result from this sub-sequence.

## Recommended next RB research

The highest-priority unresolved problem is now the absolute workload distribution. The next experiment should return to the M93/M94 opportunity architecture and explicitly model:

`team offensive plays × team rush probability × RB-room share of all team rush attempts × lead-RB share`

as a probability/distribution problem rather than a compressed mean.

It should combine the validated M94C team/game-environment signal with M93B role/concentration, and it may use the validated M95 matchup/environment information as contextual modifiers. M95D's scheme/personnel signal should be reserved primarily for 100+ / upper-tail distribution calibration unless later evidence supports a mean effect.

Primary success criteria should include 20+/25+ carry recall and precision, projected carry quantiles/maxima, false-positive high-workload games, and MAE on actual 20+/25+ games, while retaining aggregate and committee guardrails.
