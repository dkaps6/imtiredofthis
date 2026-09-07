# TE-R5P FINAL PRODUCTION MODEL V1 — Frozen Fit Contract

## Authorization
This fit is authorized only because the exact production-contract walk-forward validation passed:
- TE-R5P run `34152797603`
- artifact `10029942404`
- disposition `TE_R5P_PRODUCTION_CONTRACT_REFIT_ELIGIBLE`

This step contains **no new science and no new tuning**. It fits the already-validated TE-R5P architecture once on all eligible historical data available before the 2026 season and persists a deployable parameter artifact.

## Sources
- TE-R3 source run `34126813280`, artifact `10020423842`, used only for the historical B0 TE rows / outcomes / B0 efficiency fields. TE-R3 candidate pool remains forbidden.
- TE-R4 strict-prior participation run `34127474412`, artifact `10020700686`.
- TE-R5P validation run `34152797603`, artifact `10029942404`, must explicitly report eligible before this fit runs.

## Training cohort
Use the exact reconstructed TE-R5P merged dataset.
Training rows:
- seasons **2022–2025** inclusive;
- `actual_te_pool > 0`;
- successful one-to-one TE-R4 participation join.

No 2026 outcome data exists or is allowed.

## Frozen model / feature contract
Exactly the validated TE-R5P contract:
- StandardScaler
- Ridge(alpha=`20.0`)
- EPS=`0.02`
- target clip `[-2,+2]`
- prediction clip `[-1,+1]`
- `pool_ratio` identically `1.0`
- candidate team TE pool at inference = **current B0/production TE pool**
- room allocation = softmax of `log(b0_te_room_share + EPS) + clipped predicted residual`
- efficiency remains the current B0 production efficiency.

Feature names and ordering:
1. `b0_te_room_share`
2. `log_b0_te_pool`
3. `pool_ratio`
4. `room_size`
5. `prior1_same_team_offense_pct`
6. `prior1_same_team_offense_snaps`
7. `prior1_anyteam_offense_pct`
8. `prior3_anyteam_offense_pct`
9. `prior1_anyteam_offense_snaps`
10. `prior3_anyteam_offense_snaps`
11. `log1p_prior_count_same_team`
12. `log1p_prior_count_anyteam`
13. `prior1_same_team_available`
14. `prior3_same_team_available`
15. `snap_share_prior1_same_team`
16. `snap_share_prior3_anyteam`

## Persisted artifact
Write `model/te_r5p_production_model_v1.json`-equivalent evidence containing:
- model version;
- source run/artifact lineage;
- training seasons and row count;
- feature names/order;
- EPS, Ridge alpha, target/prediction clips;
- scaler mean and scale;
- Ridge standardized coefficients and intercept;
- `pool_ratio_contract=1.0`;
- `candidate_team_pool='b0_te_pool'`;
- feature-construction notes required for 2026 inference;
- integrity audit.

Also persist a training-row audit and coefficients table.

## Integrity requirements
- TE-R5P upstream disposition is eligible;
- training rows > 0 and seasons are exactly 2022/2023/2024/2025;
- duplicate player-team-week rate = 0;
- TE-R4 same/future count = 0;
- `pool_ratio` is exactly 1.0 for every training row;
- no TE-R3 candidate-pool value contributes to features other than the explicitly constant pool-ratio placeholder or to target mass;
- sportsbook inputs = 0;
- no 2026 outcomes used;
- fitted parameter vectors have exactly 16 entries and finite values.

## Next-step rule
If the fit passes integrity, freeze and run a **2026 Week-1 TE-R5P shadow inference/context audit** against the canonical Full Slate inputs. Do not wire Full Slate production until that shadow audit verifies feature coverage, finite team target mass, baseline parity when disabled, and sensible player-level reallocations.