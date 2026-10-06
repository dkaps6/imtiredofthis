# RB Opponent-Defender Injury Candidate V1 — Result

**STATUS: COMPLETE / RBDI-F7-1 CLOSED**

Frozen contract:
`docs/research/RB_OPPONENT_DEFENDER_INJURY_CANDIDATE_V1_CONTRACT.md`

Authority:
- branch: `research-rb-opponent-defender-injury-candidate-v1`
- run: `37544856848` — **SUCCESS**
- source SHA: `68abc2f468caddfc11ac0fb60a22a58744585f99`
- artifact: `11450645374`
- digest: `sha256:c39a971d7b81530503d22b0c60234de137f4b59fdf144e77eb04d7ba30b1c58c`
- sportsbook inputs used: **0**
- 2026 outcomes read: **0**
- production changed: **false**
- post-hoc subgroups scored: **0**

Final disposition:

`RBDI_F7_1_CLOSED`

## Candidate

Feature:
`front7_out_doubtful_snap_mass`

Form:
`baseline + beta_2024 * raw_front7_out_doubtful_snap_mass`

2024 zero-intercept fit:
- rows: 1,267
- games: 256
- beta: `9.236956888003448`
- direction: positive, as hypothesized

2024 descriptive fit metrics:
- baseline MAE: `21.437998`
- candidate MAE: `21.827621`
- baseline RMSE: `30.894916`
- candidate RMSE: `30.575121`
- baseline bias: `-7.532243`
- candidate bias: `-5.137633`
- 75+ AE: `45 -> 40`
- 100+ AE: `16 -> 16`

The fitted coefficient therefore reduced low-side bias / RMSE / large-error frequency but did not improve MAE even in training.

## Untouched 2025 confirmation

Support:
- rows: 1,277
- games: 256
- support gate: PASS

Error:
- baseline MAE: `20.701902`
- candidate MAE: `20.913293`
- MAE change: **-0.211391 worse**
- baseline RMSE: `31.005168`
- candidate RMSE: `30.532588`
- baseline bias: `-7.635227`
- candidate bias: `-4.941537`
- 75+ AE: `53 -> 50`
- 100+ AE: `22 -> 18`

Game-cluster bootstrap:
- 10,000 replicates
- P(MAE improvement): `0.036`
- 95% CI for mean paired MAE improvement:
  `[-0.434523, 0.018823]`

Zero-burden no-op:
- max absolute gap: `0.0`

## Gate disposition

Passed:
- positive 2024 coefficient;
- support;
- RMSE non-worse;
- absolute bias non-worse;
- 75+ AE non-worse;
- 100+ AE non-worse;
- exact zero-burden no-op.

Failed:
- 2025 MAE improvement;
- clustered CI lower bound > 0.

Therefore the predeclared all-gates PASS cannot be issued.

## Interpretation

The target-week opponent FRONT7 injury source is real, deployable, and directionally associated with a portion of RB underprediction, especially larger misses.

However, applying the source as one unconditional additive RB rushing-yard mean shift does not improve typical absolute error out of sample. The source appears too coarse for this exact production mechanism.

Do not rescue this result with OUT-only, DOUBTFUL-only, DL/LB splits, heavy-burden thresholds, caps, interactions, or post-hoc cohorts. Those were explicitly forbidden before scoring.

The readiness result remains valid:
`RB_OPPONENT_DEFENDER_INJURY_SOURCE_READY`.

This failure closes only `RBDI-F7-1`, not the data source itself. Any future use requires a genuinely distinct predeclared mechanism or prospective design.

No production change is authorized.
