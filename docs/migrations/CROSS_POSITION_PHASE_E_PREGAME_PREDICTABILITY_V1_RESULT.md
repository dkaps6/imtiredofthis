# CROSS-POSITION PHASE E — PREGAME SHARED-STATE + PLAYER SENSITIVITY VALIDATION V1 RESULT

## Authoritative lineage
- Run: `34150046671`
- Job: `101830234757`
- Launch SHA: `401522dbc62fc7c7d2d1335c8b76c8afd104a7c7`
- Frozen-plan SHA: `0dbfdf41b1256279bb8bd709e6f31ee43e80fd4e`
- Artifact: `10029037488`
- Artifact digest: `sha256:0d67da9d15accbdfbec3ce5499b678a37ed1b2b08289fc049802623798d06fe5`
- Workflow conclusion: `success`
- Scientific disposition: `SHARED_PASS_STATE_PREGAME_NOT_ELIGIBLE`

## Integrity
- sportsbook features used: 0
- same/future outcomes used as predictors: 0
- test rows used to create training labels: 0
- production parameters changed: 0
- Phase-C source rows: 17,887
- scored 2025 team-games: 434

## E1 shared-state scorecard
| metric | baseline MAE | corrected MAE | improvement |
|---|---:|---:|---:|
| QB attempts | 6.836088 | 6.088599 | +0.747489 |
| WR+TE target pool | 6.026892 | 5.438862 | +0.588030 |
| WR target pool | 5.342498 | 4.664419 | +0.678080 |
| TE target pool | 2.400960 | 2.763489 | -0.362529 |
| RB carry pool (356 corrected games) | 5.574873 | 5.582011 | -0.007139 |

Additional diagnostics:
- QB attempt residual sign accuracy: 0.686636
- receiver target residual sign accuracy: 0.695853
- PASS_STATE_HIGH recall: 130/137 = 0.948905
- PASS_STATE_LOW recall: 2/39 = 0.051282

Frozen gates:
- QB attempt MAE improve >=0.15: PASS
- WR+TE pool no worse >0.05: PASS
- WR or TE improves and neither worse >0.10: **FAIL** (TE worsened 0.362529)
- QB residual sign accuracy >=0.55: PASS
- RB carry MAE no worse >0.15: PASS

## Interpretation
The **shared pregame pass-state signal is real and usable as a component**, but the frozen proportional WR/TE redistribution is not a valid architecture. It materially helps QB attempt opportunity and the WR target pool while damaging the TE target pool. Do not discard the QB/WR signal; do not promote the full E1 package.

The asymmetry is also important: the current predictor catches PASS_STATE_HIGH extremely well but almost never catches PASS_STATE_LOW. A future model must not assume one symmetric residual process handles both tails.

## E2 prospective player-environment validation
Only one training-derived concept met the frozen prospective group rule:
- WR `ADVERSE_SPOT_CEILING_CANDIDATE`
  - 29 training-derived candidates
  - 28 with favorable+adverse holdout coverage
  - validation rate: 0.75
  - pooled held-out adverse residual: +9.025816 yards
  - disposition: `PROSPECTIVE_PLAYER_ENVIRONMENT_SIGNAL`

Other WR environment-sensitive/resistant/amplifier concepts failed prospective validation. TE concepts failed prospective validation. QB sample was too small for a group signal. RB remains `INSUFFICIENT_TEMPORAL_HOLDOUT` under current authoritative RB-P3 lineage.

## Next research implication
The next shared-state test should retain the same frozen pregame `delta_pass_attempts` concept but **estimate separate training-only WR and TE pool response mappings**, because Phase D already showed different QB-attempt correlations for WR targets (0.697) and TE targets (0.472). TE-R5 remains independently protected. RB carry should remain baseline unless a separate rush-state mechanism earns promotion.
