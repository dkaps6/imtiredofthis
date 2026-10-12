# PLAYER ROLE MATCHUP — DEPTH INTERACTION PREDICTIVE V1 RESULT

Date: 2026-10-11

Status: **COMPLETE — WR AND TE EXACT V1 CANDIDATES FAILED CLOSED. NO PRODUCTION CHANGE.**

Frozen contract:
- `docs/research/PLAYER_ROLE_MATCHUP_DEPTH_INTERACTION_PREDICTIVE_V1_FREEZE.md`

Canonical authority:
- workflow run: `38189687518`
- exact head: `212959a26ef15eeeaf91ec57c39dfe5306c19aa5`
- artifact: `11720818797`
- artifact digest: `sha256:cd822690c5d16c23aa2953505e79ef15b8166184c22786f63b24b6f50e0efd9f`
- workflow conclusion: `SUCCESS`

Baseline authority reproduction:
- rows: **38,258**
- max absolute gap: **1.4210854715202004e-13**
- frozen tolerance: <=1e-8
- result: **PASS**

Source-ready rows:
- **7,184**

No contamination:
- sportsbook inputs: 0
- OddsAPI calls: 0
- 2026 outcomes: 0
- fitted coefficients: 0
- threshold searches: 0
- target-game usage: false
- production change authorized: false

## WR

Candidate:
`PLAYER-DEPTH-MATCHUP-YPT-WR-V1`

Disposition:
`PLAYER_DEPTH_MATCHUP_HISTORICAL_FAIL_CLOSED`

Primary source-ready receiving-yards:

2024:
- 1,771 rows / 256 games / 177 players
- baseline MAE: **23.0985**
- candidate MAE: **23.2028**
- MAE change: **-0.1043 yards**
- baseline RMSE: 31.9795
- candidate RMSE: 32.0689
- game-cluster 95% CI: **[-0.2126, -0.0045]**
- player-cluster 95% CI: [-0.2088, +0.0002]

2025:
- 1,754 rows / 256 games / 186 players
- baseline MAE: **22.5885**
- candidate MAE: **22.6498**
- MAE change: **-0.0613 yards**
- baseline RMSE: 30.9336
- candidate RMSE: 31.0607
- game-cluster 95% CI: [-0.1747, +0.0526]
- player-cluster 95% CI: [-0.1667, +0.0401]

Broad WR protection:
- 2024 candidate produced statistically supported broad harm;
- 2025 broad MAE also worsened.

The exact individual target-depth x opponent depth-YPT multiplier does not improve
WR receiving-yard projection.

## TE

Candidate:
`PLAYER-DEPTH-MATCHUP-YPT-TE-V1`

Disposition:
`PLAYER_DEPTH_MATCHUP_HISTORICAL_FAIL_CLOSED`

Primary source-ready receiving-yards:

2024:
- 864 rows / 256 games / 89 players
- baseline MAE: **17.5767**
- candidate MAE: **17.5634**
- MAE improvement: **+0.0133 yards**
- game-cluster 95% CI: [-0.0595, +0.0867]
- player-cluster 95% CI: [-0.0798, +0.0996]
- unsupported

2025:
- 925 rows / 255 games / 98 players
- baseline MAE: **16.2503**
- candidate MAE: **16.1671**
- MAE improvement: **+0.0832 yards**
- game-cluster 95% CI: **[+0.0079, +0.1610]**
- player-cluster 95% CI: **[+0.0155, +0.1545]**
- supported improvement

Broad TE:
- 2025 also improved with supported game/player-cluster evidence;
- 2024 remained unsupported.

The frozen gate requires replicated improvement in BOTH seasons, so TE fails.

## Isolation / opportunity invariant

For both candidates:
- max receptions projection delta: **0.0**
- but the frozen untreated-projection invariant did **not** hold:
  - WR max untreated projection delta: **25.46574**
  - TE max untreated projection delta: **11.84235**

Therefore:
- WR opportunity invariant: FAIL
- TE opportunity invariant: FAIL

Even if a primary performance gate had passed, this exact implementation would
still not be promotable under the frozen contract.

## Interpretation

This result does NOT mean individual matchup context is irrelevant.

It means:
1. the public player-specific depth matchup source is real and reconstructable;
2. the exact direct YPT multiplier is not robust enough for WR;
3. TE showed a real 2025 signal but failed the required cross-season replication;
4. the candidate transmission also affected untreated projections and therefore
   failed isolation.

Do not rescue V1 by:
- changing depth bands;
- tuning multiplier bounds;
- adding a shrinkage coefficient;
- selecting deep-threat / slot / WR1 subgroups;
- combining WR and TE;
- switching post-hoc to completion-rate interaction.

The exact V1 candidates are closed.
