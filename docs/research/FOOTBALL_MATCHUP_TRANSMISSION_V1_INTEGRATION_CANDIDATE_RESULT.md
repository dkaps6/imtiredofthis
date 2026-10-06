# Football Matchup Transmission V1 — Integration Candidate Result

**STATUS: COMPLETE / ALL SIMPLE TRANSMISSION CANDIDATES CLOSED**

Authority:

- branch: `research-football-matchup-integration-candidates-v1`
- certified run: `37531333007` — **SUCCESS**
- result artifact: `11445048026`
- artifact digest: `sha256:04cbe2d1c82395db35a47d093b89e0ba8398346552869538621c5e504a4c069d`
- corrected-input artifact: `11444039829`
- corrected-input digest: `sha256:fb7213ec0589dcd4e8f0a2bdcd25bcd25ccd3ce6d3e52395670b61f11411ec13`
- source SHA: `f8cef9c22b4df2fdcb57fe4010465bbe259ab96d`
- parent 2024/2025 baseline parity: **PASS**, max gap `0.0` across 8,695 candidate rows
- sportsbook inputs used: **0**
- production changed: **false**
- combined candidate scored: **false**

Frozen parent contract:
`docs/research/FOOTBALL_MATCHUP_TRANSMISSION_V1_INTEGRATION_CANDIDATE_CONTRACT.md`

## Result

The Phase B/C residual associations were real in 2024 and 2025, but none of the three predeclared simple additive transmission mechanisms survived the independent 2023 confirmation gate.

Final disposition:

`CLOSE_ALL_INTEGRATION_CANDIDATES`

No coefficient, cohort, threshold, sign, interaction, or support gate may be retuned after this result.

## FMT-RB1 — CLOSED

Mechanism:
RB rush yards adjusted by lower opponent pass-rate-faced.

2022 frozen coefficient:
- beta: `2.638160693422653`
- rows: 1,373
- games: 255

Primary 2023 confirmation:
- baseline MAE: `19.746704`
- candidate MAE: `19.767445`
- MAE change: **-0.020741 worse**
- baseline RMSE: `28.848084`
- candidate RMSE: `28.847505`
- 75+ AE count: `39 -> 40`
- 100+ AE count: `15 -> 15`
- game-cluster P(MAE improvement): `0.3818`
- residual Spearman: `0.066163 -> -0.059223`

The frozen primary gate failed on MAE, bootstrap support, and 75+ yard tail count.

Secondary 2024/2025 consistency was directionally favorable:
- pooled MAE: `20.980530 -> 20.927400`
- 75+ AE: `100 -> 99`
- 100+ AE: `39 -> 36`

That cannot rescue a failed independent 2023 confirmation.

Disposition:
`INTEGRATION_CANDIDATE_CLOSED`

## FMT-WR1 — CLOSED

Mechanism:
WR receiving yards adjusted by offensive true PROE.

2022 frozen coefficient:
- beta: `2.860551301800433`
- rows: 1,935
- games: 255

Primary 2023 confirmation:
- baseline MAE: `22.650536`
- candidate MAE: `22.817144`
- MAE change: **-0.166608 worse**
- baseline RMSE: `31.937109`
- candidate RMSE: `32.003742`
- 75+ AE: `77 -> 77`
- 100+ AE: `25 -> 26`
- game-cluster P(MAE improvement): `0.0050`
- residual Spearman: `0.011352 -> -0.109207`

Secondary 2024/2025 pooled MAE also worsened:
- `22.034008 -> 22.042370`
- 75+ AE: `121 -> 124`

Disposition:
`INTEGRATION_CANDIDATE_CLOSED`

## FMT-TE1 — CLOSED

Mechanism:
TE receiving yards adjusted by opponent defensive pass-success allowed.

2022 frozen coefficient:
- beta: `0.5591005620979614`
- rows: 1,007
- games: 255

Primary 2023 confirmation:
- baseline MAE: `16.163428`
- candidate MAE: `16.177921`
- MAE change: **-0.014492 worse**
- baseline RMSE: `22.785872`
- candidate RMSE: `22.794175`
- 75+ AE: `14 -> 15`
- 100+ AE: `6 -> 6`
- game-cluster P(MAE improvement): `0.2042`
- residual Spearman: `0.009505 -> -0.023131`

Secondary 2024/2025 consistency was favorable:
- pooled MAE: `16.049295 -> 16.020314`
- tail counts non-worse

That cannot rescue the failed independent 2023 confirmation.

Disposition:
`INTEGRATION_CANDIDATE_CLOSED`

## Interpretation

The correct football conclusion is narrow:

1. The 2024/2025 residual associations identified in Phase B/C were not fabricated.
2. A one-feature, fixed additive yardage correction is not a sufficiently stable transmission mechanism.
3. The result does **not** prove matchup information is useless.
4. It does prove these three simple integration forms are not production candidates.
5. Do not reinterpret the hardcoded generic `0.57` pass share as authorization to reopen M16-M21, M40-M42, or M64-M65. Dynamic pass-rate / PROE / score-state repackaging is already closed by the anti-retest ledger.
6. Future football-layer work requires materially new pregame information, not another transformation of the same historical tendency fields.

## Next research boundary

The strongest genuinely new football-information families presently identified are prospective personnel/formation sources, especially private GSIS Lineup Detail / Formation Usage, because they contain exact lineup co-occurrence / personnel-state information not represented by marginal historical tendency descriptors.

Those sources remain governed by their own prospective contracts. In particular, GSIS RB Successor Lineup V1 remains sealed until its predeclared support floor is met; Week-4 alone must not be graded early.

No paid OddsAPI pull is required or authorized by this result.
