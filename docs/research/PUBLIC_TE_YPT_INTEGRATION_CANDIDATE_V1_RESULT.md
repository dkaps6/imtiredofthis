# Public TE YPT Integration Candidate V1 — Result

**STATUS: COMPLETE / PUB-TEY1 CLOSED**

Frozen contract:
`docs/research/PUBLIC_TE_YPT_INTEGRATION_CANDIDATE_V1_CONTRACT.md`

Authority:
- branch: `research-public-te-ypt-integration-candidate-v1`
- certified run: `37546447966` — **SUCCESS**
- source SHA: `cc2f0e88d7f23957958cb4859461fd6c9c429874`
- artifact: `11450274034`
- digest: `sha256:747e291262eb02776065c4d5f49a4a5d845b41a5c0f8c71f42d3089b63cf1f95`
- sportsbook inputs used: **0**
- 2026 outcomes read: **0**
- WR candidate scored: **false**
- RB candidate scored: **false**
- post-hoc rescue scored: **false**
- production changed: **false**

Final disposition:

`PUB_TEY1_CLOSED`

## Candidate

Feature:
`public_def_te_ypt_allowed`

Form:
`baseline + beta_2022 * week_z(public_def_te_ypt_allowed)`

2022 fit:
- rows: 1,007
- games: 255
- beta: `1.4534302762213023`
- support: PASS
- sign: positive / PASS

2022 descriptive fit:
- MAE: `15.658523 -> 15.579829`
- RMSE: `22.012424 -> 21.966373`
- 75+ AE: `7 -> 6`
- 100+ AE: `2 -> 2`
- residual Spearman: `0.059561 -> -0.025191`

## Primary independent 2023 confirmation

Rows: 978  
Games: 256

Baseline:
- MAE: `16.163428`
- RMSE: `22.785872`
- bias: `-4.832258`
- 75+ AE: `14`
- 100+ AE: `6`
- residual Spearman: `-0.027685`

Candidate:
- MAE: `16.252789`
- RMSE: `22.868233`
- bias: `-4.808359`
- 75+ AE: `15`
- 100+ AE: `5`
- residual Spearman: `-0.117243`

Primary MAE change:
- **-0.089361 worse**

Game-cluster bootstrap:
- 5,000 replicates
- P(MAE improves): `0.0268`
- 95% CI: `[-0.180263, 0.001318]`

Primary failed:
- MAE improvement
- RMSE non-worse
- bootstrap probability >= 0.80
- 75+ AE non-worse
- residual-Spearman reduction

100+ AE was non-worse, but that cannot rescue the failed primary gate.

## Secondary 2024/2025 consistency

The frozen 2022 coefficient was unchanged.

2024:
- MAE: `16.464566 -> 16.432064`
- improvement: `+0.032502`
- tails non-worse

2025:
- MAE: `15.644316 -> 15.620594`
- improvement: `+0.023722`
- tails non-worse

Pooled 2024/2025:
- MAE: `16.049295 -> 16.021238`
- improvement: `+0.028057`
- RMSE: `22.906943 -> 22.859713`
- 75+ AE: `27 -> 27`
- 100+ AE: `5 -> 5`

Those are consistency observations only. Because 2024/2025 were the discovery seasons, they cannot rescue a failed independent 2023 confirmation.

## Integrity

- 2022/2023 feature chronology violations: 0
- position identity missingness: 0
- 2024 baseline authority: `FROZEN_RIGHT_TAIL_FINAL_MEAN`
- 2025 baseline authority: `FROZEN_RIGHT_TAIL_FINAL_MEAN`
- public YPT source-parity result remains valid
- no sportsbook data
- no 2026 outcome use
- no production mutation

## Interpretation

The free public TE YPT source is valid and live-ready, and the 2024/2025 residual association was not fabricated.

What failed is the exact fixed additive mean-transmission mechanism.

Do not rescue with:
- TE1/high-target subsets;
- alternate history windows;
- raw YPT instead of z-score;
- caps/nonlinear transforms;
- Sharp/public blends;
- `def_pass_success_allowed` combinations;
- route/snap subgroups;
- 2024/2025 refitting;
- 2026 outcome inspection.

The public YPT source remains usable for future genuinely distinct hypotheses, but `PUB-TEY1` is closed.

No production change is authorized.
