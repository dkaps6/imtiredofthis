# WR Phase 4C Stage 1 — Predicted Realized Pass-Volume Increment V1 — Result

Status: **CLOSED NEGATIVE UNDER FROZEN GATES — RESEARCH ONLY — NO PRODUCTION/M38/R15/THRESHOLD CHANGE**

Frozen disposition:

`NO_ACTIONABLE_PREDICTED_PASS_VOLUME_INCREMENT`

## Canonical lineage

Implementation review authority:

- GPT-5.6 implementation-review request: Issue #535 comment `5688220080`
- Claude implementation verdict: Issue #535 comment `5688276336` — `IMPLEMENTATION_REVIEW_PASS`

Sealed preflight authority:

- run `35024523296` — SUCCESS
- job `104568212642` — SUCCESS
- artifact `10418149149`
- digest `sha256:1752b8b434424f79091d9db9955c9ac91cb4216d314ef3e90af7b0b9dcb9cdea`

Single frozen blind outcome exposure:

- branch: `research-wr-phase4c-stage1-pass-volume-predictor-v1`
- one-shot orchestration commit: `8b49e35f194f8dfb6538be7148590370987f8534`
- run `35025966255` — SUCCESS
- job `104572956822` — SUCCESS
- artifact `10419147232` (`wr-phase4c-stage1-pass-volume-outcomes-v1`)
- digest `sha256:09722be047456271ab2d73132cdd9c7a46dd0e771558314aba49992e096e407d`

Exact Phase4B authority reverified immediately before exposure:

- artifact `10404525877`
- digest `sha256:0dad0dbc91a4bd6aa3e3f45cbc62e8ed493ca7bf2c95daaa0c4028a54a0fd0d4`
- artifact non-expired

Focused Stage-1 synthetics were re-run immediately before the exposure: **5/5 PASS**.

The outcome workflow executed the already-reviewed evaluator unchanged with `--run-outcomes`. No model, feature, alpha, cohort, threshold, tail definition, bootstrap rule, materiality bar, or gate was altered after blind output.

GPT-5.6 posted the exact result lineage and requested Claude's independent result audit in Issue #535 comment `5688386255`. That audit is collaboration/review only; it does not reopen the frozen gate or authorize a rescue.

## Frozen design reminder

The experiment asked whether a strictly pregame football-only prediction of realized team pass volume adds blind-2024 information beyond both the Phase4B football authority and the posted game total.

Script target:

`realized_pass_volume = plays_est * dropback_rate`

Script model:

`StandardScaler -> Ridge(alpha=20.0)`

Exactly six football-only features were used:

1. own prior-8 plays
2. own prior-8 dropback rate
3. opponent-defense prior-8 plays allowed
4. opponent-defense prior-8 dropback rate allowed
5. home flag
6. rest differential

2023 script predictions were fit on 2022 only. 2024 script predictions were fit on 2022+2023 only. Market/Vegas lineage in the script target and rolling priors remained zero.

Downstream arms:

- A0 = raw Phase4B implied team target pool
- A = intercept-calibrated football authority
- B = A + `market_total` + frozen `<38` indicator
- C = B + `pred_realized_pass_volume`

The decisive comparison was C > B, with a frozen team-target materiality floor of **0.10 targets/team-game** and 10,000 paired NFL-game-cluster bootstrap repetitions using seed `20260915`.

## Coverage and integrity

- full 2024 Layer2 domain: **515 / 515 = 100%**
- amended primary tail: **208 / 208 = 100%**
- raw Layer1 primary parent remains disclosed: 222 team-games
- no production change
- challenger/production authorization: false

The amended Layer2-domain tail counts remained exactly:

- `UNDERPROJECT_30_PLUS_OPP_DOM`: 208 addressable / 222 raw
- `ACTUAL_100_PLUS_OPP_DOM`: 85 addressable / 87 raw
- `UNDERPROJECT_30_PLUS`: 294 addressable / 312 raw

## Script predictor health — PASS

The football-only script predictor itself had real blind skill:

| Metric | Naive prior-8 product | Frozen Ridge script |
|---|---:|---:|
| 2024 MAE | 6.809459 | 6.503757 |

Naive-minus-script MAE improvement:

- mean: **+0.305702**
- paired game-cluster 95% CI: **[+0.140700, +0.478409]**
- rows: 515
- game clusters: 272

Therefore `script_blind_skill = PASS`.

This is an important narrow finding: the six-feature football-only model predicted realized pass volume better than the frozen naive comparator. The Stage-1 failure occurred downstream, not because the script predictor was completely uninformative.

## Team target-pool result

| Arm | MAE | RMSE | Bias (pred - actual) |
|---|---:|---:|---:|
| A0 | 6.556943 | 8.195612 | +2.498232 |
| A | 6.137996 | 7.805585 | -0.016431 |
| B | 6.122069 | 7.776365 | +0.219207 |
| C | **6.087435** | **7.718632** | +0.097373 |

### Decisive B -> C comparison

`MAE_B - MAE_C = +0.034634` targets/team-game.

Paired game-cluster 95% CI:

`[+0.002501, +0.066427]`

The direction is statistically positive under the frozen paired bootstrap, but the magnitude is only about one-third of the prospectively frozen **0.10** materiality floor.

Therefore:

`C_gt_B_material = FAIL`

This is not a generic null: the added pass-volume scalar carries a small incremental signal beyond B. It is simply too small under the predeclared practical-materiality requirement to qualify.

### C vs A

`MAE_A - MAE_C = +0.050561`

Paired 95% CI:

`[-0.006710, +0.107871]`

This misses both the 0.10 materiality floor and the positive-CI requirement.

Therefore:

`C_gt_A_material = FAIL`

For context, B vs A was also small:

- `MAE_A - MAE_B = +0.015927`
- 95% CI `[-0.039221, +0.070861]`

## RMSE / bias tradeoff — PASS

C did not buy its small MAE improvement by worsening the frozen guardrails:

- RMSE: B `7.776365` -> C `7.718632`
- bias: B `+0.219207` -> C `+0.097373`

Therefore:

`no_rmse_bias_tradeoff = PASS`

## Frozen WR-room translation — FAIL

| Arm | WR-room MAE | WR-room RMSE | WR-room bias |
|---|---:|---:|---:|
| B | 4.698871 | 6.183831 | -1.966398 |
| C | 4.691691 | 6.181567 | -2.031435 |

B-minus-C WR-room MAE improvement:

- mean: **+0.007181**
- paired 95% CI: **[-0.010298, +0.024511]**

The average movement is tiny and the confidence interval crosses zero.

Therefore:

`wr_room_translation_positive = FAIL`

## Frozen tail diagnostics — FAIL

### Primary: `UNDERPROJECT_30_PLUS_OPP_DOM`

- canonical/addressable rows: 208 / 208
- B-minus-C team-target MAE improvement: **-0.031991**
- paired 95% CI: **[-0.078471, +0.015393]**

C moves in the wrong direction on the primary mechanism-alignment tail.

### Secondary: `ACTUAL_100_PLUS_OPP_DOM`

- canonical/addressable rows: 85 / 85
- B-minus-C improvement: **-0.084926**
- paired 95% CI: **[-0.163585, -0.006426]**

C is worse on this secondary tail, with the entire paired interval below zero.

### Secondary: full `UNDERPROJECT_30_PLUS`

- canonical/addressable rows: 294 / 294
- B-minus-C improvement: **+0.009413**
- paired 95% CI: **[-0.032431, +0.051551]**

This slice has only a tiny unstable positive direction and cannot rescue the failed primary tail.

Therefore:

`tail_mechanism_alignment = FAIL`

## Frozen gate table

| Gate | Result |
|---|---|
| coverage_integrity | **PASS** |
| script_blind_skill | **PASS** |
| C_gt_B_material | **FAIL** |
| C_gt_A_material | **FAIL** |
| no_rmse_bias_tradeoff | **PASS** |
| wr_room_translation_positive | **FAIL** |
| tail_mechanism_alignment | **FAIL** |

All seven gates were prospectively required. Four failed.

## Disposition

The exact frozen disposition is:

`NO_ACTIONABLE_PREDICTED_PASS_VOLUME_INCREMENT`

Interpretation:

1. The football-only realized-pass-volume predictor is a legitimate predictive object; it beats the naive pass-volume baseline blind.
2. Its incremental downstream contribution beyond the Gate-0 market arm is real but too small to meet the prospectively frozen practical-materiality threshold.
3. The increment does not robustly translate through the frozen WR-room mass.
4. The primary opportunity-dominant WR underprojection tail moves in the wrong direction, and the actual-100+ opportunity-dominant tail is materially worse.
5. The frozen Stage-1 contract therefore closes this lane. There is **no Stage-2 integration authorization** from this result.
6. Do not rescue with another alpha/model family, PROE, spread, margin target, alternate market threshold, interaction, subgroup, reverse rotation, or post-hoc coefficient/weight change.
7. Preserve M38 and R15. Phase4B's conclusion that R15 is healthy/improved remains intact.
8. No production authority changes.

This result does not negate Gate 0's descriptive finding that scoring environment relates to target-pool realization/tail behavior. It says the particular prospectively frozen football-only pass-volume increment is not large or mechanism-aligned enough to become an actionable Stage-1 challenger.
