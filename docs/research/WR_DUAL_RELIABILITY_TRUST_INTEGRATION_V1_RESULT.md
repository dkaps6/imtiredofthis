# WR Dual Reliability Trust Integration V1 — Result

Date: 2026-10-15

Status: **COMPLETE / HISTORICAL INTEGRATION SUPPORTED / PROSPECTIVE CONFIRMATION REQUIRED**

Frozen plan:
- `docs/research/WR_DUAL_RELIABILITY_TRUST_INTEGRATION_V1_FREEZE.md`

Canonical authority:
- workflow: `Research WR Dual Reliability Trust Integration V1`
- run: `38827426902` — **SUCCESS**
- job: `116738041310`
- exact scored head: `ca4da46fe07341900263dacc8602a7aef002ed03`
- artifact: `12015229072`
- digest: `sha256:2b27705844beaf9c35cf181e6fc5b7efff2c029df0255f035ba8c88bf24722c2`

Disposition:
`WR_DUAL_RELIABILITY_TRUST_HISTORICAL_INTEGRATION_SUPPORTED__PROSPECTIVE_REQUIRED`

## Question

Two WR reliability axes had already qualified separately around the unchanged
WR-R15 receiving-yard projection:

1. same-player opportunity difficulty;
2. same-player efficiency difficulty.

This V1 froze one no-fit integration before combination scoring:

`dual_fragility_score = (opportunity_fragility_rank + efficiency_fragility_rank) / 2`

The comparator was the already-qualified opportunity-trust axis.

No coefficient, threshold, player subgroup, model mean, distribution width, or
sportsbook input was fit.

## 2023 development

- rows: **1,357**
- WRs: **160**
- opportunity-only association: **+0.29403**
- efficiency-only association: **+0.30882**
- dual association: **+0.33308**
- dual minus opportunity: **+0.03905**
- DUAL_HIGH_TRUST final abs error: **13.08 yd**
- DUAL_LOW_TRUST final abs error: **32.83 yd**

## 2024 confirmation

- rows: **1,867**
- WRs: **183**
- opportunity-only association: **+0.28453**
- efficiency-only association: **+0.29458**
- dual association: **+0.32334**
- dual minus opportunity: **+0.03881**
- DUAL_HIGH_TRUST final abs error: **14.19 yd**
- DUAL_LOW_TRUST final abs error: **30.36 yd**

Dependence-aware gain inference:
- player-cluster 95% CI: **[+0.01875,+0.05956]**
- team-week cluster 95% CI: **[+0.01973,+0.05795]**
- P(gain > 0): **1.000** for both.

## Pooled

- rows: **3,224**
- WRs: **211**
- opportunity-only association: **+0.28853**
- efficiency-only association: **+0.30057**
- dual association: **+0.32746**
- dual minus opportunity: **+0.03893**
- DUAL_HIGH_TRUST final abs error: **13.73 yd**
- DUAL_LOW_TRUST final abs error: **31.49 yd**

Pooled dependence-aware gain inference:
- player-cluster 95% CI: **[+0.02314,+0.05494]**
- team-week cluster 95% CI: **[+0.02427,+0.05342]**
- P(gain > 0): **1.000** for both.

Every frozen gate passed.

## Interpretation

The two qualified named-WR reliability dimensions are not redundant.

A fixed 50/50 combination improves historical reliability stratification over
the already-strong opportunity-only trust axis.

This is useful player-level evidence because it says the same WR can be hard to
project for two separable reasons:
- his target/opportunity allocation is hard to predict;
- his yardage conversion/efficiency is hard to predict.

The combined state identifies final WR-R15 projection reliability better than
opportunity difficulty alone.

## Scientific boundary

This is an integration study on already-exposed historical parent seasons.

It is **not independent prospective confirmation** and does not authorize:
- production promotion;
- WR-R15 mean changes;
- width changes;
- over/under direction;
- automatic bet/fade;
- sportsbook-conditioned selection.

No Week-6 outcomes were read.

The separately frozen Week-6 opportunity-trust and efficiency-trust labels
remain immutable and separate. Do not retroactively manufacture a Week-6
composite after target-week exposure.

## Next

Freeze a multi-week prospective confirmation beginning on the next genuinely
untouched future slate.

The prospective candidate must use:
- the exact same opportunity-trust construction;
- the exact same efficiency-trust construction;
- the exact fixed 50/50 formula;
- no fitted weights;
- sufficient player-game and player support;
- dependence-aware inference;
- unchanged WR-R15 means/distributions.

A production decision can occur only after that prospective evidence.

Canonical model status remains:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.

No paid OddsAPI.
No production mutation.
