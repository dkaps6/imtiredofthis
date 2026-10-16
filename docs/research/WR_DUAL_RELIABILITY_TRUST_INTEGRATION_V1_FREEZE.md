# WR Dual Reliability Trust Integration V1 — Frozen Plan

Date: 2026-10-15

Status: **FROZEN BEFORE COMBINATION SCORING / RESEARCH ONLY / NO PRODUCTION CHANGE**

Version:
`WR_DUAL_RELIABILITY_TRUST_INTEGRATION_V1`

## Parent evidence

Opportunity-trust parent:
- `WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`
- run `38823063793`
- artifact `12013153664`.

Efficiency-trust parent:
- `WR_PLAYER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`
- run `38826418315`
- artifact `12015303274`
- digest `sha256:cbab5ac5b424861d7dfd1fc8e965b6ad579936eab5e53820019392e25f2b6cdf`.

The efficiency parent already proved that same-player efficiency difficulty adds
final receiving-yard reliability information after opportunity difficulty is
controlled.

This V1 asks a separate integration question:

> Does one fixed, no-fit combination of the two qualified WR reliability axes
> stratify final WR-R15 projection reliability better than the already-qualified
> opportunity-trust axis alone?

## Scientific boundary

This is a **historical integration diagnostic on already-exposed parent seasons**.

A positive result is NOT independent confirmation and cannot authorize
production or betting selection. At most it supports a separately frozen
prospective multi-week confirmation.

No Week-6 outcomes are used.

## Exact source

Use only:
`wr_efficiency_difficulty_trust_rows.csv`

from run `38826418315` / artifact `12015303274`.

Expected:
- 3,224 rows;
- 211 WRs;
- seasons exactly 2023 and 2024.

Required fields:
- season
- week
- team
- player_clean_key
- efficiency_fragility_rank
- opportunity_fragility_rank
- total_error_badness_rank
- current_abs_total_error.

The row panel must preserve exact identity and integrity.

## Frozen combination

No coefficient is fit.

`dual_fragility_score = (opportunity_fragility_rank + efficiency_fragility_rank) / 2`

Both inputs are already within-season/week percentile ranks on the same [0,1]
scale.

Do not test:
- max/min;
- geometric mean;
- learned weights;
- nonlinear transforms;
- WR1/WR2+ weights;
- season-specific weights.

The opportunity-only comparator is:
`opportunity_fragility_rank`.

The efficiency-only association is reported for context but is not the primary
baseline because opportunity trust is already the stronger qualified selector.

## Frozen outcomes / association

Primary outcome:
`total_error_badness_rank`.

Primary association:
Pearson correlation of the percentile ranks, equivalent here to the frozen
within-week rank association.

For each of 2023, 2024 and pooled, report:
- opportunity-only association;
- efficiency-only association;
- dual association;
- dual-minus-opportunity association gain.

## Frozen bucket diagnostic

Within each exact season/week, the dual score is already comparable because its
parents are within-week percentile ranks.

Define:
- DUAL_HIGH_TRUST: dual score <= 0.25
- DUAL_LOW_TRUST: dual score >= 0.75
- otherwise MIDDLE.

Report raw final absolute receiving-yard error by bucket.

Bucket results are secondary and cannot rescue a failed association gate.

## Dependence-aware inference

For `dual_association - opportunity_association`:

- 5,000 player-cluster bootstrap replicates;
- 5,000 team-week cluster bootstrap replicates using
  `season-week-team`;
- recompute both associations and their difference inside each replicate;
- fixed seeds;
- report pooled and 2024 confirmation.

## Frozen support

2023:
- >=1,200 rows;
- >=140 WRs.

2024:
- >=1,600 rows;
- >=160 WRs.

Pooled:
- >=3,000 rows;
- >=190 WRs.

## Frozen support disposition

Historical integration is supported only if ALL pass:

1. exact source/integrity;
2. support;
3. dual association > opportunity association in 2023;
4. dual association > opportunity association in 2024;
5. dual association > opportunity association pooled;
6. pooled player-cluster 95% CI lower bound for dual-minus-opportunity > 0;
7. pooled team-week cluster 95% CI lower bound > 0;
8. 2024 player-cluster 95% CI lower bound > 0;
9. 2024 team-week cluster 95% CI lower bound > 0;
10. DUAL_HIGH_TRUST raw abs error < DUAL_LOW_TRUST in 2023;
11. same ordering in 2024;
12. sportsbook inputs = 0;
13. OddsAPI calls = 0;
14. 2026 outcomes read = 0;
15. models fit = 0;
16. projection/distribution values changed = 0;
17. production unchanged.

Positive disposition:
`WR_DUAL_RELIABILITY_TRUST_HISTORICAL_INTEGRATION_SUPPORTED__PROSPECTIVE_REQUIRED`

Otherwise:
`NO_ACTIONABLE_WR_DUAL_RELIABILITY_TRUST_INTEGRATION`

Integrity failure:
`WR_DUAL_RELIABILITY_TRUST_INTEGRATION_INTEGRITY_FAILURE`.

## If supported

Freeze a future prospective confirmation before any new target-week outcomes.

The prospective study must:
- preserve opportunity and efficiency source locks unchanged;
- use this exact 50/50 formula;
- accumulate multiple future weeks / sufficient support;
- score reliability only;
- remain separate from over/under direction and sportsbook selection;
- require a new promotion decision after prospective evidence.

Do not retroactively create a Week-6 composite if the prospective combination
was not frozen before the relevant target outcomes.

## Anti-rescue

After scoring do not:
- tune the 50/50 weights;
- switch to max/min;
- change quartiles;
- rescue WR1/WR2+ subsets;
- pool seasons to waive a failed 2024 gate;
- remove opportunity as the comparator;
- fit a mean or width correction;
- use sportsbook lines;
- use 2026 outcomes.

The two individual WR trust axes remain valid even if this exact integration
fails.

No paid OddsAPI.
No production mutation.
