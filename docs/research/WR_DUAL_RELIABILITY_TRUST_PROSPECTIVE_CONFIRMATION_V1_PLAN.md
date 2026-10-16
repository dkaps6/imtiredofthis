# WR Dual Reliability Trust — Prospective Confirmation V1 — Frozen Plan

Date: 2026-10-15

Status: **FROZEN BEFORE TARGET-WEEK COMPOSITE EVIDENCE / RESEARCH ONLY**

Version:
`WR_DUAL_RELIABILITY_TRUST_PROSPECTIVE_CONFIRMATION_V1`

Historical integration parent:
- `WR_DUAL_RELIABILITY_TRUST_HISTORICAL_INTEGRATION_SUPPORTED__PROSPECTIVE_REQUIRED`
- run `38827426902`
- artifact `12015229072`
- digest `sha256:2b27705844beaf9c35cf181e6fc5b7efff2c029df0255f035ba8c88bf24722c2`.

## Start boundary

Week 6 is **not** part of this composite prospective confirmation.

The component Week-6 opportunity- and efficiency-trust locks remain valid and
immutable as separate prospective evidence, but the dual integration was not
designated as a target-week composite confirmation before that slate.

Prospective dual confirmation begins with **2026 Week 7**.

No Week-6 composite label may be backfilled into this confirmation.

## Fixed horizon

Target weeks:
- 2026 Weeks **7, 8, 9, 10**.

No optional stopping.

Week 10 is the first and only scientific disposition point unless the support
gate is not met. If support is insufficient after Week 10, disposition is
`WR_DUAL_TRUST_PROSPECTIVE_INSUFFICIENT_EVIDENCE`; do not silently extend the
window after seeing results.

Any extension requires a separately frozen amendment before the added target
week begins.

## Pregame lock construction

For every target week W:

1. build the exact current WR-R15 cohort from football-only authority;
2. use only completed current-season games with week < W;
3. same player only;
4. trailing max 8 qualifying games;
5. minimum 4 prior games;
6. construct opportunity error with the already-qualified opportunity-trust
   semantics;
7. construct efficiency error with the already-qualified efficiency-trust
   semantics;
8. compute same-player MAE separately for opportunity and efficiency;
9. among supported target-week WRs, percentile-rank both MAEs with
   `method="average"`;
10. freeze:
    - `opportunity_fragility_score`
    - `efficiency_fragility_score`
    - `dual_fragility_score = (opportunity + efficiency) / 2`.

No fitted weights.

The target week's WR-R15 receiving-yard projection remains unchanged.

## Pregame integrity

Every weekly lock must prove before target-week outcomes:

- exact target week and roster/slate authority;
- history source max week <= W-1;
- zero target-week outcomes read;
- zero sportsbook input;
- zero OddsAPI calls;
- no model fit;
- no projection/distribution change;
- production unchanged;
- immutable row CSV digest;
- exact source/run/SHA provenance.

If any target week cannot be frozen pregame, that week contributes zero rows and
the final support gate must still be met without replacement.

## Frozen outcome scoring

After each target week settles, join realized receiving yards only for scoring.

For each supported locked row:
- `abs_error = abs(frozen_WR_R15_projection - actual_receiving_yards)`;
- within target week:
  `error_badness_rank = average percentile rank(abs_error)`.

Primary association per week and pooled:
- opportunity:
  `corr(opportunity_fragility_score, error_badness_rank)`;
- efficiency:
  `corr(efficiency_fragility_score, error_badness_rank)`;
- dual:
  `corr(dual_fragility_score, error_badness_rank)`.

Primary incremental endpoint:
`dual_association - opportunity_association`.

## Frozen bucket diagnostic

- DUAL_HIGH_TRUST <= 0.25
- DUAL_LOW_TRUST >= 0.75
- otherwise MIDDLE.

Report raw receiving-yard absolute error by bucket.

No bucket-based rescue if the primary incremental endpoint fails.

## Fixed support gates

At final Week-10 scoring require:
- >=400 supported locked WR player-games;
- >=120 unique WR identities;
- >=40 distinct NFL games represented;
- >=80 DUAL_HIGH_TRUST rows;
- >=80 DUAL_LOW_TRUST rows;
- at least 3 of 4 target weeks contribute >=80 supported rows.

Failure => `WR_DUAL_TRUST_PROSPECTIVE_INSUFFICIENT_EVIDENCE`.

## Dependence-aware inference

At final scoring:
- 10,000 player-cluster bootstrap replicates;
- 10,000 game-cluster bootstrap replicates;
- paired resampling;
- recompute opportunity and dual associations inside each replicate;
- report 95% CI for dual-minus-opportunity;
- fixed seeds.

## Frozen confirmation gates

Prospective confirmation requires ALL:

1. support passes;
2. pooled dual association > 0;
3. pooled dual association > pooled opportunity association;
4. dual-minus-opportunity player-cluster 95% CI lower bound > 0;
5. dual-minus-opportunity game-cluster 95% CI lower bound > 0;
6. dual association > opportunity association in at least 3 of 4 target weeks;
7. DUAL_HIGH_TRUST pooled raw abs error < DUAL_LOW_TRUST pooled raw abs error;
8. same high<low ordering in at least 3 of 4 target weeks;
9. no target-week chronology violation;
10. sportsbook inputs = 0;
11. OddsAPI calls = 0;
12. projection/distribution values changed = 0;
13. production unchanged.

PASS:
`WR_DUAL_RELIABILITY_TRUST_PROSPECTIVE_CONFIRMED`

Scientific fail:
`NO_FORWARD_CONFIRMATION_WR_DUAL_RELIABILITY_TRUST`

Insufficient support:
`WR_DUAL_TRUST_PROSPECTIVE_INSUFFICIENT_EVIDENCE`

Integrity failure:
`WR_DUAL_TRUST_PROSPECTIVE_INTEGRITY_FAILURE`.

## Promotion boundary

A PASS authorizes only a separate production/reliability-annotation design.

It does not automatically:
- change WR-R15 mean;
- change width;
- create over/under direction;
- choose bets;
- feed sportsbook data upstream.

No paid OddsAPI.
No production mutation.
