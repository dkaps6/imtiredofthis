# Discrete Count Mean Alignment V1 — Result

Date: 2026-09-26

Disposition:
**DISCRETE_COUNT_MEAN_ALIGNMENT_V1_QUALIFIED_FOR_INTEGRATION_TEST**

Status:
**RESEARCH QUALIFIED — NOT YET PRODUCTION**

Branch:
`research-discrete-count-mean-alignment-v1`

Frozen plan:
`docs/research/DISCRETE_COUNT_MEAN_ALIGNMENT_V1_PLAN.md`

Mechanical amendment:
`docs/research/DISCRETE_COUNT_MEAN_ALIGNMENT_V1_MECHANICAL_AMENDMENT.md`

## Authority

Full deterministic historical reconstruction:
- run: `36275245917`
- compact artifact: `10917407106`
- compact digest: `sha256:35ad18b1a3ea96439558045a12801ea5c30a06247fce94f1ff83f6e3082c5d04`
- raw-shard artifact: `10917506697`
- raw-shard digest: `sha256:acd6bc80aaaf64872d2d3c752346d636e9480e2f2a4077f143a8fd8d5dfe2ec1`

Corrected immutable-artifact replay:
- run: `36276366140`
- head: `4aaf88c815ce050d93d8d370669b946165c53c83`
- result artifact: `10916528395`
- result digest: `sha256:0563d974f8c25de290aabec9c92665c8e509fce99c2b83c0ec10e9c5a5c69102`

No Week-3 outcomes were used.
Sportsbook inputs to football simulation: **0**.
Parameters fit: **0**.
Candidate variants scored: **1**.

## Structural contradiction confirmed

Raw Monte Carlo count distributions are integer-valued.

Current production-style multiplicative mean alignment turns those count draws into continuous fractional values whenever the final ensemble mean differs from MC.

Observed fractional-draw rate under A0:

- receptions 2024: **76.74%**
- receptions 2025: **76.03%**
- receptions pooled: **76.38%**
- rush_att 2024: **31.63%**
- rush_att 2025: **31.54%**
- rush_att pooled: **31.58%**

The frozen A1 largest-remainder projection preserved:
- nonnegative integer support;
- exact no-op semantics when production cannot mean-align;
- target sample mean within at most 0.00025 count on eligible 2,000-draw rows.

## Football-distribution result

### Receptions

| season | rows | A0 CRPS | A1 CRPS | change |
|---|---:|---:|---:|---:|
| 2024 | 4,606 | 0.997154 | **0.975917** | **-0.021237** |
| 2025 | 4,647 | 0.937276 | **0.916401** | **-0.020875** |
| pooled | 9,253 | 0.967082 | **0.946027** | **-0.021055** |

CRPS improved independently in both seasons.

Coverage moved wider:
- pooled 80% coverage: 0.8327 -> 0.9014
- pooled 90% coverage: 0.9100 -> 0.9461

Thus CRPS improved, but nominal central intervals became over-covered. Coverage was a prespecified diagnostic, not a qualification gate. This must remain visible during integration review.

### Rush attempts

| season | rows | A0 CRPS | A1 CRPS | change |
|---|---:|---:|---:|---:|
| 2024 | 5,270 | 1.037831 | **1.034776** | **-0.003056** |
| 2025 | 5,322 | 0.987607 | **0.984803** | **-0.002805** |
| pooled | 10,592 | 1.012596 | **1.009666** | **-0.002929** |

CRPS improved independently in both seasons.

Coverage also moved closer to nominal:
- pooled 80% coverage: 0.7589 -> **0.7875**
- pooled 90% coverage: 0.8046 -> **0.8226**

The rush-att effect is small but consistently favorable and passes the unchanged-season gates.

## Receptions probability result

Matched non-push archived sportsbook rows:

| season | rows | A0 Brier | A1 Brier | A0 log loss | A1 log loss |
|---|---:|---:|---:|---:|---:|
| 2024 | 2,870 | 0.270808 | **0.266672** | 0.746543 | **0.735323** |
| 2025 | 2,839 | 0.268672 | **0.264935** | 0.745385 | **0.734503** |
| pooled | 5,709 | 0.269746 | **0.265809** | 0.745967 | **0.734915** |

Pooled:
- Brier improvement: **0.003938**
- log-loss improvement: **0.011052**
- mean absolute P(over) change: **2.34 percentage points**
- rows whose >=50% side changed: **235**

Thus preserving integer count support materially changes real half-line probability calculations and improves held-out-like historical probability quality without using sportsbook information in the transform.

## Frozen gates

Mechanical:
- raw counts integer: PASS
- candidate counts integer: PASS
- candidate target-mean bound: PASS
- current fractional support observed: PASS
- zero-MC production no-op reproduced exactly: PASS
- sportsbook inputs to football: 0 — PASS

Science:
- pooled receptions CRPS improves: PASS
- pooled rush_att CRPS improves: PASS
- receptions 2024 CRPS non-worse: PASS
- receptions 2025 CRPS non-worse: PASS
- rush_att 2024 CRPS non-worse: PASS
- rush_att 2025 CRPS non-worse: PASS
- pooled receptions Brier improves: PASS
- pooled receptions log loss non-worse: PASS
- receptions 2024 Brier non-worse: PASS
- receptions 2025 Brier non-worse: PASS

All prespecified gates passed.

## Interpretation

This is not a new predictive feature.

It is a representation correction:
- simulator count outcomes begin as discrete football counts;
- final mean calibration should not destroy that count support before probabilities are computed.

The improvement is especially convincing for receptions because:
1. distribution CRPS improves in both seasons;
2. Brier improves in both seasons;
3. log loss improves in both seasons;
4. the transform has no fitted coefficient and no target-outcome access.

The rush-att gain is smaller, but directionally consistent in both seasons.

## Important separate issue

This result does **not** solve the zero-MC/nonzero-ensemble rush-att contradiction documented in:

`docs/research/RUSH_ATT_ZERO_MC_ENSEMBLE_TRANSMISSION_AUDIT_V1.md`

Rows where production cannot mean-align remain exact no-ops in both research arms.

No support was invented.

## Next authorized action

A separate production integration test may now be frozen.

It must:
- implement the exact qualified integer-preserving transform only for `receptions` and `rush_att`;
- preserve current zero-MC guard semantics;
- preserve continuous alignment bit-for-bit for all non-count markets;
- prove exact historical research-candidate parity;
- prove no football mean/weight/rule changes;
- run full repository CI;
- run the strongest available current production replay without new paid OddsAPI acquisition;
- fail closed on any unintended market drift.

Passing integration still requires an explicit production promotion decision.

No production change from this result alone.
