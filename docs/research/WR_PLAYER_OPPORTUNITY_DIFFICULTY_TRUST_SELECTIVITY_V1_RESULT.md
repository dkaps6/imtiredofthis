# WR Player Opportunity-Difficulty Trust Selectivity V1 — Result

Date: 2026-10-15

Status: **COMPLETE / WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED / RESEARCH ONLY**

Frozen contract:
- `docs/research/WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_SELECTIVITY_V1_FREEZE.md`

Canonical authority:
- run `38823063793` SUCCESS
- job `116723999254`
- exact scored head `35a10d09ba92fd1218d37e812768fda5d2f4023e`
- artifact `12013153664`
- digest `sha256:12fd0de1dc3ebceee4d356669d59f30d96776d8c0800d1fb3ac6a293af457b7c`

Parent lineage:
- WR mechanism run `37634044900`, artifact `11487359185`
- exact recovered WR-R15 authority run `37782189538`, artifact `11552447166`.

## Frozen disposition

`WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`

Every frozen gate passed.

## Exact lineage parity

The parent total-error row was proven to be the exact WR-R15 receiving-yard point error:

- matched rows: **3,224**
- players: **211**
- max signed total-error gap: **2.84e-14**
- max absolute total-error gap: **2.84e-14**
- max decomposition identity gap: **5.68e-14**

Therefore the trust test is about the final named-WR receiving-yard projection, not a proxy outcome.

## Primary result

Trust score:
within each season/week, percentile rank of the named WR's strictly-prior
opportunity difficulty
(`prior8_abs_opportunity_error_mean`).

Lower = historically more trustworthy.

2023 development:
- rows: 1,357
- WRs: 160
- primary rank association: **+0.29403**
- HIGH_TRUST absolute error: **14.37 yd**
- overall: **23.46 yd**
- LOW_TRUST: **30.50 yd**

2024 independent confirmation:
- rows: 1,867
- WRs: 183
- primary rank association: **+0.28453**
- HIGH_TRUST absolute error: **14.78 yd**
- overall: **23.09 yd**
- LOW_TRUST: **30.45 yd**

Pooled:
- rows: 3,224
- WRs: 211
- primary association: **+0.28853**
- HIGH_TRUST absolute error: **14.61 yd**
- overall: **23.25 yd**
- LOW_TRUST: **30.47 yd**

## Clustered confirmation

Pooled player cluster:
- 95% CI **[+0.24177,+0.33078]**
- P(positive) = 1.000

Pooled game cluster:
- 95% CI **[+0.25525,+0.32106]**
- P(positive) = 1.000

2024 player cluster:
- 95% CI **[+0.23014,+0.33658]**
- P(positive) = 1.000

2024 game cluster:
- 95% CI **[+0.24337,+0.32665]**
- P(positive) = 1.000

## Player-specific separation

Pooled median difficulty relative to contemporaneous WR pool:
- HIGH_TRUST: **0.470x**
- LOW_TRUST: **1.768x**

The rank is calculated within the target week, so the qualified state is not
merely identifying globally easy/hard NFL weeks.

## Role diagnostic

WR1:
- primary association +0.03185
- HIGH_TRUST sample is small (26 rows)
- descriptive only; cannot redefine the candidate.

WR2+:
- primary association +0.26885
- HIGH_TRUST abs error 14.23 yd
- LOW_TRUST 29.28 yd.

The frozen qualification is pooled player-first evidence and is not rescued by role slicing.

## Efficiency diagnostic

Prior efficiency difficulty also correlates with total projection fragility
(pooled rank association +0.30057), consistent with the earlier mixed-mechanism
result.

That does not invalidate the opportunity trust result because opportunity trust
independently cleared every prospectively frozen gate. No combined score was
fit or searched.

## Integrity

- sportsbook inputs: 0
- OddsAPI calls: 0
- 2026 outcomes: 0
- models fit: 0
- projection/distribution values changed: 0
- production changed: false

## Interpretation

This is a second clean player-level trust result, now for WR.

A named WR's own strictly-prior opportunity-projection difficulty materially
separates reliable from fragile **final WR-R15 receiving-yard projections**.

This does not authorize:
- a mean correction;
- a width formula;
- a role-specific WR1 rescue;
- an automatic fade/bet rule;
- sportsbook-conditioned trust.

The correct next step is a separately frozen prospective Week-6 WR trust shadow
using the exact same player-difficulty semantics, before reading Week-6 outcomes.

Canonical overall model status remains:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.
