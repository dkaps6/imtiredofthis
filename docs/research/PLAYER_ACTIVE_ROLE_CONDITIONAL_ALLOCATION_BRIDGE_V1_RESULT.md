# PRISM Player Active-Role -> Conditional Allocation Bridge V1 — Result

Date: 2026-10-16

Status: **HISTORICAL QUALIFICATION PARTIAL — RB RUSHING, RB RECEIVING, AND TE RECEIVING QUALIFIED; QB PASSING CLOSED; WR RECEIVING NOT SCORED. NO PRODUCTION CHANGE.**

Version:
`PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_V1_MECHANISM`

PRISM:
**Player Role & Individual Situation Model**

Scientific order remains:

`PLAYER IDENTITY -> CURRENT ROLE/STATE -> TEAMMATE COMPETITION -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

This is an individual-player result. Position aggregates are downstream diagnostics.

## Canonical authority

Frozen before scoring:
- mechanism freeze commit `87400a9c9c1db465c9c4bd70b0cb241dba27c6c9`;
- contract `docs/research/PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_V1_MECHANISM_FREEZE.md`.

Qualified source parent:
- run `38955368643` — SUCCESS;
- exact head `cb65a634a0e09296bfeabc94ffe1e77363a30f8a`;
- artifact `12070806828`;
- digest `sha256:e91a108495d7a2b6eb289e0618ab6af09e3d6cdf6895e534742a07340e4a2a83`.

Canonical scored authority:
- run `38957575645` — SUCCESS;
- job `117164201650`;
- exact scored head `eba5787c0b06aa5f975008965df7d023b3525ba7`;
- artifact `12070774378`;
- digest `sha256:e07edbad1bf97a4917ba95c58156c8941448e7b2668bd0eae2e9bcad5d4945b0`.

Overall disposition:

`PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_PARTIAL`

## Frozen mechanism

For every source-ready cell, the experiment preserved the existing canonical pregame entitlement `q_i` and consumed the already-qualified Stage-1 `p_active_i`.

For canonical-positive named players:

`A_i ~ Bernoulli(p_active_i)`.

The candidate exactly enumerates every active-set state, conditions the existing canonical entitlement on the active set, and renormalizes only among active canonical-positive players plus the existing anonymous complement/residual.

No share coefficient is fit.
No activity threshold is fit.
No team opportunity volume is changed.
Canonical-zero named-player entitlement remains zero.
The degenerate all-off state assigns finite opportunity to the anonymous complement rather than inventing a backup entitlement.

Primary score:
named-player finite-count CRPS.

Protection gates:
- expected opportunity absolute error;
- room-level allocation error;
- player-cluster, game-cluster and equal-player inference;
- exact probability/conservation integrity.

## Qualified — RB rushing

Disposition:

`RB_RUSHING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED`

2024:
- 1,386 player-games / 156 players / 416 team rooms;
- Stage-1 named-player support: **91.41%**;
- CRPS improvement: **+0.401928**;
- expected-count AE improvement: **+0.260308**;
- room-allocation improvement: **+0.005876**;
- player-cluster 95% CI low: **+0.304077**;
- game-cluster 95% CI low: **+0.355667**;
- equal-player CI low: **+0.299131**.

2025 independent confirmation:
- 1,419 player-games / 154 players / 416 team rooms;
- Stage-1 named-player support: **90.42%**;
- CRPS improvement: **+0.385893**;
- expected-count AE improvement: **+0.267115**;
- room-allocation improvement: **+0.008439**;
- player-cluster 95% CI low: **+0.297996**;
- game-cluster 95% CI low: **+0.333263**;
- equal-player CI low: **+0.277780**;
- **86.78%** of evaluable players improved on mean CRPS.

This is the strongest Stage-2 bridge result.

## Qualified — RB receiving

Disposition:

`RB_RECEIVING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED`

2024:
- 1,386 player-games / 156 players / 416 team rooms;
- CRPS improvement: **+0.040338**;
- expected-count AE improvement: **+0.054140**;
- room-allocation improvement: **+0.001571**;
- player-cluster 95% CI low: **+0.021434**;
- game-cluster 95% CI low: **+0.032162**;
- equal-player CI low: **+0.018504**.

2025 independent confirmation:
- 1,419 player-games / 154 players / 416 team rooms;
- CRPS improvement: **+0.034723**;
- expected-count AE improvement: **+0.050076**;
- room-allocation improvement: **+0.000975**;
- player-cluster 95% CI low: **+0.019115**;
- game-cluster 95% CI low: **+0.027288**;
- equal-player CI low: **+0.015982**;
- **47.11%** of evaluable players improved on mean CRPS.

The 2025 positive equal-player mean and bootstrap interval pass even though fewer than half of evaluable players improved. Preserve this heterogeneity; do not claim universal player benefit.

## Qualified — TE receiving

Disposition:

`TE_RECEIVING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED`

2024:
- 1,334 player-games / 138 players / 416 team rooms;
- CRPS improvement: **+0.025933**;
- expected-count AE improvement: **+0.026589**;
- room-allocation improvement: **+0.001186**;
- player-cluster 95% CI low: **+0.014466**;
- game-cluster 95% CI low: **+0.019696**;
- equal-player CI low: **+0.014866**.

2025 independent confirmation:
- 1,305 player-games / 141 players / 416 team rooms;
- CRPS improvement: **+0.018198**;
- expected-count AE improvement: **+0.018672**;
- room-allocation improvement: **+0.000395**;
- player-cluster 95% CI low: **+0.009962**;
- game-cluster 95% CI low: **+0.013674**;
- equal-player CI low: **+0.008875**;
- **57.89%** of evaluable players improved on mean CRPS.

## Closed — QB passing

Disposition:

`QB_PASSING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_CLOSED`

The exact bridge improved distribution CRPS at the pooled level but failed the frozen player-mean and room-allocation protection gates and failed independent 2025 inference.

2024:
- CRPS improvement: **+0.406242**;
- expected-count AE change: **-1.092245** — worse;
- room-allocation change: **-0.064128** — worse.

2025:
- CRPS improvement: **+0.187433**;
- expected-count AE change: **-1.273197** — worse;
- room-allocation change: **-0.076277** — worse;
- player-cluster CI low: **-0.128325**;
- game-cluster CI low: **-0.010985**;
- equal-player CI low: **-0.111769**;
- only **24.29%** of evaluable players improved.

Do not rescue this V1 QB bridge with:
- activity thresholds;
- correlation;
- starter carveouts;
- backup reassignment;
- new pass-share coefficients;
- pooled evidence.

QB remains protected under existing M89/M90/C2 and the previously closed same-data opportunity families.

## WR receiving — not scored

WR receiving remained outside V1 because the frozen source audit could not establish a legitimate 2025 WR-R15 OOS authority.

2024 source readiness cannot rescue the missing 2025 independent authority.

Do not relabel the WR-R15 final fit as OOS and do not score 2024 alone.

## Integrity

Canonical result:
- target-game outcomes used as predictors: **0**;
- new share models fit: **0**;
- threshold searches: **0**;
- sportsbook inputs: **0**;
- OddsAPI calls: **0**;
- 2026 outcomes: **0**;
- production changed: **false**;
- WR receiving scored: **false**.

Numerical integrity:
- max active-state weight gap <= floating-point tolerance;
- max candidate mass gap <= floating-point tolerance;
- finite team opportunity conserved in every state.

## Scientific meaning

PRISM now has historical evidence for a two-stage named-player opportunity architecture in three cells:

1. estimate whether this named player participates from his own strictly-prior same-team state;
2. condition the existing canonical entitlement on the probabilistic active set;
3. preserve teammate competition and finite team opportunity;
4. score the resulting opportunity distribution at the individual-player level.

Historical qualification exists for:
- RB rushing opportunity;
- RB receiving opportunity;
- TE receiving opportunity.

This is not production promotion.

The result does not authorize automatic deployment into the 2026 Full Slate without a separately frozen prospective/deployment contract.

## Anti-retest / next frontier

Do not reopen:
- the exact QB bridge;
- WR without a legitimate 2025 OOS parent;
- simple high-value-opportunity persistence;
- role-elasticity V1 source gate;
- generic recent-share allocators;
- generic position-average optimization.

The next research choice must reconcile against the PRISM architecture and existing terminal lanes before scoring anything new.

A legitimate next layer should come after qualified opportunity quantity and should be selected from:
- opportunity quality/topology;
- player-specific efficiency under relevant football conditions;
- opponent/environment transmission;
- projection reliability/selectivity;
- market-specific mapping.

Any new candidate must be genuinely distinct from the closed high-value persistence, generic matchup, role-elasticity and efficiency families.

No sportsbook.
No OddsAPI.
No 2026 outcomes.
No production mutation.
