# PLAYER OPPORTUNITY SHARE STABILITY V1 — RESULT

Date: 2026-10-12

Status: **SIGNAL CONFIRMED ACROSS ALL FOUR FROZEN CELLS. NO PROJECTION OR PRODUCTION CHANGE.**

Frozen plan:
- `docs/research/PLAYER_OPPORTUNITY_SHARE_STABILITY_V1_PLAN.md`

Canonical authority:
- workflow run: `38292658912`
- exact head: `1d224c8cc5804eec110f9c9750d60d0f506bc7e4`
- artifact: `11767472231`
- artifact digest: `sha256:021a9fc79f28b4568b8663be6e57d194686b785ccbfd4c6d0a088d7e44a4f5a3`
- workflow conclusion: `SUCCESS`

Disposition:
`PLAYER_OPPORTUNITY_SHARE_STABILITY_SIGNAL_CONFIRMED`

## Integrity repair before final result

The first run was invalid because the roster side used raw GSIS IDs such as
`00-0033553` while PlayerForm historical stats used canonical identity keys such
as `gsis:00-0033553`.

That namespace mismatch caused roster-to-stat merges to miss and silently filled
targets/rushes with zero.

The final run:
- canonicalizes roster identity with `historical_identity_key`;
- adds a non-degenerate actual-share integrity gate for every cell-season;
- passes all identity/chronology gates;
- has 100% scoreable stable-identity coverage.

The invalid all-zero result is superseded and must not be used.

## Final panel

- scoreable player-games: **13,064**
- distinct players: **602**
- target seasons: 2024, 2025
- target weeks: 5-18
- prior window: latest up to 8 same-season same-team games
- minimum prior games: 4
- sportsbook inputs: 0
- OddsAPI calls: 0
- model residuals: 0
- predictive models fit: 0
- projection changes: 0
- production changed: false

## RB carry share

2024:
- rows: 1,468
- players: 137
- games: 208
- Spearman rho: **+0.5766**
- game-cluster 95% CI: **[+0.5356, +0.6166]**
- player-cluster 95% CI: **[+0.4815, +0.6566]**

2025:
- rows: 1,493
- players: 139
- games: 208
- Spearman rho: **+0.6477**
- game-cluster 95% CI: **[+0.6106, +0.6834]**
- player-cluster 95% CI: **[+0.5609, +0.7205]**

Disposition:
`PLAYER_SHARE_VOLATILITY_REPLICATED`

## RB target share

2024:
- rows: 1,468
- Spearman rho: **+0.5989**
- game-cluster 95% CI: **[+0.5566, +0.6381]**
- player-cluster 95% CI: **[+0.5116, +0.6722]**

2025:
- rows: 1,493
- Spearman rho: **+0.6662**
- game-cluster 95% CI: **[+0.6288, +0.7005]**
- player-cluster 95% CI: **[+0.5870, +0.7323]**

Disposition:
`PLAYER_SHARE_VOLATILITY_REPLICATED`

## WR target share

2024:
- rows: 2,175
- players: 215
- Spearman rho: **+0.5181**
- game-cluster 95% CI: **[+0.4810, +0.5529]**
- player-cluster 95% CI: **[+0.4388, +0.5861]**

2025:
- rows: 2,232
- players: 211
- Spearman rho: **+0.5063**
- game-cluster 95% CI: **[+0.4718, +0.5399]**
- player-cluster 95% CI: **[+0.4337, +0.5707]**

Disposition:
`PLAYER_SHARE_VOLATILITY_REPLICATED`

## TE target share

2024:
- rows: 1,364
- players: 127
- Spearman rho: **+0.6268**
- game-cluster 95% CI: **[+0.5864, +0.6629]**
- player-cluster 95% CI: **[+0.5404, +0.6967]**

2025:
- rows: 1,371
- players: 127
- Spearman rho: **+0.5634**
- game-cluster 95% CI: **[+0.5212, +0.6041]**
- player-cluster 95% CI: **[+0.4664, +0.6413]**

Disposition:
`PLAYER_SHARE_VOLATILITY_REPLICATED`

## Quartile shape

The descriptive quartiles are monotonic in every reported cell-season.

Examples:

RB carry 2025:
- Q1 prior-SD ~0.0062 -> mean next-game share surprise **0.0191**
- Q4 prior-SD ~0.1752 -> mean next-game share surprise **0.1636**

WR target 2025:
- Q1 prior-SD ~0.0126 -> mean surprise **0.0220**
- Q4 prior-SD ~0.0985 -> mean surprise **0.0854**

TE target 2025:
- Q1 prior-SD ~0.0058 -> mean surprise **0.0137**
- Q4 prior-SD ~0.0735 -> mean surprise **0.0642**

These quartiles are descriptive only and authorize no threshold rule.

## Interpretation

This is a large, replicated player-specific uncertainty signal.

A player's own strictly-prior opportunity-share volatility contains substantial
information about how unstable his next-game opportunity allocation will be.

The result is strongest precisely where the broader roadmap says the model is
most vulnerable: individual opportunity allocation.

This does NOT authorize:
- changing the mean projection;
- changing carry/target allocation;
- a volatility threshold;
- a fitted widening coefficient;
- production integration.

## Next legal move

Freeze an independent holdout confirmation before any uncertainty transform.

Preferred holdout:
- 2023 regular season Weeks 5-18;
- exact same four cells;
- exact same feature and target definitions;
- exact same support floors and clustered inference;
- no rule changes after observing 2023.

Only if the signal confirms independently should a separately frozen
mean-preserving uncertainty/distribution candidate be designed.
