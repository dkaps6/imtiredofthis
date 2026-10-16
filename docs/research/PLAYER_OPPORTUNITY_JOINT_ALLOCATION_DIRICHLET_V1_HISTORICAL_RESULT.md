# PLAYER OPPORTUNITY JOINT ALLOCATION — DIRICHLET V1 HISTORICAL RESULT

Date: 2026-10-13

Status: **HISTORICAL PASS — RB CARRY SHARE AND WR TARGET SHARE PASS IN 2023, 2024, AND 2025. FORWARD SHADOW AUTHORIZED. NO PRODUCTION CHANGE.**

Frozen contract:
- `docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_HISTORICAL_FREEZE.md`

Scientific parents:
- `docs/research/PLAYER_OPPORTUNITY_SHARE_VOLATILITY_MECHANISM_V2_RESULT.md`
- `docs/research/PLAYER_OPPORTUNITY_ALLOCATION_UNCERTAINTY_BETABINOMIAL_V1_RESULT.md`
- `docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_PLAN.md`
- `docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_MECHANICS_RESULT.md`

Canonical authority:
- workflow run: `38401069806`
- exact scoring head: `5f35f96f0a43b9963691a7768a330b32b4854a5b`
- artifact: `11815490905`
- artifact digest: `sha256:2b336114dc5ef0145c0a40706266da850a0ff1d25c8935958481b996333fce29`
- workflow conclusion: `SUCCESS`

Overall disposition:
`JOINT_ALLOCATION_DIRICHLET_HAS_PASSING_CELLS`

Passing cells:
- `RB_CARRY_SHARE`
- `WR_TARGET_SHARE`

Both receive:
`JOINT_ALLOCATION_DIRICHLET_HISTORICAL_PASS_FREEZE_FORWARD_SHADOW`

## Integrity

- target seasons: 2023, 2024, 2025
- target weeks: 5-18
- scored joint team-game rooms: **2,354**
- scored eligible player marginals: **6,279**
- minimum canonical-q identity match rate: **100.0%**
- max room expected-count gap: **0.0**
- max marginal expected-count gap: **0.0**
- sportsbook inputs: 0
- OddsAPI calls: 0
- fitted coefficients: 0
- threshold searches: 0
- 2026 outcomes: 0
- production changed: false

The candidate uses the exact canonical historical model probabilities:
- RB: `rules_rush_share` after the canonical top-5 rushing pool and 0.95
  player-mass/residual semantics;
- WR: `rules_tgt_share` after canonical M38 sharpening and the same 0.95
  player-mass/residual semantics.

The candidate changes uncertainty only. Expected player counts and team totals
are unchanged.

## RB carry share

### 2023
- candidate-applied rooms: **373**
- authoritative games: **206**
- eligible players: **107**
- mean joint log-score improvement: **+3.42013**
- game-cluster 95% CI: **[+2.94566, +3.90988]**
- team-cluster 95% CI: **[+2.69551, +4.19594]**
- marginal player CRPS improvement: **+0.45697**
- marginal CRPS game-cluster CI low: **+0.40386**
- historical pass: **true**

### 2024
- rooms: **400**
- games: **208**
- eligible players: **118**
- mean joint log-score improvement: **+3.08394**
- game-cluster 95% CI: **[+2.72594, +3.46330]**
- team-cluster 95% CI: **[+2.45067, +3.74149]**
- marginal player CRPS improvement: **+0.36223**
- marginal CRPS game-cluster CI low: **+0.31041**
- historical pass: **true**

### 2025
- rooms: **387**
- games: **206**
- eligible players: **104**
- mean joint log-score improvement: **+2.88473**
- game-cluster 95% CI: **[+2.49467, +3.27033]**
- team-cluster 95% CI: **[+2.38669, +3.40813]**
- marginal player CRPS improvement: **+0.32082**
- marginal CRPS game-cluster CI low: **+0.27871**
- historical pass: **true**

Disposition:
`JOINT_ALLOCATION_DIRICHLET_HISTORICAL_PASS_FREEZE_FORWARD_SHADOW`

## WR target share

### 2023
- rooms: **391**
- games: **207**
- eligible players: **169**
- mean joint log-score improvement: **+0.78218**
- game-cluster 95% CI: **[+0.63155, +0.93797]**
- team-cluster 95% CI: **[+0.57994, +0.98635]**
- marginal player CRPS improvement: **+0.05301**
- marginal CRPS game-cluster CI low: **+0.04128**
- historical pass: **true**

### 2024
- rooms: **401**
- games: **207**
- eligible players: **169**
- mean joint log-score improvement: **+1.04293**
- game-cluster 95% CI: **[+0.83940, +1.26814]**
- team-cluster 95% CI: **[+0.72820, +1.37987]**
- marginal player CRPS improvement: **+0.07221**
- historical pass: **true**

### 2025
- rooms: **402**
- games: **207**
- eligible players: **179**
- mean joint log-score improvement: **+1.04630**
- game-cluster 95% CI: **[+0.85871, +1.24904]**
- team-cluster 95% CI: **[+0.78508, +1.31141]**
- marginal player CRPS improvement: **+0.06805**
- historical pass: **true**

Disposition:
`JOINT_ALLOCATION_DIRICHLET_HISTORICAL_PASS_FREEZE_FORWARD_SHADOW`

## Scientific interpretation

The current fixed-share multinomial allocation is too narrow for:
- individual RB carry allocation;
- individual WR target allocation.

The strictly-prior player-specific excess share variance identified by the
three-season volatility program improves both:
1. the joint team-room probability distribution; and
2. the eligible player's marginal count distribution.

This is not a new mean model.

For every supported player/category:
- canonical mean allocation remains unchanged;
- finite team opportunity volume remains conserved exactly;
- only latent role/allocation uncertainty changes.

This directly addresses a Week-5 failure family without outcome-tuning:
the model was often too certain about individual workload/role allocation.

## Boundaries

This result does NOT authorize:
- production promotion;
- changing RB/WR mean opportunity;
- RB target-share widening;
- TE target-share widening;
- fitted rho shrinkage;
- volatility buckets;
- role/subgroup carveouts;
- any sportsbook/model feedback.

RB target and TE target remain outside this candidate family.

## Next legal move

Freeze a 2026 prospective joint-allocation shadow for:
- RB carry share;
- WR target share.

The shadow must:
- use exact target-week player identity;
- consume strictly-prior excess-variance fields only;
- use canonical live player means after all existing mean rules;
- conserve exact simulated team opportunities;
- preserve expected player counts;
- use a semantically isolated allocation RNG stream;
- never mutate production pricing/board output;
- remain fail-closed while Week-6 roster authority is unavailable.

Prospective grading must not begin until the pregame shadow state is locked.
