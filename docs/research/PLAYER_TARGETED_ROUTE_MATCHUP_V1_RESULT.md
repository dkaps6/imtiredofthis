# Player Targeted-Route × Opponent Route-Family Matchup V1 — Result

Date: 2026-10-14

Status: **COMPLETE / PLAYER_TARGETED_ROUTE_MATCHUP_CONFIRMED / RESEARCH ONLY**

Frozen contracts:
- `docs/research/PLAYER_TARGETED_ROUTE_MATCHUP_V1_FREEZE.md`
- `docs/research/PLAYER_TARGETED_ROUTE_MATCHUP_V1_STAGE_B_FREEZE.md`

Authority:
- branch: `research-individual-opportunity-roadmap-2026-10-08`
- exact scored head: `b078b211415e278dbab5ef8930754395bb6447e9`
- workflow: `Research Player Targeted Route Matchup V1`
- run: `38630218270` — SUCCESS
- job: `116079627105`
- artifact: `11926191561`
- artifact digest: `sha256:c016bb9164fcb96a11464bf5bb3d9d9aa6f69a0d8b6633fef0dd19826e6e2b8d`

Source authority:
- Stage-A run: `38629731504`
- Stage-A disposition: `PLAYER_TARGETED_ROUTE_MATCHUP_SOURCE_READY`
- 46,666 valid WR/TE targeted-route rows
- 554 distinct WR/TE identities
- 10 route families with >=1,000 pooled rows
- 100% participation/PBP join on eligible route rows
- 93.18% stable receiver-identity coverage on taxonomy route rows
- all 32 defenses cleared the frozen pre-Week-8 support floor in 2023, 2024, and 2025

## Frozen overall disposition

`PLAYER_TARGETED_ROUTE_MATCHUP_CONFIRMED`

The confirmed signal is **WR-specific**.

Confirmed:
- WR yards per target family;
- WR catch-rate family.

Not confirmed:
- WR receiving-yard residual family;
- TE yards per target;
- TE catch rate;
- TE receiving yards.

No TE family may be rescued from the WR result.

## What was tested

For each named WR/TE target player-game:

1. build that player's own strictly-prior targeted-route mix from at most his prior 8 eligible games;
2. build the target opponent's strictly-prior allowed catch rate / receiving yards per target by route family from at most its prior 8 defensive games;
3. compare the opponent route-family performance with strictly-prior league-neutral route-family performance;
4. weight the opponent-minus-neutral deltas by **that named player's own route mix**.

Thus two teammates facing the same defense can receive different matchup values.

The source route label is used only as the route of the primary/targeted receiver on that play. It is not treated as total routes run, route participation, TPRR, or YPRR.

## WR YPT family — CONFIRMED

Primary x:
`matchup_delta_ypt`

Primary y:
next-game actual YPT minus the WR's own trailing strictly-prior YPT baseline.

Season Spearman:
- 2023 development: `+0.08160`
- 2024 independent confirmation: `+0.03259`
- 2025 independent confirmation: `+0.03493`

Pooled 2024-2025:
- rows: `3,821`
- named WRs: `234`
- Spearman: `+0.03302`

Dependence-aware support:
- player-cluster 95% CI lower: `+0.00404`
- game-cluster 95% CI lower: `+0.00175`

Air-yard/depth anti-proxy control:
- depth-controlled rho: `+0.03213`
- player-cluster depth-control CI lower: `+0.00314`
- game-cluster depth-control CI lower: `+0.00099`

Multiple comparisons:
- BH q: `0.02849` <= frozen `0.10`

Player-level diagnostics:
- qualifying WRs with >=6 confirmation games: `186`
- median per-player rho: `+0.02628`
- fraction with positive per-player rho: `53.23%`

Disposition:
`PLAYER_TARGETED_ROUTE_MATCHUP_FAMILY_CONFIRMED`

## WR catch-rate family — CONFIRMED

Primary x:
`matchup_delta_catch`

Primary y:
next-game actual catch rate minus the WR's own trailing strictly-prior catch-rate baseline.

Season Spearman:
- 2023 development: `+0.05998`
- 2024 independent confirmation: `+0.04735`
- 2025 independent confirmation: `+0.06502`

Pooled 2024-2025:
- rows: `3,821`
- named WRs: `234`
- Spearman: `+0.05541`

Dependence-aware support:
- player-cluster 95% CI lower: `+0.02285`
- game-cluster 95% CI lower: `+0.02392`

Air-yard/depth anti-proxy control:
- depth-controlled rho: `+0.05580`
- player-cluster depth-control CI lower: `+0.02340`
- game-cluster depth-control CI lower: `+0.02474`

Multiple comparisons:
- BH q: `0.00300` <= frozen `0.10`

Player-level diagnostics:
- qualifying WRs with >=6 confirmation games: `186`
- median per-player rho: `+0.06949`
- fraction with positive per-player rho: `58.06%`

Disposition:
`PLAYER_TARGETED_ROUTE_MATCHUP_FAMILY_CONFIRMED`

## WR receiving-yard family — CLOSED

Although 2023 development was positive, the signal did not preserve sign in 2025:
- 2024 rho: `+0.00693`
- 2025 rho: `-0.00656`
- pooled rho: `+0.00197`
- player/game clustered CIs crossed zero
- median per-player rho: `-0.00783`

Disposition:
`NO_ACTIONABLE_PLAYER_TARGETED_ROUTE_MATCHUP_FAMILY`

Do not use direct receiving-yard residual correction as a rescue.

## TE families — CLOSED FOR V1

TE YPT:
- 2024 rho `-0.03225`
- 2025 rho `+0.03407`
- pooled `-0.00323`

TE catch:
- 2024 rho `-0.06917`
- 2025 rho `+0.00615`
- pooled `-0.03150`

TE rec yards:
- 2024 rho `-0.01051`
- 2025 rho `+0.04856`
- pooled `+0.01805`, but clustered CIs crossed zero and player-level median rho was negative.

No TE family passed the frozen confirmation contract.

## Scientific interpretation

This is direct evidence for a player-specific WR matchup mechanism.

A WR's own historical targeted-route profile contains information about how that particular WR is likely to convert targets against a particular opponent, when combined with the opponent's strictly-prior performance against the same route families.

The result survives:
- independent 2024 and 2025 confirmation;
- repeated-player dependence;
- repeated-game dependence;
- a predeclared target-depth / air-yard control;
- six-family multiple-comparison protection.

Therefore this is not adequately described as:
- defense vs WR;
- a position average;
- generic target depth;
- generic man/zone;
- team personnel;
- WR-CB assignment.

It is:
`named WR route-role history × opponent route-family susceptibility`.

The effect is modest in raw rank-correlation magnitude, so this result does **not** authorize blindly adding the full observed opponent delta to the WR mean.

## Authorized next step

A separately frozen full-stack WR integration candidate is now authorized.

It must:
- use only the two confirmed WR signals (YPT and catch);
- leave WR target entitlement / team target mass unchanged;
- enter through the legitimate M38/WR-R15 receiving-efficiency / catch-conversion seam;
- preserve strict-prior construction;
- use no sportsbook inputs;
- test player-level outcomes in addition to aggregate safety guards;
- not use direct receiving-yard residual correction;
- not apply this WR result to TE.

## Integrity

- sportsbook inputs: 0
- OddsAPI calls: 0
- 2026 outcomes read: 0
- production changed: false
