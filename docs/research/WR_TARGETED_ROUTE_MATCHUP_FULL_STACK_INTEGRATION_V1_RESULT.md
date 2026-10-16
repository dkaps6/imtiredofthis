# WR Targeted-Route Matchup Full-Stack Integration V1 — Result

Date: 2026-10-14

Status: **COMPLETE / NO_ACTIONABLE_WR_ROUTE_MATCHUP_FULL_STACK_INTEGRATION / RESEARCH ONLY**

Frozen contract:
- `docs/research/WR_TARGETED_ROUTE_MATCHUP_FULL_STACK_INTEGRATION_V1_FREEZE.md`

Stage-B authority:
- disposition: `PLAYER_TARGETED_ROUTE_MATCHUP_CONFIRMED`
- canonical Stage-B run: `38641891257`
- job: `116119186261`
- exact Stage-B scored head: `fe1d8ff52bd7cfa322e095a7a6f58752acb0b55f`
- artifact: `11929999088`
- digest: `sha256:0ff2c393e0daa0fa5163d34ce35e38eb25fb7c907cdf287019d609b804ca5128`

Full-stack authority:
- branch: `research-individual-opportunity-roadmap-2026-10-08`
- exact scored head: `bdbc132e35939133e3fb9ee4fca1e85ff1857561`
- workflow: `Research WR Targeted Route Matchup Full Stack V1`
- run: `38642241032` — SUCCESS
- job: `116120366630`
- artifact: `11930768340`
- artifact digest: `sha256:a9a0198efaca76ea8967813842468894b01c5d130781f2bb5b4c71881bfc36d3`

## Frozen overall disposition

`NO_ACTIONABLE_WR_ROUTE_MATCHUP_FULL_STACK_INTEGRATION`

The underlying player-specific WR targeted-route matchup diagnostic remains scientifically confirmed.

This result closes only the exact frozen transmission of that signal into the 2024 production-order WR full stack.

Do **not** reinterpret this as evidence that the named-player route-role mechanism is false.

Do **not** rescue the exact integration by retuning:
- the 2023 coefficients;
- history windows;
- route families;
- player/defense support;
- caps;
- subgroups;
- WR1/WR2+ carveouts;
- 2024 outcomes;
- 2025 pseudo-confirmation.

## Frozen development coefficients

Fit on 2023 WR development rows only:
- rows: 1,727
- `beta_ypt = 0.4673827012`
- `beta_catch = 0.4297478511`

Neither coefficient was fit or tuned on 2024.

## Population and integrity

2024 full-stack integration population:
- 1,916 eligible WR player-games
- 187 named WRs
- 163 WRs with >=4 eligible games for player-level MAE reporting

Baseline / conservation integrity:
- all 18 audit rows passed
- hard integrity: PASS
- YPT target-share max absolute delta: 0.0
- catch target-share max absolute delta: 0.0
- YPT non-WR efficiency max absolute delta: 0.0
- catch non-WR efficiency max absolute delta: 0.0
- WR1 anchor max gap: 0.0
- TE pool max gap: ~5.55e-17
- WR2+ pool max gap: ~1.11e-16
- modeled team target-mass max gap: ~1.11e-16
- total WR room mass max gap: ~2.22e-16

The candidate therefore failed scientifically, not mechanically or because target entitlement drifted.

## YPT arm — CLOSED

Disposition:
`NO_ACTIONABLE_WR_ROUTE_MATCHUP_YPT_INTEGRATION`

Direct MC receiving-yards:
- baseline MAE: 23.18445
- candidate MAE: 23.19066
- MAE change: **-0.0268% improvement** (worse)
- baseline RMSE: 32.55549
- candidate RMSE: 32.59344 (worse)

Full-stack receiving-yards:
- baseline MAE: 22.92198
- candidate MAE: 22.93119
- MAE change: **-0.0402% improvement** (worse)
- baseline RMSE: 31.57744
- candidate RMSE: 31.59371 (worse)
- baseline correlation: 0.52249
- candidate correlation: 0.52140
- absolute bias improved slightly: 5.49876 -> 5.49086

Player-first diagnostics:
- median qualifying-player MAE delta: -0.01381 yards (slight median improvement)
- fraction qualifying players improved: 50.92%

Important subgroup/safety diagnostics:
- Weeks 5-12 MAE: 23.49633 -> 23.53146 (worse)
- Weeks 13-18 MAE: 22.08560 -> 22.08473 (slightly better)
- WR1 MAE: 29.61154 -> 29.65485 (worse)
- WR2+ MAE: 20.64846 -> 20.64609 (slightly better)
- 20+ yard miss rate: 42.171% -> 42.119% (slightly better)
- 30+ yard miss rate: 25.209% -> 25.522% (worse)
- 40+ yard miss rate: 15.658% -> 15.866% (worse)
- CRPS: 17.38953 -> 17.41827 (worse)
- 80% coverage: 61.430% -> 61.221% (nominal gap worsened)
- 90% coverage: 74.530% -> 75.157% (nominal gap improved)

Frozen failed gates:
- direct MC MAE improvement >=0.50%
- direct MC RMSE non-worse
- full-stack MAE improves
- full-stack RMSE non-worse
- Weeks 5-12 MAE non-worse
- WR1 MAE non-worse
- CRPS non-worse
- 80% coverage-gap non-worse
- 30+ miss non-worse
- 40+ miss non-worse

The player-level median being slightly favorable cannot rescue failed system and tail guards.

## Catch arm — CLOSED

Disposition:
`NO_ACTIONABLE_WR_ROUTE_MATCHUP_CATCH_INTEGRATION`

Direct MC receptions:
- baseline MAE: 1.56844
- candidate MAE: 1.56787
- MAE improvement: **+0.0364%**
- baseline RMSE: 2.19321
- candidate RMSE: 2.19193

Full-stack receptions:
- baseline MAE: 1.50869
- candidate MAE: 1.50771
- MAE improvement: **+0.0647%**
- baseline RMSE: 2.06217
- candidate RMSE: 2.06136
- baseline correlation: 0.59618
- candidate correlation: 0.59633
- absolute bias improved slightly: 0.39539 -> 0.39398

Player-first diagnostics:
- median qualifying-player MAE delta: -0.000277 receptions
- fraction qualifying players improved: 50.92%

Distribution/tail:
- CRPS: 1.16439 -> 1.16340 (better)
- 80% coverage: 80.637% -> 80.846%; nominal gap worsened
- 90% coverage: 87.213% -> 87.474%; nominal gap improved
- >=2-reception absolute-miss rate: 26.200% -> 26.514% (worse)
- >=3-reception absolute-miss rate: unchanged at 12.370%

Frozen failed gates:
- direct MC MAE improvement >=0.50%
- 80% coverage-gap non-worse
- >=2-reception miss rate non-worse

The small aggregate and player-level improvements are real but materially below the predeclared effect floor and fail safety guards.

## Scientific interpretation

The player-specific Stage-B relationship is reproducible:
`named WR targeted-route mix × opponent route-family susceptibility`
contains information about next-game WR target conversion.

However, the exact frozen mean-transmission design does not improve the promoted full stack enough to justify integration.

This distinction matters:
- **mechanism evidence remains confirmed**;
- **exact production-order mean integration is closed**.

The result is consistent with a signal that is too small, too noisy at the individual game level, or already substantially absorbed by the promoted WR stack to survive the frozen direct-mean insertion.

That is not permission to post-hoc strengthen, threshold, subgroup, or retune it.

## Integrity

- sportsbook inputs: 0
- OddsAPI calls: 0
- 2026 outcomes read: 0
- production changed: false
- TE transport: false
- target-share changes from route matchup: 0

## Next legal frontier

Do not rescue this exact WR route-matchup mean integration.

The clean player-first frontier is the separately confirmed WR/TE **player-specific opportunity uncertainty** lineage, which remains authorized for its own frozen integration/shadow work while preserving the promoted mean.

