# Player Allocation-Only Difficulty V1 — Result

Date: 2026-10-14

Status: **COMPLETE / TE PLAYER ALLOCATION DIFFICULTY CONFIRMED / WR EXACT BRIDGE CLOSED / RESEARCH ONLY**

Frozen contract:
- `docs/research/PLAYER_ALLOCATION_ONLY_DIFFICULTY_V1_FREEZE.md`

Authority:
- branch: `research-individual-opportunity-roadmap-2026-10-08`
- exact scored head: `e936e3eb03bf085c7f2bce1249cfa97435eb4b37`
- workflow: `Research Player Allocation-Only Difficulty V1`
- run: `38656442803` — SUCCESS
- job: `116167234340`
- artifact: `11936407975`
- digest: `sha256:7ded306cd0a049902130ccc5cab995231b192120c02679c3852757bc8df29cc0`

Mechanical predecessor:
- run `38656370089` failed before scoring because TE authority already carried an identity column and the merge produced suffixed keys;
- no scientific result came from that run;
- the only change before the canonical green was dropping the duplicate authority-side identity field so the frozen V2 stable identity remained canonical.

## Overall disposition

`PLAYER_ALLOCATION_ONLY_DIFFICULTY_CONFIRMED`

The confirmation is **TE-specific**.

WR exact allocation-only bridge:
`NO_ACTIONABLE_WR_PLAYER_ALLOCATION_DIFFICULTY`.

TE:
`TE_PLAYER_ALLOCATION_DIFFICULTY_CONFIRMED`.

## What this test isolated

The scored residual was not total target error.

For each player-game:
- preserve the promoted pregame team target total `T_hat`;
- preserve the promoted player target mean `mu_hat`;
- take realized player target share only as the target-game outcome;
- score
  `alloc_error = mu_hat - T_hat * actual_share`.

Therefore team pass-volume error cannot create this signal.

The candidate width used only the named player's own trailing <=8 strictly-prior absolute allocation errors, min 4 games. No coefficient was fit.

## TE — CONFIRMED

Scored:
- 1,870 TE player-games
- 121 named TEs
- 2024: 717 rows / 92 TEs
- 2025: 759 rows / 103 TEs

CRPS vs contemporaneous pooled TE allocation difficulty:

2024:
- pool: `1.46973`
- named-player: `1.42466`
- gain: `+0.04508`

2025 independent confirmation:
- pool: `1.37316`
- named-player: `1.32548`
- gain: `+0.04768`

Pooled:
- pool: `1.43036`
- named-player: `1.38784`
- gain: `+0.04253` (~2.97% relative)

Dependence-aware pooled gain:
- player-cluster 95% CI: `[+0.02258,+0.06505]`
- game-cluster 95% CI: `[+0.02756,+0.05768]`
- both P(gain>0) = `1.000`

Coverage:
- 80%: `73.58% -> 77.33%`
- 80% nominal gap: `6.42pp -> 2.67pp`
- 90%: `82.09% -> 85.24%`
- 90% nominal gap: `7.91pp -> 4.76pp`

High-difficulty Q4:
- 468 rows / 44 TEs
- CRPS: `2.12865 -> 2.03949`
- 80% coverage: `54.27% -> 88.25%`
- 90% coverage: `67.52% -> 94.87%`

Player-level 2025 confirmation:
- 69 TEs with >=6 scored games
- median per-player CRPS gain: `+0.01465`
- fraction positive: `56.52%`

All frozen TE gates passed.

### TE scientific interpretation

The earlier TE uncertainty result was not merely team pass-volume noise.

After explicitly removing team target-volume error, the named TE's own strictly-prior **allocation/share difficulty** still improves his next target-allocation distribution over pooled TE uncertainty.

This is direct support for a player-specific TE allocation-uncertainty layer while preserving TE-R5P target means.

## WR — EXACT BRIDGE CLOSED

Scored:
- 2,142 WR player-games
- 190 named WRs
- 2023: 851 rows / 144 WRs
- 2024: 1,291 rows / 162 WRs

Pooled CRPS:
- pool: `1.68244`
- named-player: `1.66916`
- gain: `+0.01328`

But the frozen replication/incrementality contract failed.

2023 development:
- pool: `1.64072`
- named-player: `1.64161`
- named-player is slightly worse.

2024:
- pool: `1.70993`
- named-player: `1.68731`
- WR V2 reference: `1.69564`
- named-player beats both point estimates.

However 2024 candidate-vs-V2 clustered inference does not clear zero:
- player-cluster CI: `[-0.00894,+0.02684]`
- game-cluster CI: `[-0.00741,+0.02453]`

Pooled candidate-vs-pool:
- player-cluster CI low: `+0.00012`
- game-cluster CI low: `-0.00013`

Coverage guards also worsened slightly, and 2024 player-level median gain was negative:
- 110 WRs with >=6 confirmation games
- median per-player gain: `-0.00178`
- fraction positive: `49.09%`

Frozen disposition:
`NO_ACTIONABLE_WR_PLAYER_ALLOCATION_DIFFICULTY`.

Do not rescue by changing windows, role slices, thresholds, or by combining post hoc with the V2 WR volatility state.

### WR interpretation

This does **not** erase:
- `WR_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`;
- WR same-player persistence;
- the independently qualified WR excess-share-volatility / Dirichlet lineage.

It says the exact attempt to reinterpret the broader WR opportunity-difficulty state as a second allocation-only width is not independently stable enough beyond the existing WR allocation-volatility machinery.

Keep the existing WR Dirichlet lineage as the allocation-uncertainty authority.

## Authorized next step

TE only:
- freeze a joint team-conserving TE target-allocation candidate;
- preserve TE-R5P target means;
- derive player overdispersion only from the confirmed same-player allocation-difficulty state;
- unsupported/under-dispersed rows remain canonical multinomial;
- score room-level joint likelihood plus marginal proper scores;
- require player-level confirmation;
- no production promotion from this result alone.

WR:
- no new integration from this exact bridge.

## Integrity

- candidate models fit: 0
- sportsbook inputs: 0
- OddsAPI calls: 0
- 2026 outcomes: 0
- projection mean changes: 0
- production changed: false
