# WR-R3 Player-Bias Persistence — Frozen Plan

## Purpose
Test whether an individual WR's own strictly-prior M38 receiving-yard residuals contain stable, deployable calibration information. This is independent of WR-R2 NGS tracking and of the failed ND3/ND5/ND6 families.

## Canonical evidence
- Exact WR-R1 paired casebook from run `34058453941`.
- Exact M38 receiving-yard rows: 12,396 across 2020-2025.
- Sportsbook data prohibited.

## Residual
`error = mc_proj_m38 - actual_m38` for WR receiving yards.
For every target player-game, use only that WR's earlier completed games.

## Frozen candidate
`WR_PLAYER_BIAS_SHRINK_V1`
- minimum prior games: 8
- expanding prior residual mean; no alternate window search
- zero-centered shrinkage weight: 8 games
- shrunken bias = `prior_mean_error * n/(n+8)`
- correction cap: +/-20 receiving yards
- candidate projection = `M38 - clipped_shrunken_bias`

## Scoreboard
Overall and by season: MAE, RMSE, bias, median AE, p90 AE, 20+/30+/40+ miss rates. Also player-level MAE delta for WRs with >=8 eligible evaluation rows.

## Frozen integration-evidence gate
ALL must hold:
1. >=6,000 eligible player-games.
2. pooled MAE improves >=1.0%.
3. RMSE does not worsen.
4. median AE does not worsen.
5. p90 AE does not worsen.
6. 20+, 30+, and 40+ miss rates do not worsen.
7. MAE improves in >=4 of 6 seasons.
8. 2024 MAE improves.
9. 2025 MAE improves.
10. median qualifying-player MAE delta <0.

Pass disposition: `WR_PLAYER_BIAS_PERSISTENCE_SIGNAL` and only then freeze a full-stack integration test under the WR full-stack protocol. Fail disposition: `NO_ACTIONABLE_WR_PLAYER_BIAS_PERSISTENCE`; no shrink/cap/min-game retuning.
