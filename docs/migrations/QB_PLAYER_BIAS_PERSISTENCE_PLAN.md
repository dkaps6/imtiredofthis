# QB Player-Bias Persistence Audit — Frozen Plan

## Purpose
Test whether a QB's own strictly-prior model residuals contain stable, out-of-sample calibration information. This is a diagnostic/calibration question, not a reopening of the closed QB feature hunt and not a sportsbook-tuning exercise.

## Canonical evidence
- M89 football-only synthesis trace from run `33331073376`.
- Evaluation seasons: 2024-2025 only, exactly as stored in the untouched M89 validation trace.
- Sportsbook/market fields are prohibited.

## Residual definition
`error = football_synthesis - actual_pass_yards`.
For each player-game, only that QB's earlier completed validation games may be used.

## Frozen candidate
`PLAYER_BIAS_SHRINK_V1`
- minimum prior games: 6
- expanding prior residual mean; no window search
- zero-centered shrinkage weight: 8 games
- shrunken bias = `prior_mean_error * n/(n+8)`
- correction cap: +/-25 passing yards
- candidate projection = `football_synthesis - clipped_shrunken_bias`

No parameter may be changed after results.

## Scoreboard
On eligible player-games compare exact M89 synthesis vs candidate:
- MAE, RMSE, bias, median absolute error, p90 absolute error
- 30+, 50+, 75+ yard miss rates
- 2024 and 2025 separately
- player-level MAE deltas for QBs with >=8 eligible evaluation rows

## Frozen evidence gate
Candidate is `PERSISTENT_PLAYER_BIAS_EVIDENCE` only if ALL hold:
1. >=400 eligible player-games.
2. overall MAE improves >=1.0%.
3. overall RMSE does not worsen.
4. median absolute error does not worsen.
5. p90 absolute error does not worsen.
6. 30+ miss rate does not worsen.
7. 50+ miss rate does not worsen.
8. MAE improves in both 2024 and 2025.
9. median player-level MAE delta among qualifying QBs is <0.

Otherwise disposition is `NO_ACTIONABLE_PLAYER_BIAS_PERSISTENCE`.

## Interpretation
A pass does NOT alter production. Because QB mean research is closed, a pass only supplies materially-new calibration evidence sufficient to justify a separately frozen production-integration review. A fail closes this calibration idea for the current cycle; no min-games/shrink/cap hunting.
