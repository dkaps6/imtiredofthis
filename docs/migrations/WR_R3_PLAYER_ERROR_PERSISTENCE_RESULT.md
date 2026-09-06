# WR-R3 — Walk-Forward Individual WR Error Persistence — Result

## Disposition

`WR_PLAYER_ERROR_PERSISTENCE_DETECTED`

All three frozen individual-error diagnostics passed. This is diagnostic evidence only; M38 remains unchanged pending separately frozen full-stack integration.

## Canonical evidence

- Run: `34064572328`
- Job: `101570966426`
- Tested SHA: `d0fc31079fcc541d487d4cc2ec2886846f697e92`
- Artifact: `9998530999` (`wr-r3-player-error-persistence`)
- Digest: `sha256:4c199ee48804f0331c259c3bce4645c2b60f21ac764923046986c111fdbba978`
- Canonical M38 source rows: `12,396`
- Scoreable rows: `10,675`
- Players represented: `478`
- History: last 8 strictly prior same-player M38 games; minimum 4
- Walk-forward leakage violations: `0`
- Sportsbook inputs: `0`
- Model fitting: `false`
- Production changed: `false`

## Frozen diagnostic results

| Diagnostic | Spearman | Quartile gap | Sign agreement / enrichment | Positive seasons | 2024 | 2025 | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| Directional bias persistence | .08889 | +9.55 signed yards | 55.56% sign agreement | 6/6 | +9.91 | +9.84 | Yes |
| Individual difficulty persistence | .25793 | +13.90 absolute yards | — | 6/6 | +14.09 | +12.87 | Yes |
| Extreme-miss persistence | — | — | 1.368x next-game 30+ miss enrichment | 6/6 | 1.435x | 1.433x | Yes |

## Scientific conclusion

The user's individual-player accuracy hypothesis is strongly supported for WRs. A WR's own **pregame-known prior model-error history** contains repeatable information about:

1. the direction of the next M38 miss;
2. how difficult that WR is for M38 to project accurately;
3. the chance of another extreme individual miss.

This is not the same as using a retrospective full-sample player MAE. The signal survived strict walk-forward construction using only prior games.

The natural next integration lanes are therefore different:

- prior signed bias may support a conservative, shrunk mean-calibration layer;
- prior difficulty / extreme-miss history may support player-specific Monte Carlo uncertainty / tail calibration.

Because these families independently passed frozen gates, a later predeclared combined full-stack candidate is legitimate. No production change occurs until those candidates are tested through the exact M38 production architecture and pass frozen aggregate + individual-error gates.
