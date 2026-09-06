# QB-PD2 — Walk-Forward Individual Player Error Persistence — Result

## Disposition

`NO_ACTIONABLE_QB_PLAYER_ERROR_PERSISTENCE`

No production change. No sportsbook input. No model fitting. No walk-forward leakage.

## Canonical evidence

- Run: `34064528914`
- Job: `101570850350`
- Tested SHA: `3d74c6c9a5feedcbce5c20e92929f6fda20852ef`
- Artifact: `9998517231` (`qb-pd2-player-error-persistence`)
- Digest: `sha256:b56df586ced1a0caf5bfe8ed9c9ffbaade659f7f877758c2d0e4031ca3f0e4a1`
- Historical source: exact M89 2024-2025 validation trace, 884 rows / 59 QBs
- Frozen history: last 8 prior same-QB games, minimum 4
- Scoreable rows: `668`

## Frozen diagnostics

| Diagnostic | Spearman | Quartile gap | Sign agreement | 2024 | 2025 | Pass |
|---|---:|---:|---:|---:|---:|---|
| Directional bias persistence | .0223 | +1.17 yards | 51.16% | +2.28 | +0.06 | No |
| Individual difficulty persistence | .0588 | +5.48 abs yards | — | -4.36 | +11.87 | No |
| Synthesis reliability persistence | .0240 | +1.22 yards | — | +5.07 | +0.25 | No |

None cleared the frozen gates. Directional bias was essentially non-persistent; individual difficulty was unstable across seasons; prior synthesis advantage did not predict next-game synthesis advantage strongly enough.

## Scientific conclusion

Retrospective per-QB MAE/bias differences are real and useful for audit, but the recent prior-only individual QB error history does **not** persist strongly enough to justify a player-specific correction or reliability adjustment under the frozen protocol.

Do not:

- feed full-sample QB MAE/bias into the 2026 projection;
- search nearby prior windows after this result;
- lower the persistence thresholds;
- create QB-specific corrections from the Week-1 market discrepancies.

The next legitimate QB reliability work should remain independent of player-error feedback and focus on mechanisms already exposed by the Week-1 pathology audit: component disagreement, synthesis-correction magnitude/cap behavior, and whether those pregame internal states predict historical out-of-sample error. This remains production-reliability/distribution work, not reopened broad mean-feature hunting.
