# RB-PD2 — Walk-Forward Individual RB Error Persistence — Result

## Disposition

`RB_PLAYER_ERROR_PERSISTENCE_DETECTED`

All four frozen carry/yard persistence diagnostics passed. No production change. No sportsbook input. No model fitting. No walk-forward leakage.

## Canonical evidence

- Run: `34064637295`
- Job: `101571138337`
- Tested SHA: `906102e47bba7bfcc887620dea4bb60a155163d3`
- Artifact: `9998549410` (`rb-pd2-player-error-persistence`)
- Digest: `sha256:3ab28410b055ef2dcf35635e86dd93a9f45cbeba971764bad313df13080996c2`
- Canonical 2025 source rows: `1,393`
- Scoreable rows: `882`
- Players: `148`
- History: last 8 strictly prior same-player games; minimum 4

## Frozen diagnostic results

| Diagnostic | Spearman | Quartile gap | Sign agreement | Weeks 5-12 | Weeks 13-18 | Pass |
|---|---:|---:|---:|---:|---:|---|
| Carry directional persistence | .15433 | +1.507 carries | 62.52% | +1.083 | +2.014 | Yes |
| Carry difficulty persistence | .24127 | +1.789 abs carries | — | +1.869 | +1.805 | Yes |
| Yard directional persistence | .14358 | +12.432 yards | 60.43% | +11.136 | +14.371 | Yes |
| Yard difficulty persistence | .33056 | +16.696 abs yards | — | +17.661 | +16.530 | Yes |

Rookie slice remained descriptive only. Among scoreable rows, rookies had carry MAE `3.886` versus `3.663` for non-rookies; rookie yard MAE was `19.58` versus `22.37`, so this result does not justify a generic rookie yard boost.

## Scientific conclusion

The user's individual-player accuracy hypothesis is strongly supported for RBs too. A player's own strictly prior model-error history predicts both the **direction** and **difficulty** of the next carry and rushing-yard projection.

This is materially different from the failed Role-Order Remap V1. The evidence does not say “make the depth-chart RB1 own the most carries.” It says the current model has **player-specific residual behavior that persists pregame** even after the existing football model has run.

The next legitimate step is a separately frozen full-stack calibration test with conservative shrinkage/minimum-sample protection:

- prior carry bias as a candidate carry-mean calibration input;
- prior yard bias only at the natural efficiency/yardage layer after carry effects are accounted for;
- prior carry/yard difficulty as uncertainty/MC-width calibration rather than a blind mean offset.

No player-specific correction is promoted from this diagnostic alone.
