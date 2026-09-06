# WR-R2 Player-Level Tracking Residuals — Result

## Disposition

`NO_ACTIONABLE_PLAYER_TRACKING_SIGNAL`

No production change. No model fitting. No sportsbook input. No target/future NGS leakage.

## Canonical evidence

- Run: `34064167212`
- Job: `101569903392`
- Tested SHA: `9c047eaaaac86d183888ffd159e393e10316c208`
- Artifact: `9998415876` (`wr-r2-player-tracking-residuals`)
- Digest: `sha256:a71077f7ecd8a6901ecd683f005b701f2041a41e5e44dd0aedb615822ce6952c`

## Integrity / source

- exact M38 multi-season WR receiving-yard rows: `12,396`
- target seasons: 2020-2025
- individual WRs profiled: `486`
- NGS raw receiving rows: `8,976`
- NGS source players: `546`
- NGS seasons available: 2019-2025
- signal coverage: approximately `86%`
- target/future NGS violations: `0`

## Frozen signal results

| Signal | Rows | Coverage | Spearman | High-low residual gap | UNDER25 | UNDER50 | ACTUAL100 | Positive seasons | 2024 gap | 2025 gap | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NGS_ADOT_PRIOR8 | 10,666 | .8604 | -.0069 | -.27 | .997x | 1.039x | .940x | 3/6 | -3.20 | -2.05 | No |
| NGS_YACOE_PRIOR8 | 10,655 | .8596 | .0306 | +3.40 | 1.073x | 1.161x | 1.283x | 5/6 | +6.11 | +5.52 | No |
| NGS_SEPARATION_PRIOR8 | 10,666 | .8604 | .0110 | +.32 | .976x | .976x | .861x | 4/6 | +2.24 | +2.11 | No |
| NGS_IAY_SHARE_PRIOR8 | 10,666 | .8604 | .0114 | +2.99 | 1.161x | 1.309x | 1.821x | 6/6 | +2.67 | +3.17 | No |

All four failed the frozen actionable gate. No threshold lowering, alternate-window search, or interaction test is authorized.

`NGS_IAY_SHARE_PRIOR8` is descriptively interesting because its residual gap was positive in all six target seasons and its UNDER50 / ACTUAL100 enrichment was strong, but it missed the frozen global Spearman, +4-yard gap, and UNDER25 requirements. It remains a failed signal, not a winner.

## Individual-player implication

WR-R2 also created a durable 486-player M38 error profile (games, MAE, bias, median absolute error, p90 absolute error, and 20+/30+/40+ miss rates). This is diagnostic evidence only; the full-sample profile cannot be used upstream.

The next independent question is whether a WR's **prior-only individual error history** is persistent enough to predict the direction or magnitude of the next M38 miss. That can be tested walk-forward without changing M38 and without relying on the failed NGS signals.
