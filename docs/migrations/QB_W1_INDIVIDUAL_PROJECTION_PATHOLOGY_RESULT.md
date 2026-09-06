# QB Week 1 Individual Projection Pathology — Result

## Disposition

`DIAGNOSTIC_ONLY_NO_PRODUCTION_CHANGE`

No sportsbook input entered the football projection. No model was fit. No production value changed.

## Canonical evidence

- Run: `34064097525`
- Job: `101569713327`
- Tested SHA: `aa0dacfc4a0a580f20c7618a853f79b56a41ab94`
- Artifact: `9998389859` (`qb-w1-individual-projection-pathology`)
- Digest: `sha256:aaf8978615d75bd90dbb3abdc08b3b23221a31ba950cb7dc881f74151d82ddf6`
- Week-1 source run: `34059395746`
- Historical M89 source run: `33331073376`

## Coverage

- Week-1 posted QB rows: `31`
- historical validation rows: `884`
- historical QBs: `59`
- Week-1 QBs with any historical profile: `30`
- Week-1 QBs with >=8 historical games: `28`

## What the individual audit exposed

Internal frozen diagnostic flags among the 31 Week-1 rows:

- large component disagreement (>=40 yards): `14`
- large synthesis move (>=30 yards): `11`
- historical player MAE >=50 yards with >=8 games: `16`
- historical directional bias >=15 yards absolute with >=8 games: `8`
- synthesis historically worse than corrected base for that player with >=8 games: `10`
- synthesis correction at the +/-45 cap: `4`

Downstream market discrepancies remained descriptive only:

- >=20 yards: `9`
- >=30 yards: `6`
- >=40 yards: `2`
- mean absolute model-minus-market gap: `15.5644`
- mean signed model-minus-market gap: `-0.7235`

## High-priority examples

- Jalen Hurts: 29 historical games, synthesis MAE `64.18`, bias `+30.53`, p90 absolute error `120.75`; Week-1 synthesis move `+31.04`, component range `48.02`. Five frozen internal flags.
- Dak Prescott: 22 historical games, synthesis MAE `44.78`, bias `-28.92`; Week-1 correction hit `+45`, component range `92.23`; market gap `-34.18` yards downstream.
- Kirk Cousins: 21 historical games, synthesis MAE `66.68`; Week-1 correction hit `+45`; synthesis historically worsened versus corrected base for this player.
- Bryce Young: 17 historical games, synthesis MAE `74.23`, bias `+29.19`; component range `49.03`; synthesis historically worsened.
- Brock Purdy: 23 historical games, synthesis MAE `61.32`; Week-1 correction hit `+45`; component range `48.08`.
- Josh Allen: 32 historical games, synthesis MAE `59.12`; Week-1 synthesis move `+42.97`, component range `53.19`; downstream market gap `+30.60`.
- Joe Burrow: 20 historical games, synthesis MAE `64.96`, bias `-35.01`, p90 absolute error `148.82`.
- Jordan Love: 28 historical games, synthesis MAE `52.68`; synthesis historically worse than corrected base; downstream market gap `-44.61`.
- Jared Goff: 33 historical games, synthesis MAE `53.86`; component range `51.91`; downstream market gap `-39.20`.

Aaron Rodgers' `+37.14` downstream market discrepancy triggered none of the frozen internal pathology flags. This is useful: a large market disagreement is not automatically an internal model pathology.

## Scientific conclusion

Aggregate QB accuracy is hiding meaningful **player-specific reliability differences**. The diagnostic supports the user's concern that a single positional MAE is insufficient to characterize projection quality.

However, the full-sample individual MAE/bias values above are retrospective diagnostics, not legal 2026 features. Using them directly would leak future information within the 2024-2025 validation period.

The next legitimate QB step is a separately frozen **walk-forward individual-error persistence diagnostic**: before each historical target game, calculate only that QB's prior model errors and test whether prior MAE/bias/reliability predicts the next-game residual or whether the synthesis layer should be trusted less in specific pregame states. No sportsbook data and no post-hoc player corrections.
