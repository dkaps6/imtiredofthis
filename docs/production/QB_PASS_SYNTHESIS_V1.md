# QB Passing-Yards Synthesis v1 — 2026 Production Promotion

## Status

`QB_PASS_SYNTHESIS_V1` is the promoted football-only QB passing-yards mean architecture for 2026 prospective production use.

Broad QB mean-projection research is frozen after M90. Future QB work should focus on production readiness and explicitly separated distribution/tail calibration unless new independent information becomes available.

## Evidence lineage

### M89 — data integrity + synthesis

Authoritative workflow run: `33331073376` (Run #13).

M89 corrected the production/research source contract before fitting the synthesis:

- official passing attempts and yards come from nflverse weekly player stats;
- PBP supplies sacks, QB scrambles, game state, EPA, xPass, pace, YAC/explosives, and pressure proxies;
- pass-attempt conversion is official attempts / (official attempts + PBP sacks + PBP QB scrambles);
- historical canonical actual attempts and passing yards reconciled 100% on the 884-game 2024-2025 cohort;
- PROE is situation-adjusted from xPass/pass probability;
- neutral pace is within-drive neutral-state pace;
- pressure is explicitly a sack-or-QB-hit proxy, not full pressure.

The frozen 2023-trained football-only synthesis improved corrected 2024-2025 MAE from approximately `57.638995` to `55.060118`, with lower RMSE, higher correlation, fewer 100+ yard misses, and approximately 99.62% paired-bootstrap support.

### M90 — temporal confirmation

Authoritative workflow run: `33333730480` (Run #11).

The exact M89 architecture was fit on corrected 2022 and prospectively evaluated on corrected 2023:

| Metric | Corrected base | Football synthesis |
|---|---:|---:|
| MAE | 60.632751 | 56.559869 |
| RMSE | 75.634635 | 69.629921 |
| Bias | -27.248932 | -6.606555 |
| Correlation | 0.172628 | 0.243467 |
| 100+ misses | 81 | 64 |
| 100+ underprojections | 69 | 37 |
| 100+ overprojections | 12 | 27 |

MAE improvement was `4.072882` yards and paired-bootstrap probability of improvement was `0.9997`. Every preregistered promotion gate passed.

Formal M90 disposition: `PROMOTE_M89_FOOTBALL_SYNTHESIS`.

## Frozen architecture

- Model: Ridge residual correction on the canonical MC/ML/State ensemble.
- Ridge alpha: `20.0`.
- Maximum predicted correction: `±45` passing yards.
- Football features: exactly the 21 M89/M90 features stored in `model/qb_pass_synthesis_v1.json`.
- Sportsbook/game-market variables: prohibited from the football synthesis.
- Postgame catastrophic-casebook variables: prohibited from prediction.

After M89/M90 validation was complete, the same architecture was refit on all corrected 2023-2025 pre-2026 canonical rows (`n=1331`) for prospective 2026 deployment. That refit is a deployment fit, not an additional historical validation result.

## Production order

For QB passing yards only:

1. Build football contexts and canonical MC/ML/State components.
2. Convert simulated pass opportunities to official attempts using the promoted M89 attempt-conversion semantic.
3. Form the canonical evidence-weighted ensemble.
4. Apply the promoted M89/M90 football-only residual synthesis.
5. Rescale the Monte Carlo distribution to the promoted synthesis mean.
6. Only after the football projection is frozen, compare with sportsbook player-prop lines/odds.

All other markets retain their existing production pathway.

## Required production audits

A successful live pricing run with QB passing-yard props must show:

- `qb_synthesis_applied == 1` for every priced `pass_yards` row;
- non-empty `qb_synthesis_version`;
- `qb_attempt_conversion` in `[0.50, 1.00]`;
- finite `qb_synthesis_proj`;
- `model_proj == qb_synthesis_proj` within numerical tolerance;
- the pre-synthesis `ensemble_proj` retained as a separate audit column.

The workflow is intentionally fail-closed for the promoted QB layer. Missing/invalid promoted context does not silently revert to the superseded QB mean.

## Known limitation carried into 2026

M90 reduced total catastrophic errors materially but shifted some error mass from severe underprojections toward severe overprojections. This does not invalidate the promoted mean architecture because MAE, RMSE, correlation, bias, and total 100+ misses all improved. It does mean future QB distribution work should model downside and upside tail risk separately rather than restarting broad mean-feature research.
