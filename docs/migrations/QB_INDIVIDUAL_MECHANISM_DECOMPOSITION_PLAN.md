# QB Individual Mechanism Decomposition — Frozen Plan

## Purpose
Turn the individual-QB MAE profiles into football/mechanism diagnostics instead of applying blanket player offsets. This does not reopen the closed QB mean-feature hunt and cannot change production.

## Canonical evidence
- M89 run `33331073376`.
- `reconciliation/m89_corrected_qb_common_trace.csv` for actual attempts/actual YPA and pregame predicted attempts/YPA.
- `synthesis/m89_2024_2025_synthesis_trace.csv` for base projection and final football synthesis.
- Expected aligned rows: 884, seasons 2024-2025.
- Sportsbook/market fields prohibited.

## Exact decomposition
Define pregame mechanics projection `M = pred_attempts * pred_ypa`.
Actual passing yards `A = actual_attempts * actual_ypa` (must reconcile to official actual passing yards).

Use two-factor Shapley decomposition of `A - M`:
- attempt contribution = `(actual_attempts - pred_attempts) * (actual_ypa + pred_ypa) / 2`
- YPA contribution = `(actual_ypa - pred_ypa) * (actual_attempts + pred_attempts) / 2`

Then define:
- stack adjustment = `base_proj - M`
- synthesis adjustment = `football_synthesis - base_proj`

Exact final residual identity must hold:
`actual_pass_yards - football_synthesis = attempt_contribution + ypa_contribution - stack_adjustment - synthesis_adjustment`.

## Player profiles
For each QB with >=8 aligned games report:
- final synthesis MAE/bias
- mean and mean-absolute attempt contribution
- mean and mean-absolute YPA contribution
- mean and mean-absolute stack adjustment
- mean and mean-absolute synthesis adjustment
- dominant mean-absolute component among ATTEMPTS / YPA / STACK / SYNTHESIS
- 30+/50+/75+ final miss rates

Also report aggregate and season-level component summaries and the highest-MAE individual QBs.

## Integrity
- 884 aligned rows.
- official actual-yard reconciliation max error <=1e-6.
- decomposition identity max error <=1e-6.
- no model fitting, no sportsbook, no production change.

Disposition is diagnostic only: `QB_INDIVIDUAL_MECHANISMS_MAPPED`. Any later correction must be a separately frozen, generalizable pregame mechanism test; no player-specific hand tuning.
