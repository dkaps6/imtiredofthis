# RB Individual Mechanism Decomposition — Frozen Plan

## Purpose
Determine whether recurring individual RB rushing-yard errors are primarily carry-volume errors or yards-per-carry errors. This follows the individual-MAE strategy and is independent of the rejected blunt current-depth remap.

## Canonical evidence
- RB individual role-transition diagnostic run `34064260950`.
- Exact 1,393-row STACK1 production-equivalent 2025 RB/FB casebook.
- Sportsbook data prohibited.

## Exact decomposition
For each player-game define actual/projected YPC as rushing yards divided by carries when carries >0; define YPC=0 only when both carries and yards are zero. Inconsistent zero-carry/nonzero-yard rows are excluded and counted.

Use two-factor Shapley decomposition of `actual_rush_yards - projected_rush_yards`:
- carry contribution = `(actual_carries - projected_carries) * (actual_ypc + projected_ypc) / 2`
- YPC contribution = `(actual_ypc - projected_ypc) * (actual_carries + projected_carries) / 2`

The two contributions must reconstruct rushing-yard residual within 1e-6.

## Player profiles
For each player with >=8 scoreable games report:
- carry MAE/bias
- rushing-yard MAE/bias and 20+/30+/40+/50+ miss rates
- mean and mean-absolute carry contribution
- mean and mean-absolute YPC contribution
- contribution shares
- dominant mechanism: `CARRIES` if mean-absolute carry contribution >=1.25x YPC; `YPC` if reverse; otherwise `MIXED`.

Also report rookie, mismatch+rookie, depth-RB1, and aggregate component summaries.

## Integrity
- source rows 1,393.
- decomposition identity <=1e-6 on scoreable rows.
- no fitting, sportsbook, or production change.

Disposition: `RB_INDIVIDUAL_MECHANISMS_MAPPED`. Any later fix must be a separately frozen generalizable mechanism test; no manual per-player corrections.
