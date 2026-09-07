# CROSS-POSITION PHASE F — SEPARATE WR/TE SHARED-STATE RESPONSE V1

## Purpose
Phase D established that QB attempt residuals co-move differently with WR targets (Pearson ~0.697) and TE targets (~0.472). Phase E then prospectively showed that one shared pregame pass-state signal materially improved QB attempts and WR targets, but **failed** when the receiver correction was split proportionally between WR and TE: TE target-pool MAE worsened by 0.3625 targets.

Phase F extracts the useful shared-state component without relaxing Phase-E gates. It asks one predeclared architecture question:

> Can the exact same pregame shared pass-state signal be mapped separately to WR and TE pools using training-only response slopes, rather than assuming WR and TE absorb pass-state changes in proportion to their baseline target pools?

This is diagnostic opportunity research only. No production mean, yardage efficiency, individual entitlement, sportsbook input, or simulation parameter changes.

## Frozen lineage
- Phase C authoritative source: run `34147777341`, artifact `10028332887`.
- Phase E authoritative source: run `34150046671`, artifact `10029037488`.
- Exact Phase-E `delta_pass_attempts` values for 2025 team-games must be reused unchanged. Phase F is not allowed to refit or alter the pass-state predictor.

## Cohort
Use the exact 434 scored 2025 team-games from Phase E with primary QB + WR + TE overlap.

## Frozen separate response mappings
For each scored 2025 week, estimate only from chronologically earlier Phase-C team-games:

- `beta_wr = zero-intercept slope(WR target residual ~ QB attempt residual)`
- `beta_te = zero-intercept slope(TE target residual ~ QB attempt residual)`

Training may use 2024 plus prior 2025 games. Same-week and future games are prohibited.

Apply the **unchanged Phase-E delta**:
- QB corrected attempts = exact Phase-E corrected QB attempts.
- WR corrected target pool = baseline WR pool + `beta_wr * delta_pass_attempts`.
- TE corrected target pool = baseline TE pool + `beta_te * delta_pass_attempts`.
- RB carries remain exactly baseline. No inverse RB correction is used in Phase F.

Feasibility floor only: a corrected WR or TE pool may not be below zero. Report any floor hits. No upper cap and no tuning.

## Frozen scorecard
Baseline vs Phase-F candidate on the exact same rows:
- QB attempt MAE/RMSE/bias/correlation; must reproduce Phase E corrected values within 1e-9.
- WR target-pool MAE/RMSE/bias/correlation.
- TE target-pool MAE/RMSE/bias/correlation.
- combined WR+TE target-pool MAE/RMSE/bias/correlation.
- RB carry MAE must be numerically identical to baseline because Phase F does not change it.
- Q4 absolute target-error miss counts using thresholds frozen from the Phase-E baseline distribution.
- positive-delta and negative-delta scorecards separately.
- actual PASS_STATE_HIGH and PASS_STATE_LOW rows separately, using Phase-D sign definitions where RB overlap exists.

## Frozen decision rule
Disposition `SEPARATE_PASS_CATCHER_POOL_RESPONSE_ELIGIBLE` only if all are true:
1. exact Phase-E QB corrected-attempt parity within 1e-9.
2. WR target-pool MAE improves by >=0.15 targets.
3. TE target-pool MAE does not worsen by >0.05 targets.
4. combined WR+TE target-pool MAE does not worsen by >0.05 targets.
5. RB carry predictions unchanged exactly.
6. no same/future outcome leakage and zero sportsbook inputs.

Otherwise `SEPARATE_PASS_CATCHER_POOL_RESPONSE_NOT_ELIGIBLE` or mechanical failure.

No gate, mapping form, coefficient, or cohort may be changed after results.

## Required outputs
- `phase_f_result.json`
- `phase_f_casebook.csv`
- `phase_f_scorecard.csv`
- `phase_f_delta_direction_scorecard.csv`
- `phase_f_actual_state_scorecard.csv`

## Interpretation
A Phase-F pass would support a future player-level opportunity integration that preserves M38 within-WR hierarchy and TE-R5 within-TE entitlement while allowing the shared pass-state to move the **WR and TE team pools by different amounts**. A Phase-F fail does not invalidate the pregame QB/WR state signal from Phase E; it means the current mapping from team pass state into pass-catcher pools remains incomplete.
