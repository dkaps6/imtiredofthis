# CROSS-POSITION PHASE E — PREGAME SHARED-STATE + PLAYER SENSITIVITY VALIDATION V1

## Purpose
Phase A-D reverse-engineered the largest individual-player misses and identified two potentially actionable mechanisms:

1. one latent pass-vs-run game state can move QB attempts, WR/TE targets, and RB carries together;
2. individual high-opportunity players may differ materially in how strongly their yardage responds to generic favorable/adverse environments.

Phase E is the first **prospective/pregame validation** of those findings. It is not a feature hunt and it does not change production.

No Phase-E target or label may use sportsbook information. All predictors must be available before the evaluated game. All model fitting is walk-forward.

---

# E1 — Shared pass-state predictability

## Frozen source
Use authoritative Phase-C `cross_position_all_rows_game_spot.csv` unchanged.

Construct team-games from frozen model rows.

### Primary pass-state cohort
Use team-games with:
- one primary QB (largest predicted QB opportunity),
- >=1 WR,
- >=1 TE.

Primary evaluation is 2025. 2024 rows may train 2025 predictions but may never be scored as though they had future data.

### Full pass/run cohort
Where RB-P3 rows also exist, add RB opportunity evaluation. Current authoritative overlap is expected to be 2025 only; do not fabricate earlier RB authority.

## Frozen pregame feature set
Team-level predictors are restricted to fields already present in Phase C and known pregame:
- `pass_opportunity_spot`
- `pass_efficiency_spot`
- `rush_opportunity_spot`
- `rush_efficiency_spot`
- predicted primary-QB attempts
- predicted WR+TE targets
- predicted WR targets
- predicted TE targets
- predicted RB carries when available
- week number encoded numerically

No actual same-game outcome field may be a predictor.

## Shared latent correction
Target for the primary model is primary-QB `actual_opportunity - pred_opportunity`.

Model is frozen as:
- StandardScaler
- Ridge(alpha=20)

Walk-forward fitting: for every scored 2025 team-game, train only on chronologically earlier eligible team-games. Require >=128 prior team-games; otherwise emit baseline/no correction.

The model produces `delta_pass_attempts`.

Within each training fold only, estimate zero-intercept linear mappings:
- receiver target residual ~ beta_receiver * QB attempt residual
- RB carry residual ~ beta_rb * QB attempt residual (when prior RB overlap >=64 games)

Apply:
- corrected QB attempts = baseline + delta_pass_attempts
- corrected WR+TE target pool = baseline + beta_receiver * delta_pass_attempts
- corrected RB carries = baseline + beta_rb * delta_pass_attempts

This is a diagnostic opportunity correction only. No yardage efficiency, entitlement share, or production simulation is changed.

## Frozen scorecard
Report baseline vs corrected, on the exact same scored rows:
- QB attempt MAE/RMSE/bias/correlation
- WR+TE target-pool MAE/RMSE/bias/correlation
- WR target-pool MAE
- TE target-pool MAE
- RB carry-pool MAE where correction is available
- sign accuracy of QB attempt residual
- sign accuracy of WR+TE target residual implied by the shared correction
- catastrophic opportunity misses by position/pool (Q4 absolute-error threshold, frozen from baseline distribution)

Also report PASS_STATE_HIGH / PASS_STATE_LOW recall using Phase-D sign definitions, but these are secondary diagnostics.

## E1 decision rule
`SHARED_PASS_STATE_PREGAME_ELIGIBLE` only if all are true:
1. QB attempt MAE improves by >=0.15 attempts.
2. WR+TE target-pool MAE does not worsen by >0.05 targets.
3. At least one of WR or TE target-pool MAE improves and neither worsens by >0.10.
4. QB attempt residual sign accuracy >=0.55 on rows receiving a correction.
5. RB carry MAE does not worsen by >0.15 where RB correction is available.
6. No leakage/integrity failure.

Otherwise disposition is `SHARED_PASS_STATE_PREGAME_NOT_ELIGIBLE` or mechanical failure.

No threshold may be changed after results.

---

# E2 — Prospective player environment-sensitivity validation

Phase-D full-sample labels are descriptive only. Phase E must create labels from earlier games and evaluate them strictly later.

## Frozen splits
- WR: train 2020-2023, test 2024-2025.
- TE: train 2023-2024, test 2025.
- QB: train 2024, test 2025 exploratory only.
- RB: no prospective label validation from current authoritative RB-P3 lineage because only 2025 is available; report `INSUFFICIENT_TEMPORAL_HOLDOUT` rather than inventing one.

## Eligibility and label definitions
Use the exact Phase-D eligibility and candidate-label definitions on **training rows only**. No test-game outcome may influence eligibility, thresholds, medians, slopes, or labels.

## Frozen test metrics
For each training-derived label and player, evaluate later held-out games:
- held-out favorable mean residual
- held-out adverse mean residual
- held-out favorable-minus-adverse residual
- held-out spot slope
- held-out catastrophic-under rate by spot
- label-direction consistency

Group scorecards by position and training-derived label.

### Validation interpretation
- `ENVIRONMENT_SENSITIVE_CANDIDATE` validates if held-out favorable-minus-adverse residual > 0 and held-out slope > 0.
- `ENVIRONMENT_RESISTANT_CANDIDATE` validates if absolute held-out favorable-minus-adverse residual <=20% of the position catastrophic threshold.
- `ADVERSE_SPOT_CEILING_CANDIDATE` validates if held-out adverse mean residual >=0.
- `FAVORABLE_SPOT_AMPLIFIER_CANDIDATE` validates if held-out favorable mean residual > held-out adverse mean residual.

These are diagnostic validations, not automatic production multipliers.

## E2 decision rule
A player-environment concept becomes `PROSPECTIVE_PLAYER_ENVIRONMENT_SIGNAL` for a position/label only if:
- >=5 eligible held-out players for the label,
- >=60% of eligible players validate the label direction,
- pooled held-out residual contrast has the expected sign,
- no leakage/integrity failure.

Otherwise retain as descriptive only.

---

# Integrity gates
- sportsbook features used = 0
- same-game/future outcomes used as predictors = 0
- test rows used to create training-derived player labels = 0
- production parameters changed = 0
- exact Phase-C source lineage recorded

# Required outputs
- `phase_e_result.json`
- `phase_e_shared_state_casebook.csv`
- `phase_e_shared_state_scorecard.csv`
- `phase_e_shared_state_signature_recall.csv`
- `phase_e_player_train_labels.csv`
- `phase_e_player_holdout_casebook.csv`
- `phase_e_player_holdout_scorecard.csv`
- `phase_e_player_holdout_by_player.csv`

## Interpretation rule
A descriptive forensic relationship is not enough. Phase E is specifically asking whether it was **available and useful before kickoff**. Failure here does not invalidate Phase D's football relationship; it means the current pregame feature representation cannot exploit it reliably yet.
