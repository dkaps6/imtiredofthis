# TE-R3 Target-Pool Context Model — Frozen Plan

## Status
Frozen before implementation/results. Research-only. Production unchanged.

## Purpose
TE-R1 found TARGETS are the largest TE receiving-yard error mechanism. TE-R2 then located the majority of TE target error at the **team TE target-pool layer**:
- pooled TEAM_TE_POOL absolute mass share: 60.4050%;
- highest-error quartile: 72.0726%;
- every 2020-2025 season independently routed `TE_TARGET_POOL_FIRST`.

TE-R3 therefore tests whether a leakage-safe **pregame football-context model** can improve the amount of target opportunity assigned to the team TE room and, through the existing within-room player shares and efficiency mechanics, improve individual TE projections.

This is not a generic target lift. The candidate must demonstrate differential game/team/opponent corrections and improve individual projection quality.

## Lineage
- Parent TE-R2 branch/result commit: `research-te-r2-pool-vs-individual-allocation` / `debe5efa59dc998c3bef8a7cc1141002302f3555`.
- TE-R2 canonical run: `34126026512`; artifact `10020131404`; digest `sha256:54309eda0886cd0b327333ea89bde12a2fc506d141e2b72ddc3b1d1d778b874d`.
- Exact source: Joint Pass/Receiving Conservation V1 run `34081764151`, artifact `10004223287`, digest `sha256:753aa191e6c80a059918553d8567499c9cdbce82b12a26b4d5b19b21225764ac`.
- Baseline: B0/current TE opportunity and receiving-yard path.
- Sportsbook inputs: 0.

## Modeling unit
Team-game TE target pool.

The source casebook already contains B0 pregame expected targets and actual receiving outcomes for every receiving position. Opponent is inferred only from the other team in the same `event_id`; no market information is used.

## OOS design
Expanding-season walk-forward:
- train 2020-2021 -> predict 2022;
- train 2020-2022 -> predict 2023;
- train 2020-2023 -> predict 2024;
- train 2020-2024 -> predict 2025.

Every historical-outcome feature for a target team-game must be computed with a strict `shift(1)` or equivalent chronological cutoff. Target-game outcomes are forbidden.

## Exact target and baseline
For team-game g:
- `b0_te_pool = sum(B0 expected targets for TE players)`;
- `actual_te_pool = sum(actual TE targets)`;
- target residual = `actual_te_pool - b0_te_pool`.

Candidate predicts the residual and forms:
`candidate_te_pool = max(0, b0_te_pool + clip(predicted_residual, -3.0, +3.0))`.

The +/-3 target correction cap is frozen before results.

## Frozen feature set
All features are football-only.

### Current pregame model/game-environment features
1. `b0_te_pool`
2. `b0_total_target_pool` = B0 expected targets summed over all receiving positions on team-game
3. `b0_te_target_share` = b0_te_pool / b0_total_target_pool
4. `b0_te_room_size` = count of TEs with B0 expected targets >=0.25
5. `b0_top_te_share` = largest individual B0 TE expected-target share of the TE room
6. `b0_te_hhi` = sum of squared individual B0 TE room shares

These preserve the existing model's player role/room structure and game-level expected pass opportunity.

### Strict-prior team usage/history
7. `team_te_pool_prior1`
8. `team_te_pool_prior4`
9. `team_te_pool_season_to_date`
10. `team_total_targets_prior4`
11. `team_total_targets_season_to_date`
12. `team_te_share_prior4`
13. `team_te_share_season_to_date`
14. `team_te_rec_yards_prior4`

### Strict-prior opponent/defensive matchup
15. `opp_te_targets_allowed_prior4`
16. `opp_te_targets_allowed_season_to_date`
17. `opp_total_targets_allowed_prior4`
18. `opp_te_target_share_allowed_prior4`
19. `opp_te_rec_yards_allowed_prior4`
20. `opp_te_receptions_allowed_prior4`

These are defensive matchup statistics from games strictly prior to the target game.

No same-game result, sportsbook line, price, or target-game NGS observation is allowed.

## Missing-history handling
- Rolling features require at least one strictly prior observation.
- Missing prior-history numeric values are imputed using the **training-fold median only**.
- Missingness indicators are not added in this candidate.

## Frozen model
`StandardScaler + Ridge(alpha=20.0, fit_intercept=True)`.

No hyperparameter search. No feature selection after results. One frozen candidate only.

## Translation back to individual TE projections
Candidate team TE pool is allocated using the existing B0 within-room shares:
`candidate_player_targets = candidate_te_pool * b0_player_te_room_share`.

Existing B0 target-conditioned conversion is preserved:
- `b0_rec_per_target = b0_receptions / b0_expected_targets` when denominator >0;
- `b0_rec_yards_per_target = b0_rec_yards / b0_expected_targets` when denominator >0.

Then:
- `candidate_receptions = candidate_player_targets * b0_rec_per_target`;
- `candidate_rec_yards = candidate_player_targets * b0_rec_yards_per_target`.

This isolates the TE target-pool mechanism. TE-R3 does not alter catch-rate or YPR mechanics.

## Frozen scorecard
Pooled OOS and by target season:

### Team TE pool
- MAE, RMSE, bias, correlation;
- median/p75/p90 absolute error;
- 3+/5+ miss rates.

### Individual TE targets
- MAE, RMSE, bias, correlation;
- median/p75/p90 absolute error;
- 2+/4+/6+ miss rates.

### Individual TE receiving yards
- MAE, RMSE, bias, correlation;
- median/p75/p90 absolute error;
- 20+/30+/40+ miss rates.

### Player-tier stability
Evaluate individual results separately for B0 TE opportunity quartiles, including the high-opportunity Q4 tier.

### Correction behavior
Report correction mean, SD, p10/p50/p90, cap-hit rate, fraction positive, fraction negative, and maximum absolute correction.

## Frozen integrity gates
1. OOS team-games >=1,800.
2. OOS TE player-games >=3,500.
3. all four OOS seasons represented.
4. exact B0 reconstruction/parity gap <=1e-9 before candidate correction.
5. every prior-history feature demonstrably uses only rows earlier than target row; zero same/future outcomes.
6. sportsbook inputs = 0.

## Frozen scientific gates
All must pass for `TE_TARGET_POOL_CONTEXT_MODEL_ACTIONABLE`:

1. pooled team TE-pool MAE improvement >= **0.15 targets**;
2. pooled individual TE target MAE improvement >= **0.05 targets**;
3. pooled individual TE receiving-yard MAE improvement >= **0.40 yards**;
4. team-pool MAE improves in at least **3 of 4** OOS seasons;
5. individual receiving-yard MAE improves in at least **3 of 4** OOS seasons, including combined 2024-2025;
6. individual receiving-yard p90 does not worsen by > **0.50 yards**;
7. 30+ receiving-yard miss rate does not worsen by > **0.5 percentage points**;
8. 40+ receiving-yard miss rate does not worsen by > **0.5 percentage points**;
9. high-opportunity Q4 receiving-yard MAE does not worsen by > **0.25 yards** and 30+/40+ miss rates each do not worsen by >0.5pp;
10. correction is meaningfully differential rather than a generic lift: correction SD >= **0.50 targets**, positive corrections >=15% of OOS team-games, and negative corrections >=15%.

Integrity passes but any scientific gate fails -> `TE_TARGET_POOL_CONTEXT_MODEL_FAIL`.
Integrity failure -> `TE_TARGET_POOL_CONTEXT_INTEGRITY_FAIL`.

## Interpretation discipline
A lower RMSE/p90 cannot rescue worsened MAE/individual miss rates. A generic positive correction cannot be called player-centric even if aggregate bias improves. No thresholds may change after results.

## Authorized next step
- If actionable: integrate only the validated TE-pool mechanism into a full-stack walk-forward `simulation_v2` candidate, preserving the separate individual allocation/catch/YPR layers.
- If fail: do not retune Ridge/cap. Route next work to a genuinely new source/mechanism, most likely current TE participation/route involvement or coverage/personnel matchup information, after source integrity is proven.
