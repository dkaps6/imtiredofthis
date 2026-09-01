# M96C — M94C-Anchored RB Efficiency Residual Synthesis

## Status

Research-only. No sportsbook inputs. No production changes. No carry adjustment.

## Question

Can leakage-safe player/offense/defense rushing-efficiency information explain the rushing-yard residual left by the frozen M94C opportunity/yard point, without materially damaging ordinary or high-workload regimes?

## Frozen source contract

- M94C authoritative source: run `33353485070`, artifact `migration-94c-rb-game-environment`, file `m94c_2025_rb_trace.csv`.
- M95D/M95C feature source: run `33359898917`, artifact `migration-95d-rb-rushing-environment-scheme`, file `m95d_rb_environment_trace.csv`.
- The frozen M94C artifact persists a player-level rushing-yard point for 2025 only. M96C therefore uses strict expanding-week 2025 out-of-fold evaluation rather than reconstructing a synthetic 2024 M94C player point.
- 2025 is development/inspected evidence, not pristine confirmation. Any survivor remains research-only until genuinely prospective 2026 confirmation.

## Frozen target / opportunity contract

M94C carries and M94C rushing-yard point are frozen.

For training rows with at least 5 actual carries:

`M94C implied YPC = M94C rush yards / M94C rush attempts`

`actual YPC = actual rush yards / actual rush attempts`

`efficiency residual = actual YPC - M94C implied YPC`

Each test-week candidate predicts only that efficiency residual. The final candidate yard point is:

`M94C rush yards + M94C rush attempts * predicted efficiency residual`

No candidate may alter M94C rush attempts.

For each expanding-week fit, the training efficiency residual is winsorized at that training set's 5th/95th percentiles; predicted residuals are clipped to those same training-only bounds. No validation-week outcomes enter this clipping.

## Temporal protocol

- Evaluate Weeks 6–18 of 2025.
- For each test week `w`, train only on 2025 Weeks `< w`.
- Minimum 150 eligible training rows required.
- Ridge alpha is frozen at `10.0` for all point-residual arms.
- Median imputation + standardization are fit on training rows only.
- No hyperparameter search, no coefficient search, no after-result retuning.

## Predeclared capability blocks

### E — blocking / rushing environment

- `pfr_ybc_per_att_avg3`, `pfr_ybc_per_att_avg5`
- `ngs_expected_yards_per_att_avg3`, `ngs_expected_yards_per_att_avg5`
- `ngs_percent_attempts_gte_eight_defenders_avg3`, `ngs_percent_attempts_gte_eight_defenders_avg5`
- `ngs_avg_time_to_los_avg3`, `ngs_avg_time_to_los_avg5`
- `team_pfr_ybc_per_att_avg3`, `team_pfr_ybc_per_att_avg5`
- `team_pbp_stuff_rate_avg3`, `team_pbp_stuff_rate_avg5`
- `rel_ybc_vs_team_avg3`, `rel_ybc_vs_team_avg5`

### P — player-created efficiency

- `pfr_yac_per_att_avg3`, `pfr_yac_per_att_avg5`
- `pfr_brk_tkl_per_att_avg3`, `pfr_brk_tkl_per_att_avg5`
- `ngs_ryoe_per_att_avg3`, `ngs_ryoe_per_att_avg5`
- `ngs_rush_pct_over_expected_avg3`, `ngs_rush_pct_over_expected_avg5`
- `rel_yac_vs_team_avg3`, `rel_yac_vs_team_avg5` when available

### D — opponent run efficiency / resistance

- `def_rush_ypa_allowed_avg3`, `def_rush_ypa_allowed_avg5`
- `def_rush_epa_allowed_avg3`, `def_rush_epa_allowed_avg5`
- `def_rush_success_allowed_avg3`, `def_rush_success_allowed_avg5`
- `def_rush_first_down_rate_allowed_avg3`, `def_rush_first_down_rate_allowed_avg5`
- `def_non_scramble_ypa_allowed_avg3`, `def_non_scramble_ypa_allowed_avg5`
- `def_stuff_rate_allowed_avg3`, `def_stuff_rate_allowed_avg5`
- `def_rb_ypc_allowed_avg3`, `def_rb_ypc_allowed_avg5`
- `def_rb_over_prior5_rush_yards_allowed_avg3`, `def_rb_over_prior5_rush_yards_allowed_avg5`

### X — explosive/upside context (tail-only primary role)

- player 10+/15+/20+ explosive rushing rates, avg3/avg5
- team 10+/20+ explosive rushing rates, avg3/avg5
- defense 10+/15+/20+ explosive rates allowed, avg3/avg5

X is tested against 75+/100+ rushing-yard events using an expanding-week logistic model. Baseline is M94C rushing-yard point only; candidate is M94C point + X. Logistic `C=0.1` is frozen. X is not allowed to alter the point estimate in M96C.

## Frozen point arms

- `C` — M94C baseline only
- `C+E`
- `C+P`
- `C+D`
- `C+E+P`
- `C+E+D`
- `C+P+D`
- `C+E+P+D`

These are ablations, not a search over arbitrary subsets or weights.

## Point retention gate

A point-residual arm is globally retainable only if all are true on Weeks 6–18 OOF:

1. all-RB MAE gain vs C is at least `0.25` rushing yards;
2. all-RB RMSE does not worsen;
3. absolute bias does not worsen by more than `1.0` yard;
4. no workload slice with at least 50 rows among 0–5, 6–10, 11–14, 15–19, and 20+ actual carries regresses MAE by more than `1.0` yard;
5. Weeks 13–18 all-RB MAE does not regress by more than `0.50` yard.

If multiple arms pass, choose the smallest feature-count arm within `0.10` MAE yards of the best passing arm. This is development selection only and still requires prospective confirmation.

If an arm improves important regimes but fails the global non-degradation gate, classify it as a conditional-expert clue; do not globally promote it.

## Tail-only X retention gate

For 75+ and 100+ rushing yards, compare expanding-week probability models C vs C+X.

X is retainable as a tail-only module only if:

- at least one threshold improves AUC by `>=0.01` or Brier by `>=0.001`; and
- neither threshold regresses AUC by more than `0.01`; and
- neither threshold regresses Brier by more than `0.002`.

Report full Weeks 6–18 and late Weeks 13–18 metrics.

## Required reporting

- source/join coverage and exact truth parity;
- feature availability by block;
- all point arms: MAE, RMSE, bias, correlation;
- workload slices: 0–5, 6–10, 11–14, 15–19, 20+, 25+ actual carries;
- Weeks 13–18 stability;
- 75+/100+ AUC/Brier/logloss for C and C+X;
- module capability ledger: RETAIN / REJECT / CONDITIONAL_CLUE;
- representative casebook of largest positive/negative corrections;
- exact disposition and next migration.

## Stopping logic

- If a global point arm passes, freeze the smallest compatible winner for prospective/conditional confirmation.
- If blocks help different regimes but fail global non-degradation, do not average them blindly; the next step may be a precommitted conditional-expert/routing audit.
- If no efficiency block has useful global or conditional signal, retain M94C and move toward prospective confirmation rather than opening unlimited retrospective variants.
