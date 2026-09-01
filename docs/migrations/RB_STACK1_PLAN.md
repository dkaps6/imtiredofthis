# RB-STACK1 — Production-Equivalent RB Historical Baseline + Integration Audit

## Purpose

Establish the leakage-safe historical RB baseline for the complete canonical football stack before integrating new RB capability modules. This is not a replacement contest against M94C. It creates the correct parent system for later `full stack + enriched M94C/backfield allocation` tests.

## Frozen questions

1. What do the canonical MC (Bayesian + empirical football rules + joint Monte Carlo), supervised ML v2, and Markov State v2 components achieve for RB rushing attempts and rushing yards on 2025?
2. What does the canonical nonnegative MC/ML/State ensemble achieve when weights are learned only from prior out-of-sample football results?
3. How does that full-stack parent compare with the M91 ML-only baseline and frozen M94C on the exact common 2025 RB/FB games?
4. Which important production-context fields are actually present in the historical reconstruction, especially role/depth, injuries, weather, box context and rule application?

## Temporal protocol

- Reconstruct 2024 OOS component predictions using only 2023 + pre-target-2024 information.
- Reconstruct 2025 OOS component predictions using only 2024 + pre-target-2025 information.
- `ensemble_preseason_2024`: fit canonical market-specific nonnegative weights on all eligible 2024 OOS component rows, freeze those weights, apply them to every 2025 game.
- `ensemble_expanding`: for each 2025 week, fit the same canonical weight function on all 2024 OOS rows plus only earlier 2025 OOS weeks, then apply to the target week. This is a leakage-safe adaptive diagnostic; it is labeled separately from the preseason-frozen deployable baseline.
- Sportsbook lines are not used anywhere in prediction construction, weight fitting or gates.
- Target-game actuals are attached only after prediction generation and are used only for evaluation/slicing.

## Production-equivalence caveat

The repository's actual full-slate pricing path explicitly falls back to MC-only when no calibrated ensemble artifact is present. Therefore STACK1 must report both:

- `MC_CANONICAL` — the canonical production fallback parent;
- calibrated ensemble variants — what the canonical evidence-weighted ensemble framework does when valid historical weights are supplied.

Do not silently call a calibrated research ensemble the current production behavior if the production checkout lacks its weights artifact.

## 2025 primary evaluation population

- positions RB/FB/HB where present;
- markets `rush_att` and `rush_yards`;
- primary all-player-game metrics: MAE, RMSE, bias, correlation, actual mean and prediction mean;
- exact-common comparison with frozen M94C using its authoritative 2025 trace.

## Diagnostic slices

Evaluation-only slices include:

- Week 1 vs Weeks 2-18;
- actual carries 0-5 / 6-10 / 11-14 / 15-19 / 20+ / 25+ for rushing-yards diagnosis;
- RB vs FB/HB where sample permits;
- rows with/without historical role/depth labels;
- rows with/without injury context;
- rows with/without box-rate context.

Actual-carry slices are postgame diagnostics only and may never become pregame routing features.

## Context coverage audit

Report coverage for at least:

- `rules_applied`;
- player `role` and `rules_role`;
- `ctx_rush_share_available`;
- `ctx_success_rate_available`;
- `ctx_pace_available`;
- `ctx_proe_available`;
- `ctx_pressure_available`;
- `ctx_explosive_available`;
- `ctx_def_epa_available`;
- `ctx_box_rates_available`;
- `ctx_injury_available`;
- `ctx_weather_available`.

No silent placeholder behavior is acceptable.

## Frozen comparison arms

For each supported RB market:

1. `MC_CANONICAL`
2. `ML_V2_M91_COMPONENT`
3. `STATE_V2`
4. `ENSEMBLE_2024_FROZEN`
5. `ENSEMBLE_EXPANDING`
6. `M94C` on exact common rows only

No new model features, coefficients, threshold search, sportsbook input or M94C retuning occurs in STACK1.

## Decision / next step

STACK1 does not promote a new RB production model by itself. It identifies the strongest legitimate parent and the information already present or missing. Immediately afterward, preserve both complementary tracks:

- enrich M94C/team opportunity with timestamp-safe role/backfield football information;
- test that enriched M94C/backfield-allocation capability as an add-on to the strongest full-stack parent through precommitted ablations.

Retained M95F workload-distribution, M95I vacancy/transition, and M95C/M96 efficiency/environment capabilities follow only after the opportunity/allocation integration is understood.
