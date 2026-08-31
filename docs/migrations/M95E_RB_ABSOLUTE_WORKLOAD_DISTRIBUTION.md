# M95E — Absolute RB Workload Distribution

## Question

Can we stop compressing true 20–30 carry RB games toward ~15 carries by explicitly modeling the missing allocation layer between total team rushing volume and the RB room?

M95E tests this football decomposition:

`full team rush attempts × RB-room share of team rushes × player share of RB-room rushes = player carries`

It also builds a workload distribution around the resulting mean and direct pregame 20+/25+ carry probability models.

## Frozen inputs

- M95B leakage-safe RB matchup trace: run `33357785600`.
- M94C football-only game-environment output: run `33353485070`.
  - M95E uses `structured_team_rush_att` as the upstream estimate of **all non-kneel team rush attempts**, not the M94C blended projection-universe total.
- M94D trace: run `33354419964`.
  - Used only to recover the exact M94C RB projection for apples-to-apples comparison on the 2024 holdout and 2025 validation.
- nflverse PBP is used to construct the actual all-team non-kneel rushing denominator for the RB-room-share target.

No sportsbook fields are inputs. No production projection code is changed.

## Pregame features

RB-room share uses lagged team RB-pool, total-rush, QB-rush-share, RB-count/concentration, rush-rate, neutral/early-down run tendency, play volume, QB scramble share, opponent rushing-volume history, and home/away context.

Player share uses lagged carries, RB-room share, 15+/20+ workload frequency, role flags, team concentration/QB-rush context, rush-rate/play-volume context, and opponent RB-volume history.

All feature values come from pregame rolling history already frozen in M95B.

## Protocol

1. Fit candidate share architectures on 2023 through 2024 W12.
2. Select architecture/blend only on 2024 W13–18.
3. A candidate is development-eligible only if it improves the 20+ and 25+ carry slices while keeping all-RB MAE within 0.05 carries of M94C and both 6–10 and 11–14 slices within 0.10 carries.
4. Freeze the selected mean architecture, tail classifier family, probability thresholds, and distribution calibration.
5. Refit share/tail models on all 2023–2024.
6. Score untouched 2025 once.

Pre-specified mean families are Ridge, gradient boosting, and random forest. Each is tested as (a) explicit RB-room × player-share decomposition and (b) direct absolute team-rush-share prediction, with fixed prior/model blends of 0.50, 0.75, and 1.00.

Tail models are class-balanced logistic regression and random forest classifiers for 20+ and 25+ carries. Thresholds are selected on the 2024 holdout only.

The distribution layer uses the frozen M94C structured team-rush mean plus effective team-volume and absolute-share dispersion calibrated on the 2024 holdout. It emits p50/p75/p90/p95 carries and simulated 20+/25+ probabilities. This is uncertainty modeling, not an arbitrary tail boost.

## Required diagnostics

- all RB; actual 0–5, 6–10, 11–14, 15+, 20+, 25+; bellcow-60 slices
- exact M94C vs M95E MAE, bias, RMSE, correlation
- projected counts at 18/20/22/25 carries
- actual 20+/25+ mean projected carries and p90/p95 values
- tail AUC, precision, recall, and frozen-threshold F1
- RB-room-share and lead-RB absolute-share calibration
- all-team RB-pool allocation accuracy
- false high-workload predictions and actual-25+ false negatives
- distribution coverage

## Promotion rule

Research-only. `ADVANCE_M95E_COMPONENT_FOR_INTEGRATION_REVIEW` requires the development constraint to pass and untouched 2025 to improve all-RB, 20+, and 25+ carry MAE without more than a 0.10-carry regression in either middle slice. Even that status does **not** modify production automatically.
