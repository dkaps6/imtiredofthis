# Migration 81 — Frozen FTN Feature Dictionary

This file freezes M81 feature construction before the first predictive run. It is subordinate to `M81_QB_FTN_NOVEL_MECHANISM_DEVELOPMENT.md` and may not be changed after results to rescue a failed family.

## Temporal boundary

- Target outcomes: 2024 only.
- Fit window: 2024 Weeks 1-9.
- Development holdout: 2024 Weeks 10-18.
- Historical feature source: all 2023 regular-season FTN/PBP plus 2024 games strictly before the target week.
- 2025 target outcomes are not parsed into the modeling frame and are reserved for M82.
- Target-game FTN charting is prohibited.

## Source bridge

FTN charting joins to nflverse play-by-play using `nflverse_game_id -> game_id` and `nflverse_play_id -> play_id`. Required regular weeks are 1-18 in both 2023 and 2024. FTN-PBP play join rate must be at least 95%. Each approved raw field must be at least 80% populated. Canonical-QB to nflverse passer identity mapping must be at least 95%. Any source-contract failure is fail-closed.

## Fixed transforms

For each permitted raw metric and permitted aggregation level, M81 creates exactly:

- trailing-8 game mean;
- trailing-3 game mean;
- trailing-3 minus trailing-8 trend.

If an entity has no prior history, the value falls back to the strictly-prior league mean for the same metric; the trend is therefore zero. No target outcome is used for imputation.

Allowed aggregation levels remain those frozen in the M81 preregistration: target QB history, target offense history, opponent-defense prior allowed/induced history, and same-family QB/offense x opponent interactions only.

## Family definitions

### TACTICAL_CALL_STRUCTURE

Raw FTN fields: `is_motion`, `is_screen_pass`, `is_rpo`.

Features:
- offense trailing-8/trailing-3/trend rates for all three fields;
- opponent-defense prior allowed/induced trailing-8/trailing-3/trend rates for all three fields;
- three fixed trailing-8 offense x defense same-field interactions.

### PRESSURE_RESPONSE

Raw FTN fields: `n_blitzers`, `is_qb_out_of_pocket`, `is_throw_away`, `is_qb_fault_sack`.

Pass/dropback-play denominator is fixed from nflverse `pass_attempt == 1`.

Features:
- QB trailing-8/trailing-3/trend for mean blitzers faced, out-of-pocket rate, throwaway rate, and QB-fault sack rate;
- opponent-defense trailing-8/trailing-3/trend for the corresponding induced/allowed metrics;
- four fixed trailing-8 interactions: QB out-of-pocket x opponent blitzers, QB throwaway x opponent blitzers, QB-fault sack x opponent blitzers, and QB historical blitz exposure x opponent blitzers.

### THROW_DECISION_QUALITY

Raw FTN fields: `is_interception_worthy`, `read_thrown`, `is_catchable_ball`.

Throw denominator excludes sacks. `read_thrown` is parsed only through its documented numeric read index when present; no learned categorical encoding is introduced.

Features:
- QB trailing-8/trailing-3/trend for interception-worthy rate, catchable rate, mean numeric read, and second-read-or-later rate;
- opponent-defense trailing-8/trailing-3/trend for corresponding allowed/induced values;
- four fixed trailing-8 same-metric QB x defense interactions.

### RECEIVER_ERROR_ATTRIBUTION

Raw FTN field: `is_drop`.

Throw denominator excludes sacks.

Features:
- QB trailing-8/trailing-3/trend drop rate;
- offense trailing-8/trailing-3/trend drop rate.

No receiver-created-reception, contested-ball, YAC, separation, coverage-shell, injury, inactive, or generic-pressure variables may enter this family.

## Frozen model and component bounds

Every family uses exactly:

- `StandardScaler`;
- `Ridge(alpha=20, fit_intercept=False)` for attempt residual;
- `Ridge(alpha=20, fit_intercept=False)` for YPA residual.

No alpha tuning, nonlinear fallback, feature subset search, or algorithm search is allowed.

Corrected components use one common outer safety contract for every family:
- attempts: 18 to 48;
- YPA: 4.5 to 10.5.

The canonical baseline itself is always scored from its frozen `pred_pass_yards`; candidates are judged only by incremental correction versus that baseline.

## M82 refit contract

If M81 freezes a candidate, M82 may refit the exact frozen candidate architecture on all 2024 canonical target rows using only 2023 plus strictly-prior 2024 FTN history, then score 2025 once. M82 may not change fields, transforms, interactions, alpha, component bounds, or candidate composition after seeing 2025.
