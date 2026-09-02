# RB STACK6N — Deep-Late Persistence Qualification

## Purpose

STACK6M established that, conditional on realized trailing occupancy, the largest recoverable rushing-tendency error comes from **late multi-score deficits** (`deep_late`: Q4+, score differential <= -9). This is especially strong in P3 team-pool false-high games.

STACK6N asks a narrower pregame question:

> Is the conditional deep-late run-abandonment behavior sufficiently stable and timestamp-safe to estimate from prior football history alone?

This is a team-level mechanism qualification only. It does not authorize player-level recomposition or production changes.

## Frozen lineage

- Parent result: STACK6M `DEEP_LATE_DOMINANT`
- Parent canonical run: `33639765082`
- Parent results commit: `ca463805294de57b301567774eba8d3f785de38f`
- M94C frozen artifact: run `33353485070`
- STACK6H frozen pool-error bins: run `33632678179`
- Sportsbook inputs: forbidden

## Temporal contract

- PBP source seasons loaded: 2023, 2024, 2025 regular season.
- Evaluation: 2025 W6–18 only.
- Any candidate rate for a target `(season, week, team)` may use only games strictly before that target week.
- Target-game PBP may be used only for labels and the conditional-occupancy grading scaffold.
- No target-game injury, participation, box-score, or sportsbook information may enter a candidate estimate.

## Context definitions

- `trail`: score differential < -3.
- `deep_late`: score differential <= -9 and quarter >= 4.
- Offensive play: `rush_attempt == 1` or `qb_dropback == 1`.

## Why the evaluation uses actual occupancy

STACK6N is intentionally **conditional**. STACK6M already isolated the rate problem after supplying realized state occupancy. Therefore STACK6N continues to hold actual lead/neutral/trail/deep-late shares fixed only for grading so that the experiment measures whether a pregame estimate of **conditional deep-late rushing tendency** is useful.

This does not make actual occupancy an allowed production input.

## Frozen candidate estimators

The M94C parent `gs_team_trail_rush_rate_shrunk` remains the generic trailing-state rate.

### Arm A — `LEAGUE_DEEP_LATE_CONTEXT`

For each target game, compute from all strict-prior league PBP available in 2023–target week:

- league trailing rush rate
- league deep-late rush rate
- `league_context_delta = league_deep_late_rate - league_trail_rate`

Then:

`pred_deep_late_rate = clip(parent_trail_rate + league_context_delta, 0.05, 0.75)`

This tests whether M94C mainly omitted a universal football-context penalty.

### Arm B — `TEAM_SHRUNK_DEEP_LATE`

Use the **last 8 prior team games** (spanning seasons if necessary), with no search over window length.

Within that fixed window compute:

- team trailing rush rate
- team deep-late rush rate
- `team_context_delta = team_deep_late_rate - team_trail_rate`

Use the M94B carry-forward pseudo-sample of **24 deep-late plays**:

`w = team_deep_late_plays / (team_deep_late_plays + 24)`

`shrunk_context_delta = league_context_delta + w * (team_context_delta - league_context_delta)`

`pred_deep_late_rate = clip(parent_trail_rate + shrunk_context_delta, 0.05, 0.75)`

If the team has no usable prior deep-late sample, `w = 0` and the arm collapses to the league-context estimate.

No coefficient, window, pseudo-sample, threshold, or clipping bound may be tuned after seeing 2025 results.

## Reconstruction

Keep the frozen M94C 75/25 structured/baseline blend.

Using actual target-game state occupancy only as the conditional grading scaffold:

- lead contribution = actual lead share × parent lead rate
- neutral contribution = actual neutral share × parent neutral rate
- non-deep-late trailing contribution = actual non-deep-late trail share × parent trail rate
- deep-late contribution = actual deep-late share × candidate pregame deep-late rate

Candidate total team rushing:

`0.25 * baseline_team_rush_att + 0.75 * pred_off_plays * summed_rate_contributions`

Also construct `ORACLE_DEEP_LATE` by replacing only the deep-late contribution with realized deep-late rushes / actual offensive plays. This defines the available headroom for this exact component.

## Frozen populations

Primary:
- all W6–18 team-games

Guardrails from STACK6H:
- `POOL_OVER_5`
- `POOL_UNDER_5`
- `POOL_ABS_5`
- `NON_EXTREME_ABS_LT3`

## Required reporting

For parent occupancy baseline, both candidate arms, and deep-late oracle report:

- n
- MAE
- RMSE
- bias
- correlation
- MAE gain vs occupancy baseline

Also report:

- strict-prior coverage
- team prior deep-late-play coverage
- mean / median prior deep-late plays
- direct deep-late-rate MAE where target deep-late plays > 0
- weighted deep-late-rate absolute error weighted by target deep-late plays
- fraction of perfect deep-late oracle headroom recovered

## Frozen retention gates

An arm qualifies as a **retainable conditional-tendency mechanism** only if all are true:

1. Overall W6–18 team-rush MAE gain vs occupancy baseline >= **0.10** attempts.
2. Recovers >= **20%** of the available `ORACLE_DEEP_LATE` MAE headroom.
3. `POOL_OVER_5` MAE gain > **0.20** attempts.
4. `POOL_UNDER_5` MAE regression <= **0.10** attempts.
5. `NON_EXTREME_ABS_LT3` MAE regression <= **0.05** attempts.
6. Overall RMSE does not worsen.
7. Absolute overall bias <= **0.75** attempts.
8. Strict-prior construction coverage = **100%** for evaluated rows.

Interpretation:

- If league arm qualifies but team-shrunk does not materially improve it, the missing signal is mostly universal context physics rather than persistent team identity.
- If team-shrunk improves the league arm and qualifies, team-specific late-deficit behavior is a usable pregame state variable.
- If neither qualifies, historical deep-late play-calling tendency is insufficient and the next search must move to other pregame football information rather than retuning this history formula.

## Prohibitions

- No sportsbook or Vegas inputs.
- No 2025 hyperparameter, feature, threshold, window, or blend search.
- No target-game PBP in candidate construction.
- No player-level correction in STACK6N.
- No production change from a passing conditional-only result; predicted occupancy must still be integrated and validated in a later stack before promotion.
