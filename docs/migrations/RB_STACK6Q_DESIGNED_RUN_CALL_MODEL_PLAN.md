# RB STACK6Q — Pregame Designed-Run-Call Model Qualification

## Status

**FROZEN BEFORE 2025 SCORING.**

No production change is authorized by this plan. STACK6Q is a temporal research qualification of the predictive family identified by STACK6P.

## Why this exists

STACK6P showed that designed-run behavior explains roughly 90-92% of the remaining within-state rushing-tendency oracle recovery, robustly across two strict-prior attribution schemes and in both large false-high and false-low P3 pool misses.

A naive decomposition of the existing M94C state rush rate into designed + scramble + kneel rates would be algebraically identical if all three components used the same historical window and shrinkage. STACK6Q therefore does **not** test that identity. It tests whether designed-run tendency deserves its own pregame predictive mechanism.

## Frozen football architecture

Preserve from M94C without retuning:

1. frozen baseline team-rush projection;
2. frozen pregame offensive-play prediction;
3. frozen pregame predicted lead / neutral / trail occupancy;
4. frozen final blend weight `alpha = 0.75` between baseline and structured team-rush projection.

Replace only the state-conditioned total rush-tendency term.

For each score state `s`:

`pred_total_rush_rate_s = pred_designed_rate_s + prior_scramble_rate_s + prior_kneel_rate_s`

Then:

`structured_team_rush = pred_off_plays * sum_s(pred_state_share_s * pred_total_rush_rate_s)`

and:

`STACK6Q_team_rush = 0.25 * M94C_baseline_team_rush + 0.75 * structured_team_rush`

No blend-weight search is allowed.

## Event definitions

Using nflverse play-by-play:

- offensive play: `rush_attempt == 1 OR qb_dropback == 1`
- scramble: `qb_scramble == 1`
- kneel: `qb_kneel == 1`
- designed rush: `rush_attempt == 1 AND qb_scramble != 1 AND qb_kneel != 1`

Designed QB runs remain in the designed component. This is not an RB-share model.

Score states use the exact M94B/M94C definition:

- lead: offensive score differential `> +3`
- neutral: `-3 <= score differential <= +3`
- trail: offensive score differential `< -3`

## Temporal protocol

- source seasons: 2023-2025 PBP plus frozen M94C artifacts;
- all predictors for a target team-week must use games strictly before that target week;
- fit designed-rate models on 2024 target rows only;
- evaluate 2025 only;
- primary scoring population: 2025 W6-18;
- late stability population: 2025 W13-18;
- no 2025 outcome may select features, model family, hyperparameters, clips, or blend weights.

Target-game PBP is label/evaluation truth only.

## Frozen designed-rate learner

Three separate state models: `lead`, `neutral`, `trail`.

Each is:

- `SimpleImputer(strategy="median")`
- `StandardScaler()`
- `Ridge(alpha=10.0)`

No model-family search and no alpha search.

Predicted designed rate is clipped to `[0.00, 0.80]` before recomposition.

Rows with zero realized plays in the target state do not supply a state-rate training label for that state.

## Frozen feature set

Exactly the following pregame concepts are allowed for each state model. Rolling values are strict-prior.

### Team designed-run history

1. team state designed rate — prior 1 game
2. team state designed rate — prior 3 games
3. team state designed rate — prior 5 games
4. team state play count over prior 5 games
5. team overall designed-run rate — prior 3 games
6. team overall designed-run rate — prior 5 games
7. team neutral early-down designed-run rate — prior 3 games

### Opponent defensive designed-run history

8. opponent state designed-run rate allowed — prior 1 game
9. opponent state designed-run rate allowed — prior 3 games
10. opponent state designed-run rate allowed — prior 5 games
11. opponent overall designed-run rate allowed — prior 3 games
12. opponent overall designed-run rate allowed — prior 5 games

### QB-designed-run environment, team aggregate only

13. team QB designed-run share of offensive plays — prior 3 games
14. opponent QB designed-run share allowed — prior 3 games

These are team-level strict-prior rates. No target-week QB participation or postgame QB identity is permitted.

### Frozen M94C pregame environment

15. predicted mean margin
16. predicted final margin
17. M94C margin blend
18. absolute M94C margin blend
19. home indicator
20. predicted state occupancy for the state being modeled

Exactly 20 features are permitted. No feature search.

## Scramble and kneel nuisance rates

Scramble and kneel rates are **not fitted** in STACK6Q.

For each state, each is estimated from the team's prior five games and shrunk to the strict-prior league state rate with the same pseudo-play count used by M94B (`24.0`).

This keeps the low-value components stable while allowing the STACK6P-dominant designed component to vary through its own predictive model.

## Comparators

Primary comparator:

- frozen `M94C candidate_team_rush_att` on the identical 2025 team-games.

STACK6Q also reports the frozen M94C baseline team-rush projection for context but does not change the benchmark.

## Frozen scoring gates

STACK6Q qualifies for downstream RB-pool recomposition only if **all** of the following pass on 2025 W6-18 versus M94C:

1. team-rush MAE gain `>= 0.20` attempts;
2. team-rush RMSE gain `> 0`;
3. team-rush correlation gain `>= +0.02`;
4. absolute bias may not worsen by more than `0.25` attempts;
5. MAE gain in frozen `POOL_OVER_5` team-games `> 0`;
6. MAE gain in frozen `POOL_UNDER_5` team-games `> 0`;
7. W13-18 MAE gain `> 0`.

`POOL_OVER_5` / `POOL_UNDER_5` are inherited from the frozen STACK6H/P3 grading trace and may be used only for grading, never as model features or target selection.

No gate may be waived after scoring.

## Dispositions

- `STACK6Q_DESIGNED_RUN_MODEL_QUALIFIED` — all gates pass; authorize a separate downstream P3 RB-pool recomposition test, not production.
- `STACK6Q_DESIGNED_RUN_MODEL_NOT_QUALIFIED` — any scientific gate fails; no retune on 2025.
- `STACK6Q_INTEGRITY_FAILURE_DO_NOT_INTERPRET` — temporal, reconstruction, coverage, or artifact identity failure.

## Integrity requirements

The run must report and enforce:

- exact 2025 M94C team-game join coverage;
- W6-18 sample count matching the current frontier (`388` unless an explicitly documented upstream identity correction changes it);
- strict-prior feature coverage / construction flag;
- zero sportsbook inputs;
- zero target-game participation or injury inputs;
- target-game PBP used only as labels/grading truth;
- zero feature search;
- zero hyperparameter search;
- zero model-family search;
- fixed M94C alpha `0.75`;
- untouched frozen M94C play and state-occupancy predictions.

## Promotion boundary

Even a full STACK6Q pass does **not** promote a production RB model. It only authorizes the next controlled step: apply the qualified team-rush prediction through the already-frozen P3 RB-share bridge and determine whether the improvement survives at the team RB carry pool and then player level.

Vegas / sportsbook information remains downstream benchmark only and is prohibited from this predictive model.
