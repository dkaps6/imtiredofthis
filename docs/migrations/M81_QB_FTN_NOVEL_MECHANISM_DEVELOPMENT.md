# Migration 81 — QB FTN Novel Mechanism Predictive Development Screen

## Status

`PREREGISTERED / DEVELOPMENT ONLY`

Migration 81 is the first predictive migration allowed by the Migration 80 source-frontier contract. It must not change production logic and it must not inspect 2025 target outcomes while choosing a candidate.

## Canonical foundation

- Canonical QB foundation: `qb_frontier_canonical_v3_football_only`
- Canonical snapshot SHA256: `c4a481a760657bb7516d52b9aa9ba84af096063a4e78660b12a541f59fd7b742`
- Canonical rows: 884 total (2024=444, 2025=440)
- No sportsbook / market columns may enter football features, cohort selection, fitting, or promotion gates.
- M80 remains the authoritative M1-M79 no-retest crosswalk.

## Frozen scientific boundary

M81 is **development only**.

- Historical FTN source history may use 2023 plus strictly-prior 2024 charting to construct pregame features.
- Predictive development and family selection use **2024 canonical target rows only**.
- **2025 canonical target outcomes are untouched and prohibited from candidate selection.**
- A single frozen M81 winner, if any, is reserved for a separate M82 confirmation run on 2025.
- Target-game FTN charting is never a pregame feature.

## M80-approved information families

Only the four preregistered M80 families below may enter M81.

### 1. `TACTICAL_CALL_STRUCTURE`

Fields:
- `is_motion`
- `is_screen_pass`
- `is_rpo`

Closest prior work: M67/M68 generic formation/opening tendency and M70 YAC/explosive decomposition.

Material novelty: explicit tactical call identity and pre-snap motion, not generic formation/pass tendency.

### 2. `PRESSURE_RESPONSE`

Fields:
- `n_blitzers`
- `is_qb_out_of_pocket`
- `is_throw_away`
- `is_qb_fault_sack`

Closest prior work: M9, M16, M22-M23, M45, M56, M69, M70, M72 aggregate pressure/dropback work.

Material novelty: exact blitz construction plus quarterback response/attribution, not another aggregate pressure-strength transform.

### 3. `THROW_DECISION_QUALITY`

Fields:
- `is_interception_worthy`
- `read_thrown`
- `is_catchable_ball`

Closest prior work: M70 completion/CPOE/interception decomposition and M71 volatility.

Material novelty: charted decision/process quality and read progression, not ordinary INT rate, CPOE, YPA, or volatility.

### 4. `RECEIVER_ERROR_ATTRIBUTION`

Fields:
- `is_drop`

Closest prior work: M34 catch conversion and M70 completion decomposition.

Material novelty: manual receiver-error attribution separating receiver failure from quarterback throw quality.

## Explicitly prohibited retests

M81 must not reopen, rename, or recombine as standalone candidates:

- shotgun / QB location
- backfield count / generic personnel tendency
- no-huddle
- generic play action
- generic pass-rusher count / pressure rate
- generic game-script / DBR / pass-rate state
- generic coverage-shell frequencies
- contested-ball as a standalone receiver-matchup family
- created-reception / generic receiver YAC creation
- M72/M75 receiver matchup/tracking families
- official inactives / depth-chart discontinuity / generic injury burden
- new Ridge/HGB/XGB/ensemble variants on already-rejected feature universes

## Feature construction contract

Every M81 feature must be observable before kickoff.

For each target QB-game, FTN features are built only from games strictly before the target `(season, week)`.

Allowed aggregation levels:

1. QB historical behavior for the target quarterback.
2. Offense historical behavior for the target team.
3. Opponent-defense allowed / induced behavior from prior games.
4. Predeclared QB × opponent interactions only when both terms come from the same approved M80 family.

No target-game charting, target-game attempts, target-game YPA, target-game passing yards, or postgame participation variable may be used as a feature.

All feature families must meet the M80 source contract before fitting:

- required historical seasons available
- regular weeks 1-18 source coverage complete
- family fields >=80% populated in every required source season
- source errors fail closed

## Prediction targets

M81 preserves the canonical component decomposition.

- Attempt residual: `actual_attempts - pred_attempts`
- YPA residual: `actual_ypa - implied_pred_ypa`
- Corrected attempts: canonical predicted attempts + predicted attempt residual
- Corrected YPA: canonical implied predicted YPA + predicted YPA residual
- Corrected passing yards: corrected attempts × corrected YPA

No direct sportsbook line fitting.

## Frozen model discipline

M81 is an **information-family test, not a model-zoo test**.

Each family receives the same single standardized linear residual architecture:

- `StandardScaler`
- `Ridge(alpha=20, fit_intercept=False)`
- one attempt-residual model
- one YPA-residual model

No per-family alpha search, nonlinear fallback, feature subset search, post-result clipping search, threshold tuning, or algorithm replacement is allowed inside M81.

The same component safety bounds used by the canonical research framework must be applied consistently to every family.

## 2024 development split

M81 uses a frozen temporal development structure:

- feature history may include 2023 and earlier 2024 games strictly before each target week
- fit window: 2024 Weeks 1-9 canonical rows
- development holdout: 2024 Weeks 10-18 canonical rows

If the early fit window does not contain sufficient rows for a family, the family returns `INSUFFICIENT_DEVELOPMENT_HISTORY`; M81 must not borrow 2025 outcomes to rescue it.

## Independent family gates

A family is a `DEVELOPMENT_SURVIVOR` only if **all** gates pass on 2024 Weeks 10-18:

1. passing-yard MAE gain vs canonical >= **0.75 yards**
2. passing-yard correlation gain >= **0.015**
3. passing-yard RMSE is non-worse
4. 100+ yard misses do not increase
5. attempt MAE improves by >= **0.10 attempts** **or** YPA MAE improves by >= **0.03 YPA**
6. paired bootstrap `P(pass-yard MAE gain > 0)` >= **0.70**
7. source/feature coverage contract passes with no fail-open condition

These are development-screen gates only. Passing them does not promote production logic.

## Survivor-stack rule

- If zero families survive: M81 closes `NO_FTN_DEVELOPMENT_SURVIVOR`; no M82 predictive confirmation is run.
- If exactly one family survives: freeze that family as the M82 candidate.
- If two or more genuinely distinct families survive independently: M81 may fit **one** preregistered combined survivor stack using the union of only the independently surviving families, with the same Ridge architecture.
- The combined stack is scored only on the same 2024 development holdout.
- The lowest 2024 holdout passing MAE among the independently surviving single families and the allowed combined stack becomes the single frozen M82 candidate; RMSE then correlation then 100+ misses are tie-breakers.

No second combined-stack search is allowed.

## M82 boundary

M82, not M81, owns untouched 2025 confirmation.

The frozen M81 candidate must enter M82 unchanged:

- same source fields
- same aggregation definitions
- same model architecture
- same alpha
- same safety bounds
- same candidate composition

M82 may reject the candidate but may not tune it on 2025.

## M81 outputs required

- source snapshot / hashes
- source-contract summary
- feature dictionary with prior-migration crosswalk
- per-family coverage report
- 2024 development predictions
- component metrics by family
- passing-yard metrics by family
- paired bootstrap summary
- gate table
- survivor decision
- if applicable, one survivor-stack result
- frozen M82 candidate contract JSON

## Production disposition

`production_actionable = false`

M81 changes no production football projection logic regardless of result.
