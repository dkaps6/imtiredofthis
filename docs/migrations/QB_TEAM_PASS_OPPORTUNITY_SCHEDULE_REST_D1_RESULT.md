# QB Team Pass Opportunity Schedule/Rest D1 — Result

## Disposition

`SCHEDULE_REST_D1_FAIL_NO_CONFIRMATION`

2025 confirmation is **not authorized**. Production is unchanged.

## Canonical lineage

- Branch: `research-qb-team-pass-opportunity-schedule-rest-d1`
- Frozen plan: `e42db98b4ba06e52af8e76dd0e76f3548db8c4be`
- Evaluator commit: `fbc388a638da8b2e6ec4f8152f3b5fba2519774d`
- Tested/workflow head: `4a40ae2ebdd9b13755360ea92fe43e0f451dc669`
- Run: `34533794810`
- Job: `103060549944`
- Artifact: `10174577290` (`qb-team-pass-opportunity-schedule-rest-d1`)
- Artifact digest: `sha256:e7c5d1359bb9db6fae4d850321a75b17dfb728d5c839327a6ac90f925c97095b`
- Source-audit parent run: `34533408818`
- Opportunity-chain parent run: `34523313743`

## Integrity

All frozen integrity gates passed.

- 2024 Weeks 1-9 fit rows: `235`
- 2024 Weeks 10-18 holdout rows: `209`
- 2025 target rows used for fit/selection/scoring: `0`
- sportsbook/result features used: `0`
- frozen feature count: `8`
- Ridge alpha: `20.0`
- baseline attempt identity max gap: `1.7763568394002505e-14`
- production changes: `false`

## Development result

### Team pass opportunity

Baseline vs candidate on 2024 Weeks 10-18:

- MAE: `7.430990 -> 6.629129` (**+0.801861 improvement**)
- RMSE: `9.401454 -> 8.442953`
- bias: `-4.270967 -> -0.772702`
- correlation: `0.051195 -> 0.095262`
- p90 absolute error: `14.368626 -> 13.637886`

The frozen upstream opportunity gate passed.

### QB attempts

- MAE: `7.480993 -> 6.578211` (**+0.902782 improvement**)
- RMSE: `9.641849 -> 8.574700`
- bias: `-5.025480 -> -2.061295`
- correlation: `-0.026540 -> 0.014260`
- 8+ attempt miss rate: `41.1483% -> 32.0574%`
- 10+ attempt miss rate: `27.2727% -> 21.5311%`

The frozen attempt gates passed.

### QB passing yards after propagation through unchanged M89/M90 downstream correction

- MAE: `59.542131 -> 63.218888` (**3.676756 yards worse**)
- RMSE: `74.326809 -> 78.189179` (worse)
- bias: `-1.772961 -> +19.490501`
- correlation: `0.121186 -> 0.107229` (worse)
- p90 absolute error: `118.568280 -> 127.615505` (worse)
- 100+ miss rate: `17.2249% -> 21.0526%` (worse)

Paired bootstrap on passing-yard MAE gain:

- observed gain: `-3.676756`
- `P(gain > 0) = 0.0089`
- 5th / 50th / 95th bootstrap gain: `-6.192575 / -3.670233 / -1.136774`

## Why D1 failed

The candidate learned an overwhelmingly positive opportunity correction:

- mean correction: `+3.498265` team pass opportunities
- mean absolute correction: `3.498265`
- minimum correction: `+0.242426`
- maximum correction: `+7.202349`
- p90 absolute correction: `5.600634`

This repaired a real raw M89 opportunity/attempt underprojection in the development holdout, but the promoted M89/M90 synthesis mean had already corrected most of the downstream passing-yard bias. Applying the new positive opportunity correction **on top of** the unchanged synthesis adjustment therefore double-counted correction and pushed passing yards high.

This is not evidence that team pass opportunity is unimportant. The parent diagnostic remains authoritative: team pass opportunity is the primary remaining attempt-error mechanism and carries the shared QB/WR opportunity miss.

D1 instead shows that a new upstream correction cannot simply be added beneath the existing synthesis adjustment without accounting for overlap.

## Scientific interpretation

The next question is architectural, not another schedule/rest retune:

> Does the existing pregame M89/M90 synthesis mean already contain an implicit opportunity-volume correction that can be re-expressed upstream while preserving the exact QB passing-yard mean?

If yes, that would create a path to a shared QB/receiver opportunity state without adding another correction to the QB mean.

## Stopping rule honored

- no 2025 confirmation;
- no alpha retune;
- no feature subset search;
- no schedule/rest transformation search;
- no penalty/fourth-down rescue;
- no production change.
