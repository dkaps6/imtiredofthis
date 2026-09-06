# RB Individual Error / Role-Transition Audit — Result

## Disposition

`DIAGNOSTIC_ONLY_NO_PRODUCTION_CHANGE`

No sportsbook input, no model fitting, no production change.

## Canonical evidence

- Run: `34064260950`
- Job: `101570151670`
- Tested SHA: `0c17511718cd61bbd757befa9da30dd7311ba4b6`
- Artifact: `9998437758` (`rb-individual-error-role-transition`)
- Digest: `sha256:9177e2688517efcce05ea0be7667ac4c8321a813bbadf7fdb26b1f3c418af191`

## Integrity / population

- exact canonical rows: `1,393`
- individual RB/HB/FB profiles: `148`
- depth coverage: `0.949748743718593`
- depth-vs-projected-carry-order mismatch rows: `413`
- limited-prior-history rows: `106`
- no-prior-same-team rows: `71`
- rookie rows: `268`
- injury-created-context rows: `70`

## Role-state findings

The previous failed Role-Order Remap V1 is reinforced: **depth-order mismatch itself is not the error concentration**.

- mismatch rows carry MAE `3.320` vs non-mismatch `3.551`
- mismatch rows rush-yard MAE `18.02` vs non-mismatch `21.44`

So we should not rescue a direct depth-order correction.

The clearest adverse state in this diagnostic is **rookie status for carries**:

- rookies carry MAE `3.776` vs veterans `3.413`
- rookie carry bias `-1.337` vs veterans `-0.750` (projection minus actual), i.e. the baseline underprojects rookie carries more strongly
- rookie carry 5+ miss rate `26.49%` vs `23.11%`

The combination `depth mismatch + rookie` is worse still:

- carry MAE `3.939` vs `3.458`
- carry bias `-1.813` vs `-0.811`

But this is diagnostic, not a frozen correction gate, and the sample is only 72 rows for the combined state.

Depth missing entirely is also associated with worse error:

- carry MAE `4.194` vs `3.445` when depth is present
- rush-yard MAE `24.61` vs `20.20`

Limited prior history, no-prior-same-team, and injury-created context did **not** show broad MAE deterioration in this 2025 diagnostic. Injury-created rows were actually nearly unbiased on average.

## Individual-player profiles

The individual scorecard exposes large persistent differences hidden by positional MAE. Examples among 2025 high-game players:

- Christian McCaffrey: 17 games, carry MAE `6.97`, carry bias `-5.35`, rush-yard MAE `33.69`, yard bias `-25.02`
- Ashton Jeanty: 17, carry MAE `6.33`, bias `-3.44`, rush-yard MAE `25.65`
- Rico Dowdle: 17, carry MAE `6.08`, rush-yard MAE `36.84`
- Derrick Henry: 17, carry MAE `5.96`, rush-yard MAE `44.46`, yard bias `-31.01`
- James Cook: 17, carry MAE `5.92`, rush-yard MAE `46.53`, yard bias `-32.22`
- Bijan Robinson: 17, rush-yard MAE `42.81`, yard bias `-23.87`
- Jahmyr Gibbs: 17, rush-yard MAE `41.40`
- Jonathan Taylor: 17, yard bias `-23.16`

These are retrospective full-sample 2025 profiles. They must not be inserted into a target-game projection as known information.

## Scientific conclusion

The user's individual-player concern is materially supported. Some RBs are repeatedly much harder for the current model than others, and several high-volume backs show persistent underprojection in carries and/or yards.

The next legitimate RB test is a **walk-forward individual-error persistence diagnostic**: before each target game, calculate only that player's prior model MAE/bias and test whether those prior-only quantities predict the next miss. Rookie state can be retained as a predeclared context slice, but we should not create a rookie boost from this retrospective result without a separately frozen test.
