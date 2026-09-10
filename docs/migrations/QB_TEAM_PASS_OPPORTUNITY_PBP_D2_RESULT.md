# QB Team Pass Opportunity PBP D2 — Result

## Disposition

`D2_NO_INDEPENDENT_SURVIVOR`

Neither source-eligible PBP family cleared the frozen independent development gates. The combined candidate was therefore not authorized or run. 2025 confirmation is not authorized. Production is unchanged.

## Canonical lineage

- Branch: `research-qb-team-pass-opportunity-pbp-d2`
- Frozen plan: `e8b06bbab7fc96b2fda008815f5cf92a25736a92`
- Evaluator commit: `65f04cfbef1ff3f73fff81831a3024ebaa46a98d`
- Tested/workflow head: `8346a68e8b60256359901192fb47a1d948f80203`
- Run: `34534830257`
- Job: `103063893342`
- Artifact: `10174978673` (`qb-team-pass-opportunity-pbp-d2`)
- Artifact digest: `sha256:cc481741a7480cce011db3828a23a5c8c534725eded52a845fd4df187b833a6c`
- V1B source run: `34534419071`
- Opportunity-chain run: `34523313743`
- Shared QB/WR run: `34066549394`

## Integrity

All frozen integrity gates passed.

- 2024 M89 rows: `444`
- fit rows Weeks 1-9: `235`
- holdout rows Weeks 10-18: `209`
- 2025 target outcomes used: `0`
- sportsbook/result features used: `0`
- exact last-8 history: PASS
- penalty features: exact 2
- fourth-down features: exact 6
- Ridge: alpha `20.0`, `fit_intercept=False`
- strict-prior history verification: PASS
- baseline attempt identity max gap: `1.7763568394002505e-14`
- production changed: `false`

## PENALTY_DRIVE_EXTENSION

Disposition: `D2_INDEPENDENT_FAIL`

This family showed a small upstream signal but did not survive the complete football/share protection gates.

### Team pass opportunity
- MAE: `7.430990 -> 7.237135` (**+0.193855 improvement**)
- RMSE: `9.401454 -> 9.216307`
- bias: `-4.270967 -> -3.800196`
- correlation: `0.051195 -> 0.043092`
- p90 abs error: `14.368626 -> 14.061522`

### QB attempts
- MAE: `7.480993 -> 7.362848` (**+0.118145 improvement**)
- RMSE: `9.641849 -> 9.487888`
- 10+ miss rate: `27.2727% -> 24.4019%`

### QB passing yards
- MAE: `59.542131 -> 59.680636` (**0.138504 worse**)
- RMSE: `74.326809 -> 74.624220` (worse)
- correlation: `0.121186 -> 0.114418` (worse)
- p90 abs error: `118.568280 -> 118.058632`
- 100+ miss rate unchanged at `17.2249%`

### Information transfer
- correction vs actual team-pass-opportunity residual Spearman: `0.039405`
- correction vs 2024 WR reception-mass residual Spearman: `0.052081`
- bootstrap `P(pass-yard MAE gain > 0) = 0.3495`

The family therefore failed the residual-information, shared-receiver, and passing-yard gates despite modest raw opportunity/attempt improvement.

## FOURTH_DOWN_AGGRESSION

Disposition: `D2_INDEPENDENT_FAIL`

### Team pass opportunity
- MAE: `7.430990 -> 7.347024` (**+0.083966 improvement**, below gate)
- RMSE: `9.401454 -> 9.386859`
- correlation: `0.051195 -> 0.008132`

### QB attempts
- MAE: `7.480993 -> 7.438554` (**+0.042439**, below gate)
- 10+ miss rate: `27.2727% -> 28.2297%` (worse)

### QB passing yards
- MAE: `59.542131 -> 62.339599` (**2.797467 worse**)
- RMSE: `74.326809 -> 77.533971`
- correlation: `0.121186 -> 0.058103`
- p90 abs error: `118.568280 -> 125.522411`
- 100+ miss rate: `17.2249% -> 19.6172%`

### Information transfer
- correction vs actual team-pass-opportunity residual Spearman: `0.078903`
- correction vs 2024 WR reception-mass residual Spearman: `-0.105819`
- bootstrap `P(pass-yard MAE gain > 0) = 0.0009`

This family clearly failed.

## Combined candidate

Not run. The frozen plan permitted `PENALTY_PLUS_FOURTH_DOWN` only if both independent families survived. Independent survivors: `[]`.

## Scientific meaning

Penalty-created first downs contain a small amount of aggregate opportunity information, but not enough stable game-level directional information to support a shared QB/receiver team-pass-opportunity correction. Fourth-down aggression is not supported as a useful opportunity predictor under the frozen test.

Together with Schedule/Rest D1, the project has now tested and rejected three genuinely new possession/opportunity candidate families without disturbing M89/M90 production.

The parent mechanism result remains authoritative: `TEAM_PASS_OPPORTUNITY` is still the primary physical source of remaining attempt error and the shared QB/receiver opportunity miss. D2 says these particular pregame explanatory families do not predict that state strongly enough.

## Stopping rule honored

- no combined model;
- no feature/window/alpha retune;
- no 2025 confirmation;
- no schedule/rest retest;
- no production mutation.
