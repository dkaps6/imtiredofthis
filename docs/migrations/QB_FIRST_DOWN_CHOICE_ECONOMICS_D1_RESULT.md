# QB First-Down Choice Economics D1 — Result

## Canonical execution

- Branch: `research-qb-first-down-choice-economics-d1`
- Frozen plan commit: `a915d46db36cddb69650ea932ca49f98a3dacf42`
- Evaluator commit: `2516ae353970ee2326ad50b1256438c66042beed`
- Canonical execution head: `777012a4be50a59d34f8944164ad5d3722f4e269`
- Run: `34545121616`
- Job: `103095961998`
- Artifact: `10178725142`
- Artifact name: `qb-first-down-choice-economics-d1`
- Digest: `sha256:8839cdcb74d942c68779e959f88e825a2f693bb0e6b1ebfe13955efc261e5020`
- Disposition: `FIRST_DOWN_CHOICE_ECONOMICS_D1_FAIL_NO_CONFIRMATION`
- Confirmation authorized: `false`
- Production actionable: `false`

## Integrity

All scientific-boundary/integrity requirements passed:

- exact 544 2023 source rows, unique by team-week, with all four qualified descriptors finite;
- exact 2023 target-PBP alignment and finite first-down DBR outcomes;
- all baseline values finite and strictly prior;
- 2022 plus strictly-prior 2023 PBP only;
- no 2024/2025 target PBP or outcomes read;
- no QB/WR target outcomes or parent residuals read;
- zero sportsbook/game-market inputs;
- zero production changes.

The run is therefore a valid scientific failure, not a mechanical failure.

## Frozen 2023 development design

- fit: Weeks 1-9, 272 team-weeks;
- holdout: Weeks 10-18, 272 team-weeks;
- one equal-weight weekly-percentile `CHOICE_EDGE` from the four qualified offense/opponent first-down pass-minus-run EPA/success descriptors;
- one non-negative no-intercept coefficient;
- no alternative feature weighting, intercept, model family, or hyperparameter.

Fitted coefficient:

- `beta_raw = 0.0039004976640005097`
- `beta = 0.0039004976640005097`

## Holdout result

### Baseline

- MAE: `0.09766808438457152`
- RMSE: `0.12467684666125418`
- bias: `+0.01074199156606688`
- correlation: `0.3046972232200466`
- p90 absolute error: `0.20446511447019247`
- mean predicted first-down DBR: `0.5121930149194076`
- mean actual first-down DBR: `0.5014510233533407`

### Choice-economics candidate

- MAE: `0.09769250172660522`
- RMSE: `0.12468356046755949`
- bias: `+0.01080652185830218`
- correlation: `0.3043818527611781`
- p90 absolute error: `0.20448433567262322`
- mean predicted first-down DBR: `0.5122575452116429`

MAE gain:

- `-0.000024417342033705713` — candidate slightly worse.

## Residual relationship

Holdout correction magnitude was tiny:

- mean correction: `0.00006453029223530255`
- mean absolute correction: `0.0006271895621645615`
- p90 absolute correction: `0.0012493781580001633`

Correction versus baseline holdout residual:

- Pearson: `0.001566950557873544`
- Spearman: `-0.03005549739430238`
- same-sign rate: `0.4852941176470588`

This is effectively no out-of-sample relationship.

## Stability / bootstrap

- holdout weeks with positive MAE gain: `4 / 9`
- paired bootstrap support for positive MAE gain: `0.303`

## Frozen advancement gates

Passed:

- source integrity;
- target-PBP alignment;
- strict-prior baseline;
- positive beta;
- all leakage/sportsbook/production boundaries.

Failed:

- holdout MAE gain >=0.005;
- RMSE non-worse;
- absolute bias non-worse;
- p90 absolute error non-worse;
- correction/residual Spearman >=0.15;
- >=6/9 holdout weeks won;
- paired-bootstrap support >=0.90.

## Scientific conclusion

The source itself was exceptionally dense and clean, but the frozen first-down **relative pass-vs-run efficiency economics** score did not predict week-specific first-down pass-origin choice out of sample in 2023.

Therefore:

- do not inspect 2024/2025 to rescue the family;
- do not retune descriptor weights;
- do not replace percentile ranks with z-scores;
- do not add an intercept;
- do not change the coefficient form;
- do not loosen gates;
- do not promote anything to production.

The exact first-down choice-economics family is closed.

The broader mechanism finding remains intact: first-down within-state play selection is still the primary shared QB/receiver opportunity-error layer. This D1 result only rejects one candidate explanation for that layer.
