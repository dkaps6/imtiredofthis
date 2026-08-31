# M95L — Sealed Temporal Confirmation Plan

Research-only. M95K advanced its two-regime workload-tail architecture to sealed confirmation. This migration must not search new features, shrinkage values, regularization, transforms, thresholds, or probability-mass rules.

## Frozen architecture

- stable-workhorse feed model spec: `feed_compact_env`
- empirical-Bayes shrink: `k=4`
- logistic regularization: `C=.03`
- stable 20+ probability-mass-preserving rerank
- stable 25+ method: frozen conditional 25|20 ratio plus probability-mass anchor
- vacancy branch: frozen M95I recipient-specific deep-concentration + tail architecture
- other RBs: frozen M95F tail architecture
- central carry estimate: M94C football-only opportunity architecture; no tail mean boost
- no sportsbook input
- no production change

## Sealed temporal rotation

Primary confirmation period: **2023 Weeks 13-18**.

Every target-row feature must be available before kickoff. Models that require fitted coefficients are trained/calibrated using **2023 Weeks 1-12 only**, with temporal OOF construction where calibration requires it. No 2023 W13-18 labels may be used for fitting, calibration, feature selection, hyperparameter selection, probability anchoring to outcomes, or architecture changes.

The required 2023 M91 component projections are rebuilt prospectively from 2022 history using the original walk-forward pipeline. The M94C football-environment layer is reconstructed with the already-frozen M94C model families and blend (`mean margin GBR`, `final margin RF`, `plays RF`, `state mapper Ridge`, `alpha=.75`) rather than reselecting them on 2023. M95F/M95H/M95I/M95K model families and constants are likewise frozen from the prior research sequence.

Probability-mass anchoring may use only the baseline predicted probability mass in the confirmation population, never confirmation outcomes.

## Confirmation gates

Report at minimum:

- stable-workhorse 20+ AUC/Brier/log-loss versus frozen baseline;
- stable-workhorse 25+ AUC/Brier/log-loss;
- vacancy 25+ ranking/calibration where sample size supports a meaningful metric;
- all-RB 20+/25+ metrics;
- stable probability-mass preservation;
- M94C central-carry preservation and ordinary carry slices;
- source/data coverage and any reconstruction differences.

A scientific confirmation requires stable 20+ and stable 25+ to improve ranking without worsening Brier, full-population 20+/25+ not to regress materially, and the vacancy mechanism not to be contradicted. Tiny rare-event samples must be labeled as such rather than waived.

If a faithful 2023 reconstruction cannot be completed, M95L must fail closed rather than substitute a weaker test.

## Authoritative sealed result — Run #5

- workflow run: `33429747106`
- job: `99611940386`
- SHA: `caa9401eb50f6980e2a2c35ddd8e54467f57cbef`
- execution conclusion: **success**
- confirmation period: **2023 Weeks 13-18**
- M94C player join: **1.000000** after the source-verified GSIS identity bridge
- feature search: **0**
- coefficient search: **0**
- sportsbook inputs: **0**
- probability-mass preservation: **passed**
- M94C central carry reference: **preserved**
- scientific confirmation: **failed**
- disposition: `M95K_SEALED_TEMPORAL_CONFIRMATION_FAILED`

The sealed result is a scientific failure, not a mechanical failure. Mechanical reconstruction issues were fixed before Run #5 without changing the frozen modeling architecture or confirmation gates.

### Stable-workhorse 20+

- `n=73`, positive events `24`
- M95F AUC `0.727041` -> M95L `0.545068` (`-0.181973`)
- M95F Brier `0.233221` -> M95L `0.244446` (worse by `0.011225`)
- M95F log loss `0.673392` -> M95L `0.710790`
- mean probability remained `0.161895`, confirming the mass-preserving rerank did not create extra aggregate tail probability.

### Stable-workhorse 25+

- `n=73`, positive events `10`
- M95F AUC `0.533333` -> M95L `0.442857` (`-0.090476`)
- M95F Brier `0.123614` -> M95L `0.126356` (worse by `0.002742`)
- M95F log loss `0.461608` -> M95L `0.478042`
- mean probability remained `0.060139`.

### Full population

20+ (`n=453`, 44 positives):
- AUC `0.880974` -> `0.856746`
- Brier `0.074541` -> `0.076462`

25+ (`n=453`, 14 positives):
- AUC `0.813537` -> `0.808005`
- Brier `0.028838` -> `0.029277`

Both full-population nonregression gates failed.

### Vacancy branch

Vacancy 25+ had `n=24` and **zero positive events** in the sealed period, so AUC is undefined. The branch is correctly labeled `inconclusive_small_n`; it was not contradicted, but this period does not independently validate it.

### Central-carry preservation

M95L did not alter the M94C central carry projection. MAE and bias are identical to M94C in every carry slice. The sealed sample still shows the known high-workload underprojection:

- actual 20+ games (`n=44`): MAE/bias magnitude `7.937524`
- actual 25+ games (`n=14`): MAE/bias magnitude `11.632081`

## Scientific disposition

Do **not** promote the M95K stable-workhorse feed/carry-ceiling rerank. Its 2025 research improvement did not generalize to the untouched 2023 W13-18 confirmation period and materially worsened stable-workhorse ranking and calibration despite preserving probability mass.

Do not tune M95K against the now-exposed 2023 confirmation labels. Any follow-up hypothesis must be treated as a new research migration with a new validation design. M95I vacancy evidence remains promising but unconfirmed by this sealed period because vacancy 25+ produced no positive events.
