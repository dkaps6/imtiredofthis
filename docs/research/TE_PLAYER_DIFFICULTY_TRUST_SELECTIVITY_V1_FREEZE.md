# TE Player Difficulty Trust / Selectivity V1 — Frozen Plan

Date: 2026-10-15
Status: FROZEN BEFORE TRUST-SCORE SCORING / RESEARCH ONLY

Version:
`TE_PLAYER_DIFFICULTY_TRUST_SELECTIVITY_V1`

## Question

The TE same-player allocation-difficulty signal is confirmed, while two exact
distribution-transmission laws are closed:

1. Normal/MAE -> joint Dirichlet/Beta-Binomial;
2. Normal target sigma -> pregame YPT -> independent receiving-yard marginal width.

The second closure nevertheless showed that named-player difficulty improves the
canonical TE-R5P marginal versus canonical in both 2024 and 2025, while a pooled
TE width control is better on CRPS.

Do **not** force the signal into another width law.

Test a distinct player-first question:

> Does strictly-prior named-TE allocation difficulty identify when the existing
> canonical TE-R5P receiving-yard projection/distribution is reliable versus
> fragile, after removing the contemporaneous week-wide TE environment?

This is a trust/selectivity shadow only.

It does not change:
- projection mean;
- distribution width or shape;
- target entitlement;
- teammate allocation;
- any sportsbook price;
- any production decision.

## Frozen source authority

Use only the exact successful historical marginal result rows from:
- run `38800926331`;
- job `116649432539`;
- exact head `e7961d5bf6abe9097a8fbf21d776fb1696a57bba`;
- artifact `12003898778`;
- digest `sha256:0c59e6e551e6ac025ac21e4098c5298a3a6ac3e16ecef9f7cd5e8d2bb0db73a4`.

Required parent disposition:
`NO_ACTIONABLE_TE_PLAYER_ALLOCATION_MARGINAL_REC_YARDS`.

The source rows already contain:
- strictly-prior same-player `player_mae`;
- contemporaneous `week_pool_mae`;
- canonical TE-R5P receiving-yard mean `baseline_mean`;
- canonical receiving-yard MC SD `baseline_sd`;
- canonical empirical CRPS `baseline_crps`;
- realized receiving yards;
- player/game/week identities.

No new historical model fit is authorized.

## Population

Primary:
- TE rows in the canonical marginal-result artifact;
- 2024 development;
- 2025 independent confirmation within this frozen trust test;
- finite `player_mae > 0`;
- finite `week_pool_mae > 0`;
- finite `baseline_sd > 0`;
- finite nonnegative actual receiving yards;
- exact player/game/week identity.

The parent already enforces minimum same-player prior support.

No TE1/TE2 or player subgroup carveout.

## Frozen trust score

For each season/week independently:

1. order eligible TEs by `player_mae`;
2. assign average percentile rank:
   `trust_fragility_score = rank(player_mae, method=average, pct=true)`.

Interpretation:
- lower score = historically more stable / higher trust;
- higher score = historically more difficult / more fragile.

Because `week_pool_mae` is common to the contemporaneous TE week state, the
within-week rank removes that pooled level by construction. Report
`player_mae / week_pool_mae` descriptively, but it is **not** an alternate
candidate and may not replace the frozen score.

No coefficient is fit.

## Frozen reliability outcomes

Primary:
`standardized_crps = baseline_crps / baseline_sd`.

Secondary:
`standardized_abs_error = abs(actual_rec_yards - baseline_mean) / baseline_sd`.

Also report raw:
- canonical CRPS;
- absolute mean error.

The standardized outcomes are primary/secondary because they test reliability
beyond the scale of the canonical player's own distribution.

## Within-week ranking control

For inference, also construct within each season/week:

`crps_badness_rank = percentile_rank(standardized_crps)`

`mean_error_badness_rank = percentile_rank(standardized_abs_error)`.

Primary monotonic statistic:
- Pearson correlation between `trust_fragility_score` and
  `crps_badness_rank`.
  Since both are ranks, this is the frozen within-week rank association.

Secondary monotonic statistic:
- same construction using `mean_error_badness_rank`.

Positive correlation means more difficult named players are less reliable even
after removing week-level pooled conditions and canonical distribution scale.

## Frozen trust buckets

Within each season/week:
- HIGH_TRUST = `trust_fragility_score <= 0.25`;
- LOW_TRUST / FRAGILE = `trust_fragility_score >= 0.75`;
- middle rows are retained for correlation but excluded from the bucket contrast.

These thresholds are fixed before scoring and are prospectively computable from
the current TE cohort.

## Dependence-aware inference

For the primary rank association:
- player-cluster bootstrap;
- game-cluster bootstrap;
- 5,000 replicates;
- resample whole clusters with replacement;
- retain all rows in each sampled cluster;
- recompute the correlation on the sampled rows;
- fixed seeds in implementation.

Report both:
- pooled 2024+2025 bootstrap;
- 2025-only bootstrap.

For bucket contrasts, report:
- mean standardized CRPS HIGH_TRUST vs LOW_TRUST;
- mean standardized absolute error HIGH_TRUST vs LOW_TRUST;
- raw CRPS and raw absolute error for interpretation.

No fitted regression coefficient is required or allowed for qualification.

## Frozen qualification gates

Support/integrity:
1. 2024 >=600 rows and >=60 TEs;
2. 2025 >=600 rows and >=60 TEs;
3. pooled >=1,300 rows and >=90 TEs;
4. each season has >=120 HIGH_TRUST rows and >=120 LOW_TRUST rows;
5. sportsbook inputs = 0;
6. OddsAPI calls = 0;
7. 2026 outcomes read = 0;
8. production changed = false;
9. projection/distribution values changed = 0.

Primary CRPS trust:
10. 2024 primary rank association > 0;
11. 2025 primary rank association > 0;
12. pooled player-cluster 95% CI low > 0;
13. pooled game-cluster 95% CI low > 0;
14. 2025 player-cluster 95% CI low > 0;
15. 2025 game-cluster 95% CI low > 0;
16. 2024 LOW_TRUST standardized CRPS > HIGH_TRUST standardized CRPS;
17. 2025 LOW_TRUST standardized CRPS > HIGH_TRUST standardized CRPS;
18. 2024 HIGH_TRUST standardized CRPS < season overall standardized CRPS;
19. 2025 HIGH_TRUST standardized CRPS < season overall standardized CRPS;
20. 2024 LOW_TRUST standardized CRPS > season overall standardized CRPS;
21. 2025 LOW_TRUST standardized CRPS > season overall standardized CRPS.

Secondary mean-error alignment:
22. 2024 secondary rank association > 0;
23. 2025 secondary rank association > 0;
24. 2024 LOW_TRUST standardized abs error > HIGH_TRUST standardized abs error;
25. 2025 LOW_TRUST standardized abs error > HIGH_TRUST standardized abs error.

Disposition:
- all gates pass:
  `TE_PLAYER_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`
- otherwise:
  `NO_ACTIONABLE_TE_PLAYER_DIFFICULTY_TRUST_SELECTIVITY`.

## If qualified

This is still research-only.

Immediately freeze 2026 Week 6 pregame TE trust labels using football-only
authority, before reading any Week 6 outcomes.

Preferred Week 6 football source:
- no-odds Full Slate run `38790425676`;
- artifact `11999618082`;
- digest `sha256:d7473705d4534c3172b58843272e7d95f5fbbd9aa910238d48916a8567cde1de`.

The Week 6 shadow may only label/rank player trust. It may not alter mean,
distribution, price, or bet signal.

Production promotion requires prospective scoring after outcomes settle.

## If failed

Close this exact trust score from this exact TE allocation-difficulty state.

Do not rescue with:
- alternate rank transforms;
- fitted thresholds;
- fitted regression coefficients;
- TE1/TE2 or post-result player subgroups;
- different prior windows/minimum support;
- raw-error-only qualification;
- alternate normalization;
- combining seasons to hide a failed 2025 confirmation.

Then move to the next genuinely distinct named-player mechanism.

## Leakage / reuse note

2024/2025 outcomes were used in prior TE experiments, including the marginal
width test. The trust statistic and gates above are frozen before computing this
new trust/selectivity score, but these seasons are not untouched external data.

Therefore even a historical qualification cannot promote production by itself;
the Week 6 prospective lock is mandatory before any production consideration.

No paid OddsAPI.
No production mutation.
