# QB Pass-Opportunity Rate Designed-Run D1 — Frozen Development Plan

## Purpose

Test exactly one mechanistic use of the newly source-qualified, previously orphaned QB designed-run signal against the current fixed 57% pass-opportunity-rate foundation.

This is a 2024 development screen only. The 2025 cohort remains reserved for a separate confirmation run and may not influence candidate selection, thresholds, coefficient choice, or stopping rules.

The experiment asks:

> Does strict-prior QB designed-run burden explain enough week-specific departure from the fixed 57% pass-opportunity rate to improve the physical team pass-opportunity state and QB attempts, while preserving the promoted M89/M90 passing-yard mean exactly?

This is not a generic pass-rate model. It uses one fixed conservation identity with no fitted coefficient.

## Canonical lineage

### Parent play/rate decomposition
- Branch: `research-qb-team-pass-opportunity-play-rate-decomp-v1`
- Result commit: `7dbbc93e42eae68e6032ec6cb2299d357be6cb20`
- Run: `34535405829`
- Job: `103065737323`
- Artifact: `10175200512`
- Digest: `sha256:f2a79b330c7cc6f24d6b83479806478e5534afd2bd09961d37304a2c247b6bb4`
- Disposition: `PASS_OPPORTUNITY_RATE_PRIMARY_DIAGNOSTIC`

### Designed-run source qualification
- Branch: `research-qb-pass-rate-designed-run-source-audit-v1`
- Result commit: `add1162974de5c791aa4df0da69e9b5f08e59150`
- Frozen source plan: `f9ad9dae8226c614609f504232e3c57995b9ff0a`
- Tested head: `5044cc4be6e3b0eb0ddc209bee4ad151cfd82ec9`
- Run: `34537717639`
- Job: `103073096550`
- Artifact: `10176060151`
- Digest: `sha256:1a3f5c908ce81731526904166fc334b6421229288dbaf09b29690d6dbaacf441`
- Disposition: `QB_DESIGNED_RUN_LINKAGE_SOURCE_ELIGIBLE`

### Shared receiver reference
- Run: `34066549394`
- Artifact: `9999119623`
- 2024-2025 QB-attempt residual vs WR reception-mass residual Spearman: `0.4994040246`.

## Frozen development cohort

Primary development cohort:
- exact 2024 M89/M90 QB-game rows from the parent play/rate artifact: `444` rows.

2025 is confirmation-only and must not be scored in D1.

The 2024 shared-receiver secondary cohort may be used only for the preregistered WR reception-mass diagnostic.

## Frozen source input

Use the immutable designed-run source-audit artifact `10176060151`, specifically `qb_designed_run_prior_history_audit.csv`.

For each resolved target QB with at least one prior game, define:

`qb_prior_designed_runs_per_game = recent8_designed_runs / prior_games`

where both quantities are already constructed only from games strictly before the target `(season, week)`.

Rows with unresolved QB identity, zero prior games, or non-finite required values receive **no designed-run adjustment** and remain exactly at the 0.57 baseline. They may not be imputed to zero designed-run burden.

## Frozen weekly center

For each 2024 target week, compute:

`weekly_reference_designed_runs_per_game`

as the simple arithmetic mean of `qb_prior_designed_runs_per_game` across that week's resolved target QBs with at least one prior game.

This weekly center is derived only from each QB's strict-prior completed-game history. It uses no target-game outcome.

The center exists so the candidate preserves the league-level 57% foundation rather than globally shifting the entire league more run-heavy merely because designed QB runs exist.

No median, trimmed mean, player weighting, team weighting, or alternate center may be tried.

## Frozen candidate formula

Let:
- `R0 = 0.57` = current promoted pass-opportunity-rate foundation;
- `P = pred_plays` = parent M89 projected team offensive plays;
- `Q = qb_prior_designed_runs_per_game`;
- `L = weekly_reference_designed_runs_per_game`.

For eligible rows:

`designed_run_excess = Q - L`

`candidate_pass_rate_raw = R0 - designed_run_excess / P`

`candidate_pass_rate = clip(candidate_pass_rate_raw, 0.35, 0.75)`

For ineligible source rows:

`candidate_pass_rate = R0`.

Coefficient is exactly `1.0` by football conservation: one designed QB run above the weekly QB reference displaces one otherwise league-average offensive-play opportunity from the pass/dropback side of the fixed pool. No fitted multiplier, shrinkage factor, cap search, or blend is authorized.

The fixed `[0.35, 0.75]` clip is a mechanical sanity boundary only and may not be changed after results.

## Frozen propagation

### Team pass opportunity

`candidate_D = pred_plays * candidate_pass_rate`

Baseline remains:

`baseline_D = pred_plays * 0.57`.

### QB official attempts

Use the already-frozen M89 opportunity-chain factors for attempt conversion and primary-QB share:

`candidate_qb_attempts = candidate_D * pred_C * pred_S`

No attempt-conversion or QB-share coefficient may change.

### QB passing-yard mean

The promoted M89/M90 `football_synthesis` mean remains authoritative and **must not change in D1**.

D1 is testing whether the internal opportunity state becomes more physically accurate while preserving the already-promoted QB yard mean. Therefore:

`candidate_qb_pass_yards_mean = football_synthesis`

exactly.

Passing-yard mean identity gap must be zero within `1e-12` for every row. A future integration, if this signal confirms, would need to reconcile opportunity and efficiency around the protected M89/M90 mean rather than stack a second yard correction.

## Frozen 2024 development metrics

Report baseline vs candidate on all 444 2024 rows for:

1. pass-opportunity-rate MAE, RMSE, bias, correlation, p90 absolute error;
2. team pass-opportunity `D` MAE, RMSE, bias, correlation, p90 absolute error;
3. QB official-attempt MAE, RMSE, bias, correlation, p90 absolute error;
4. 8+ and 10+ absolute QB-attempt miss rates;
5. candidate adjustment distribution: mean, mean absolute, p90 absolute, min/max;
6. correlation of `candidate_pass_rate - 0.57` with the realized pass-rate residual `actual_rate - 0.57`;
7. correlation of `candidate_D - baseline_D` with the 2024 WR reception-mass residual from the immutable shared QB/WR artifact;
8. source-eligible subset and full-population metrics separately.

Target outcomes are used only after the candidate formula is frozen and only for this development evaluation.

## Frozen uncertainty test

Use paired bootstrap over the 444 2024 QB-games:
- seed: `5601`
- draws: `5000`
- statistic: baseline MAE minus candidate MAE.

Report bootstrap `P(gain > 0)` for pass-opportunity rate, team pass opportunity, and QB attempts.

No alternate seed/draw count may be tried.

## Frozen development advance gates

D1 advances to a separately frozen 2025 confirmation only if **all** gates pass:

### Integrity
1. exact 444 2024 parent rows;
2. 2025 rows are not scored or summarized;
3. zero sportsbook inputs;
4. no production change;
5. no model fitting;
6. candidate coefficient exactly 1.0;
7. M89/M90 passing-yard mean preserved within `1e-12` every row;
8. source history is strictly prior by the immutable source-audit contract.

### Football performance
9. pass-opportunity-rate MAE gain >= `0.0030`;
10. team pass-opportunity MAE gain >= `0.15` opportunities;
11. QB official-attempt MAE gain >= `0.10` attempts;
12. candidate pass-rate p90 absolute error is non-worse;
13. candidate team pass-opportunity p90 absolute error is non-worse;
14. candidate QB-attempt p90 absolute error is non-worse;
15. 10+ QB-attempt miss rate is non-worse;
16. Spearman(`candidate_pass_rate - 0.57`, `actual_rate - 0.57`) >= `0.10`;
17. Spearman(`candidate_D - baseline_D`, 2024 WR reception-mass residual) >= `0.10`;
18. bootstrap P(pass-rate MAE gain > 0) >= `0.80`;
19. bootstrap P(team-D MAE gain > 0) >= `0.80`;
20. bootstrap P(QB-attempt MAE gain > 0) >= `0.80`.

If any gate fails, D1 disposition is `QB_DESIGNED_RUN_PASS_RATE_D1_FAIL_NO_CONFIRMATION` and 2025 must remain unopened for this candidate.

If all pass, disposition is `QB_DESIGNED_RUN_PASS_RATE_D1_PASS_READY_FOR_2025_CONFIRMATION`.

## Stopping rule

- One candidate formula only.
- No fitted coefficient.
- No alternate recent window.
- No alternate weekly center.
- No nonlinear transform.
- No special threshold for mobile QBs.
- No combining with schedule/rest, personnel inactives, PROE, game script, market data, or M89 synthesis reparameterization.
- No 2025 scoring unless D1 passes all gates.
- A D1 fail is preserved as a scientific fail and may not be rescued by retuning this family.
