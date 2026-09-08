# RB R17 Mean-Preserving Tail Mixture V1 — Frozen Plan

Date frozen: 2026-09-08
Parent diagnostic: `RB_R16_UPSIDE_TAIL_STATE_V1`
Parent run: `34286363931`
Parent artifact: `10079630404`
Status: PLAN FROZEN BEFORE EXECUTION

## Research question

R16 supported a strict-prior signal for large positive RB receiving-yard residuals while R13-R15 did not support forcing weak efficiency information into the mean. R17 therefore tests a distribution-only mechanism:

> Can OOS R16 probabilities for >=30-yard and >=50-yard baseline underprojection improve the RB receiving-yard predictive distribution while preserving the frozen baseline receiving mean exactly?

This is not a new mean model and does not promote R12.

## Frozen data lineage

Use the mechanically corrected RB-only R12 prediction artifact from run `34273055095` for player-game means/outcomes and the immutable R16 artifact from run `34286363931` for OOS tail probabilities.

R16 probabilities are OOS:
- train 2023 -> score 2024
- train 2023-2024 -> score 2025

R17 uses the same test folds:
- train residual pools from 2023 -> test 2024
- train residual pools from 2023-2024 -> test 2025

No current-game outcome is used to choose a test-game distribution. Training outcomes are allowed only in the historical residual pools.

## Frozen comparator

`GLOBAL_EMPIRICAL_MEAN_PRESERVED`

For each fold, sample residuals from the unconditional training residual distribution:

`residual = actual_rec_yards - baseline_pred_rec_yards`

Add sampled residuals to the frozen test-game baseline mean, floor at zero, then rescale the draw vector so its sample mean equals the frozen baseline mean exactly.

This comparator is deliberately empirical rather than a weak parametric straw man.

## Frozen candidate

`R16_NESTED_TAIL_EMPIRICAL_MEAN_PRESERVED`

Training residual pools are frozen as:
- NON_TAIL: residual < 30
- TAIL_30_49: 30 <= residual < 50
- TAIL_50_PLUS: residual >= 50

For each OOS test player-game, use R16 OOS probabilities:
- `p30 = P(residual >= 30)`
- `p50 = P(residual >= 50)`

Nested mixture weights:
- `w50 = min(p50, p30)`
- `w30 = max(p30 - w50, 0)`
- `wnon = 1 - p30`

Draw the residual component using those probabilities and sample within the corresponding historical training residual pool. Add to the frozen baseline mean, floor at zero, then rescale the draw vector so its sample mean equals the frozen baseline mean exactly.

No post-result tuning of component boundaries, weights, or mean-preservation rule is permitted.

## Monte Carlo settings

- deterministic seed: 917
- 2,000 draws per player-game
- same draw count for comparator and candidate
- all reported metrics computed from immutable simulated draw vectors before any downstream pricing

## Frozen scoring

Per player-game and aggregated by fold/combined:
- CRPS
- Brier score for actual >= frozen mean + 30
- Brier score for actual >= frozen mean + 50
- pinball loss at q90
- pinball loss at q95
- central 80% interval coverage
- central 90% interval coverage
- max absolute difference between simulated mean and frozen baseline mean

## Frozen support gates

R17 is supported only if ALL gates pass:

1. `mean_preservation`: candidate max absolute mean delta <= 1e-6 yards.
2. `combined_crps`: candidate combined CRPS <= comparator combined CRPS.
3. `fold_crps_guard`: candidate CRPS no worse than comparator by more than 1.0% in either OOS fold.
4. `cat30_brier`: candidate combined 30+ Brier < comparator.
5. `cat50_brier`: candidate combined 50+ Brier < comparator.
6. `q90_pinball`: candidate combined q90 pinball < comparator.
7. `q95_pinball`: candidate combined q95 pinball < comparator.
8. `coverage80_guard`: candidate absolute error from 80% coverage <= comparator absolute coverage error + 0.02.
9. `coverage90_guard`: candidate absolute error from 90% coverage <= comparator absolute coverage error + 0.02.
10. `sportsbook_zero`: no sportsbook inputs.

These gates were frozen before R17 execution and must not be changed after results are visible.

## Governance

PASS authorizes only a subsequent production-parity/distribution integration candidate. It does not:
- change or promote an RB receiving mean,
- promote R12,
- change target entitlement,
- change RB-room mass,
- alter WR/TE/QB production stacks,
- add sportsbook inputs upstream.

FAIL means the R16 classification signal is scientifically interesting but the frozen tail-mixture implementation did not convert it into a better probabilistic yardage distribution. Diagnose before trying another mechanism.
