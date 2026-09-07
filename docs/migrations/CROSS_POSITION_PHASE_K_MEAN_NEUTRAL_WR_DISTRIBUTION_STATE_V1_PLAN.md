# CROSS-POSITION PHASE K — MEAN-NEUTRAL WR DISTRIBUTION STATE V1

## Namespace / lineage note
The WR distribution-state design was already frozen at commit `5f13f311535c5636a361a5456599f99cbdbeadce` in `CROSS_POSITION_PHASE_J_MEAN_NEUTRAL_WR_DISTRIBUTION_STATE_V1_PLAN.md`.

A concurrent valid research path subsequently occupied the generic Phase-J script/workflow namespace with the deployable QB state-selector study, which completed successfully as run `34151640191` with disposition `DEPLOYABLE_QB_STATE_SELECTOR_ELIGIBLE`.

**Phase K is therefore a namespace-only carry-forward of the already-frozen WR experiment.** No scientific rule, threshold, feature, gate, pool-size requirement, seed, cohort, or interpretation has changed after seeing any WR-distribution result.

## Frozen scientific design
All scientific rules below are copied from the previously frozen Phase-J WR plan and remain authoritative.

### Purpose
Test whether the strict-pregame positive pass-state signal can improve WR receiving-yard distribution shape while preserving every M38 WR point mean and hierarchy exactly.

### Cohort
Use the exact 2025 WR player-games from the authoritative Phase-C casebook whose `(season, week, team)` aligns to the authoritative Phase-G team-game casebook. No outcome-based scoring-row filtering.

### Strict-prior training information
For scored 2025 week `w`, training rows may use only:
- all Phase-C WR rows from seasons `< 2025`;
- 2025 WR rows from weeks `< w`.

The frozen `opportunity_quartile` is the WR stratum.

For prior 2025 games only, the realized `PASS_STATE_HIGH` training label is the Phase-D condition:
- QB opportunity residual > 0;
- combined WR+TE opportunity residual > 0;
- RB carry residual < 0.

These are training labels only. The scored-game selector is exclusively strict-pregame Phase-G `delta_pass_attempts > 0`.

### B0 empirical WR distribution
For each scored WR:
1. use strict-prior residuals `actual_yards - M38_pred_yards` from the same opportunity quartile;
2. center the residual pool to zero mean;
3. add centered residuals to the unchanged M38 point mean;
4. floor samples at 0;
5. apply one deterministic scale factor so sample mean returns exactly to the unchanged M38 point mean;
6. if the raw sample mean is zero, use a degenerate distribution at the M38 point mean.

### Candidate WR distribution
If Phase-G `delta_pass_attempts <= 0`, candidate = B0 exactly.

If `delta_pass_attempts > 0`:
1. prefer strict-prior WR residuals from prior realized `PASS_STATE_HIGH` games in the same opportunity quartile;
2. require at least **100 WR residual rows**;
3. if fewer than 100 are available, use all prior realized `PASS_STATE_HIGH` WR residuals across quartiles if that pool has at least 100 rows;
4. otherwise candidate = B0;
5. apply the exact same centering, nonnegative floor, and mean-restoring scale.

No threshold search, variance multiplier, interpolation, reweighting, player-specific tuning, matchup coefficient, sportsbook input, or postgame scored-game selector is allowed.

### Frozen scorecard
Report B0 vs candidate for ALL, Q1/Q2/Q3/Q4, Phase-G positive/nonpositive, and realized PASS_STATE_HIGH/PASS_STATE_LOW diagnostic slices.

Metrics:
- point-mean MAE/RMSE/bias/correlation;
- max mean-anchor gap;
- empirical CRPS;
- 10,000-resample paired player-game bootstrap probability with seed **5610**;
- 50/80/90 coverage, calibration error, interval widths;
- p05/p10/p25/p50/p75/p90/p95 averages;
- p90 point-mean absolute error;
- unchanged 50+ yard point-mean miss count and under/over split;
- Brier score for `actual_yards >= M38_mean + 50`;
- diagnostic state-pool usage and Phase-I QB-vs-WR 80% width relationship.

### Frozen eligibility gates
`MEAN_NEUTRAL_WR_DISTRIBUTION_STATE_ELIGIBLE` only if all are true:
1. max mean-anchor gap <= **0.01 yard**;
2. point-mean MAE delta <= **0.01 yard**;
3. overall CRPS improves >= **0.10 yard**;
4. bootstrap probability >= **0.90**;
5. Q4 CRPS improves >= **0.10 yard**;
6. 80% coverage absolute error does not worsen by > **0.02**;
7. 90% coverage absolute error does not worsen by > **0.02**;
8. catastrophic-upside Brier does not worsen by > **0.002**;
9. p90 point-mean absolute error unchanged within **0.01 yard**;
10. 50+ yard point-mean miss count unchanged exactly;
11. M38 point means/ranks unchanged; QB/TE/RB point means unchanged; sportsbook inputs = 0; same/future scored-game outcomes used as selectors = 0; production parameters changed = 0.

## Interpretation / next step
- Pass: one joint QB+WR Monte Carlo assembly test, then 2026 Week-1 player-by-player production audit.
- Fail: stop WR distribution-state tuning; retain Phase-G WR opportunity flag diagnostically only and move directly to the 2026 Week-1 audit.
