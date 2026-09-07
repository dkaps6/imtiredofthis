# CROSS-POSITION PHASE J — MEAN-NEUTRAL WR DISTRIBUTION STATE V1

## Purpose
Phase G prospectively improved QB-attempt and WR-target opportunity, Phase H proved that multiplying WR/QB yard means by that opportunity correction is harmful, and Phase I showed that the same pass-state signal can improve QB distribution calibration while preserving the promoted QB mean exactly.

Phase J tests the analogous WR question: **can the Phase-G positive pass-state signal improve WR receiving-yard distribution shape while preserving every M38 WR point mean and hierarchy exactly?**

This is a distribution-only test. It is not a smaller Phase-H mean multiplier, it does not reopen WR-R11, and it does not authorize any TE/RB change.

## Frozen lineage
- Phase G authoritative run: `34150685094`, disposition `POSITIVE_SHARED_PASS_STATE_ELIGIBLE`.
- Phase H authoritative run: `34150887575`, disposition `PLAYER_YARD_PROPAGATION_NOT_ELIGIBLE`.
- Phase I authoritative run: `34151186485`, disposition `MEAN_NEUTRAL_QB_DISTRIBUTION_STATE_ELIGIBLE`.
- WR point mean / hierarchy authority remains M38.
- QB point mean authority remains M89/M90; Phase-I QB distribution remains separate evidence.
- TE authority remains TE-R5.
- RB rushing authority remains RB-P3.
- C2 receiver integration remains scientifically failed under its original frozen macro-p90 gate and is not activated here.

## Cohort
Use the exact 2025 WR player-games from the authoritative Phase-C casebook whose `(season, week, team)` aligns to the authoritative Phase-G team-game casebook.

Expected cohort: approximately 1,700 WR player-games across 434 team-games.

No outcome-based scoring-row filtering is allowed.

## Frozen training information
For a scored 2025 week `w`, training rows may use only:
- all Phase-C WR rows from seasons `< 2025`; and
- Phase-C WR rows from 2025 weeks `< w`.

The scored game itself and all later games are forbidden from distribution construction.

The existing Phase-C `opportunity_quartile` is the frozen WR opportunity stratum.

### Prior realized pass-state label
For prior 2025 team-games only, reconstruct the same Phase-D realized `PASS_STATE_HIGH` label:
- primary QB opportunity residual > 0;
- combined WR+TE opportunity residual > 0;
- RB carry residual < 0.

These are **training labels only** from already-completed earlier games. They are never used to select the scored game's distribution.

The scored game's selector is exclusively strict-pregame Phase-G `delta_pass_attempts > 0`.

## Frozen distribution construction
For every scored WR player-game:

### B0 empirical distribution
1. Take all strict-prior WR residuals `actual_receiving_yards - M38_pred_receiving_yards` from the same frozen `opportunity_quartile`.
2. Center that residual pool by subtracting its training-pool mean.
3. Add the centered residuals to the scored player's unchanged M38 mean.
4. Floor raw samples at 0 yards.
5. Multiply all nonnegative samples by one deterministic scale factor so their sample mean equals the unchanged M38 point mean exactly.
6. If the resulting raw sample mean is zero, use a degenerate distribution at the M38 point mean.

### Positive-state candidate distribution
If Phase-G `delta_pass_attempts <= 0`, candidate = B0 exactly.

If Phase-G `delta_pass_attempts > 0`:
1. Prefer strict-prior WR residuals from prior realized `PASS_STATE_HIGH` team-games in the same `opportunity_quartile`.
2. Require at least **100 WR residual rows** in that state/quartile pool.
3. If fewer than 100 are available, use all strict-prior WR residuals from prior realized `PASS_STATE_HIGH` team-games across quartiles if that pool has at least 100 rows.
4. If fewer than 100 prior high-state WR residuals exist in total, candidate = B0.
5. Apply the exact same centering, nonnegative floor, and deterministic mean-restoring scale as B0.

No threshold search, variance multiplier, quantile interpolation, distribution reweighting, player-specific tuning, matchup coefficient, sportsbook input, or postgame selector is allowed.

## Frozen scorecard
Report B0 vs candidate for:
- all aligned WR player-games;
- M38 opportunity quartiles Q1/Q2/Q3/Q4;
- Phase-G positive vs nonpositive team-games;
- realized `PASS_STATE_HIGH` and `PASS_STATE_LOW` diagnostic slices.

Metrics:
- point-mean MAE / RMSE / bias / correlation (must be identical by construction);
- max absolute mean-anchor gap;
- empirical CRPS;
- 10,000-resample paired player-game bootstrap probability candidate CRPS improves vs B0, fixed seed **5610**;
- empirical 50%, 80%, 90% coverage and absolute calibration error;
- average interval widths;
- average p05/p10/p25/p50/p75/p90/p95;
- p90 absolute error of the unchanged point mean;
- 50+ yard point-mean miss count and under/over split (must be unchanged);
- Brier score for the catastrophic-upside event `actual_receiving_yards >= M38_mean + 50` using each empirical distribution's probability of exceeding that threshold.

Also report diagnostically:
- count of scored rows using state-specific residual pools versus B0 fallback;
- state-pool sizes by week and opportunity quartile;
- same-team aggregate WR candidate 80% width versus Phase-I QB candidate 80% width when alignment is available; this is descriptive only and is not a gate.

## Frozen eligibility gates
`MEAN_NEUTRAL_WR_DISTRIBUTION_STATE_ELIGIBLE` only if all are true:
1. max absolute WR mean-anchor gap <= **0.01 yard**;
2. WR point-mean MAE differs from M38/B0 by <= **0.01 yard**;
3. overall candidate CRPS improves vs B0 by >= **0.10 yard**;
4. paired bootstrap probability of CRPS improvement >= **0.90**;
5. Q4 WR candidate CRPS improves vs B0 by >= **0.10 yard**;
6. absolute 80% interval coverage error does not worsen by > **0.02**;
7. absolute 90% interval coverage error does not worsen by > **0.02**;
8. catastrophic-upside Brier score does not worsen by > **0.002**;
9. p90 point-mean absolute error unchanged within **0.01 yard**;
10. 50+ yard point-mean miss count unchanged exactly;
11. M38 player ordering / point means unchanged exactly; TE/RB/QB point means unchanged; sportsbook inputs = 0; same/future outcomes used as selectors = 0; production parameters changed = 0.

## Interpretation
- A pass means Phase-G positive pass-state is useful for **mean-neutral WR distribution shaping**, not for raising M38 WR means.
- A fail means retain Phase G as WR opportunity evidence / diagnostic flag and retain M38 point means; do not tune a nearby pool-size threshold or variance multiplier.
- Even a pass does not activate C2 receiver means or alter TE-R5/RB-P3.

## Next-step rule
If Phase J passes, freeze one joint QB+WR Monte Carlo assembly test that preserves:
- M89/M90 QB mean;
- Phase-I QB distribution state;
- M38 WR means/hierarchy;
- Phase-J WR distribution state;
- TE-R5 and RB-P3 means unchanged;
then score same-game conservation/correlation and move directly into the 2026 Week-1 player-by-player production-readiness audit.

If Phase J fails, stop WR distribution-state tuning and move directly to the 2026 Week-1 audit with Phase-G WR opportunity flag retained diagnostically only.