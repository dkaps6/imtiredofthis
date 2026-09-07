# CROSS-POSITION PHASE I — MEAN-NEUTRAL QB DISTRIBUTION STATE V1

## Purpose
Phase G prospectively identified a positive pass-volume state and improved QB attempts / WR target opportunity, but Phase H proved that deterministically multiplying yard means by that opportunity correction is harmful.

Phase I tests the materially different architecture implied by those results: **preserve the promoted QB yard mean exactly and use the positive pass-state signal only to select a broader, conservation-derived QB outcome distribution.**

This is not a smaller Phase-H multiplier and it does not reinterpret H as a pass. No player mean is changed.

## Frozen lineage
- Phase G authoritative positive shared-state run: `34150685094`, disposition `POSITIVE_SHARED_PASS_STATE_ELIGIBLE`.
- Phase H authoritative yard-propagation run: `34150887575`, disposition `PLAYER_YARD_PROPAGATION_NOT_ELIGIBLE`.
- C2 authoritative full-stack conservation run: `34142510405`, scientific disposition remains `CONSERVATION_INTEGRATION_CANDIDATE_FAIL` because its frozen receiver macro-p90 gate failed.
- C2 independently demonstrated materially better QB CRPS / interval calibration while preserving the M89/M90 mean exactly.
- QB mean authority remains M89/M90.
- WR mean/hierarchy remains M38.
- TE authority remains TE-R5.
- RB rushing authority remains RB-P3.

## Cohort
Use the exact 2025 Phase-G team-games that align to the authoritative C2 QB casebook. Expected primary cohort: 434 team-games.

Any C2 QB row without a Phase-G row is excluded. No outcome-based filtering is allowed.

## Frozen candidate
For each aligned team-game:

- If strict-pregame Phase-G `delta_pass_attempts > 0`, use the existing C2 QB Monte Carlo distribution for that team-game.
- If `delta_pass_attempts <= 0`, use the existing B0 QB Monte Carlo distribution.
- The selected distribution's mean must equal the existing M89/M90 mean anchor; both B0 and C2 were already mean-anchored in the authoritative C2 run.
- No quantile interpolation, variance multiplier, threshold optimization, reweighting, resampling, or new Monte Carlo generation is allowed.

`ALL_C2` is reported as a frozen reference only: use C2 for every aligned game. It is not the primary Phase-I candidate.

No WR/TE/RB mean, target, yard, hierarchy, or rushing prediction changes in Phase I.

## Frozen scorecard
For B0, Phase-I candidate, and ALL_C2 reference report:
- mean MAE / RMSE / bias / correlation;
- mean-anchor max absolute gap;
- mean paired CRPS;
- paired CRPS improvement vs B0;
- 10,000-resample paired team-game bootstrap probability that candidate CRPS improves vs B0, fixed seed **5609**;
- empirical 50%, 80%, 90% interval coverage and absolute calibration error;
- average 50%, 80%, 90% interval widths;
- p10/p25/p50/p75/p90 average quantiles;
- p90 absolute error of the unchanged mean;
- 100+ yard mean-miss rate of the unchanged mean.

Also report, diagnostically only:
- Phase-G positive vs nonpositive slices;
- realized Phase-D `PASS_STATE_HIGH` / `PASS_STATE_LOW` slices reconstructed from the Phase-G casebook residual signs;
- per-game CRPS win/loss/tie counts for candidate vs B0 and candidate vs ALL_C2.

## Frozen eligibility gates
`MEAN_NEUTRAL_QB_DISTRIBUTION_STATE_ELIGIBLE` only if all are true:
1. candidate mean-anchor max gap <= **0.01 yard**;
2. candidate mean MAE differs from B0 by <= **0.01 yard**;
3. candidate CRPS improves vs B0 by >= **0.25 yard**;
4. paired bootstrap probability of CRPS improvement >= **0.90**;
5. absolute 80% interval coverage error does not worsen vs B0 by > **0.02**;
6. absolute 90% interval coverage error does not worsen vs B0 by > **0.02**;
7. candidate p90 mean absolute error is unchanged within **0.01 yard**;
8. candidate 100+ yard mean-miss rate is unchanged within **1e-9**;
9. no WR/TE/RB/player mean changes are made;
10. sportsbook inputs = 0; same/future outcomes used as selectors = 0; production parameters changed = 0.

The candidate does **not** need to beat ALL_C2 to pass. ALL_C2 is a reference showing whether Phase-G gating preserves, improves, or sacrifices some of C2's QB distribution value while avoiding any receiver-side C2 activation.

## Interpretation
- A pass means Phase G is useful as a **mean-neutral QB distribution-state selector**, not as a deterministic yard-mean correction.
- A fail means preserve C2 as generic QB distribution evidence and Phase G as opportunity-state evidence separately; do not search a nearby threshold or variance multiplier.
- No C2 receiver change is authorized by a Phase-I pass.

## Next-step rule
If Phase I passes, the deadline production candidate may combine:
- M89/M90 QB point mean unchanged;
- Phase-I selected QB distribution;
- M38 WR point means/hierarchy unchanged, with Phase-G upside state retained as a diagnostic/distribution flag only until a separately validated WR distribution model exists;
- TE-R5 unchanged;
- RB-P3 unchanged.

Then move to the final 2026 Week-1 player-by-player audit / production-readiness assembly rather than another broad feature hunt.
