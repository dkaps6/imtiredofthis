# Pass/Receiving Conservation Integration V1 — Frozen Plan

## Purpose

Advance only the independently supported C2 completed-pass/receiving-yard conservation mechanism from `JOINT_PASS_RECEIVING_CONSERVATION_V1` into a full-stack research integration test. This migration does not promote to production by itself.

## Lineage

- Parent branch: `research-joint-pass-receiving-conservation-v1`
- Parent SHA: `4f620f1ea24a10cdf840d52d42a8930e5991f1ee`
- Scientific source run: `34081764151`
- Scientific source job: `101618243530`
- Scientific source artifact: `10004223287`
- Frozen disposition: `CONSERVATION_ONLY_SUPPORTED`
- C2 QB CRPS improvement vs B0: `-1.2927925027427682` yards, paired bootstrap probability of improvement `1.0`
- C2 macro WR/TE/RB receiving-yard MAE: `16.95936203611147 -> 16.910356643667505`
- Promoted QB mean anchor remains M89/M90.
- Promoted WR hierarchy remains M38.
- RB rushing production remains P3.
- Sportsbook inputs: **0**.

## What is authorized

Integrate only the C2 mechanism:

1. finite player + residual targets;
2. catches generated from frozen pregame catch-rate inputs;
3. receiver yards generated from the same completed-pass process;
4. explicit residual receiver bucket;
5. exact iteration-level identity: QB passing yards = modeled receiver yards + residual receiver yards;
6. one constant pregame team-game scale factor preserving the existing promoted QB mean where an M89/M90 anchor exists;
7. outside the exact M89/M90 2024-2025 common cohort, preserve the exact current B0 pregame QB Monte Carlo mean for receiver evaluation only and exclude those rows from the M89 QB scoreboard.

No C1 position-group target-mass calibration is authorized. C1 and C3 failed their frozen player-level protection gates and must not be smuggled into this migration.

## Frozen comparison

- `B0_CURRENT`: exact current production/research architecture.
- `C2_INTEGRATED`: exact C2 conservation mechanism embedded in the full-stack simulation path with all other promoted means, hierarchy logic, rushing outputs, injuries, weather, defense, pace, PROE, matchup and rules/Bayesian logic unchanged.

No alternate residual defaults, YPR transforms, target allocation rules, scale factors, or coefficients may be introduced after results are visible.

## Historical scope

### Receiver markets
- 2020-2025 regular seasons.
- 2020 Weeks 1-17; 2021-2025 Weeks 1-18.
- WR, TE and RB reported separately and pooled.

### QB distribution
- Exact available M89/M90 common-era cohort in 2024-2025 only.
- QB mean must remain numerically identical to the M89/M90 football synthesis within 0.01 yards for every aligned team-game.

### RB rushing
- Exact parity check against P3 rushing outputs. The conservation integration is receiving-only.

## Frozen individual-player evaluation

For WR, TE and RB receiving, report B0 vs C2 at pooled, season, player-role tier and projection-volume tier where sample permits:

- target MAE/RMSE/bias/correlation;
- reception MAE/RMSE/bias/correlation;
- receiving-yard MAE/RMSE/bias/correlation;
- median, p75 and p90 absolute receiving-yard error;
- 20+, 30+ and 40+ absolute receiving-yard miss rates;
- directional over/under bias;
- top projected player strata separately from low-volume depth players.

For RB additionally report rush+receiving total-yard MAE/RMSE/bias and confirm rushing-yard projections themselves are unchanged.

For QB report:

- mean-anchor parity;
- MAE/RMSE/bias/correlation of the mean projection;
- CRPS;
- 50%, 80% and 90% empirical interval coverage and absolute calibration error;
- median, p75 and p90 absolute passing-yard error;
- 50+, 75+ and 100+ absolute miss rates;
- distribution p10/p25/p50/p75/p90.

## Integrity gates

Scientific interpretation stops unless all pass:

1. exact B0 receiving baseline reproduction on the established 2025 all-receiver cohort (`n=4647`, MAE within 0.05 of `17.099904733366`);
2. exact M89/M90 aligned team-week uniqueness;
3. zero sportsbook inputs;
4. no target-game outcome in any pregame feature;
5. M38 within-WR hierarchy unchanged;
6. RB P3 rushing outputs unchanged;
7. zero player-iterations with zero receptions and positive receiving yards;
8. max iteration-level conservation gap <= `1e-6` yards.

## Scientific gates

`CONSERVATION_INTEGRATION_CANDIDATE_PASS` requires all:

1. pooled 2024-2025 QB mean differs from M89/M90 anchor MAE by <= 0.01 yards;
2. mean paired QB CRPS improves by >= 0.25 yards vs B0;
3. paired 10,000-resample team-game bootstrap probability that QB CRPS improves >= 0.90, fixed seed 5601;
4. absolute 80% QB interval coverage error does not worsen by more than 0.02;
5. pooled macro-average WR/TE/RB receiving-yard MAE is no worse than B0;
6. no individual receiving position worsens pooled receiving-yard MAE by more than 0.50 yards;
7. no single season/position receiving-yard MAE worsens by more than 1.50 yards;
8. latest-era 2024-2025 pooled macro receiving-yard MAE is no worse than B0;
9. pooled macro p90 absolute receiving-yard error does not worsen;
10. pooled macro 40+ yard receiving miss rate does not worsen by more than 0.5 percentage points;
11. QB p90 absolute error does not worsen;
12. QB 100+ yard miss rate does not worsen;
13. RB rush+receiving total-yard MAE does not worsen by more than 0.50 yards;
14. all integrity gates pass.

If any gate fails, preserve B0 production and retain C2 only as research-supported distribution evidence. Do not relax a threshold or search a nearby variant.

## Production rule

Passing this migration authorizes a separate production-promotion/full-slate confirmation migration. It does not directly change production.

## Stopping rule

Run B0 vs exact C2 integrated only. No C1/C3 target-group calibration, coefficient search, residual-default search, alternate YPR transform, or post-result tuning.