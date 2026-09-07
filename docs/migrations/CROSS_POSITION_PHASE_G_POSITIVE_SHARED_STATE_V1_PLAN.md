# CROSS-POSITION PHASE G — POSITIVE-ONLY SHARED PASS STATE V1

## Purpose
Phases D-F established a reproducible shared pass-state relationship and showed that the current pregame representation is strongly asymmetric:

- predicted positive pass-state deltas materially improve QB attempts and WR target-pool opportunity;
- predicted negative pass-state deltas do not improve WR opportunity and are not a reliable representation of PASS_STATE_LOW;
- forcing TE opportunity to follow the shared delta worsens TE target-pool accuracy;
- RB carry opportunity should remain unchanged in this component.

Phase G is a materially new, frozen prospective candidate. It does **not** reinterpret Phase F as a pass. It tests whether the reusable component is specifically a positive-only QB+WR opportunity state while preserving TE-R5 and RB-P3 opportunity baselines exactly.

No sportsbook inputs. No production changes.

## Frozen lineage
- Phase C authoritative all-row game-spot artifact from run `34147777341`.
- Phase E authoritative shared-state casebook from run `34150046671`.
- Phase F authoritative scientific run `34150525462` is the predeclared diagnostic basis for this new hypothesis; its disposition remains `SEPARATE_PASS_CATCHER_POOL_RESPONSE_NOT_ELIGIBLE`.

## Frozen candidate
For every scored 2025 team-game:

1. Reuse Phase E `delta_pass_attempts` unchanged.
2. Define `delta_up = max(delta_pass_attempts, 0)`.
3. QB opportunity:
   - `candidate_qb_attempts = pred_qb_attempts + delta_up`.
   - Negative predicted deltas are ignored; those games retain baseline QB attempts.
4. WR target pool:
   - Within each walk-forward fold, estimate zero-intercept `beta_wr` using only chronologically prior team-games: `wr_target_residual ~ beta_wr * qb_attempt_residual`.
   - `candidate_wr_targets = max(0, pred_wr_targets + beta_wr * delta_up)`.
5. TE target pool:
   - **unchanged exactly** from TE-R5/Phase-C baseline.
6. RB carry pool:
   - **unchanged exactly** from RB-P3/Phase-C baseline.
7. No individual WR entitlement, TE entitlement, efficiency, YPA/YPR, or yardage parameter is changed in Phase G.

This is not a universal pass-state model. It is specifically a test of a pregame-identifiable upside/pass-volume state.

## Frozen scorecard
Exact same 434 2025 team-games as Phase E/F where available.

Report baseline vs candidate:
- QB attempt MAE, RMSE, bias, correlation, p90 absolute error.
- WR target-pool MAE, RMSE, bias, correlation, p90 absolute error.
- WR+TE target-pool MAE and p90 absolute error.
- TE target-pool exact parity.
- RB carry-pool exact parity.
- Q4 baseline-absolute-error catastrophic opportunity miss counts for QB and WR; threshold frozen from the baseline 75th percentile absolute error over the scored cohort.
- directional slices for `delta_pass_attempts > 0` and `<= 0`.
- actual Phase-D `PASS_STATE_HIGH` and `PASS_STATE_LOW` slices for interpretation only.

## Frozen eligibility gates
`POSITIVE_SHARED_PASS_STATE_ELIGIBLE` only if all are true:
1. QB attempt MAE improves by >= **0.50 attempts**.
2. WR target-pool MAE improves by >= **0.50 targets**.
3. WR+TE target-pool MAE improves by >= **0.30 targets**.
4. QB p90 absolute error does not worsen.
5. WR p90 absolute error does not worsen.
6. QB catastrophic opportunity miss count does not increase.
7. WR catastrophic opportunity miss count does not increase.
8. TE target predictions are unchanged to <=1e-9.
9. RB carry predictions are unchanged to <=1e-9.
10. sportsbook features used = 0; same/future outcomes used as predictors = 0; production parameters changed = 0.

Otherwise disposition is `POSITIVE_SHARED_PASS_STATE_NOT_ELIGIBLE`.

No gate, threshold, delta clipping rule, beta method, or cohort may be changed after results.

## Next-step rule
If Phase G is eligible, the next experiment must propagate this exact frozen opportunity component into actual player yardage outputs while preserving the current position authorities:
- QB mean authority M89/M90;
- WR hierarchy M38;
- TE entitlement TE-R5 unchanged;
- RB-P3 unchanged;
- C2 distribution/conservation evidence handled explicitly rather than silently promoted.

That next experiment must score individual-player yard MAE and catastrophic tails. Phase G itself cannot change production.
