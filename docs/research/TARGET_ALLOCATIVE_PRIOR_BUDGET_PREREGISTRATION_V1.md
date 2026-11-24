# Receiving: allocative target prior + team budget 1.0 through the production target chain: PREREGISTRATION V1

**Date:** 2026-11-19, frozen before any arm runs. **Research only. No production change.**

## Prior-work check, and what is new

- **Alpha compression is real and stable:**
  - `PRISM_ALPHA_COMPRESSION_BOARD_CHECK_V1.md`: alphas -15 to -20% vs the line every week;
    bench receivers +8 to +30%.
  - `PRISM_ALPHA_TIER_RESULTS_V1.md`: B1 passed every gate. **Its licensed follow-up, a
    flagged implementation plus prospective grading, never happened.**
  - **Root cause, triangulated by those studies:** group-mean shrinkage toward a marginal
    position default.
- **A multiplicative de-compression was falsified** out of sample (board check §3). It
  absorbed a season-specific market regime. **Not retried.**
- **The role-rank anchor** (`PRISM_ROLE_STATE_FRONTIER_RESULTS_V2.md`) is close to an
  allocative target. It passed 8 of 9 gates and was blocked by its own G7 threshold. It acted
  on Bayesian shares only: no pregame universe, no rules multipliers, no cap/budget stage.
- **Team receiving mass is low by 15-16%** (`PRISM_RECEIVING_MASS_DECOMPOSITION_V1.md`), with
  the simulator component the most biased. Diagnosed, no fix tested.
- **Measured today, on archived production inputs (wk 8, 9, 11):**
  - `rules_tgt_share` equals `bayes_tgt_share` for WR and QB;
  - it is about x1.265 for TE and about x1.20 for RB/FB (correlation ≥ 0.99).

  So a Bayesian fix reaches every position. The TE/RB multipliers inflate team mass before the
  0.95 cap.
- **New here:**
  - the rushing D1 result (`ALLOCATIVE_PRIOR_PLUS_BUDGET_NORMALIZATION_RESULTS_V1.md`) applied
    to targets;
  - run through the full target chain on a pregame universe.

## Chain (offline reproduction of production)

1. Bayes `tgt_share` posterior: group target x 3.0, prior x min(prior_games, 6), current x
   current_games, clipped to [0,1].
2. The rules multiplier: TE x1.265, RB/FB x1.20, WR/QB x1.0. These are the measured modal
   values, a disclosed approximation of `simulation_rules`.
3. `simulation_v2._sharpen_wr_target_shares` (promoted M37, WR mass preserved), called
   directly.
4. The team stage: clip 0.95 per player; production caps the team sum at 0.95.
5. x **actual** team targets, so only the share differs.

**Universe:** the pregame roster, ACT minus Out/Doubtful, QB/RB/WR/TE (FB counts as RB),
identical to the rushing studies.

## Arms

- **C0:** production.
- **T3:** an allocative target prior for every block, `alloc_P[K_P][k] x block_P`, ranked by
  the pregame proxy (current share, else prior, else 0), with the cap at 0.95. Tables are
  fitted the same way as the rushing study:
  - 2019-2024 tables for the 2025 holdout;
  - 2019-2025 tables for the 2026 wk 4-10 holdout;
  - a 20-observation cell floor.
- **TD1 (candidate):** T3 + the team stage renormalized to 1.0.
- **TN0 (control):** production priors + renormalized to 1.0.

## Metrics and gates (both holdouts; weeks 4+; players who played)

- **Primary:** the MAE gain in expected targets, WR + TE, C0 minus TD1. **Pass** needs a
  game-clustered 95% CI excluding 0.
- **NI-1, depth receivers:** the WR/TE pregame block rank ≥ 3. Fail if the CI of
  (TD1 MAE - C0 MAE) lies entirely above 0.
- **NI-2, team target mass |proj - actual|:** fail if significantly worse (CI of the worsening
  entirely above 0) **or** the point worsening exceeds **0.5 targets per team-game**.
- **M1, mechanism:** the under-projection bias of WR1s (pregame rank 1 WRs) must shrink in
  both holdouts. If it does not, the mechanism is not what the prior studies say it is.
- **Reported:**
  - RB and TE targets separately;
  - T3 and TN0;
  - the fitted tables and block masses.
- **PASS:** primary, NI-1, NI-2 and M1. A pass is a candidate only, for:
  - a prospective test;
  - an end-to-end rec-yards and receptions test, where side accuracy must be scored separately
    (board check §3);
  - an Ourlads replay.

## Prediction

- TD1 passes, with a smaller relative gain than rushing (19-21%). The irreducible-floor study
  (`PRISM_IRREDUCIBLE_FLOOR_AND_ALPHA_HAIRCUT_V1.md`) puts target-share headroom above the
  physical floor at only about 0.0096.
- WR1 bias shrinks by more than half.
- Team target mass improves.

**Named risk:** target share sits near its multinomial noise floor, so the gain is too small to
clear the CI (primary null). Or the M37 WR sharpening already captures the WR shape, leaving
the allocative prior nothing to add for WRs.
