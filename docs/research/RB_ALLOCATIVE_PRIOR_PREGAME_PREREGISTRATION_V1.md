# RB allocative prior on a PREGAME universe, through the production carry chain: PREREGISTRATION V1

**Date:** 2026-11-19, frozen before any arm runs. **Research only. No production change.**

## Prior-work check

- **`PRISM_PLAYER_EVIDENCE_SHRINKAGE_RESULTS_V1.md`** (2014-2025, 18,809 player-weeks):
  - **The allocative prior (leg a) is a strong candidate:** within-team share error +0.026,
    and it closes the lead-back flattening (C0 -0.105, allocative -0.017).
  - **Evidence-precision weighting (leg b) is closed negative.**
  - The study said leg (a) needs its own preregistration: one free structure, a prospective
    2026 check, and a test on a universe that contains non-playing backs.
  - `PRISM_WEEK8_PREREGISTERED_GRADING_V1.md` records it as authorized on 2026-10-28 and
    never completed.
- **Validity gap in that study.** Its universe and back count K were the RBs who recorded a
  stat in the game, which is known only after the game. Production must allocate across the
  **pregame** universe, including backs who will not play. This study closes that gap.
- **`PRISM_ROLE_STATE_FRONTIER_RESULTS_V2.md`** (role-rank anchor, λ frontier) ran on the
  P/S objectives with the reduced-form projector. It did not run through Bayes → top-5 →
  0.95 cap on a pregame universe.
- **Closed families, not touched:**
  - Rush Pool Evidence Guard (pool selection);
  - evidence-precision weights;
  - top-5 or residual rescues;
  - Bayesian prior strengths (`GROUP_STRENGTH` and the caps are unchanged here).
- **Harness:** `evaluate_rb_recency_end_to_end_v1.py`. It reproduces production's -3.5
  carries/team RB mass deficit (`RB_CARRY_SHARE_RECENCY_END_TO_END_RESULTS_V1.md`).

## Design

**Universe, per team-week, all pregame:**
- `load_rosters_weekly` status ACT, QB/RB/WR/TE (FB counts as RB);
- minus players with an official Out or Doubtful report.

**Production chain, unchanged from the harness:**
- Bayes posterior: group target x 3.0, prior x min(prior_games, 6), current x
  current_games, clipped to [0,1];
- `_top_n_shares(5)`;
- clip 0.95, then scale to 0.95 if the sum exceeds 0.95;
- x actual team rush attempts.

**Arms:**
- **C0 (production):** the RB group target is the prior_games-weighted mean of RB
  `rush_share_prior` across that week's universe.
- **C1 (allocative):** for RBs only, the shrinkage target becomes `alloc[K][k] x rb_block`.
  - **K:** the number of RBs in the team's pregame universe; K ≥ 5 is pooled as 5.
  - **k:** the rank by the pregame proxy, which is `rush_share_current` if current_games >
    0, else `rush_share_prior`, else 0. Ties are broken by order.
  - **`alloc[K][k]`:** the mean, over training team-weeks, of the rank-k universe RB's
    carries / the universe RBs' total carries. It uses the same universe and proxy
    definitions. RBs who did not play count as 0.
  - **Thin cells:** a cell with fewer than 20 observations falls back to C0's group target.
  - **`rb_block`:** the mean, over training team-weeks, of universe RB carries / team
    carries.
  - **Unchanged:** the weights; QB, WR and TE targets; all other stages.
- **C2 (reported, not gating):** C1 plus the recency input, a = 0.85, from
  `RB_CARRY_SHARE_RECENCY_END_TO_END_RESULTS_V1.md`.

**Training and holdouts:**
- Tables fitted on 2019-2024 are applied to the 2025 holdout.
- Tables fitted on 2019-2025 are applied to the 2026 holdout (weeks 4-10).
- Scored weeks: 4+.
- Scored rows: universe RBs who took an offensive snap.

## Metrics

- **Primary:** the MAE gain in expected RB carries, C0 minus C1, with a 95% bootstrap CI
  clustered by team-game.
- **Reported:**
  - lead-back (rank 1) MAE and bias;
  - committee backs (K ≥ 3, rank ≥ 2) MAE;
  - within-team RB shape error;
  - team RB carries projected vs actual;
  - the fitted tables.

## Kill and pass

- **K1:** the primary CI includes 0 in either holdout → null.
- **K2:** committee backs are worse with the CI excluding 0 in either holdout → rejected,
  whatever the headline (mirrors the parent's G5).
- **K3:** the team RB mass error |projected - actual| gets worse in either holdout →
  mis-specified.
- **PASS:** a positive primary result with the CI excluding 0 in both holdouts, and no K2 or
  K3. A pass is a candidate only. Promotion requires:
  - a production replay on the real Ourlads universe;
  - a check of its interaction with the INERT concentration transform and the RB P3 Week-1
    route;
  - prospective grading.

## Prediction (scored after)

C1 wins clearly, smaller than the box-score study's gain but well outside noise:
- lead-back bias falls by more than half;
- team RB mass moves toward actual.

**Named risk:** in a pregame universe, K counts backs who never play. The allocative table
then gives the real backs less than the box-score table did, and the cap rescales, so the gain
could shrink toward noise (K1), or committee backs could be over-credited (K2).
