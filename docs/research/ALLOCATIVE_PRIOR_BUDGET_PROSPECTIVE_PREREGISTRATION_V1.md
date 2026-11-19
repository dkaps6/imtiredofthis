# Allocative prior + team budget 1.0 (D1): PROSPECTIVE PREREGISTRATION V1

**Frozen:** 2026-11-19, about 14:40 ET. That is **before week 11 kicks off** (TNF IND@HOU, 20:15
ET), so weeks 11-18 of 2026 are unseen.

**Parent:** `ALLOCATIVE_PRIOR_PLUS_BUDGET_NORMALIZATION_RESULTS_V1.md`.
- D1 cleared K1-K3 on 2025 and 2026 wk 4-10, with large margins.
- K4 fired on a 0.024-carry point worsening (CI [-0.107, +0.061]), because that gate had no
  noise tolerance.
- Re-scoring the same holdouts under a new gate would be fitting the gate to a known result.
  **Only unseen data can settle it.**

## Frozen candidate, no changes allowed

- **Arm D1:** exactly as in the parent.
  - all-block allocative targets;
  - team stage renormalized to 1.0;
  - top-5 and the per-player 0.95 clip unchanged.
- **Tables:** the 2026-holdout tables fitted on 2019-2025, frozen in
  `data/research/allocative_prior_all_blocks_pregame_v1_tables.csv`.
  - sha256 `e4c5737882de1074bd6cb0149e5a41a915e8eacdf58b0fa2d390adaf1d2b2ce9`.
  - **They are not refit on any week-11+ data.**
- **Code, frozen at these hashes:**
  - `evaluate_allocative_prior_budget_normalization_v1.py`
    `1f94f58ade03948a0f140cdb2bb28cf549b1b2b165a0e7ba6639a96fff194b5e`
  - `evaluate_allocative_prior_all_blocks_pregame_v1.py`
    `6374b96ab7ccf7a07f5029cf764020877ce4edcaff2a4128b73063156996c2d4`
  - `evaluate_rb_allocative_prior_pregame_v1.py`
    `e5e7aa3be1a103b2e190b77f9e7fa324a0446bcd85c3044980705e8c52c5481c`
- **Comparator:** C0, production as reproduced.
- **Reported only:** D1 + recency (a = 0.85).

## Data

- 2026 weeks 11-18 REG.
- The universe for each week comes from that week's pregame roster: ACT, minus official
  Out/Doubtful.
- Current-season inputs use weeks before the target week only.

## Primary and gates, with margins fixed now

- **Primary:** RB expected-carries MAE gain, D1 vs C0. **Pass** needs a 95% CI, clustered by
  team-game, that excludes 0.
- **Non-inferiority (each must hold):**
  - **NI-1, committee RBs (K ≥ 3, k ≥ 2):** fail if the CI of (D1 MAE - C0 MAE) lies
    entirely above 0, i.e. significantly worse.
  - **NI-2, team RB mass |proj - actual|:** fail if significantly worse (CI of worsening
    entirely above 0) **or** the point worsening exceeds **0.25 carries per team-week**.
  - **NI-3, team QB mass |proj - actual|:** the same rule, margin **0.25 carries per
    team-week**.
- **Why 0.25:** about 10-12% of C0's team-mass errors (2.0-2.75), and well under half of D1's
  smallest holdout primary gain (0.62). It is set before any week-11 data exists.

## Schedule

- **Interim after week 14:** reported, with **no decision.**
- **Final after week 18:** the decision. If nflverse data lags, the final runs as soon as week
  18 is published.

## Outcomes

- **PASS** (primary and NI-1 to NI-3): D1 becomes a promotion candidate for:
  - a receiving-side test of the same team-budget change;
  - an Ourlads-universe production replay;
  - interaction checks with RB P3 and the INERT concentration transform.
- **FAIL:** recorded as a prospective failure. The retrospective holdout result stands as
  research evidence only.

No sportsbook variable is used. No production code changes until a separate integration
preregistration exists.
