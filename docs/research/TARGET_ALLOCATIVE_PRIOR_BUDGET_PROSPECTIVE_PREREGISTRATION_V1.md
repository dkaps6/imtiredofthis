# Receiving TD1: PROSPECTIVE PREREGISTRATION V1

**Frozen:** 2026-11-19, 15:40 ET, **before week 11 kicks off** (20:15 ET). Weeks 11-18 are
unseen.

**Parent:** `TARGET_ALLOCATIVE_PRIOR_BUDGET_RESULTS_V1.md`. TD1 passed retrospectively.
- WR+TE target MAE improved 7-8% on 2025 and 2026 wk 4-10.
- One parent gate (NI-2) was uninformative. **This test replaces it with a block-level
  gate.**

## Frozen candidate, no changes

- **Arm TD1:** the all-block allocative target prior, the rules multipliers (TE 1.265, RB/FB
  1.20), M37 WR sharpening, and the team stage renormalized to 1.0. It uses
  `scripts/research/evaluate_target_allocative_prior_budget_v1.py`, sha256
  `900d129d42e1f55d3dce973068c3e6a8b0569effa40bd020f2da4bc9482aec2d`.
- **Tables:** the 2026-holdout tables in
  `data/research/target_allocative_prior_budget_v1_tables.csv`, sha256
  `ef0532e10db1f4a0e4020ca499e2727cb0141bdd783a85348f260a6314244a5a`. **Not refit.**
- **Comparator:** C0. T3 is reported.

## Gates, with margins fixed now (weeks 11-18, players who played)

- **Primary:** the WR+TE expected-targets MAE gain, TD1 vs C0, with a CI clustered by team-game
  that excludes 0.
- **NI-1, depth WR/TE (rank ≥ 3):** not significantly worse (CI of the worsening not entirely
  above 0).
- **NI-2b, WR block mass per team-game |proj - actual|:** fail if significantly worse, or if the
  point worsening exceeds **0.5 targets**. This is a block gate, so it can fail under
  renormalization.
- **NI-3, TE block mass:** the same rule, margin **0.5 targets**.
- **M1:** |WR1 bias| (pregame rank 1 WR) is smaller under TD1 than under C0.

## Schedule

- **Interim after week 14:** no decision.
- **Final after week 18:** the decision.

**PASS** = primary, NI-1, NI-2b, NI-3 and M1.

No sportsbook variable is used. No production change.
