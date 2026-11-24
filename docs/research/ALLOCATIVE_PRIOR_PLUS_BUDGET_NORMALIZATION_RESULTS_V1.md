# Allocative prior + corrected team budget: RESULTS V1

**Date:** 2026-11-19.

**Preregistration:** `ALLOCATIVE_PRIOR_PLUS_BUDGET_NORMALIZATION_PREREGISTRATION_V1.md`, frozen at
c72936e. It reopens the closed "residual/top-five normalization" family under operator
authorization.

**Script:** `scripts/research/evaluate_allocative_prior_budget_normalization_v1.py`.

**Rows:** `data/research/allocative_prior_budget_normalization_v1{,_rows}.csv`.

**Verdict: K4 FIRED, narrowly, in 2025 only. Not a pass, as written. Nothing promoted.**
- Every other gate clears with wide margins.
- The candidate fixes the RB team-mass defect that blocked the previous two studies.
- The K4 failure is a 0.02-carry point worsening, with a CI that includes 0. The gate was
  written without a noise tolerance. That is a gate specification error (mine), recorded as
  such and **not** reinterpreted.
- The candidate goes to a **prospective** test on weeks 11-18 with margins frozen before any of
  that data exists: `ALLOCATIVE_PRIOR_BUDGET_PROSPECTIVE_PREREGISTRATION_V1.md`.

## Results vs C0 (production)

| | 2025 (wk 4-18) | 2026 (wk 4-10) |
|---|---|---|
| **RB carries MAE, C0 → D1** | 3.256 → **2.633** | 3.457 → **2.721** |
| **D1 gain [95% CI]** | **+0.623 [+0.519, +0.725]** (-19%) | **+0.736 [+0.558, +0.908]** (-21%) |
| lead back MAE, C0 → D1 | 4.78 → 3.47 | 5.23 → 3.75 |
| lead back bias, C0 → D1 | +3.74 → **+0.68** | +4.12 → **+0.77** |
| committee (K≥3, k≥2) MAE, C0 → D1 | 2.48 → 2.20 | 2.55 → 2.19 |
| **team RB mass \|proj - actual\|, C0 → D1** | 2.547 → **1.914** [+0.47, +0.80] | 2.753 → **1.915** [+0.61, +1.06] |
| team RB mass bias, C0 → D1 | +1.81 → **+0.23** | +2.05 → **+0.55** |
| **team QB mass \|proj - actual\|, C0 → D1** | 2.000 → **2.023** (worse by 0.024, CI [-0.107, +0.061]) | 2.202 → 2.074 (better) |
| QBs who played, MAE C0 → D1 | 1.71 → 1.69 (CI crosses 0) | 1.67 → 1.71 (CI crosses 0) |

**Gates:**
- **K1, clear.**
- **K2, clear.**
- **K3, clear.** The team RB mass defect is fixed: bias falls from about 2 carries per team
  to about 0.2-0.5.
- **K4, FIRED** in 2025 on a point worsening of 0.024 carries per team-week, with a CI that
  includes 0. It is clear in 2026.

**K5 (control) not fired.** N0, the budget fix alone:
- primary +0.11 / +0.10;
- **committee backs worse** (CI excludes 0);
- **QB mass worse** (CI excludes 0).

The budget fix without the allocative prior is harmful. **Both pieces are needed.** Neither
piece alone gets close.

**D2** (cap raised to 1.0, scaling down only) sits between C3 and D1:
- primary +0.55 / +0.62;
- RB mass about unchanged.

Full renormalization (D1) is what closes the RB mass gap.

## The arms in one line each (2026 primary gain)

| arm | primary gain |
|---|---|
| recency alone (earlier study) | +0.05 |
| N0, budget alone | +0.10 |
| C1, RB-only prior | +0.55 |
| C3, all-block prior | +0.60 |
| D2, all-block + cap 1.0 | +0.62 |
| **D1, all-block + budget 1.0** | **+0.74** |

## Prediction scored

- "D1 passes": **wrong, narrowly** (K4, 2025, by 0.02).
- "RB mass bias near 0": **right** (+0.23 / +0.55).
- "Primary beyond C3": **right** (+0.62 / +0.74 vs +0.50 / +0.60).
- "N0 well below D1": **right.**
- **Named risk** (renormalization over-credits committee backs or the QB block):
  - committee backs: **did not happen**;
  - QB block: **happened in a trivial amount** (2025 QB mass +0.02). QB over-projection of
    about 1 carry per team persists in both C0 and D1.

## What is established, and what is not

**Established on two blind holdouts with a pregame universe and production's own chain:**
the allocative prior together with a team budget of 1.0 cuts RB expected-carries error by
19-21%, removes most of the RB team-mass deficit and lead-back under-projection, and harms no
RB subgroup.

**Not established:**
- Formal passage. K4 fired as written.
- The receiving side of the same cap change. `_allocate_counts` is shared and was not tested.
- The real Ourlads universe.
- Interaction with RB P3 Week-1 and the INERT concentration transform.
- Rushing yards. These are carries only; YPC is unchanged by these arms.
- **The open QB over-projection** (about 1 carry per team in every arm) is its own defect.

**Next:** the prospective test, frozen before week-11 kickoff on 2026-11-19 at about 14:40 ET.
Nothing was promoted and no production code changed. No sportsbook variable was used.

## Addendum: where the ~1 carry/team of QB over-projection lives (diagnostic, not a test)

Per QB-game, projected minus actual carries, D1:

| | 2025 | 2026 |
|---|---|---|
| pregame QB1 who played | **-0.05** | **+0.31** |
| backup QB who played | -0.42 | -0.56 |
| backup QB who did **not** play (300 / 155 rows) | +1.27 | +1.17 |
| pregame QB1 who did not play (10.5% / 8.8% of team-weeks) | +2.81 | +3.24 |

**Starting QBs' carries are accurate under D1.** The over-projection is mass on QBs who do not
take a snap:
- backups carry a mean prior-season share of about 0.09, often earned as a starter elsewhere;
- QB1s ruled out late.

Props on non-players void, so the betting impact is small. The cost is the finite carries
those players hold.

**Prior work closes the obvious fix.** Weighting down prior-season evidence was killed in
`PRISM_PRIOR_ANCHORING_RESULTS_V1.md`: the prior earns its weight, even for team changers. The
remaining lever is availability: knowing who will not play. That is the roster-coverage
problem in `PRISM_RB_RUSH_SHARE_MASS_BUDGET_V1.md` Addendum 2, not a prior problem. No QB study
is opened here.
