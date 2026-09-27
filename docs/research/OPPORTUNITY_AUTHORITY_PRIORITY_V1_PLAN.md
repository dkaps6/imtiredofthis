# Opportunity Authority Priority V1 — Frozen Full-Stack Plan

Status: **FROZEN BEFORE CANDIDATE SCORING**

Branch:
`research-bayesian-current-state-transmission-v1`

Parent scientific authority:
- `BAYESIAN_CURRENT_STATE_TRANSMISSION_V1`
- run `36330210860`
- disposition `BAYESIAN_CURRENT_STATE_TRANSMISSION_SYSTEMIC_MISMATCH_CONFIRMED`

Production remains unchanged.

## Question

The prior audit established that the exact production PlayerForm blend is more
accurate than the downstream empirical-Bayes posterior for:

- RB rush share;
- WR target share;
- TE target share;

in both 2024 and 2025 and in the pooled two-current-game analogue.

This candidate asks one production-order question:

> Does using the already-validated PlayerForm blend as the rule-layer authority
> for those three fast-moving opportunity states improve the downstream football
> stack without breaking other positions or markets?

This is **not Bayesian retuning**. No Bayes constant changes.

Version:

`OPPORTUNITY_AUTHORITY_PRIORITY_V1`

## Baseline

Exact current production semantics for the tested historical week:

1. exact PlayerForm four-game pseudo-prior blend;
2. exact empirical-Bayes baseline;
3. rule layer prefers Bayesian target/rush shares;
4. existing contextual multipliers;
5. M38 explicit target entitlement;
6. TE-R5P inside conserved TE room;
7. WR-R15 inside conserved WR room;
8. canonical joint Monte Carlo;
9. fixed existing ML/State components;
10. fixed current production ensemble weights;
11. RB Rush+Receiving Conservation V2 for non-Week-1 RB combined yards.

Week 1 is excluded from scientific scoring because current Week-3 production
does not use RB P3 and the question is current-state transmission after at least
one completed current-season game.

## Candidate

Everything is identical to baseline except the **source priority at the rule
input seam**.

For a production-normalized position family:

- RB: `rules_rush_share` starts from exact PlayerForm `rush_share`;
- WR: `rules_tgt_share` starts from exact PlayerForm `tgt_share`;
- TE: `rules_tgt_share` starts from exact PlayerForm `tgt_share`.

All other opportunity metrics keep baseline production authority.

All efficiency metrics remain empirical-Bayes:
- YPC;
- YPT;
- YPA;
- catch/reception rate;
- YPRR / route metrics where applicable.

No Bayes group strength, player-prior cap, default, coefficient, threshold or
position grouping is changed.

### Injury-target consistency

Any complete-player target-share authority used by the legacy injury rule must
use the same candidate source priority for WR/TE target shares and baseline
Bayes for every other position. Row-local and complete-player rule paths must
not disagree.

This is an internal authority-consistency requirement, not new vacancy science.

## Production-exact historical reconstruction

Do **not** use the generic historical-context PlayerForm shortcut as the
scientific baseline for this test.

For each target week, reconstruct:

- previous-season production totals;
- current-season-to-date production totals strictly before target week;
- target-week weekly-roster universe;
- stable identity from evidence available before target week;
- exact production PlayerForm `w_current = current_games/(current_games+4)`;
- exact production empirical Bayes;
- exact current rule layer;
- current target-entitlement order (M38 -> TE-R5P -> WR-R15);
- canonical MC;
- existing ML and State components;
- fixed production ensemble weights;
- RB Rush+Receiving Conservation V2 semantics.

Target-game outcomes are joined only after baseline and candidate projections
are complete.

No sportsbook data is used.

## Historical design

Independent seasons:

- 2024 Weeks 2-18, prior season 2023;
- 2025 Weeks 2-18, prior season 2024.

Use the same pregame weekly-roster universe, feature rows, ML/State predictions,
team state, simulation iteration count and random seed for baseline and
candidate.

Candidate variants = **1**.
Fit parameters = **0**.

## Primary opportunity endpoints

### RB

- rush-attempt MAE and p90 absolute error;
- RB-only and all-position rushing opportunity effects reported separately.

### WR / TE

Construct expected target count from the same simulated team passing-opportunity
state and the final promoted entitlement state.

Report target-count MAE and p90 separately for:
- WR;
- TE.

## Downstream market endpoints

Point-projection MAE, bias and p90 AE for:

- RB rush attempts;
- RB rush yards;
- RB rush+receiving yards using V2 conservation;
- RB receptions;
- RB receiving yards;
- WR receptions;
- WR receiving yards;
- TE receptions;
- TE receiving yards.

Guard separately:
- QB rush attempts;
- QB rush yards;
- OTHER-position rush attempts/yards.

QB passing is expected to be mean-invariant to this candidate; verify exact
point-mean invariance rather than use it as an improvement target.

## Frozen qualification gates

`OPPORTUNITY_AUTHORITY_PRIORITY_V1_QUALIFIED` requires ALL:

### Direct opportunity
1. RB rush-attempt MAE strictly improves in 2024.
2. RB rush-attempt MAE strictly improves in 2025.
3. WR expected-target MAE strictly improves in 2024.
4. WR expected-target MAE strictly improves in 2025.
5. TE expected-target MAE strictly improves in 2024.
6. TE expected-target MAE strictly improves in 2025.
7. RB/WR/TE primary opportunity p90 AE is non-worse in both seasons.

### Downstream player markets
8. RB rush-yard MAE is non-worse in both seasons.
9. RB rush+receiving-yard MAE is non-worse in both seasons.
10. WR receiving-yard MAE is non-worse in both seasons.
11. WR reception MAE is non-worse in both seasons.
12. TE receiving-yard MAE is non-worse in both seasons.
13. TE reception MAE is non-worse in both seasons.
14. Across those six downstream families, at least **four** must strictly
    improve in pooled 2024-2025 MAE.

### Cross-position guards
15. QB rush-attempt MAE is non-worse in both seasons.
16. QB rush-yard MAE is non-worse in both seasons.
17. OTHER-position rush-attempt MAE is non-worse in both seasons.
18. OTHER-position rush-yard MAE is non-worse in both seasons.
19. RB receiving-yard and reception MAE are non-worse in both seasons.
20. QB passing point means are numerically invariant (max gap <= 1e-10).

### Integrity
21. team play/pass/rush volume inputs are unchanged baseline vs candidate;
22. Bayesian efficiency columns are unchanged;
23. ML and State component predictions are unchanged;
24. ensemble weights are unchanged;
25. M38 / TE-R5P / WR-R15 model assets and coefficients are unchanged;
26. RB Rush+Receiving Conservation V2 remains exact;
27. zero sportsbook inputs;
28. zero target/future outcomes in features;
29. no Week-1 P3 transport;
30. only the three frozen opportunity source-priority cells may change before
    downstream propagation.

If any qualification gate fails:

`OPPORTUNITY_AUTHORITY_PRIORITY_V1_FAILED_CLOSED`

No rescue.

## Additional reporting

Report the two-current-game analogue separately, but it is **not** allowed to
override the independent 2024/2025 gates.

Report high-opportunity players separately using only pregame baseline
entitlement/rush-share rank. This is descriptive and cannot create a subgroup
carveout.

## Stopping rule

After results are visible, do not:

- tune Bayes strengths/caps;
- add RB/WR/TE-specific blend weights;
- add current-games thresholds;
- add depth-role exceptions;
- add rookie/injury exceptions;
- change top-N rushing support;
- change team opportunity volume;
- rescue only the market families that happen to improve;
- use 2026 Week-3 outcomes;
- use sportsbook lines upstream.

A PASS authorizes only a separate production-integration proposal.
It does not authorize a merge to production.
