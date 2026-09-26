# Rush Post-Ensemble Reconciliation V1 — Frozen Historical Test Plan

Date: 2026-09-26

Status: **FROZEN BEFORE HISTORICAL OUTCOME SCORING**

Branch: `research-rush-post-ensemble-reconciliation-v1`

Parent diagnostic:
- `POST_SPECIALIST_CROSS_MARKET_CONSISTENCY_V1_CONTRADICTION_CONFIRMED`
- run `36252584883`
- artifact `10909602741`
- result commit `5d79ccd5920032dc9f4268119fd955b8e97eaae1`

## Question

Can one parameter-free post-ensemble reconciliation restore the finite rushing-opportunity identity **without changing any player's existing final implied rushing efficiency** and improve historical full-stack rushing projections?

This is not a rescue of Rush Pool Evidence Guard V1.

Rush Pool V1 changed **which players entered the simulator's top-five carry allocator**.

This candidate leaves the simulator, player membership, raw shares, ML, State, ensemble weights and every underlying football input unchanged. It acts only on the already-produced final rushing means after independent market blending.

## Historical authority

Use the preserved exact full-stack integration artifact:

- run `36049144898`
- job `107800031207`
- artifact `10830277740`
- digest `sha256:b6a0715d38984fa91b8815561b4e72679684affcc142ee773a8b7428342412a5`
- source production main at that integration: `b5b1816e7d11cadca412faf841fbfe5336798c2e`

The baseline columns only are scientific authority:
- `baseline_mc_proj`
- `baseline_proj`
- `actual`

The old Rush Pool candidate columns are explicitly ignored.

A repository comparison from `b5b1816...` to current Week-3 production source `f7d2011...` shows no production/model implementation changes; intervening files are research/docs/workflows only. Therefore this preserved baseline remains the current production rushing architecture for the tested 2024-2025 scope.

Historical seasons:
- 2024 Weeks 2-18
- 2025 Weeks 2-18

Week 1 remains protected by the promoted P3 Week-1 route and is outside this candidate.

## Frozen candidate

Version:

`RUSH_POST_ENSEMBLE_RECONCILIATION_V1`

Scope:
- all positions with paired final `rush_att` and `rush_yards` rows;
- team-game reconciliation;
- Weeks 2-18 only;
- one candidate only;
- no fitted parameters.

For each team-game:

`M = sum_i(baseline_mc_rush_att_i)`

This is the joint simulator's finite modeled **player carry mass** after its residual bucket.

`E = sum_i(baseline_final_rush_att_i)`

This is the sum of independently blended final player carry means.

If `E > 0`:

`F = M / E`

For every rusher `i` on the team:

`candidate_rush_att_i = baseline_final_rush_att_i * F`

`candidate_rush_yards_i = baseline_final_rush_yards_i * F`

Therefore:

- candidate player carry sum equals joint-MC player carry mass exactly;
- all player relative final carry shares are preserved;
- every player's baseline final implied YPC is preserved exactly because carries and yards receive the same factor;
- no position carveout exists;
- no threshold or router exists.

If `E == 0` and `M == 0`, use `F = 1`.
If `E == 0` and `M > 0`, fail closed.

## RB Rush+Receiving Conservation V2

For RB-family rows only, preserve the already-promoted receiving component:

`baseline_rec_component = baseline_rush_rec_yards - baseline_rush_yards`

`candidate_rush_rec_yards = candidate_rush_yards + baseline_rec_component`

Thus the protected V2 identity remains:

`candidate_rush_rec_yards = candidate_rush_yards + unchanged receiving component`

No receiving mean is refit or altered.

## Explicit no-op / protected scope

Do not change:
- Week 1 / P3;
- simulator allocation;
- top-five pool selection;
- raw `rules_rush_share`;
- Bayesian evidence state;
- ML models;
- State models;
- ensemble weights;
- player-specific YPC / implied final YPC;
- receiving projections;
- target/reception projections;
- QB passing;
- sportsbook information;
- vacancy logic;
- Rush Pool Evidence Guard V1;
- M96 / P3 / prior closed RB routers.

## Frozen integrity gates

All must pass:

1. sportsbook inputs used = 0;
2. parameters fit = 0;
3. candidate variants scored = 1;
4. old Rush Pool `candidate_*` columns are never used as candidate inputs;
5. paired rush-att/rush-yard identity is unique;
6. candidate team player-carry sum equals baseline joint-MC player-carry sum within `1e-10`;
7. max player implied-YPC change from baseline final means <= `1e-10` wherever baseline/candidate carries are positive;
8. one identical team factor is applied to carries and yards for every player;
9. RB receiving component changes by <= `1e-10`;
10. Week-1 candidate rows = 0;
11. no production file is mutated.

## Frozen scoring populations

Report independently for 2024 and 2025:

- ALL positions;
- RB/FB/HB family;
- QB;
- OTHER positions.

Markets:
- rush attempts;
- rush yards.

Also report RB-family rush+receiving yards downstream.

Metrics:
- MAE;
- RMSE;
- bias;
- p90 absolute error;
- 10+ attempt misses for rush attempts;
- 30+ yard misses for rush yards / rush+receiving;
- changed-row candidate-closer rate.

## Frozen qualification gates

`RUSH_POST_ENSEMBLE_RECONCILIATION_V1_QUALIFIED` requires **all**:

1. ALL rush-att MAE strictly improves in 2024;
2. ALL rush-att MAE strictly improves in 2025;
3. ALL rush-yard MAE strictly improves in 2024;
4. ALL rush-yard MAE strictly improves in 2025;
5. RB-family rush-att MAE strictly improves in both seasons;
6. RB-family rush-yard MAE strictly improves in both seasons;
7. QB rush-att MAE is non-worse in both seasons;
8. QB rush-yard MAE is non-worse in both seasons;
9. OTHER rush-att MAE is non-worse in both seasons;
10. OTHER rush-yard MAE is non-worse in both seasons;
11. ALL rush-att p90 is non-worse in both seasons;
12. ALL rush-yard p90 is non-worse in both seasons;
13. RB-family rush-att p90 is non-worse in both seasons;
14. RB-family rush-yard p90 is non-worse in both seasons;
15. RB rush+receiving MAE is non-worse in both seasons;
16. RB rush+receiving p90 is non-worse in both seasons;
17. changed-row candidate-closer rate > 50% for rush attempts in both seasons;
18. changed-row candidate-closer rate > 50% for rush yards in both seasons;
19. all integrity gates pass.

If any gate fails:

`RUSH_POST_ENSEMBLE_RECONCILIATION_V1_FAILED_CLOSED`

No rescue.

## Stopping rule

After scores are visible, do not search:
- partial reconciliation factors;
- caps/floors on `F`;
- RB-only routing;
- QB/OTHER exclusions;
- high-volume-only routing;
- player thresholds;
- alternate team pools;
- alternate residual percentages;
- different factors for carries vs yards;
- tuned efficiency blends;
- 2026 outcome-based corrections.

A failure closes this exact reconciliation family.

A qualification still does not authorize production; it would require a separate exact-current-stack integration certification.
