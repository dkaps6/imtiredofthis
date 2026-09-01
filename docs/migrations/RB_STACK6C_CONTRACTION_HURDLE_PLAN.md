# RB STACK6C — Rotation-Gated Contraction Hurdle Plan

## Why this architecture

Broad STACK6 and compact STACK6B both failed as bidirectional continuous carry-residual models. The frozen STACK6B directional postmortem showed that negative/contraction corrections contain useful signal while positive/expansion corrections are harmful. A naive contraction-only replay is still unacceptable because it overcontracts the eligible population and violates the original carry-bias safety constraint.

STACK6C source audit then established a live-capable completed-prior-game PBP rotation family:

- touch-opportunity share vs historical RB on-field presence share: `r=.8674`;
- top-RB identity agreement: `84.74%`;
- touch-derived lead-drive share vs historical lead-drive share: `r=.9051` descriptively;
- prior-3 coverage: `96.38%`;
- strict-prior leakage pass: `100%`.

The predeclared simple drive-presence proxy gate failed (`r=.4610`), so `touch_drive_share` is excluded from this model on **source-fidelity grounds**, not because of rushing outcome performance.

This experiment changes the statistical question from:

> How much positive or negative residual should be added to every eligible back?

into:

> Is P3 overallocating this secondary back at all, and if so how much should workload be contracted?

No expansion is possible by construction.

## Frozen parent and population

Parent remains P3 exactly:

- Week 1: full production-equivalent stack;
- Weeks 2-18: enriched M94C opportunity/allocation x full-stack efficiency/context.

Eligible correction domain:

- Week 6+;
- frozen M95F non-risk only;
- depth rank 2+;
- at least one strictly prior same-team PBP rotation-proxy game;
- strict as-of source safety required.

Protected exactly:

- M95F-risk rows;
- depth-rank-1 rows;
- all rows outside the eligible domain.

## Live-capable rotation feature block

Use exactly five PBP-derived rotation concepts, each represented by prior-1 and prior-3 same-team values = **10 features** total:

1. `touch_opp_share`
2. `touch_lead_drive_share`
3. `opening_drive_touch_share`
4. `team_touch_leader_switch_rate`
5. `team_touch_hhi`

Do not use delayed participation/on-field features in either fitted arm.

Do not use `touch_drive_share` because its predeclared source-fidelity gate failed.

## Existing aggregate block

The same 14 timestamp-safe aggregate pregame features from STACK6/STACK6B may be used unchanged:

1. `depth_rank`
2. `depth_slot`
3. `prior1_snap_pct`
4. `prior3_snap_pct`
5. `prior3_rb_share`
6. `credible_competitors`
7. `prior_backfield_hhi`
8. `injury_reported`
9. `injury_out_doubtful`
10. `injury_questionable`
11. `rookie_flag`
12. `prior1_rb_share`
13. `prior1_carries`
14. `prior3_carries`

## Two arms only

### ROTATION_HURDLE

- exactly 10 rotation features.

### AGG_PLUS_ROTATION_HURDLE

- 14 aggregate features + 10 rotation features = exactly 24.

No other arm is allowed.

## Frozen two-stage model

For each expanding target week from Week 6 through Week 18:

### Stage 1 — over-allocation classifier

Training target:

`overallocated = 1 if actual_rush_att - parent_att < 0 else 0`

Model:

- `LogisticRegression`
- L2 penalty
- `C=1.0`
- `solver='lbfgs'`
- `max_iter=1000`
- no class weighting
- median imputation
- standard scaling
- no missingness indicator features

Contraction gate:

- use the model's default probability threshold `0.50`;
- no threshold search.

### Stage 2 — contraction magnitude

Fit only training rows where `overallocated=1`.

Target:

`contraction_magnitude = -(actual_rush_att - parent_att)`

Model:

- Ridge
- alpha `10.0`
- median imputation
- standard scaling
- no missingness indicators.

Prediction is clipped to `[0,4]` carries.

### Final correction

If classifier probability `< .50`:

`delta = 0`

If classifier probability `>= .50`:

`delta = -clipped_predicted_contraction_magnitude`

Thus:

`delta <= 0` always.

No row can receive a carry expansion.

## Expanding-week fit rules

- start evaluation: Week 6;
- train on strictly earlier 2025 weeks only;
- minimum total training rows: `40`;
- minimum overallocated/magnitude rows: `20`;
- both classifier classes must exist;
- no 2024 outcome fitting in this experiment;
- 2025 remains exposed retrospective development data.

## Retention gates — unchanged from STACK6B

For each arm independently:

1. eligible carry MAE gain >= `0.20`;
2. eligible yard MAE gain >= `0.15`;
3. eligible W13-18 yard MAE gain > `0`;
4. all-RB W6-18 yard MAE regression <= `0.05`;
5. eligible absolute carry-bias worsening <= `0.25`;
6. M95F-risk maximum point change = `0`;
7. depth-rank-1 maximum point change = `0`.

If both pass, prefer the 10-feature `ROTATION_HURDLE` if its eligible yard-MAE gain is within `0.05` of the best passing arm.

## Sportsbook separation

Football-first disposition must be frozen before sportsbook data is loaded.

Sportsbook is downstream benchmark only and may not affect:

- feature inclusion;
- classification threshold;
- contraction magnitude;
- retention decision;
- model selection.

## No search contract

- hyperparameter search: `0`
- feature search: `0`
- threshold search: `0`
- weight search: `0`
- delta-cap search: `0`
- population search: `0`

## Dispositions

If no arm passes:

`STACK6C_NO_RETAINABLE_ROTATION_CONTRACTION_INCREMENT`

Do not retune the hurdle on exposed 2025 data. Proceed to exact availability/competitor-state source work or another genuinely new football-information family.

If an arm passes:

`STACK6C_ROTATION_CONTRACTION_RETAINED_RESEARCH_ONLY`

Freeze the winner and require prospective 2026 confirmation plus a production-grade live identity/PBP pipeline before promotion.
