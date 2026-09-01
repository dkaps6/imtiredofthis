# RB STACK6B — Compact Secondary-Back Role Concepts

## Why STACK6B is justified

STACK6's 230-feature flat situational representation failed materially. A no-fit failure atlas then showed that compact football relationships still order future carry residuals: the predeclared rushing-vs-passing role-balance concept had a +0.862 carry-residual top-vs-bottom quartile spread, while passing-down specialization had a -1.205 spread. Raw historical rush ownership moved negatively versus P3 residual, indicating redundancy/overreaction rather than a need for more raw volume weight.

STACK6B is therefore an architecture compression, not hyperparameter retuning. 2025 remains exposed development data and any winner must be frozen for prospective 2026 confirmation.

## Parent and protected domain

Same frozen P3 parent and same STACK6 domain:
- correct only Week 6+;
- frozen M95F-nonrisk only;
- depth-rank 2+ only;
- at least one strictly prior participation-history game;
- M95F-risk and depth-rank 1 unchanged exactly.

## Fixed compact concepts

The new block contains exactly these football-defined concepts, built with same-team prior-3 history when available and any-team prior-3 fallback otherwise:

1. `role_balance`: mean(rush ownership, early-down presence, short-yardage presence, red-zone presence) minus mean(third-down, third-long, two-minute presence).
2. `passing_role`: mean(third-down, third-long, two-minute presence).
3. `goal_line_role`: mean(inside-10, inside-5 presence).
4. `rush_vs_presence`: rush-attempt ownership minus overall RB-presence ownership.
5. `rush_momentum`: prior-1 rush ownership minus prior-3 rush ownership.
6. `early_momentum`: prior-1 early-down presence minus prior-3 early-down presence.
7. `role_stability`: negative mean absolute prior-1 vs prior-3 change for rush, early-down, and third-down role.
8. `team_concentration`: prior-3 team RB-presence HHI.

No raw 230-column block is eligible in STACK6B.

## Arms

- `COMPACT_ROLE`: eight concepts only.
- `AGG_PLUS_COMPACT`: the existing 14 aggregate pregame role/snap features plus the eight compact concepts.

Fixed expanding-week Ridge alpha 10; same Week 6 start; same training-only residual clipping intersected with [-4,+4] carries; no hyperparameter, threshold, feature, or weight search.

## Retention gate

Same STACK6 football-first gate:
- eligible carry MAE gain >= 0.20;
- eligible yard MAE gain >= 0.15;
- eligible W13-18 yard gain > 0;
- all-RB W6-18 yard MAE regression <= 0.05;
- eligible absolute carry bias worsening <= 0.25;
- zero change to M95F-risk and depth-rank 1.

If both pass, prefer smaller arm within 0.05 eligible yard MAE gain of best.

Vegas remains downstream-only after disposition is frozen.
