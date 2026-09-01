# RB STACK6 / ND3 — Secondary-Back Situational Role Model

## Frozen hypothesis

STACK5 localized the remaining 2025 market deficit primarily to M95F-nonrisk depth-rank 2/3 backs and large false-high opportunity disagreements. STACK6 tests whether strictly prior-game situational participation explains carry allocation that aggregate depth rank and total snap share miss.

## Parent

P3 is frozen:
- Week 1: full historical stack.
- Weeks 2-18: STACK2 enriched M94C opportunity/allocation × full-stack implied efficiency/context.

No STACK6 arm changes P3 efficiency. A carry correction is converted to rushing yards using frozen P3 implied YPC.

## Target domain

STACK6 point corrections are allowed only when all are true:
- week >= 6;
- frozen M95F workload-risk state is false;
- pregame depth_rank >= 2.

M95F-risk rows and depth-rank 1 rows are unchanged by construction. Rows without sufficient prior situational history remain at P3.

## Pregame feature construction

The target row universe comes only from the existing P3 pregame casebook. The participation feed never determines whether a target row exists.

For every target player-game, STACK6 performs an as-of lookup into completed historical RB/FB player-games with source season-week ordinal strictly less than target season-week ordinal. No target-game or future-game participation may create, fill, or select a feature.

New situational features include prior-1 and prior-3 player share of team RB presence/usage in:
- all RB presence;
- rushing attempts;
- early downs;
- third downs;
- third-and-long;
- two-minute;
- short yardage;
- red zone;
- inside 10;
- inside 5;
- shotgun;
- under center;
- 11/12/21/22 personnel.

Also include prior team backfield concentration/rotation descriptors computed from completed games only. Same-team and any-team player histories are kept distinct where available.

The source is historical/postseason only. It can prove the information family but cannot be silently promoted as a live-2026 dependency.

## Fixed model protocol

Outcome: `actual_rush_att - P3_parent_att`.

Evaluation uses expanding-week 2025 OOF starting Week 6. Each test week fits only earlier 2025 eligible target rows. Fixed Ridge alpha = 10. No alpha grid, feature selection, threshold search, weight search, sportsbook feature, or market-driven model choice.

Residual predictions are clipped to the intersection of training-only 5th/95th residual quantiles and fixed [-4,+4] carries. Corrected carries are clipped at zero.

Ablations:
1. `AGG_ROLE` — existing pregame aggregate role/snap context only.
2. `SITUATIONAL_ROLE` — new as-of situational role/rotation history only.
3. `AGG_PLUS_SITUATIONAL` — both fixed blocks.

## Football-first retention gate

An arm is scientifically retainable only if all are true on the eligible Week 6-18 target subset:
- carry MAE gain versus P3 >= 0.20 carries;
- rushing-yard MAE gain versus P3 >= 0.15 yards;
- late Week 13-18 eligible-subset yard MAE gain > 0;
- all-RB Week 6-18 yard MAE regression <= 0.05 yards;
- eligible-subset absolute carry bias does not worsen by > 0.25 carries;
- M95F-risk rows and depth-rank 1 rows have exactly zero projection change.

If multiple arms pass, prefer the smallest feature block within 0.05 eligible-subset yard MAE of the best passing arm; otherwise retain none.

## Market audit

Only after the candidate predictions and scientific disposition are fixed, join the same archived 899-game Vegas benchmark and report:
- all 899;
- M95F risk/nonrisk;
- depth rank 1/2/3+;
- absolute disagreement buckets and signed 10+ disagreement directions.

Sportsbook values never affect fit, clipping, eligibility, arm selection, or retention gates.

## Interpretation

2025 is exposed retrospective development evidence. A positive STACK6 result must be frozen for prospective 2026 confirmation and requires a qualified live-capable equivalent source before production use.
