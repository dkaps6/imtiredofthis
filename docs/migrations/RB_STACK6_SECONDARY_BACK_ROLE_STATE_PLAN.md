# RB STACK6 / ND3 — Secondary-Back Role & Substitution State

## Why this migration exists

STACK5 localized the remaining retrospective Vegas gap in the current P3 rushing-yard parent to ordinary / non-M95F-risk backs, especially depth-rank 2 and 3+ players. The clearest systematic miss is large false-high rushing-yard projections where predicted carries exceed actual carries. This is an opportunity-allocation problem, not justification for another generic global RB model.

Current P3 parent:

- Week 1: frozen production-equivalent full stack.
- Weeks 2-18: STACK2 enriched M94C opportunity/allocation multiplied by frozen full-stack implied efficiency/context.
- 2025 all-RB MAE: 19.949524.
- Exact 899 market-listed games: P3 MAE 24.315798 vs Vegas consensus 23.701891.
- M95F-risk subset already slightly beats Vegas; do not disturb that regime without evidence.

## Hypothesis

Aggregate snap percentage and depth rank do not identify *what kind of snaps* a secondary RB owns. Two RBs with the same 35% snap share may have very different rushing opportunity if one plays third-down/two-minute/pass-protection snaps while the other alternates early-down series.

A timestamp-safe, strictly lagged player-role layer may improve allocation by reconstructing prior-game participation by situation and by backfield rotation pattern.

## Candidate football information

Using historical play-level participation + PBP, audit and, only if source integrity passes, build strictly prior-game features for:

1. offensive on-field snap share;
2. early-down participation;
3. third-down participation;
4. two-minute participation;
5. short-yardage participation and rush role;
6. red-zone, inside-10, and inside-5 participation and rush role;
7. single-back vs multi-back personnel participation;
8. per-drive / per-series participation and rotation concentration where reconstructable;
9. historical ball-carrier rate conditional on being on field;
10. historical competitor overlap and backfield substitution pattern.

Exact target-week active/inactive status may only be used from a genuine pregame source. Target-week participation itself is postgame evidence and is prohibited upstream. Participation-derived features must be shifted so only completed prior games can inform a target game.

## Scientific protocol

- Sportsbook inputs upstream: 0.
- Target-game participation upstream: 0.
- Target-game actual carries/rush yards upstream: 0.
- 2025 is exposed retrospective development data, not pristine confirmation.
- First run is a source/schema/derivability audit only.
- If the audit passes, build a narrow role/allocation module. Do not re-fit full-stack efficiency or reopen generic M96 router searches.
- Evaluate first on player share / carry allocation, then on downstream rush-yard MAE using the frozen P3 efficiency/context layer.
- Primary evaluation slices: M95F non-risk, depth-rank 2, depth-rank 3+, and 10+ yard model-over-Vegas disagreements. The market remains downstream benchmark only.
- Non-degradation: M95F-risk and depth-rank-1 performance must not materially deteriorate.
- Any promising retrospective architecture must be frozen for prospective 2026 confirmation before production promotion.

## Source audit pass conditions

The participation/PBP source may advance only if:

- play-level join coverage is at least 95%;
- offensive player and position arrays are populated and align well enough to identify RB/FB on-field participation;
- ball-carrier identity can be reconciled to on-field participants at high coverage on rush plays;
- down/distance, field-position and clock descriptors support the intended situational buckets;
- prior-game shifting can be performed at player/team/game grain without using target-game participation.

If any required family is unavailable, preserve the failure and build only the subset that is source-valid.
