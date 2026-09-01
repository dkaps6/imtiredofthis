# RB STACK7 — Official Game-Day Availability Reallocation

## Motivation

STACK6 proved that strictly lagged play-level situational role contains real but mostly redundant allocation information. Its frozen 75/25 secondary/non-risk role arm failed materiality gates. The no-retune failure atlas showed the role signal is asymmetric: contraction calls help materially, while expansion calls hurt. This supports a genuinely different current-week state family rather than retuning the role weight.

The repository already contains the corrected frozen M78 reconstruction of official NFL.com game-day inactive reports for every 2024 and 2025 regular-season team-game:

- source run `33288381864`
- source artifact `9725181841`
- source head `426fde17668aad049e887ba8da0776a28a4dc9ca`
- frozen team-week CSV SHA256 `d39aaf0feea101f3e0d2721ebd4118ef33fb1a4d3c76670e2a4f17734e37b609`
- 1,088 team-week rows, 544 per season.

This is genuine pregame identity information and is not reconstructed from target-game participation.

## Frozen primary mechanism

Use current P3 as the parent. For each team-game:

1. identify projected RB/FB rows that appear on the official inactive list;
2. set their current P3 opportunity share to zero;
3. proportionally renormalize the remaining projected RB/FB parent shares to conserve the exact parent team RB opportunity pool;
4. preserve each player's frozen parent implied efficiency; only opportunity is changed.

No fitted coefficient, threshold, feature search, or sportsbook input is used.

The correction applies to all weeks because official game-day inactive status is available before kickoff. If no projected RB/FB is officially inactive for a team-game, the parent is unchanged.

## Source integrity gates

Before scoring the candidate:

- every projected player matched to official inactive identity must have actual carries = 0 and actual rushing yards = 0; otherwise fail source/player reconciliation;
- official inactive team-week coverage must be complete for the 2025 parent schedule;
- team RB opportunity pool conservation max absolute difference must be <= 1e-6.

## Scientific retention gates

Frozen before first result is exposed:

- all-RB rushing-yard MAE gain >= 0.10;
- all-RB carry MAE gain >= 0.03;
- secondary/non-M95F-risk rushing-yard MAE gain >= 0.10;
- active teammates on team-games with a projected inactive RB may not regress by > 0.10 rushing-yard MAE;
- M95F-risk rushing-yard MAE regression <= 0.10;
- all-RB RMSE regression <= 0.10;
- all-RB absolute-bias deterioration <= 0.50.

The exact 899-game sportsbook benchmark is downstream only and cannot select or retain the football module.

## Transition diagnostics

Without using them as point adjustments in this migration, record pregame-known transition states:

- player was officially inactive in the team's previous game but is not inactive now (`returned_from_official_inactive`);
- count/share of currently projected teammates that returned from official inactive status;
- current projected inactive competitor count/share.

These diagnostics decide whether a separate role-return/activation mechanism is justified after STACK7. They are not allowed to alter the frozen primary candidate.

## Validation status

2025 is exposed retrospective development evidence. Any retained architecture is frozen for prospective 2026 confirmation before production promotion.
