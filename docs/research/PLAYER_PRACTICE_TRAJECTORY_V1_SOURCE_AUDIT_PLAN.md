# Player Practice Trajectory V1 — Source Audit Plan

Date: 2026-09-24

Status: FROZEN SOURCE AUDIT — NO OUTCOME SCORING

## Question

Does the maintained historical injury/practice feed preserve enough within-week, timestamp-safe player-level practice participation detail to construct a genuinely new pregame trajectory feature for RB/FB/TE/WR?

Examples:
- DNP -> LIMITED -> FULL
- DNP -> DNP -> LIMITED
- first practice back after prior-game absence
- consecutive LIMITED days
- final practice upgrade/downgrade

This is distinct from prior work that used single weekly binary flags such as practice_dnp/practice_limited. Do not treat those old binaries as a test of within-week trajectory.

## Audit only

Inspect 2023-2026 raw nflreadpy injury data and player registry.

Report:
- raw columns;
- report-date/timestamp fields available;
- rows by season/week;
- rows per player-week;
- share of RB/FB/TE/WR player-weeks with >=2 and >=3 distinct report dates;
- practice-status vocabulary;
- game-status vocabulary;
- position-resolution rate;
- 2026 W1-W3 current coverage.

No football outcomes, projection residuals, sportsbook fields, or predictive fitting.

## Gate

Advance only if:
1. a real report-date field exists;
2. >=70% of historical skill-position player-weeks with any report have >=2 distinct pregame report dates;
3. >=50% have >=3;
4. 2026 current-season data are available in the same schema;
5. timestamps/dates are structurally pregame-report dates rather than postgame reconstructions.

Otherwise disposition is SOURCE_NOT_DENSE and this exact trajectory lane closes.

## Standing protections

No M96 reopening. No retrospective RB router search. No production change. No sportsbook input.
