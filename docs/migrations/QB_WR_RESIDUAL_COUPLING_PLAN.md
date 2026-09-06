# QB-WR Residual Coupling Audit — Frozen Plan

## Question
When our WR receiving-yard model misses a team's WR production high/low, does the football-only QB passing-yard model tend to miss the same team's QB passing yards in the same direction?

This is deliberately different from M72. M72 tested pregame aggregate explosive-weapon x defense features as a predictor of QB residuals and failed. This audit first measures direct historical error coupling; it does not claim a deployable pregame signal.

## Canonical evidence
- M89 football-only QB synthesis trace, run `33331073376`, seasons 2024-2025.
- WR-R1 exact M38 paired casebook, run `34058453941`; use 2024-2025 WR receiving-yard rows only.
- Join key: season/week/team.
- Sportsbook data prohibited.

## Residual definitions
- `qb_residual = actual_pass_yards - football_synthesis`.
- `wr_all_residual = sum(actual WR rec yards) - sum(M38 WR rec-yards projections)` within team-week.
- `wr_top2_projected_residual`: aggregate residual of the two WRs selected strictly by highest pregame M38 projection in that team-week.
- `wr_top1_projected_residual`: residual of the WR selected strictly by highest pregame M38 projection.

Outcome values may be used only to calculate residuals after the pregame selection is frozen. No realized WR outcome may select the player subset.

## Metrics
For all three WR residual aggregates vs QB residual:
- Pearson correlation
- Spearman correlation
- same-sign rate excluding exact zeros
- mean QB residual by WR residual quartile
- Q4-minus-Q1 QB residual gap
- 2024 and 2025 correlations separately

Also report identity coverage and team-week counts.

## Frozen diagnostic interpretation
`STRONG_QB_WR_ERROR_COUPLING` only if the all-WR residual satisfies ALL:
1. >=700 aligned QB team-weeks.
2. Pearson >=0.35.
3. Spearman >=0.30.
4. same-sign rate >=0.60.
5. Q4-minus-Q1 QB residual gap >=30 yards.
6. Pearson >0 in both 2024 and 2025.

Otherwise disposition `NO_STRONG_QB_WR_ERROR_COUPLING`.

Even a pass does not authorize WR outcomes upstream of QB. It would only show that the two position models share meaningful error structure. A deployable QB bridge would still require a separately validated *pregame* WR signal; M72 and WR-R2 cannot be retroactively rescued.
