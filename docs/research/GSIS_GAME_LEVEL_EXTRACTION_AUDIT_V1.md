# GSIS game-level extraction audit v1

Status: research-only source audit  
Audit date: 2026-09-28 UTC  
GSIS source version: v11.3.0

This document records a bounded authenticated rendered-DOM audit. It contains no
GSIS report cell values, player-level results, credentials, cookies, tokens,
authorization headers, browser storage, session state, or sensitive request
metadata.

## Result

`Player -> Game by Game` and `Team -> By Game` can both be collected
systematically through native rendered-page selectors. The same method was
verified on two current teams. A single 2025 REG team was used to confirm
historical game granularity and schema consistency. No broad historical
collection was performed.

The extraction path is:

1. select season, phase, team, and report mode through native `select` elements;
2. wait for the rendered report table to refresh;
3. confirm the selected filter values;
4. read table cells from the rendered DOM;
5. retain the raw table orientation and source strings before any long-form
   transformation.

No network interception, direct internal endpoint, or persisted browser session
material is required.

## Player -> Game by Game

### Filters and coverage

| Filter | Observed behavior |
|---|---|
| Season | 2026 through 1981 |
| Phase | Pre Season, Regular Season, Post Season |
| Team | 35 labels, including current and legacy franchise labels |
| Mode | Rushing, Passing, Receiving, Punting, Kickoff Returns, Punt Returns, Place Kicking |

The team list is not season-normalized. A collector must use a season-aware
franchise map or explicitly preserve source-empty results. Current and legacy
labels must not be silently merged.

### Rendered structure

The page contains an outer layout table and one `table.linedTable` per player.
Within each player table:

- row 0 contains the report mode and displayed player label;
- row 1 contains the column schema;
- subsequent rows contain scheduled games for the selected phase;
- the final row is `TOTALS` and must be tagged or excluded from a game-level
  materialization.

The bounded 2025 REG validation produced 17 scheduled game rows in each player
table. A player can have a scheduled-game row without a statistical appearance.
GSIS encodes those states in report cells using source strings including:

- `Active, Did Not Play`
- `Inactive`
- `Not on Team`
- `-`

These values are source semantics. They must remain strings and must not be
coerced to zero or null during raw capture.

### Exact displayed schemas

| Mode | Columns |
|---|---|
| Rushing | Date, Opponent, No, Yds, Avg, LG, TD |
| Passing | Date, Opponent, Att, Cmp, Yds, Cmp%, Yds/Att, TD, TD%, INT, INT%, LG, Sack/Lost, Rating |
| Receiving | Date, Opponent, Tar, Rec, Yds, Avg, LG, TD |
| Punting | Date, Opponent, No, Yds, Avg, TB, In20, LG, Net |
| Kickoff Returns | Date, Opponent, No, Yds, Avg, LG, FC, TD |
| Punt Returns | Date, Opponent, No, Yds, Avg, LG, FC, TD |
| Place Kicking | Date, Opponent, KO, KO Yds, TB, XP Att, XP, FG Att, FG, Long, Points |

## Team -> By Game

### Filters and coverage

| Filter | Observed behavior |
|---|---|
| Season | 2026 through 1981 |
| Phase | Pre Season, Regular Season, Post Season |
| Team | Same 35-label current-plus-legacy selector |
| Side | Offense, Defense |
| Display | None, alternate rows, alternate groups, bold first in group |

The display selector changes presentation only and is not part of the source
data key.

### Rendered structure

The report is a stable `table#tblTeamStats` matrix:

- the first column contains metric names;
- each remaining column represents one game and is labeled with date and
  opponent;
- offense and defense use the same displayed metric schema;
- a completed 2025 REG validation contained 17 game columns;
- bounded current-season tests succeeded for two teams and for both offense and
  defense.

### Exact displayed metric schema

1. Points
2. 1st Qtr
3. 2nd Qtr
4. 3rd Qtr
5. 4th Qtr
6. Overtime
7. TDs (Ru-P-Ret)
8. PATs (M/A)
9. 2PT Convs (M/A)
10. FGs (M/A)
11. Safeties
12. First Downs
13. Rushing
14. Passing
15. Penalty
16. 3rd Down Conv (M/A)
17. 3rd Down Conv Pct
18. 4th Down Conv (M/A)
19. 4th Down Conv Pct
20. Red Zone Conv (M/A)
21. Red Zone Conv Pct
22. Goal to Go Conv (M/A)
23. Goal to Go Conv Pct
24. Total Net Yards
25. Total Off. Plays
26. Avg. Gain Per Play
27. Net Yards Rushing
28. Total Rushing Plays
29. Avg. Gain Per Rush
30. Net Yards Passing
31. Times Sacked
32. Yards Lost on Sacks
33. Gross Yards Passing
34. Pass Attempts
35. Pass Completions
36. Completion Pct
37. Avg. Gain Per Pass
38. Interceptions
39. Fumbles / Fum. Lost
40. Penalties
41. PenaltyYards
42. Punts
43. Gross Punting Average
44. Touchbacks
45. Inside20
46. Punts Blocked
47. Net Punting Average
48. Punt Returns
49. Punt Return Yards
50. Punt Return Avg.
51. Fair Catches
52. Kickoff Returns
53. Kickoff Return Yards
54. Kickoff Return Avg.
55. Time of Possession

## Leakage and archive contract

These reports expose true game-level observations rather than only season
aggregates. They can support historical research when each game is keyed by
season, phase, team, displayed date, displayed opponent, side or mode, and the
source capture metadata. Pregame features must be built only from games that
precede the target game. Totals rows and completed-season aggregates are not
pregame states.

The frozen 2026 Week-3 boundary remains in force. The current rows may be
archived and schema-audited, but must not be used to tune, diagnose, grade, or
modify the production model before the established postmortem window opens.

## Recommended next acquisition step

Run a private, bounded 2025 REG pilot on the same predeclared small team sample:

- Player Game by Game: Rushing, Passing, Receiving;
- Team By Game: Offense and Defense.

Preserve the rendered source tables unchanged, then materialize a separate
long-form layer. Validate game identity, legacy-franchise handling,
nonparticipation strings, totals-row exclusion, and agreement with nflverse
before considering a broader historical collection.
