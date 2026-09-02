# RB STACK6D Availability Source Audit — Implementation Correction

Status: FROZEN BEFORE CORRECTED RERUN

This note corrects one implementation defect in the first STACK6D availability-source audit. It does **not** change the research question, source gates, thresholds, eligible P3 population, or any model behavior.

## Defect found

The first source-audit run (`33571944162`) applied a generic `season_type == REG` filter independently to nflverse source tables. The injury source does not carry that field consistently across the audited seasons. That schema drift caused timestamp-bearing injury rows to be discarded while retaining rows without usable historical modification timestamps.

The resulting 0% injury `date_modified` parse rate is therefore not authoritative evidence about the source.

## Frozen correction

1. The schedule remains the canonical regular-season authority.
2. Non-schedule sources (weekly rosters, injuries, participation benchmark) are restricted to regular-season team-games by joining/filtering against the canonical regular-season schedule keys `(season, week, team)`.
3. No outcome columns are used to select or tune this correction.
4. No sportsbook data are loaded.
5. Target-game participation remains delayed benchmark truth only and never becomes a fitted pregame feature.
6. All original STACK6D source gates and thresholds remain unchanged.
7. Season-level injury timing diagnostics are emitted so timestamp availability cannot be hidden by cross-season schema differences.

## Authority

For STACK6D source qualification, the corrected rerun supersedes the timestamp-related conclusions of run `33571944162`. All other prior STACK6/STACK6B/STACK6C dispositions remain unchanged.
