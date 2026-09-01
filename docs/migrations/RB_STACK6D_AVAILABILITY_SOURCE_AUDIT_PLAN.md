# RB STACK6D / ND5 — Pregame Backfield Availability Source Audit Plan

## Motivation

STACK6C's frozen one-sided rotation hurdle failed because it overcontracted an eligible P3 population whose carry bias was essentially centered before correction. Lagged role/rotation information is therefore insufficient to determine which secondary backs should actually lose target-game workload.

The next missing football state is target-game backfield composition:

- which credible competing RBs are definitely unavailable;
- which previously unavailable competitors are returning;
- how many usable RBs remain;
- whether nominal depth rank changes meaning because of target-game availability.

Do not retune STACK6C. Audit the source family first.

## Candidate historical sources

### 1. nflverse weekly rosters

`load_rosters_weekly()` is available from 2002 onward and is documented as including weekly player status changes, injury designations, and roster moves. The roster-status dictionary includes:

- `ACT`: on active roster;
- `INA`: under contract but inactive/not on active roster;
- reserve/PUP/suspension and other roster states.

The audit must determine whether the weekly record contains enough timestamp/provenance information to treat `INA` as a target-game pre-kickoff inactive indicator. A week label alone is **not sufficient** timing proof.

### 2. nflverse injuries

`load_injuries()` includes official game-status reports plus `date_modified`.

Candidate pregame availability fields:

- `report_status` (`Out`, `Doubtful`, `Questionable`, etc.);
- practice status/injury fields;
- `date_modified`.

Only a row whose historical modification timestamp is demonstrably before the game's decision deadline may be used as a target-game pregame feature.

### 3. nflverse participation — benchmark truth only

Historical participation may be used only to validate source semantics (for example, whether an `INA` RB is absent from target-game offensive participation). It is delayed/postgame truth and is forbidden as a fitted target-game pregame feature.

## Timing standard

NFL game-day inactive lists are normally due approximately 90 minutes before kickoff. For this audit:

- `decision_deadline = scheduled kickoff - 90 minutes`;
- injury rows must have parseable `date_modified` and be `<= decision_deadline` to qualify as pregame evidence;
- weekly-roster `INA` can qualify as exact inactive state only if the source exposes a historical timestamp/provenance that proves the relevant weekly state existed by the decision deadline, or if the data documentation/source contract unambiguously identifies it as the official game-day inactive declaration.

Do not infer timing from target-game participation.

## Population

Audit 2024 and 2025 regular seasons.

Primary population:

- RB/FB rows by GSIS identity;
- team-week/game aligned to schedule;
- focus reporting on player-games and team backfields.

No rushing outcome is used.

## Predeclared audits

### A. Schema / timing

Report all candidate columns for weekly rosters and injuries, especially:

- player IDs;
- week/team;
- position;
- roster `status`;
- injury `report_status`;
- date/timestamp fields.

Measure:

1. schedule kickoff parse rate;
2. injury `date_modified` parse rate;
3. share of injury rows last modified by the 90-minute deadline;
4. share of `Out` injury rows last modified by the deadline;
5. whether weekly roster data contains any usable timestamp/provenance column.

### B. Semantic validation — benchmark only

Using delayed participation only as truth:

6. `INA` RB/FB target-game offensive-participation absence rate;
7. `ACT` RB/FB target-game offensive-participation presence rate;
8. pre-deadline injury `Out` offensive-participation absence rate;
9. pre-deadline `Questionable` / `Doubtful` participation rates descriptively;
10. agreement between roster `INA` and pre-deadline injury `Out` where both exist.

### C. Coverage / backfield state

11. RB/FB weekly-roster coverage among scheduled team-games;
12. injury-report coverage among RB/FB weekly-roster player-games;
13. percentage of team-games where a usable active/definitely-out RB count can be constructed without postgame information;
14. identity join coverage across roster, injury, schedule, and participation sources.

## Frozen source gates

### Infrastructure

1. scheduled team-game coverage for weekly RB/FB roster state >= `0.95`;
2. identity mapping/join coverage >= `0.95`;
3. participation benchmark join coverage for roster RB/FB rows >= `0.90` at player-game level (absence itself is a valid benchmark outcome; this gate refers to resolvable game/player identity rather than on-field presence);
4. injury timestamp parse rate >= `0.95` among injury rows used for timing claims.

### Exact inactive source gate

Weekly-roster `INA` qualifies as exact target-game inactive state only if BOTH:

5. historical pre-kickoff timing/provenance is demonstrably valid;
6. `INA` offensive-participation absence rate >= `0.98`.

If gate 5 fails, `INA` may remain benchmark/context evidence but cannot be fitted as a target-game pregame feature.

### Definite-out injury source gate

Official injury `Out` qualifies as a pregame definitely-unavailable signal if:

7. >= `0.95` of `Out` rows have a parseable modification timestamp at or before the 90-minute deadline;
8. pre-deadline `Out` offensive-participation absence rate >= `0.98`;
9. 2025 RB/FB team-game coverage is sufficient to create at least one useful availability-state field on >= `0.50` of P3-eligible team-games.

The `0.50` threshold concerns presence of usable availability information, not missing=healthy imputation. Missing injury report is not automatically interpreted as active unless source semantics prove that interpretation.

## Allowed dispositions

### Exact inactive qualifies

`GO_STACK6D_EXACT_INACTIVE_COMPETITOR_STATE`

This authorizes a frozen architecture using exact game-day active/inactive competitor counts/state.

### Exact inactive fails but official pregame Out qualifies

`GO_STACK6D_DEFINITE_OUT_COMPETITOR_STATE_ONLY`

This authorizes a narrower architecture using only timestamp-qualified definitely-out state plus existing pregame information. Do not infer exact active state for questionable/healthy scratches.

### Neither qualifies

`STACK6D_AVAILABILITY_SOURCE_NOT_TIMESTAMP_SAFE`

Do not use target-game roster/inactive state. Move to another genuinely new live source or stay with lagged state.

## Forbidden

- no carry or yard outcome model fit;
- no sportsbook data;
- no target-game participation as a feature;
- no using weekly `INA` merely because it matches postgame nonparticipation if timing proof fails;
- no treating `Questionable` or `Doubtful` as inactive without an exact source;
- no missing-injury-report = active assumption unless explicitly validated by source contract;
- no feature / threshold / hyperparameter / weight search;
- no production change.
