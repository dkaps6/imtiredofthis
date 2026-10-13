# WR-CB Free Historical Archive Source Readiness V1

Date: 2026-09-29  
Status: **SOURCE FRONTIER REOPENED — FREE HISTORICAL ASSIGNMENT ARCHIVE IDENTIFIED; PARITY/COVERAGE AUDIT REQUIRED**  
Branch: `research-week3-postmortem-execution-v1`

## Why this audit exists

The repo's prior WR-CB source conclusion was:

`NO_GO_TRUE_ASSIGNMENT`

but that conclusion was specifically about reconstructing true receiver-to-defender
responsibility from nflverse participation/PBP. It correctly established that
nflverse does not identify explicit historical CB-to-WR responsibility.

It did **not** establish that no free public historical assignment archive exists
outside nflverse.

That distinction matters because M82/M83 explicitly left a source-blocked
frontier open:

`TOP_WEAPON_ESCAPE_HATCH`

requiring materially new pregame route/responsibility-level
receiver-defender exposure or equivalent individual matchup information.

## Newly identified free archive

Public FantasyAlarm weekly WR/CB matchup reports have been verified for at least:

- 2021 regular-season weekly reports;
- 2022;
- 2023;
- 2024;
- 2025;
- current 2026.

Verified examples include:
- 2021 Week 5 and Week 11;
- 2022 Week 5;
- 2023 Week 5 and Week 10;
- 2024 Week 1, Week 5, Week 10 and Week 18;
- 2025 Week 1, Week 5, Week 10 and Week 18;
- 2026 Weeks 1-3.

The archived reports contain explicit pregame receiver/cornerback pairings,
typically separated by alignment:
- left WR vs right CB;
- right WR vs left CB;
- slot WR vs slot CB.

Recent reports also preserve WR team, opponent, and a categorical matchup label.

## Critical interpretation correction

This discovery changes the source frontier from:

`NO_FREE_HISTORICAL_WR_CB_SOURCE_KNOWN`

to:

`FREE_HISTORICAL_WR_CB_ASSIGNMENT_ARCHIVE_IDENTIFIED_NEEDS_COVERAGE_SEMANTIC_AUDIT`

It does **not** promote any WR-CB model feature yet.

## What may be used from FantasyAlarm

### Assignment / alignment — eligible for source audit

Potentially valid factual fields:
- season;
- week;
- publication timestamp;
- WR name;
- WR team;
- opponent;
- CB name;
- alignment bucket (left / right / slot).

These are the fields relevant to the previously source-blocked responsibility
frontier.

### Editorial matchup grade — NOT football-model eligible by default

Do not treat FantasyAlarm's `Safe / Moderate / Risky` or
`Upgrade / Neutral / Downgrade` label as a pure CB-quality feature.

The article commentary can blend:
- WR role / target expectation;
- injuries;
- quarterback quality;
- fantasy ranking judgment;
- current Vegas/game environment;
- external coverage grades.

Therefore using the editorial label directly upstream in football generation
would violate the project's sportsbook/football-information boundary unless a
separate audit proves a clean, sportsbook-free construction.

The first source build must preserve the raw label for provenance but mark it:
`EDITORIAL_COMPOSITE_NOT_MODEL_ELIGIBLE`.

## Historical completeness caveat

Older reports explicitly describe:
- all outside matchups;
- only a selected subset of slot matchups in some seasons.

Therefore missing rows must **never** be interpreted as "no matchup" or zero
exposure.

Historical source completeness must be measured separately by:
- season;
- week;
- alignment bucket;
- NFL team;
- WR identity.

A historical predictive test may use only explicitly observed matchup rows.

## Proposed scientific use if source clears

The cleanest no-paid-data use is **not** to import somebody else's CB grade.

Instead:

1. use the archived WR-CB pair identity as the pregame assignment observable;
2. join the WR's realized receiving outcome after the game;
3. estimate CB effect strictly from that CB's prior observed assignments only;
4. control against the project's own pregame baseline / opportunity state;
5. test on an untouched later season.

Candidate design:
- discovery / training: 2021-2024 assignments;
- untouched confirmation: 2025;
- 2026 reserved for live/prospective use.

Exact seasons may move only if archive coverage audit shows a natural source
boundary; no outcome may choose the split.

## Relation to prior closed defensive work

Do not confuse this source with closed families:

- M56: richer static/lagged defense, pressure, coverage, box and QB-defense
  interactions — SIGNAL_SCREEN_FAILED.
- M83: comparable-opponent prediction of target-game tactical adaptation —
  `NO_DEFENSIVE_ADAPTATION_MECHANISM`.
- team-level `coverage_man_rate / coverage_zone_rate` ablation — near-neutral.
- aggregate explosive-weapon proxies M72/M75 — failed.

True WR-CB responsibility identity is materially new information relative to
those closed tests and is explicitly the type of source M82/M83 said would be
required for a legitimate revisit.

## Current production heavy-zone rule

The free WR-CB archive does **not** justify the existing legacy
`coverage_penalty()` fallback.

Current rule:
- heavy man / tough shadow: YPT x0.94, target share x0.92;
- heavy zone: YPT x1.04, target share x1.06.

The exact preserved Week-3 artifact showed:
- 161 WR rows;
- 131 affected;
- 27/30 teams affected;
- zero nonblank frozen `primary_cb` rows;
- all active adjustments came from the generic heavy-zone branch.

Therefore current Week-3 behavior was effectively a broad team-zone tilt, not
player-level WR-CB matchup modeling.

That legacy rule has since been **retired from production on main via PR #664**.
The newly identified free WR-CB source must not be used to restore, rescue, or
rationalize the deleted static coefficients. Any future WR-CB contribution must
be introduced under a new frozen contract after this source gate clears.

## Next source actions

1. Build an explicit public-URL manifest by season/week.
2. Parse only factual matchup/alignment rows.
3. Preserve publication/update timestamp and page URL.
4. Canonicalize WR/CB/team identity with fail-close ambiguity handling.
5. Audit coverage by season/week/alignment/team.
6. Confirm every source page was published before the games it describes.
7. Freeze a historical predictive protocol only after source coverage is known.

No paid source is required for this source audit.
No production change is authorized.
