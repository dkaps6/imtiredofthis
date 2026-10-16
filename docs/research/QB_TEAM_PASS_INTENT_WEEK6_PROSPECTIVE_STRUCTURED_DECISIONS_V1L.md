# QB Team Pass Intent — Week 6 Prospective Structured Decisions V1L

Date: 2026-10-15

Status: **FROZEN PREGAME SEMANTIC DECISIONS / RESEARCH ONLY**

Version:
`QB_TEAM_PASS_INTENT_WEEK6_PROSPECTIVE_STRUCTURED_DECISIONS_V1L`

## Collection authority

- frozen collector: `QB_TEAM_PASS_INTENT_PROSPECTIVE_STRUCTURED_SOURCE_COLLECTION_V1L`
- Thursday refresh run: `38827689707` — SUCCESS
- job: `116738872975`
- exact head: `6c694209323a5af646e0b2d80c175d3fec281dd4`
- artifact: `12016115756`
- digest: `sha256:c741cbd9a94c17d0996c2a00e789710da0964cfe5c6a2e0ec3fcf0ac380c3b01`
- packet rows: **38**
- scheduled team-weeks: **28**
- teams with retained first-party source: **15**
- source coverage: **53.57%**

The original artifact preserves exact source URL, publication timestamp, kickoff,
source family, title and review excerpts for every `review_id`.

Semantic decisions below were frozen at:
`2026-10-15T20:13:54-04:00`.

## Frozen semantic standard

Exactly one decision per retained source:
- `EXPLICIT_TARGET_GAME_INTENT_FOUND`
- `NO_EXPLICIT_TARGET_GAME_INTENT`
- `INSUFFICIENT_CONTEXT`

Allowed structured tags remained exactly those defined by V1L.

Precision rule:
- no opponent-inference masquerading as intent;
- no analyst prose;
- no semantic auto-finalization;
- no directional pass/run tag without explicit language;
- video-only pages without enough directly attributable visible text were
  `INSUFFICIENT_CONTEXT`, not guessed.

## Decision ledger

| review_id | team | decision | tags | speaker / evidence |
|---|---|---|---|---|
| 2026W6-JAX-1 | JAX | INSUFFICIENT_CONTEXT | — | retained OC/DC video lacked enough directly attributable visible text |
| 2026W6-JAX-2 | JAX | INSUFFICIENT_CONTEXT | — | retained QB video lacked enough directly attributable visible text |
| 2026W6-JAX-3 | JAX | INSUFFICIENT_CONTEXT | — | retained player media video lacked enough directly attributable visible text |
| 2026W6-ATL-1 | ATL | INSUFFICIENT_CONTEXT | — | retained player media lacked enough directly attributable visible text |
| 2026W6-ATL-2 | ATL | INSUFFICIENT_CONTEXT | — | retained player media lacked enough directly attributable visible text |
| 2026W6-ATL-3 | ATL | INSUFFICIENT_CONTEXT | — | retained player media lacked enough directly attributable visible text |
| 2026W6-CAR-1 | CAR | EXPLICIT_TARGET_GAME_INTENT_FOUND | PROTECTION_ADJUSTMENT | Brad Idzik: “somebody that you've got to account for in both phases” |
| 2026W6-CHI-1 | CHI | INSUFFICIENT_CONTEXT | — | retained coordinator video lacked enough directly attributable visible text |
| 2026W6-CHI-2 | CHI | INSUFFICIENT_CONTEXT | — | retained HC video lacked enough directly attributable visible text |
| 2026W6-CHI-3 | CHI | INSUFFICIENT_CONTEXT | — | retained HC video lacked enough directly attributable visible text |
| 2026W6-NE-1 | NE | NO_EXPLICIT_TARGET_GAME_INTENT | — | no qualifying target-game offensive intent in retained assistant-coach source |
| 2026W6-NE-2 | NE | INSUFFICIENT_CONTEXT | — | retained player media lacked enough directly attributable visible text |
| 2026W6-NE-3 | NE | NO_EXPLICIT_TARGET_GAME_INTENT | — | full Drake Maye transcript contained general offensive principles but no qualifying target-game directional intent |
| 2026W6-NO-1 | NO | NO_EXPLICIT_TARGET_GAME_INTENT | — | defensive-player source |
| 2026W6-NO-2 | NO | NO_EXPLICIT_TARGET_GAME_INTENT | — | team “Five Things” page announced upcoming media but contained no qualifying intent statement |
| 2026W6-NYG-1 | NYG | EXPLICIT_TARGET_GAME_INTENT_FOUND | TEMPO_UP | Matt Nagy: “For sure. Absolutely. I think that helped us last week. I know it helped us.” |
| 2026W6-NYG-2 | NYG | EXPLICIT_TARGET_GAME_INTENT_FOUND | PERSONNEL_USAGE_CHANGE | John Harbaugh: “Now we're starting to get adapted, I think, a little bit to the game plan and what we're doing” |
| 2026W6-NYJ-1 | NYJ | NO_EXPLICIT_TARGET_GAME_INTENT | — | defensive coordinator source |
| 2026W6-NYJ-2 | NYJ | INSUFFICIENT_CONTEXT | — | retained HC video lacked enough directly attributable visible text |
| 2026W6-PHI-1 | PHI | INSUFFICIENT_CONTEXT | — | retained player media lacked enough directly attributable visible text |
| 2026W6-PHI-2 | PHI | INSUFFICIENT_CONTEXT | — | retained player media lacked enough directly attributable visible text |
| 2026W6-PHI-3 | PHI | INSUFFICIENT_CONTEXT | — | retained OC video description identified discussion topics but not a precision-safe explicit target-game intent statement |
| 2026W6-PIT-1 | PIT | INSUFFICIENT_CONTEXT | — | retained HC/show video lacked enough directly attributable visible text |
| 2026W6-PIT-2 | PIT | INSUFFICIENT_CONTEXT | — | retained RB media lacked enough directly attributable visible text |
| 2026W6-PIT-3 | PIT | INSUFFICIENT_CONTEXT | — | retained HC video lacked enough directly attributable visible text |
| 2026W6-TB-1 | TB | INSUFFICIENT_CONTEXT | — | retained OL media lacked enough directly attributable visible text |
| 2026W6-ARI-1 | ARI | INSUFFICIENT_CONTEXT | — | retained OC video lacked enough directly attributable visible text |
| 2026W6-ARI-2 | ARI | NO_EXPLICIT_TARGET_GAME_INTENT | — | special-teams coordinator source |
| 2026W6-ARI-3 | ARI | NO_EXPLICIT_TARGET_GAME_INTENT | — | defensive coordinator source |
| 2026W6-KC-1 | KC | NO_EXPLICIT_TARGET_GAME_INTENT | — | defensive-player source |
| 2026W6-KC-2 | KC | NO_EXPLICIT_TARGET_GAME_INTENT | — | defensive-player source |
| 2026W6-KC-3 | KC | INSUFFICIENT_CONTEXT | — | retained HC video lacked enough directly attributable visible text |
| 2026W6-LAC-1 | LAC | NO_EXPLICIT_TARGET_GAME_INTENT | — | defensive-coach source |
| 2026W6-LAC-2 | LAC | INSUFFICIENT_CONTEXT | — | retained OC description discussed personnel/offensive state but did not expose a precision-safe explicit target-game intent statement |
| 2026W6-LAC-3 | LAC | NO_EXPLICIT_TARGET_GAME_INTENT | — | special-teams coordinator source |
| 2026W6-DAL-1 | DAL | NO_EXPLICIT_TARGET_GAME_INTENT | — | defensive coordinator source |
| 2026W6-DAL-2 | DAL | EXPLICIT_TARGET_GAME_INTENT_FOUND | RUN_VOLUME_UP | Klayton Adams first-party source description: “a stronger commitment to the running game” |
| 2026W6-DAL-3 | DAL | NO_EXPLICIT_TARGET_GAME_INTENT | — | special-teams coordinator source |

## Frozen counts

- `EXPLICIT_TARGET_GAME_INTENT_FOUND`: **4**
- `NO_EXPLICIT_TARGET_GAME_INTENT`: **13**
- `INSUFFICIENT_CONTEXT`: **21**

Explicit tag counts:
- `PROTECTION_ADJUSTMENT`: **1**
- `TEMPO_UP`: **1**
- `PERSONNEL_USAGE_CHANGE`: **1**
- `RUN_VOLUME_UP`: **1**

## Explicit source notes

### CAR — Brad Idzik
Retained source:
`2026W6-CAR-1`

Opponent-specific offensive protection/accounting statement concerning Eagles
DT Jalen Carter. No pass-volume or run-volume direction inferred.

### NYG — Matt Nagy
Retained source:
`2026W6-NYG-1`

The OC directly confirmed tempo helped the prior week and discussed making it an
advantage in the current offensive approach. Tagged only `TEMPO_UP`; no
pass-volume direction inferred.

### NYG — John Harbaugh
Retained source:
`2026W6-NYG-2`

The HC explicitly described adapting the game plan and practice reps toward
“Jameis type things.” Tagged `PERSONNEL_USAGE_CHANGE`; no pass-volume direction
inferred.

### DAL — Klayton Adams
Retained source:
`2026W6-DAL-2`

The official Cowboys first-party source description directly attributes to the
OC a need for “a stronger commitment to the running game” while preparing for
Green Bay. Tagged `RUN_VOLUME_UP`.

## Prospective integrity

At decision freeze:
- football outcomes read: **0**
- model residuals read: **0**
- sportsbook inputs used: **0**
- paid OddsAPI calls: **0**
- football predictive models fit: **0**
- production changed: **false**

SEA and DEN had **zero retained V1L sources**, so tonight's game contributes no
semantic decision and cannot contaminate this packet.

## Scientific boundary

These are source-level semantic labels only.

Week 6 alone is not enough to test predictive value or alter QB projections.
Do not:
- turn the four signals into a QB mean correction;
- infer direction from `NO_EXPLICIT...`;
- convert `INSUFFICIENT_CONTEXT` to negative evidence;
- use target-week results to revise these labels;
- relax source families to increase coverage.

The existing V1L contract permits additional **source-only** snapshots later in
the target week under the exact same rules, provided each new decision is frozen
before that team's kickoff.

No paid OddsAPI.
No production mutation.
