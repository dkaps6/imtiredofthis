# GSIS novelty / existing-source audit v1

**Status:** COMPLETE — ACQUISITION GATE

**Date:** 2026-09-28
**Scope:** Source and prior-research comparison only. No additional GSIS data
was acquired, no model was fit, and no production or Full Slate path changed.

## Decision

The GSIS reports divide into three practical groups:

1. **Genuinely new live information:** exact-player lineup co-occurrence and
   current personnel deployment by situation. The current stack has individual
   snap shares and current roster/depth information, but it does not have an
   in-season play-linked source for who shared the field or how often each
   personnel grouping was used in specific down/distance states.
2. **Authoritative duplicates:** official denominators and classifications that
   are useful for source QA, including total plays, official pass attempts,
   sacks, red-zone conversions, and direction buckets.
3. **Adequate existing information:** player box scores, team game statistics,
   down/distance tendencies, field position, quarter, score differential, and
   generic run/pass propensity. These already exist in nflverse PBP/player
   stats and the repository's PlayerForm, TeamForm, Sharp, and research layers.

The next GSIS acquisition, if separately authorized, should be limited to the
first group. `Player -> Game by Game` and `Team -> By Game` do not justify a
parallel historical pipeline.

## Novelty classes

- **A — genuinely new:** absent at the required live/pregame granularity.
- **B — duplicate but potentially superior/current/authoritative:** comparison
  or QA candidate, not a new football hypothesis.
- **C — already adequate:** do not ingest or build a parallel pipeline unless a
  concrete QA discrepancy requires it.

Class A means the field is eligible for a bounded source-quality test. It does
not authorize model integration or imply predictive value.

## Existing source inventory

| Existing source/path | Current information already available |
|---|---|
| nflverse PBP | Play-level down, yards to go, field position, quarter, score differential, play type, QB dropback, official-attempt construction, sacks, scrambles, rush/pass location, pass depth/location, first downs, conversions, red-zone/goal-to-go state, turnovers, penalties, clock, EPA and success. |
| nflverse weekly player stats | Game/week-level passing, rushing and receiving volume/production, including official attempts, carries, targets and receptions; defensive and special-teams box-score fields where supplied. |
| nflverse snap counts | Player/team/week offensive snaps and offensive snap percentage. The production WR-R15/TE-R5P path uses strict-prior 2020–2026 observations and, for 2026 Week 3+, fails closed unless the immediately completed prior week has schedule-complete team coverage. |
| nflverse participation | Historical play-linked participants, personnel/formation context, defenders in box and partial coverage labels. Repository audits establish that 2023+ participation is postseason-only and therefore not an in-season 2026 source. It also lacks explicit defender-to-receiver coverage responsibility. |
| PlayerForm / historical player logs | Canonical season/week/player/team/game observations; targets, receptions, receiving yards, carries, rushing yards, pass attempts and passing yards; derived target/rush shares and efficiency. Historical construction is explicitly strict-prior at prediction time. |
| TeamForm / historical team-week PBP | Plays, dropback rate, official attempts per dropback, pace, PROE/pass tendency, red-zone rate, personnel-12 rate when present, box rates, offensive/defensive success, explosive rate, pass/rush EPA and pressure/sack context. |
| Sharp feeds | Offensive tendencies/PROE, neutral pass rate, pace, coverage scheme, defensive/offensive line and other team-level tendency enrichments. |
| Ourlads/current availability | Current roster/depth role state and availability authority for the live player universe. |
| FTN-derived research | Separate advanced mechanism families such as exact blitz construction and receiver-error attribution. The audited GSIS reports do not expose those fields. |

## Field/report novelty matrix

| GSIS report | Field/group | Current existing source | Already have it? | Live/pregame freshness difference | Granularity difference | Class | Recommended action |
|---|---|---|---|---|---|---|---|
| Player → Game by Game | Season/phase, date, opponent and game identity | nflverse schedule; historical player logs | Yes | No meaningful advantage | Same game grain | C | Use existing schedule/game keys. |
| Player → Game by Game | Rushing: No, Yds, Avg, LG, TD | nflverse weekly player stats and PBP; PlayerForm | Yes | GSIS may publish an official correction sooner, but the stack already has completed-game weekly rows | Same game/player grain | C | No historical GSIS pipeline; compare only if a concrete discrepancy appears. |
| Player → Game by Game | Passing: Att, Cmp, Yds, rates, TD, INT, LG, Sack/Lost, Rating | nflverse weekly player stats plus PBP; QB pass synthesis | Yes | No established pregame advantage. Official attempts/yards already reconciled on the canonical historical cohort | Same game/player grain | C | Keep nflverse authority. GSIS can be a one-off QA comparator for attempt semantics. |
| Player → Game by Game | Receiving: Tar, Rec, Yds, Avg, LG, TD | nflverse weekly player stats; PlayerForm; WR/TE/RB entitlement research | Yes | No material advantage after the completed game | Same game/player grain | C | Do not duplicate. Targets and receiving history are already core inputs. |
| Player → Game by Game | Punting, returns and place-kicking modes | nflverse weekly stats and PBP | Substantially yes | No documented model need or freshness gap | Same game/player grain | C | Do not acquire for the current player-prop stack. |
| Player → Game by Game | `Active, Did Not Play`, `Inactive`, `Not on Team`, `-` | weekly rosters, official-inactives/current-availability pipeline | Yes, through different fields | GSIS states are retrospective report semantics, not a new pregame availability signal | Player/game status string | B | Preserve only if GSIS is later used for QA; never coerce to zero. Do not make it a new availability authority without timing proof. |
| Player → Game by Game | `TOTALS` rows | PlayerForm aggregation | Yes | Completed-season total can leak if used pregame | Season aggregate | C | Exclude from game-level materialization and pregame research. |
| Team → By Game | Points and scoring by quarter; TD/PAT/2PT/FG/safety | nflverse PBP and schedule/team scores | Yes | No meaningful advantage | Same game/team grain | C | Use existing PBP. |
| Team → By Game | First downs; 3rd/4th-down; red-zone and goal-to-go made/attempted and rates | nflverse PBP; TeamForm/PBP builders | Yes | GSIS is an official aggregate and may be useful for denominator QA | Matrix instead of play rows; no new state | B | One-time semantic reconciliation only if needed; no parallel history. |
| Team → By Game | Total offensive plays, rush plays/yards, pass attempts/completions/yards, sacks/lost | nflverse PBP; historical team-week builder; QB attempt-conversion path | Yes | GSIS is authoritative, but existing fields are current and already used pregame after lagging | Same game/team facts | B | QA official denominators only. Existing PBP remains the feature source. |
| Team → By Game | Turnovers, fumbles, interceptions, penalties/yards | nflverse PBP/team stats | Yes | No meaningful advantage | Same game/team grain | C | No acquisition. |
| Team → By Game | Punts, touchbacks, inside-20, returns, fair catches | nflverse PBP/player stats | Yes for current needs | No documented model gap | Same game/team grain | C | No acquisition. |
| Team → By Game | Time of possession | nflverse PBP clock/schedule-derived team stats | Yes or directly derivable | GSIS may be an authoritative QA value | Same game/team grain | B | QA only if a clock reconciliation is required. |
| Play Time → Lineup Detail | Exact 11-player offensive lineup | Historical nflverse participation can reconstruct on-field players; current snap counts and Ourlads cannot reconstruct co-occurrence | **No for live 2026** | Material: GSIS is available in season; participation is not an in-season 2026 source | Exact 11-player combination rather than individual snap share | **A** | Preserve prospective weekly snapshots. First test incremental role-state information over current snap share/Ourlads; no immediate model use. |
| Play Time → Lineup Detail | Exact 11-player defensive lineup | Historical participation; current availability/depth and team defensive context | **No for live 2026** | Same material in-season advantage | Exact defensive unit co-occurrence; still no coverage assignment | **A** | Archive prospectively for continuity/personnel-state research. Do not infer WR-CB assignments. |
| Play Time → Lineup Detail | Plays, Passing Plays, Rushing Plays by exact lineup | Historical PBP + participation join; no current play-linked participation feed | **No for live 2026** | Material in-season advantage | Player-combination-conditioned play choice | **A** | Candidate for a bounded novelty test against aggregate snaps/personnel. Keep counts as observations through the prior completed week only. |
| Play Time → Lineup Detail | Avg Gain, pass/rush gain, first downs, TD, fumbles lost, INT by exact lineup | Derivable historically from PBP + participation; unavailable live as an exact-lineup join | Not live at this grain | GSIS supplies current combination-conditioned outcomes | Exact lineup efficiency/outcome split | B | Retain as raw semantics with the A-class lineup archive, but do not treat small-sample efficiency as a new model signal without a separate frozen test. |
| Play Time → Formation Usage | Down, yards to go, #TE, #WR | Historical PBP/participation; current snap counts have no formation co-occurrence | **No for live 2026** | Material: live/current-season personnel deployment is available before the next kickoff | Situation × personnel grouping | **A** | Preserve prospectively. This is the clean current personnel-mix candidate. |
| Play Time → Formation Usage | Play Count, rushing plays, passing plays and Pass% by personnel/down/distance | Derivable historically; current TeamForm has aggregate tendency rather than live personnel-conditioned tendency | **No at this live grain** | Material in-season advantage | Personnel-conditioned pass/run use | **A** | Test only as incremental information over current snaps, PROE and existing PBP tendencies. Do not relabel it as generic pass-tendency research. |
| Play Time → Formation Usage | Avg gain by play/rush/pass and NFL comparisons/ranks | PBP/participation historically; league ranks derivable | Concept already available | GSIS current aggregation may be fresher than participation | Personnel-conditioned efficiency and presentation rank | B | Archive only alongside the A-class raw table; ranks are display metadata, not a new feature. |
| Team → Down Analysis | Exact/grouped down and distance; rush/pass/scramble play counts | nflverse PBP (`down`, `ydstogo`, play type, `qb_scramble`) | Yes | No material gap for completed prior games | GSIS aggregate is coarser than PBP | C | Do not build a GSIS feature pipeline. |
| Team → Down Analysis | Yards, average, first downs and conversion rate by state; offense/defense | nflverse PBP and historical team-week/state research | Yes | GSIS can serve as an official aggregation check | Coarser aggregate; no new timing state | B | QA only. Do not reopen within-state pass-choice or efficiency hypotheses. |
| Team → Play Propensity | Down/distance run/pass percentages | nflverse PBP; TeamForm PROE/pass rate; prior QB/WR shared-opportunity research | Yes | No new pregame timing: these are outcomes through completed games | Coarser season aggregate | C | Stop. The concept and its residual structure have already been researched. |
| Team → Play Propensity | Field-position and red-zone/goal-to-go splits | nflverse PBP (`yardline_100`, goal-to-go); TeamForm red-zone rate | Yes | No material advantage | Coarser bins than PBP | C | Do not ingest. Prior first-down field-position work found occupancy was not the answer. |
| Team → Play Propensity | Quarter splits | nflverse PBP (`qtr`, clock); script-escalator/team-history builders | Yes | No material advantage | Coarser season aggregate | C | Do not ingest. |
| Team → Play Propensity | Winning/tied/losing score-differential splits | nflverse PBP (`score_differential`); script/volatility/game-state research | Yes | No material advantage | Coarser score bins | C | Do not ingest. Prior score-state decomposition already isolated the remaining within-state miss. |
| Team → Play Direction | Rushing left/middle/right and gap counts/efficiency | nflverse PBP run location/gap fields | Yes/derivable | No material advantage for completed prior games | GSIS official buckets versus play rows | B | If definitions differ, run a small QA reconciliation. Do not create a parallel source. |
| Team → Play Direction | Passing Short/Deep × Left/Middle/Right counts/efficiency | nflverse PBP pass location plus pass length/air yards | Yes/derivable | No material advantage | GSIS official charting categories may differ at boundaries | B | Definition QA only. It does not supply route, alignment, separation, or coverage responsibility. |
| Player → Defense | Tackles, assists, sacks/yards, TFL, QH, INT, PD, FF, FR | nflverse player stats and PBP; defensive enrichment paths | Substantially yes | GSIS is official/current but no documented pregame gap | Same player/game or season aggregate | C | No acquisition for current offensive-player markets. |
| Player → Defense | Rush/pass tackle splits | Derivable by joining PBP tackle IDs to play type; not currently a core materialized feature | Concept available, pipeline absent | GSIS may simplify current official aggregation | Player × play-type defensive split | B | Defer. No documented model gap requires it; require a preregistered use case before acquisition. |
| Play Time → Lineup Combinations | Exact-player combination frequency/usage | Lineup Detail; historical participation | Potentially duplicate of Lineup Detail | Could be live/current | Exact combination, schema not yet audited | **A provisional** | Do not acquire yet. First prove that its fields add something beyond Lineup Detail; otherwise omit it. |
| Play Time → Player Counts | Player participation/count summaries | Current nflverse snap counts and Lineup Detail/Formation Usage | Likely yes | A freshness advantage has not been demonstrated | Schema not yet audited | **B provisional** | Default to no acquisition. Give no novelty credit until a schema-only comparison proves an incremental field. |

## Answers to the seven gate questions

### 1. Player Game by Game

It has no demonstrated field or timing advantage that the current football
stack truly lacks. Passing, rushing, receiving and targets are already available
as weekly player-game observations and are already wired into PlayerForm and the
historical walk-forward path. GSIS could be an authoritative QA comparator for
official attempt/status semantics. It is not a new research source.

**Decision: do not acquire a GSIS historical Player Game-by-Game archive.**

### 2. Team By Game

Nothing in the 55-field matrix is materially new versus nflverse PBP and the
existing historical team-week builder. GSIS can validate official denominators,
especially total plays, pass attempts, sacks and conversion attempts, but the
PBP source is more granular and already supports strict-prior construction.

**Decision: no parallel Team By Game pipeline; QA only.**

### 3. Lineup Detail

Yes. Exact live 2026 11-player co-occurrence, split into pass/rush usage, is
genuinely new relative to aggregate snap counts and current depth/availability.
Historical participation can represent on-field players, but repository audits
establish that the recent participation feed is postseason-only and not a live
2026 deployment source.

The value is current role/personnel state, not the small-sample lineup efficiency
columns by themselves.

**Decision: retain as the highest-priority prospective GSIS source.**

### 4. Formation Usage

Yes, narrowly. Current #TE/#WR personnel use by down/distance and its play counts
adds live grouping information that aggregate player snap shares do not contain.
Historical personnel concepts are already derivable from PBP/participation, so
the novelty is current-season timing and personnel-conditioned granularity.

It does not provide routes, alignments, blocking assignments or player identity
inside the grouping.

**Decision: retain as the second A-class prospective source.**

### 5. Down Analysis and Play Propensity

They are overwhelmingly duplicates of PBP-derived information. Down, distance,
field position, quarter, score differential, scrambles, conversions and
run/pass choice already exist at play level. The project has also already
localized the QB/receiver opportunity miss to within-state first-down choice and
found that generic field position and score-state slicing did not explain it.

**Decision: no GSIS modeling pipeline. Down Analysis may be used only for
authoritative aggregation QA; Play Propensity is class C.**

### 6. Play Direction

It is not a new information family. nflverse PBP contains rushing location/gap
and passing location/depth inputs from which these splits can be produced. GSIS
may use slightly different official bucket definitions, which makes it a class-B
semantic comparator rather than a new research source.

It does not resolve the documented individual matchup gap because it contains no
route, alignment, separation, coverage responsibility or blocker-rusher
assignment.

**Decision: QA only; no acquisition unless a classification discrepancy is
first identified.**

### 7. Fields that can fill a documented gap without reopening closed science

Only these fields presently qualify:

- exact offensive and defensive 11-player lineup identity;
- plays and pass/rush counts by exact lineup;
- #TE/#WR personnel group by down and yards to go;
- personnel-conditioned play counts and pass/rush use.

Their legitimate initial use is a **live role/personnel-state comparison**
against current snap share, Ourlads depth/availability and existing entitlement
state. This is distinct from the closed generic down/distance, field-position,
score-state, receiver-room targets-per-play, static snap-depth, and team-level
coverage hypotheses.

The GSIS surfaces audited so far do **not** fill the remaining route/alignment,
explicit coverage assignment, separation, receiver-error, QB-delivery,
blocker-rusher assignment or pregame play-call-intent gaps.

## Anti-reinvention rules

- Do not rerun generic down/distance, field-position, quarter or score-state
  pass-tendency research under a GSIS label.
- Do not treat completed-season GSIS aggregates as historical pregame states.
- Do not rebuild PlayerForm or team game history from GSIS.
- Do not rerun the failed Receiver Room Targets-Per-Play, static snap/depth, NGS
  target model, team-level coverage or first-down choice-economics families.
- Do not infer WR-CB assignments, routes, blocking duties or play calls from
  shared lineup presence.
- Do not replace the current snap-count authority silently. Any A-class GSIS
  test must measure incremental information over that source.
- Preserve the 2026 Week-3 boundary. Current GSIS data may remain archived but
  cannot be used for tuning, diagnosis, grading or production modification
  before the established postmortem protocol allows it.

## Acquisition gate after this audit

No broad historical acquisition is justified.

If the user later authorizes the next collection step, the recommended scope is:

1. continue immutable prospective weekly snapshots only for Lineup Detail and
   Formation Usage;
2. perform a schema-only comparison of Lineup Combinations against Lineup Detail
   before retaining it;
3. omit Player Game by Game, Team By Game, Play Propensity and Play Direction
   from historical acquisition;
4. retain Down Analysis only if a bounded official-denominator QA task is
   specified;
5. preregister an incremental source test against current snap share/Ourlads
   before any A-class field reaches a model experiment.

## PR replay note

The current PR #662 replay failure is the known expired artifact
`run_35282021679`. It is not evidence of a GSIS implementation failure and does
not authorize changes to the GSIS materializer or archive contract.
