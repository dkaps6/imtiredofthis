# Week 1 Live Full Slate Preserved-Artifact Replay — 2026-09-11

## Scope

This report closes the preserved-artifact/player-identity audit gate from
`NFL_HANDOFF_2026-09-11_FULL_SLATE_LIVE_REPAIR_CURRENT.md`.

- repository: `dkaps6/imtiredofthis`
- replay starting `main`: `27986f563a0c66b7f697c3a8d26194a8ddefe971`
- original failed paid Full Slate run: `34602328548`
- failed job: `103272747139`
- preserved artifact ID: `10265557165`
- preserved archive SHA256: `08727a37cf174a5be9767855a8a6aa3854c0f56cc5e650099ec3dced127e12b1`
- model science changed: **NO**
- new paid OddsAPI calls used for this replay: **0**

## Preserved roster / event-scope replay

The repaired event-scoped roster invariant passes on the preserved artifact.

- current `roles_ourlads.csv`: 406 rows / 28 teams
- active sportsbook player-event universe: 14 events / 28 teams
- missing current-event roster teams: **0**
- intentionally absent already-played teams: `LAR`, `NE`, `SEA`, `SF`
- modeled core player rows checked: 393
- modeled core unique players: 183
- modeled core unresolved team/opponent rows: **0**
- modeled core PlayerForm join misses: **0**

The fatal all-32 assertion from the original incident is therefore gone without
weakening the required current-event coverage gate.

## Additional mechanical event-scope defect found by replay

`run_live_odds_gate.py` previously admitted provider events using only the
unordered team pair. The preserved game-odds artifact contained four later-season
rematches with the same opponents:

| Provider event | Matchup | Commence time |
|---|---|---|
| `f2a036d5edf483a277880014861e78ad` | DEN-KC | 2026-11-01T21:25:00Z |
| `b9c60db63b1a19ad319f24b33c0defe8` | WAS-PHI | 2026-11-02T01:20:00Z |
| `5d8d85473b87f772a979ecf294acab94` | GB-MIN | 2026-11-15T18:00:00Z |
| `c8945d476449633912cf51a346ac98ca` | DAL-NYG | 2027-01-03T18:00:00Z |

Those four events also produced 32 bookmaker-missing compact placeholder rows.
They did not change the upstream football model, but they violate the active-week
sportsbook boundary.

The repair is mechanical: require both the canonical Week-1 team pair and a
provider `commence_time` within 36 hours of the authoritative
`team_week_map.kickoff_utc` anchor. The preserved legitimate Week-1 events are
17.0-24.33 hours from their date-level schedule anchors; the leaked rematches are
at least 1,173 hours away. Matching active-pair rows with an invalid commence time
fail closed.

Preserved replay result after the rule: **14 allowed current events; 4 rematches
rejected**.

## Player-identity classification

Before the one verified alias repair, the replay finds 84 unresolved actual
sportsbook labels, all in `player_anytime_td`.

- modeled core markets unresolved: **0**
- all other non-anytime markets unresolved: **0**
- synthetic defense entities: **56**
- real player names: **28**

The 56 synthetic labels are the expected `TEAM D/ST` / `TEAM Defense` non-player
entities. `materialize_live_props_for_model_v1.py` already quarantines these as
`NON_PLAYER_PROP_ENTITY`; they are not player-identity defects.

Of the 28 player names:

- 15 have stable historical identity on one of the event teams but are absent from
  the current authoritative roster/slate;
- 3 have a stable historical identity only on another team and are absent from the
  current authoritative roster/slate;
- 9 have no exact stable historical-name match and are absent from the current
  authoritative roster/slate;
- 1 is a positively verified current-name alias: `Zonovan Knight` -> `Bam Knight`.

The 27 current-roster-absent names remain unassigned. They are non-core anytime-TD
offers and are correctly quarantined as `UNMODELED_NONCORE_PLAYER_IDENTITY`. No
historical team is used to force a current assignment.

### Exhaustive 28-player classification

| player | event_pair | category | evidence |
|:--|:--|:--|:--|
| Ameer Abdullah | JAX-CLE | noncore_absent_current_historical_other_team | registry last IND 2025 W18 id=00-0032104 |
| Audric Estime | DET-NO | noncore_absent_current_no_stable_history_match | no current roster/slate match and no exact stable historical-name match |
| Ben Yurosek | MIN-GB | noncore_absent_current_historical_event_team | historical MIN 2025 W18 id=00-0040509 |
| Brevin Jordan | HOU-BUF | noncore_absent_current_historical_event_team | historical HOU 2025 W20 id=00-0036556 |
| Cade Stover | HOU-BUF | noncore_absent_current_historical_event_team | historical HOU 2025 W20 id=00-0039359 |
| Cam Brown | NYG-DAL | noncore_absent_current_no_stable_history_match | no current roster/slate match and no exact stable historical-name match |
| Colson Yankoff | PHI-WAS | noncore_absent_current_historical_event_team | historical WAS 2025 W18 id=00-0039686 |
| Dallen Bentley | KC-DEN | noncore_absent_current_no_stable_history_match | no current roster/slate match and no exact stable historical-name match |
| David Martin-Robinson | TEN-NYJ | noncore_absent_current_historical_event_team | historical TEN 2025 W18 id=00-0039648 |
| E.J. Jenkins | PHI-WAS | noncore_absent_current_historical_event_team | historical PHI 2025 W19 id=00-0038498 |
| J. Michael Sturdivant | MIN-GB | noncore_absent_current_no_stable_history_match | no current roster/slate match and no exact stable historical-name match |
| Ja'Tavion Sanders | CAR-CHI | noncore_absent_current_historical_event_team | historical CAR 2025 W19 id=00-0039356 |
| Jack Endries | CIN-TB | noncore_absent_current_no_stable_history_match | no current roster/slate match and no exact stable historical-name match |
| Jackson Meeks | DET-NO | noncore_absent_current_historical_event_team | historical DET 2025 W18 id=00-0040390 |
| Jake Briningstool | KC-DEN | noncore_absent_current_historical_event_team | historical KC 2025 W18 id=00-0040081 |
| Jelani Woods | TEN-NYJ | noncore_absent_current_historical_event_team | historical NYJ 2025 W18 id=00-0037755 |
| Josh Cuevas | IND-BAL | noncore_absent_current_no_stable_history_match | no current roster/slate match and no exact stable historical-name match |
| Julius Chestnut | TEN-NYJ | noncore_absent_current_historical_event_team | historical TEN 2025 W18 id=00-0037594 |
| Justin Joly | LV-MIA | noncore_absent_current_no_stable_history_match | no current roster/slate match and no exact stable historical-name match |
| Keleki Latu | HOU-BUF | noncore_absent_current_historical_event_team | historical BUF 2025 W20 id=00-0040363 |
| Kene Nwangwu | TEN-NYJ | noncore_absent_current_historical_event_team | historical NYJ 2025 W18 id=00-0036842 |
| Mark Redman | MIN-GB | noncore_absent_current_historical_other_team | registry last LAR 2025 W21 id=00-0040598 |
| Matthew Hibner | IND-BAL | noncore_absent_current_no_stable_history_match | no current roster/slate match and no exact stable historical-name match |
| Najee Harris | NYG-DAL | noncore_absent_current_historical_other_team | registry last LAC 2025 W19 id=00-0036893 |
| Tanner Koziol | JAX-CLE | noncore_absent_current_no_stable_history_match | no current roster/slate match and no exact stable historical-name match |
| Thomas Fidone | NYG-DAL | noncore_absent_current_historical_event_team | historical NYG 2025 W18 id=00-0040225 |
| Tyler Badie | KC-DEN | noncore_absent_current_historical_event_team | historical DEN 2025 W21 id=00-0037085 |
| Zonovan Knight | LAC-ARI | verified_alias | current ARI Bam Knight and historical Zonovan Knight share GSIS 00-0037157 |

## Verified alias repair

The preserved artifact proves:

- sportsbook name: `Zonovan Knight`
- current Ourlads/current availability/current slate name: `Bam Knight`
- current team: `ARI`
- stable GSIS identity: `00-0037157`
- historical roster history contains both `Zonovan Knight` and `Bam Knight` under
  that same GSIS identity.

This is a deterministic person-identity alias, not a fuzzy guess. The verified
manual name override is therefore extended with `Zonovan Knight -> Bam Knight`.
No other unresolved player is auto-attached.

## Requested identity buckets

- unmatched sportsbook names: 84 before alias = 56 synthetic defenses + 28 real
  anytime-TD player names; 83 after alias = 56 synthetic defenses + 27 quarantined
  non-core player names.
- unresolved canonical names: **0** in the modeled core markets; the 27 remaining
  real names retain their source/canonical text but intentionally have no current
  team assignment.
- Tier-1 / PlayerForm unmatched: **0** for the 393 modeled core rows in the replay.
  The original failed run never reached the downstream metrics debug-report stage;
  direct replay of the current suffix-insensitive team/name join finds zero misses.
- role-chain depth/fallback failures: **0 observed** in the preserved production
  roster. All 406 rows have nonblank role, position, position group, depth index,
  raw depth role, availability authority, and final availability state; depth
  indices are 1-3.
- alias/name-normalization defects: **1 positively verified**, `Zonovan Knight`
  -> `Bam Knight`; repaired surgically.
- genuine current-roster defects: **0 proven**. The 27 other names are absent from
  both the current authoritative roster and current identity slate; they remain
  quarantined rather than being guessed onto a team.

## Gate state after this replay

The preserved-artifact replay establishes that the original event-scoped roster
repair is correct and that no modeled core player identity failure remains.
Before a paid run, the rematch-scope regression and the single verified alias
repair must pass targeted tests, repository CI, the broader regression suite, and
a no-paid-odds Full Slate run. Only after those gates are green should the one
controlled paid Full Slate validation run be dispatched.
