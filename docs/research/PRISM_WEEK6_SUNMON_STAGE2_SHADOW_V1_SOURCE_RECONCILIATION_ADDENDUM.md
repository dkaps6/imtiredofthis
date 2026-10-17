# PRISM Week-6 Sunday/Monday Stage-2 Shadow V1 — Source Reconciliation Addendum

Date frozen: 2026-10-16

Status: **FROZEN SOURCE-MECHANICS RECONCILIATION AFTER SOURCE-BLOCKED DRY RUN / BEFORE REPAIR CODE / NO OUTCOME USE / NO GATE CHANGE**

Parent freeze:
`docs/research/PRISM_WEEK6_SUNMON_STAGE2_SHADOW_V1_FREEZE.md`

Parent freeze commit:
`f954bf392c133c2507ff1b4bfa4d8325030abcc5`

Source-blocked diagnostic authority:
- run `38990267886`;
- exact head `7a8f4ae8b6478fe1fddb51ed75e4b22212c9d898`;
- artifact `12084487655`;
- digest `sha256:230374e7a9de57e88a9cce8d8447eb491f67bf43f770245b5c276283ae0d8e38`;
- disposition `PRISM_WEEK6_SUNMON_STAGE2_SHADOW_SOURCE_BLOCKED`.

This addendum changes **source reconciliation mechanics only**. It does not alter:
- the qualified Stage-1 probability formula;
- the qualified Stage-2 active-set law;
- any canonical entitlement;
- any opportunity volume;
- any support/gate threshold;
- any target row based on an outcome;
- the exclusion of SEA-DEN.

Week-6 target outcomes read before this addendum: **0**.

## 1. Parent-population parity for source bridge metrics

The dry run incorrectly measured two identity bridges across all PFR/NFL inactive positions while the qualified Stage-1 parent is a skill-family model over QB/RB/WR/TE.

Repair is frozen as:
- retain all-position values as diagnostics;
- calculate the gate value on the same Stage-1 skill-family population:
  QB, RB/FB/HB, WR/LWR/RWR/SWR, TE;
- do not drop a difficult skill-family row after seeing whether it matches;
- keep the original gates:
  - official-inactive -> roster identity bridge >= 95%;
  - positive-snap -> roster bridge >= 98%.

The source-blocked dry run already showed that the all-position denominator was semantically broader than the Stage-1 parent. Correcting the denominator is not a threshold change.

## 2. Fixed official-inactive source completion

The dry run's 93.04% team-game coverage was caused by article discovery/parser omissions, not permission to assume missing players active.

The following NFL.com pages are fixed pre-kickoff inactive facts and are authorized as explicit source URLs:

Week 1:
- `https://www.nfl.com/news/nfl-kickoff-game-inactives-new-england-patriots-at-seattle-seahawks`
- `https://www.nfl.com/news/australia-game-inactives-san-francisco-49ers-at-los-angeles-rams`
- `https://www.nfl.com/news/inactive-reports-sunday-week-1-2026-nfl-season`
- `https://www.nfl.com/news/week-1-monday-night-inactives-denver-broncos-at-kansas-city-chiefs`

Week 2:
- `https://www.nfl.com/news/week-2-thursday-night-inactives-detroit-lions-at-buffalo-bills`
- `https://www.nfl.com/news/nfl-week-2-inactives-players-ruled-out-sunday-14-games-2026`
- `https://www.nfl.com/news/week-2-monday-night-inactives-new-york-giants-at-los-angeles-rams`

Week 3:
- `https://www.nfl.com/news/week-3-thursday-night-inactives-atlanta-falcons-at-green-bay-packers`
- `https://www.nfl.com/news/nfl-week-3-inactives-players-ruled-out-sunday-14-games-2026`
- `https://www.nfl.com/news/week-3-monday-night-inactives-philadelphia-eagles-at-chicago-bears`

Week 4:
- `https://www.nfl.com/news/week-4-thursday-night-inactives-pittsburgh-steelers-at-cleveland-browns`
- `https://www.nfl.com/news/nfl-week-4-inactives-players-ruled-out-sunday-14-games-2026`
- `https://www.nfl.com/news/week-4-monday-night-inactives-atlanta-falcons-at-new-orleans-saints`

Week 5:
- `https://www.nfl.com/news/week-5-thursday-night-inactives-tampa-bay-buccaneers-at-dallas-cowboys`
- `https://www.nfl.com/news/nfl-week-5-inactives-players-ruled-out-sunday-2026-season`
- `https://www.nfl.com/news/week-5-monday-night-inactives-buffalo-bills-at-los-angeles-rams`

The repair may:
- force the known target week for a URL whose title omits the literal word "Week";
- broaden DOM heading recognition and use nearest preceding unique team heading for player bullets;
- deduplicate exact team/player rows.

It may **not**:
- infer an inactive from an injury designation;
- infer a missing inactive from a game result;
- assume an uncovered team has no inactives;
- lower the 99% team-game coverage gate.

Every fetched page must retain URL/hash provenance.

## 3. Week-6 target identity metadata

The preserved pregame origin artifact already contains:

`data/player_identity_roster_history.csv`

It is authorized for **identity metadata only**.

It may provide:
- stable GSIS/player ID;
- canonical/full/base name keys;
- position;
- historical team aliases.

It may not provide opportunity, efficiency, target-game result, or sportsbook information.

Two source-only aliases are frozen from official identity evidence and existing stable historical identity:

1. preserved Week-6 `Audric Estim` / key `audricestim`
   -> `Audric Estimé` / base key `audricestime`
   -> stable GSIS `00-0039373`.

   Official current identity source:
   `https://www.nfl.com/players/audric-estime/stats/logs/2026/`

2. preserved Week-6 `Drew Ogletree` / key `drewogletree`
   -> historical `Andrew Ogletree`
   -> stable GSIS `00-0037292`.

   Official current Colts roster uses Drew Ogletree while the same official player route is `/andrew-ogletree/`:
   `https://www.colts.com/team/players-roster/andrew-ogletree/`

The repository also already documented Drew Ogletree -> Andrew Ogletree as research-only official-source identity evidence in the Week-3 postmortem.

Resolution hierarchy is frozen:
1. exact preserved target player ID if present;
2. exact same-team Week-6 weekly-roster stable ID;
3. exact alias/base-key match to the preserved immutable identity-history file, requiring one unique stable GSIS ID and position-family compatibility;
4. unresolved.

No fuzzy edit-distance match.
No manually invented player ID.
Unresolved rows remain unresolved and the >=99% target identity gate decides the disposition.

## 4. Repository audit dependency

The dry-run strict repository audit failed only because the research workflow did not install PyYAML.

The workflow may add `pyyaml` to its dependency install.
No scientific code/gate changes follow from that repair.

## 5. Unchanged stopping rule

After these exact source-mechanics corrections:
- if every original source and target gate passes, the Sunday/Monday shadow may lock;
- otherwise the shadow remains source-blocked.

No additional rescue is authorized from the result.

No sportsbook upstream.
No OddsAPI call.
No Week-6 outcome.
No production mutation.
