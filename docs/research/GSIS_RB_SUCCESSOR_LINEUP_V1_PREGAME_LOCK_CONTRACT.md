# GSIS RB Successor Lineup V1 — Pregame Three-Arm Lock Contract

**STATUS: FROZEN BEFORE THE FIRST ELIGIBLE FUTURE LOCK.**

Parent science:
`docs/research/GSIS_RB_SUCCESSOR_LINEUP_V1_PLAN.md`

## Purpose

Operationalize the already-frozen prospective experiment so the first eligible
future RB/FB vacancy can be locked before kickoff with no postgame reconstruction.

This contract does not change the candidate formula.

## Required authorities

A scientific lock requires all of the following:

1. an immutable pre-kickoff private GSIS Lineup Detail snapshot;
2. its SHA-256 and capture timestamps;
3. the current canonical availability ledger identifying definitive-unavailable
   RB/FBs;
4. the frozen RB Vacancy Opportunity V1 no-outcome state using the exact
   pregame availability/roles/logs/snaps inputs;
5. an exact no-live-odds Full Slate football source checkout/artifact from the
   same pregame state;
6. official target game identity and kickoff timestamp.

All source SHAs / run IDs / artifact IDs / digests are frozen into the lock
manifest.

## Snapshot timing

For each target team-game, the relevant **team offense Lineup Detail capture**
must be strictly earlier than that target game's kickoff UTC.

A snapshot captured after kickoff is invalid even if it contains only season
aggregates.

The existing Week-3 private snapshot may be used only for parser/schema tests.
It can never generate a retrospective Week-3 scientific lock.

## Frozen arms

For every eligible target event, create exactly three football arms:

1. `BASELINE`
   - current production football state unchanged.

2. `VACANCY_V1_SNAP`
   - add the exact already-frozen Vacancy V1 snap-weight
     `transfer_rush_share` to surviving successor `rules_rush_share`.

3. `GSIS_LINEUP_V1`
   - use the same exact frozen vacated share;
   - replace only the successor weights with the frozen GSIS absent-lineup
     exposure weights;
   - add `gsis_transfer_rush_share` to successor `rules_rush_share`.

No other football field may differ across arms before simulation.

## Production semantics

The lock runs from the exact production-source checkout associated with the
pregame Full Slate source artifact.

For Week > 1 RB/FB rushing markets:
- do not invoke the Week-1-only RB-P3 mean override;
- simulate each arm with the canonical production simulator and the same
  iteration count / seed;
- compute the same calibrated generic MC + ML + State ensemble used by current
  production;
- count representation may be aligned downstream exactly as production does,
  but the locked scientific point mean is the final calibrated football mean.

The candidate never reads sportsbook line, price, market probability, edge, or
bet signal.

## Locked cohort

Freeze every active production-eligible RB/FB on every qualifying vacancy team,
not only direct transfer recipients.

This is required because finite team rushing allocation / normalization can move
nonrecipient teammates.

## Locked outputs

### Private lock

May contain player identity and all three arms:
- target season/week;
- event/game id;
- team/opponent;
- kickoff UTC;
- player / stable player key;
- direct recipient flags;
- unavailable set;
- frozen vacated share;
- snap successor weight / transfer;
- GSIS successor weight / transfer;
- baseline / snap / GSIS `rules_rush_share`;
- frozen YPC;
- baseline / snap / GSIS MC means;
- frozen ML / State means;
- baseline / snap / GSIS final ensemble means;
- simulation metadata;
- exact GSIS snapshot hash / team capture timestamp.

This private artifact must not be committed to the public repository.

### Public-safe manifest

May contain only:
- hashes;
- run/artifact/source SHAs;
- target season/week;
- number of events / teams / locked players;
- arm names;
- aggregate conservation gaps;
- aggregate timing checks;
- count of abstained events and reasons;
- booleans proving no sportsbook / target outcome usage.

No raw GSIS cells or player-level private candidate rows.

## Hard invariants

A lock fails closed if any of the following occurs:

- relevant GSIS Lineup Detail capture timestamp >= kickoff;
- GSIS snapshot SHA does not match the declared private authority;
- source target season/week disagrees with the target event;
- unavailable identities overlap successors;
- ambiguous / fuzzy player identity mapping is required;
- snap and GSIS arms do not use the same frozen vacated share;
- successor weights are negative or do not sum to 1;
- transfer mass does not conserve within 1e-10;
- a candidate changes any field other than `rules_rush_share`;
- YPC / efficiency differs across arms;
- ML / State inputs differ across arms;
- sportsbook or target outcome fields appear;
- exact active RB/FB cohort coverage differs across arms;
- any output is non-finite.

## No-event / no-exposure states

A pregame execution may legitimately produce:
- `NO_QUALIFYING_VACANCY_EVENT`;
- `NO_GSIS_SUCCESSOR_EXPOSURE`;
- `SOURCE_TIMING_INVALID`.

These are valid fail-closed observations. Do not fabricate fallback weights.

## Grading boundary

Outcomes are attached only after games are final and only to an immutable
pregame lock.

The parent prospective support and pass gates remain unchanged:
- >=6 future weeks;
- >=10 vacancy team-games;
- >=20 successor player-games;
- paired attempt-AE improvement vs VACANCY_V1_SNAP;
- 10,000 game-cluster bootstrap, seed 20261001;
- all secondary guards from the parent plan.

No post-outcome formula, blend, threshold, or cohort rescue.
