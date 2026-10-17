# PRISM Week-6 Prior-State Source Recovery V1 — Freeze

Date frozen: 2026-10-16

Status: **FROZEN SOURCE-REPAIR CONTRACT / NO MODEL RETUNE / NO WEEK-6 OUTCOMES**

Parent:
- `PRISM_WEEK6_SUNMON_STAGE2_SHADOW_V1`
- first complete source-gated build run `38990267886`
- exact head `7a8f4ae8b6478fe1fddb51ed75e4b22212c9d898`
- artifact `12084487655`
- digest `sha256:230374e7a9de57e88a9cce8d8447eb491f67bf43f770245b5c276283ae0d8e38`
- disposition `PRISM_WEEK6_SUNMON_STAGE2_SHADOW_SOURCE_BLOCKED`.

Observed source blockers before this repair contract:
- completed Weeks 1-5 official-inactive team-game coverage: **147 / 158 = 93.03797%**, frozen gate >=99%;
- official-inactive -> weekly-roster identity bridge: **94.42623%**, frozen gate >=95%;
- reported positive-snap -> roster bridge: **66.50602%**, frozen gate >=98%;
- Week-6 target stable identity gate also failed.

No Week-6 target-game outcome was read to discover or define any blocker.

## Gates remain unchanged

Do NOT lower:
- roster team-game coverage >=99%;
- official-inactive team-game coverage >=99%;
- official-inactive -> roster identity bridge >=95%;
- stable identity >=99%;
- positive-snap -> roster identity bridge >=98%;
- target stable identity >=99%;
- Stage-1 p_active attach =100%;
- probability/conservation tolerances.

## Allowed repair 1 — official inactive source completeness

Only the NFL's official game-day inactive reports may fill missing completed Weeks 1-5 inactive team-games.

Allowed:
- recover standalone Kickoff, international, Thursday or Monday official-inactive articles omitted by the generic sitemap/title parser;
- use an explicit target week for an official inactive article whose title omits the literal "Week N", when the game identity and schedule establish the week;
- loosen HTML heading recognition only enough to recognize the same NFL team section when records/year/image text surround the team heading;
- deduplicate by season/week/team/player identity.

Forbidden:
- infer an inactive from postgame snaps;
- substitute injury-report OUT/QUESTIONABLE status for the official inactive fact;
- infer from box score/stat absence;
- use sportsbook information.

Known missing team-games from the parent source audit are diagnostic only; source recovery must parse the official reports, not manually type player outcomes.

## Allowed repair 2 — inactive identity bridge

Identity resolution may use only the same-week, same-team official weekly roster.

Order:
1. exact normalized name;
2. deterministic unique high-similarity same-team/week roster name if exact matching fails.

A fuzzy recovery is allowed only when:
- exactly one candidate is clearly best;
- similarity is high enough to reject weak nickname guesses;
- no target-game performance field is consulted.

Resolved roster identity is used only to connect the official inactive fact to the weekly roster identity.

## Allowed repair 3 — positive-snap bridge denominator correction

The parent 66.5% diagnostic mixed all positive offensive snap rows, including offensive linemen, with a roster universe intentionally restricted to QB/RB/WR/TE.

The repaired diagnostic must evaluate the frozen Stage-1 bridge on **positive offensive snap rows whose PFR snap-count position belongs to QB/RB/WR/TE families**.

Do not remove a difficult skill player merely because it fails the bridge.
Do not include OL or other out-of-scope position rows in the denominator.

This is a denominator correction, not a relaxed gate.

## Allowed repair 4 — Week-6 stable player identity

For a preserved Week-6 target row whose football artifact has a temporary/name-only identity:
- first use exact same-team Week-6 weekly-roster identity;
- then a deterministic unique high-similarity same-team Week-6 weekly-roster identity.

No postgame identity evidence.
No Week-6 result.
No manual one-off assignment without source provenance.

If target identity remains below 99%, the shadow stays source-blocked.

## Prospectivity remains frozen

- SEA-DEN remains excluded because it was played before the Stage-2 mechanism freeze.
- Only Sunday/Monday Week-6 games may be prospectively locked.
- Week-6 target outcomes read during repair/lock: 0.
- Sportsbook input upstream: 0.
- New OddsAPI calls: 0.
- New share models fit: 0.
- Threshold search: 0.
- Production mutation: false.

The purpose of V1 source recovery is to determine whether the already-frozen PRISM shadow can be supported by honest pregame football data. It may still fail closed.
