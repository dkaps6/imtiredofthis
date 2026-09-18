# OL/DL Protection Context Contract V1

**Status:** FROZEN ENGINEERING CONTRACT  
**Scope:** football-context engineering only; no predictive-lift claim; no production integration authorization.

## Purpose

Define a leakage-safe, identity-aware OL/DL and pass-protection context layer that can be joined to the canonical historical player-game/team-week base without rebuilding existing history or pretending geometric/blocking interactions are universal assignment labels.

## Scientific boundary

This contract is descriptive engineering. It does not retune production models, alter simulations, select bets, reopen failed research families, or touch Issue #535.

## Canonical grain

Two linked grains are permitted:

1. **team × game × offensive/defensive unit** for pregame personnel/continuity context;
2. **player × game** for blocker/rusher strict-prior historical context.

Play-level source rows may be used to materialize histories, but target-game outcomes must never enter a pregame snapshot.

## Identity requirements

Prefer stable nflverse/NFL IDs. Name-only joins are forbidden when a stable ID exists. Every materialized table must retain source identity, normalized team, season, week and game ID where available.

## Pregame OL personnel fields

Minimum contract:

- projected/known starting LT, LG, C, RG, RT identities when source-supported;
- returning starter count from previous team game;
- returning snap-weighted OL share;
- starts-together count for the current five-man combination using strict-prior games only;
- rolling OL snap continuity;
- position-specific replacement flags;
- center change flag;
- left/right tackle change flags;
- injury/availability evidence timestamp when used;
- source provenance and confidence class.

Unknown is distinct from unchanged.

## Pregame DL/pass-rush personnel fields

Minimum contract:

- active/expected edge and interior rusher identities when source-supported;
- returning snap-weighted front share;
- top-rusher continuity from previous team game;
- missing high-share rusher flags;
- rolling pressure/sack contribution shares using strict-prior games;
- personnel-change count;
- injury/availability evidence timestamp when used;
- source provenance and confidence class.

## BDB 2023 protection interaction history

The already-certified BDB 2023 materializer may supply blocker interaction history. Its semantic firewall is mandatory:

`pff_nflIdBlockedPlayer` identifies a blocking interaction in the source. It MUST NOT be promoted to a universal primary blocker-rusher assignment label.

Allowed strict-prior aggregates include:

- blocker interaction count;
- blocker interaction diversity;
- blocker snap-geometry stability;
- blocker time-to-engagement distribution;
- blocker/rusher repeated-interaction count when source identity supports it;
- pressure allowed / beaten indicators only when explicitly source-supported;
- rolling and expanding histories excluding the target game.

## Matchup composition layer

Pregame OL/DL context may describe composition, not guaranteed one-on-one responsibility. Allowed examples:

- expected OL continuity versus opposing front continuity;
- tackle-change exposure versus opponent edge pressure concentration;
- interior-change exposure versus opponent interior pressure concentration;
- protection continuity percentile;
- front continuity percentile;
- historical repeated-interaction availability flag.

Do not label a specific defender as assigned to a specific blocker unless an authoritative source explicitly supplies that responsibility.

## Leakage rules

For target game G:

- all rolling/expanding statistics end strictly before kickoff of G;
- injury, lineup and availability evidence must have a timestamp <= G kickoff;
- no target-game snap counts may determine a pregame starter label;
- no postgame depth chart may backfill pregame certainty;
- no target-game BDB geometry may enter a pregame feature;
- unknown personnel must remain unknown rather than inferred from target-game participation.

## Provenance classes

Every personnel fact receives one of:

- `OFFICIAL_PREGAME` — team/NFL official pregame source;
- `AUTHORITATIVE_REPORT` — timestamped credible report before kickoff;
- `STRICT_PRIOR_INFERRED` — inferred only from games before target kickoff;
- `UNKNOWN` — insufficient evidence.

Downstream code must be able to distinguish these classes.

## Historical reuse rule

This layer joins onto existing canonical history. It must not independently redownload ordinary player game logs when the required base rows can be reused or deterministically rehydrated under `HISTORICAL_DATA_REUSE_POLICY_V1`.

## QA gates

A materialization is valid only if it reports:

- row counts by season/week;
- stable-ID coverage;
- team/game join coverage;
- provenance-class counts;
- unknown-rate by field;
- duplicate-key count = 0 at published grain;
- target-game-history rows used = 0;
- target-game snap-derived starter labels = 0;
- post-kickoff evidence used = 0;
- semantic-firewall tests for interaction-vs-assignment language.

## Join contract

Published pregame tables must be joinable through canonical game/team/player keys defined by `FOOTBALL_CONTEXT_FEATURE_JOIN_CONTRACT_V1.md` and must never silently fan out the canonical base.

## Recommended implementation order

1. inventory already-available nflverse participation/roster/depth/injury fields;
2. materialize strict-prior OL/DL continuity from existing historical sources;
3. attach certified BDB 2023 blocker-history summaries where identity joins are valid;
4. add timestamped current-game availability evidence;
5. publish QA/manifest only;
6. defer any predictive test to a separately authorized scientific plan.

## Disposition

`OL_DL_PROTECTION_CONTEXT_CONTRACT_V1_FROZEN`
