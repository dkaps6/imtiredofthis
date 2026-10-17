# PRISM Week-6 Sunday/Monday Stage-2 Shadow V1 — Freeze

Date frozen: 2026-10-16

Status: **FROZEN BEFORE SUNDAY/MONDAY WEEK-6 OUTCOMES / RESEARCH SHADOW ONLY / NO PRODUCTION CHANGE**

Model:
**PRISM — Player Role & Individual Situation Model**

Scientific parents:
- `PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED_ALL`;
- `PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_PARTIAL`.

Historical Stage-2 authority:
- run `38957575645` — SUCCESS;
- artifact `12070774378`;
- exact head `eba5787c0b06aa5f975008965df7d023b3525ba7`;
- result `docs/research/PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_V1_RESULT.md`.

Preserved Week-6 pregame football origin:
- Full Slate run `38791377571`;
- artifact `11999923880`;
- digest `sha256:0916cd1cbec93f9a2529190b3df7c39172eee011a31eb9df429a35cdb9b41f61`;
- artifact created 2026-10-15 before the Thursday SEA-DEN kickoff.

## Prospectivity boundary

The Thursday game `2026_06_SEA_DEN` is **EXCLUDED**.

Reason:
the PRISM Stage-2 mechanism was frozen/scored on 2026-10-16, after SEA-DEN had already been played. It would be scientifically false to call that game prospective.

The shadow population is limited to Week-6 games scheduled on:
- Sunday 2026-10-18;
- Monday 2026-10-19.

No SEA-DEN result or other Week-6 target-game outcome may be read during construction.

## Eligible cells

Only historically qualified Stage-2 cells:
1. `RB_RUSHING`
2. `RB_RECEIVING`
3. `TE_RECEIVING`

Do not shadow-promote:
- QB passing — exact V1 bridge is closed;
- WR receiving — 2025 OOS source authority is blocked.

## Frozen target football state

Use only football-side files from preserved artifact `11999923880`.

Permitted target files:
- `data/football_simulation_universe.csv`;
- `data/current_player_availability.csv`;
- `data/target_entitlement_v1_trace.csv`;
- `data/te_r5p_full_slate_entitlement_trace.csv`;
- `data/schedules/schedule_2026.csv`;
- football-only rule/team state needed solely to prove denominator/lineage parity.

Forbidden as upstream PRISM inputs:
- sportsbook lines;
- prop lines;
- over/under odds;
- no-vig probabilities;
- market-implied state;
- any Week-6 result.

The preserved artifact may contain sportsbook files because it is a Full Slate artifact. The shadow script must explicitly reject/avoid those columns/files.

## Stage-1 Week-6 probability source

Reconstruct the qualified Stage-1 prior state from completed 2026 Weeks 1-5 only.

Historical input contract:
- weekly nflverse roster authority for 2026;
- scheduled team-games only;
- roster status `ACT` where the usable status field exists;
- stable GSIS/player identity;
- exact fixed pre-kickoff NFL.com official inactive identity for Weeks 1-5;
- PFR/nflverse offensive snap authority for Weeks 1-5.

Target-game Week-6 snap/result data are forbidden.

For each current Week-6 target player:
- position baseline = exact Stage-1 Jeffreys strictly-prior same-season family participation probability;
- named-player state = most recent up to 4 eligible same-team prior games;
- minimum 3 named-player games;
- candidate prior strength = 2.0;
- if named-player support is absent, use the exact Stage-1 position fallback.

No thresholding or recalibration.

## Source gates before shadow construction

Weeks 1-5 historical prior source must pass:
- scheduled team-game roster coverage >= 99%;
- stable identity >= 99%;
- official-inactive team-game coverage >= 99%;
- official-inactive -> roster identity bridge >= 95%;
- snap-source team-game coverage >= 99%;
- positive-snap -> roster identity bridge >= 98%;
- chronology violations = 0;
- no Week-6 outcomes read.

Week-6 target join must pass:
- preserved football target rows only;
- target player stable identity >= 99%;
- Stage-1 p_active attach = 100% for shadow rows after allowed fallback;
- current availability marks row eligible for opportunity;
- source-qualified canonical q is finite;
- canonical team/room mass conservation within `1e-12`.

If any gate fails, disposition is source-blocked. Do not lower gates.

## Frozen Stage-2 shadow transform

Use the exact qualified V1 active-set law.

For canonical-positive player i:
`A_i ~ Bernoulli(p_active_i)`.

Preserve canonical `q_i`.
Enumerate all active states exactly.
Condition/renormalize only existing canonical entitlement among active named players plus the existing complement.

Canonical-zero named rows remain zero.
No backup entitlement is invented.
No share coefficient is fit.

For each locked row persist:
- season/week/game/team;
- player/player_id/family;
- cell;
- `p_active`;
- whether named-player Stage-1 history or fallback was used;
- canonical `q_i`;
- candidate expected conditional share;
- baseline/candidate expected opportunity mean using the same preserved football-only team opportunity mean where available;
- current availability provenance;
- source hashes/run identities.

## Outcome-blind lock

At lock time:
- Week-6 target outcomes read = 0;
- sportsbook inputs upstream = 0;
- new OddsAPI calls = 0;
- new share models fit = 0;
- threshold searches = 0;
- production mutation = false.

The shadow is descriptive until the Sunday/Monday games settle.

## Future grade

After all included Week-6 games settle, grade under a separately frozen result step using:
- actual finite team opportunity N;
- actual named-player opportunity y;
- the already-locked `p_active` and `q_i`;
- the exact historical V1 count-CRPS and mean/room diagnostics.

Week-6 is a single prospective shadow week and cannot independently production-promote the mechanism.

No result may change this frozen construction.
