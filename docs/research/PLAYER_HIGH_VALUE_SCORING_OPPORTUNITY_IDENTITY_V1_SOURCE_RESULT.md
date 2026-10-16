# Player High-Value Scoring Opportunity Identity V1 — Source Audit Result

Date: 2026-10-15

Status: **SOURCE READY / MECHANISM TEST AUTHORIZED / RESEARCH ONLY**

Canonical run:
- run `38848080453` — SUCCESS
- exact head `1448fcce54f803d9737a7b3f7d9ff451b5189add`
- artifact `12023852453`
- digest `sha256:1d8a22bf6240b4d4443ab7432d3224b63eaef97439639e7b47f536aed174adda`

Disposition:
`PLAYER_HIGH_VALUE_SCORING_OPPORTUNITY_SOURCE_READY`

Optional end-zone target source:
**READY**.

## What qualified

The source contract passed across every frozen regular season from 2022 through
2025.

The canonical nflverse/nflfastR PBP path provides:
- stable rusher and receiver identity;
- team/week identity;
- `yardline_100`;
- rushing and target events;
- inside-20, inside-10 and inside-5 rushing topology;
- red-zone receiving opportunity;
- sufficiently complete `air_yards` for optional end-zone-target derivation.

Across all four seasons:
- week/team coverage passed the frozen 99% gate;
- yardline coverage passed the frozen 99% gate;
- rusher identity passed 99%;
- receiver identity passed 99%;
- position-family bridge passed 98%;
- every regular-season week contained red-zone rushing and receiving events;
- inside-10 and inside-5 rushing events were present;
- duplicate player-play identity contradictions were zero;
- optional `air_yards` coverage exceeded 95%.

Selected support:
- 2022: 47,157 PBP rows; 14,800 rush rows; 17,391 target rows;
  1,251 inside-10 rushes; 750 inside-5 rushes.
- 2023: 47,399 PBP rows; 14,651 rush rows; 17,558 target rows;
  1,338 inside-10 rushes; 767 inside-5 rushes.
- 2024: 47,274 PBP rows; 14,723 rush rows; 17,103 target rows;
  1,380 inside-10 rushes; 817 inside-5 rushes.
- 2025: 46,452 PBP rows; 14,633 rush rows; 16,699 target rows;
  1,361 inside-10 rushes; 806 inside-5 rushes.

Target-row air-yards coverage was approximately 99.46%-99.57% in every season.

## Scientific meaning

The repository has enough historical information to build a named player's
**high-value scoring-opportunity identity** without using sportsbook prices or
post-hoc availability inference.

This does not prove predictiveness.

It authorizes the already-frozen next experiment:
`PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_V1`.

That experiment asks whether the named player's strictly-prior scoring-area
opportunity share forecasts his next-game scoring-area share better than his
ordinary overall opportunity share.

This is intentionally upstream of touchdown conversion.

## Boundaries

- sportsbook inputs: 0
- OddsAPI calls: 0
- 2026 outcomes: 0
- predictive models fit: 0
- projection changes: 0
- production changes: 0

The current generic ATD model remains uncertified.

No ATD promotion is authorized from this source audit.
