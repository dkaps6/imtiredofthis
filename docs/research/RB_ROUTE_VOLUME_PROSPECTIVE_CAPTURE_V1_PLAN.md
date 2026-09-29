# RB Route-Volume Prospective Capture V1 — Frozen Plan

Date frozen: 2026-09-29  
Status: **FROZEN BEFORE WEEK-4+ OUTCOME USE — RESEARCH SHADOW ONLY**  
Branch: `research-week3-postmortem-execution-v1`

Parent source-readiness result:
`docs/research/RB_ROUTE_VOLUME_SOURCE_READINESS_V1.md`

Parent disposition:
`LIVE_RB_ROUTE_VOLUME_CONFIRMED_HISTORICAL_WEEKLY_PARITY_NOT_CLEARED`

## Purpose

Build a leakage-safe, no-cost, immutable 2026 route-volume history for RB/HB/FB
players beginning with the Week-3 -> Week-4 boundary, without purchasing
historical custom splits and without fitting any model to Weeks 1-3 outcomes.

This is data capture / source validation only.

## Sources

### Primary weekly semantic source: HeatRadar

Capture the public route table when available for each completed NFL week:
- player;
- team;
- position;
- week;
- Routes;
- source-defined Route %;
- source timestamp/capture time;
- source URL/page identity.

The site's own definition of Route % must be preserved. Do not silently replace
its denominator with snap share, dropbacks, or another provider's definition.

### Independent cumulative cross-check: StatRankings

Capture the free current-season cumulative Routes Run table:
- player;
- team;
- position;
- cumulative Routes Run;
- displayed update timestamp;
- capture time.

Do not purchase or invoke premium custom splits under this plan.

## Snapshot timing

A capture may be used as prior information for target Week W only when:
1. it was obtained after the player's/team's Week W-1 game was final;
2. it was obtained before that player's Week W kickoff;
3. the source itself had updated through Week W-1;
4. no Week W result was used to repair or infer the snapshot.

If a source is not updated in time, mark that source/week `NOT_FRESH`; do not
backfill after kickoff and pretend it was pregame.

## Weekly route derivation from cumulative snapshots

For StatRankings only, a week-over-week cumulative difference may be emitted
prospectively:

`derived_week_routes = cumulative_routes_t - cumulative_routes_t_minus_1`

Only when all of the following hold:
- same canonical player identity;
- same season;
- team identity is consistent or an explicit trade/team-change state is
  preserved;
- both snapshots are immutable and predate the next target game;
- cumulative totals are finite;
- delta is nonnegative;
- no provider correction flag / impossible team total is detected.

Any negative delta, identity ambiguity, team-change ambiguity, or missing prior
snapshot => `DERIVATION_FAIL_CLOSED`.

Never coerce missing to zero.

## Cross-source parity

Where HeatRadar direct weekly Routes and StatRankings derived weekly Routes both
exist for the same canonical player-week:
- compare exact route count;
- compare presence/missingness;
- compare team/position identity;
- report absolute difference and exact-match rate.

This comparison is source validation only. No realized receiving/rushing outcome
enters source selection.

## Canonical identity

Prefer existing repo player identity / `player_clean_key` and GSIS mappings.

Ambiguous names must be quarantined rather than fuzzy-matched into a row.
Trades and team changes remain explicit state, not silent aliases.

## Required immutable capture metadata

Each capture must record:
- season / through-week;
- captured UTC timestamp;
- source;
- source displayed last-updated text if present;
- page identity;
- row count;
- canonicalization version;
- content hash or deterministic normalized-table hash when the raw provider
  table cannot be publicly committed;
- source-empty / not-fresh / parse-failure states.

Never overwrite an earlier capture.

## Source-quality checkpoints

After each weekly capture report:
- source freshness;
- RB/HB/FB row count;
- teams represented;
- unresolved identities;
- cross-source matched rows;
- exact route-count agreement rate;
- median / p90 absolute route-count gap;
- provider-only row counts;
- cumulative-delta failures.

No predictive metrics.

## Minimum source-parity gate before predictive science

Do not freeze a route-volume predictive candidate until at least:
- 4 distinct prospectively captured weekly transitions;
- >=200 cross-source RB/HB/FB player-week matches;
- >=95% exact route-count agreement OR a documented deterministic definitional
  mapping that explains >=99% of absolute discrepancies;
- >=30 NFL teams represented;
- unresolved identity rate <=1%;
- zero silent missing-to-zero conversions.

If the two sources remain definitionally different, preserve both and do not
blend them.

## Predictive science boundary

Even after source parity clears, this capture plan does not itself authorize a
model change.

A later frozen study must separately define whether strict-prior route volume
adds information beyond:
- current snap share;
- RB rush share / PlayerForm opportunity state;
- availability/vacancy state;
- existing receiving target/reception opportunity;
- current production mean.

No sportsbook input may enter that football feature.

## No-outcome / no-retuning rules

- Weeks 1-3 outcomes cannot define thresholds or source choice.
- Week-4+ outcome data cannot alter this capture schema.
- No provider may be selected because its route values correlate better with
  realized yards.
- No route-rate threshold is defined here.
- No target-game route data may enter its own projection.
- No paid source acquisition is authorized.
