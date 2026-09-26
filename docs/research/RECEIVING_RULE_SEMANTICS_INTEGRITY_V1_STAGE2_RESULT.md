# Receiving Rule Semantics Integrity V1 — Stage 2 Result

Date: 2026-09-26

Disposition: **HISTORICAL_SOURCE_UNAVAILABLE_PROSPECTIVE_ONLY**

Status: **NO PRODUCTION CHANGE / STAGE 3 NOT AUTHORIZED**

## Frozen parent

Plan:
`docs/research/RECEIVING_RULE_SEMANTICS_INTEGRITY_V1_PLAN.md`

Stage-1 structural result:
`docs/research/RECEIVING_RULE_SEMANTICS_INTEGRITY_V1_STAGE1_RESULT.md`

Stage-1 authority:
- run `36276736046`
- artifact `10917254062`
- digest `sha256:c7ff51d17adfaf0f96e18808cec5948fa314f126f5c8b0392c8bed5568fa66fa`

Stage 1 confirmed:
1. middle_open unit mismatch;
2. slot alignment loss before rule labeling.

## Stage-2 source audit

Frozen source-audit plan:
`docs/research/RECEIVING_RULE_SEMANTICS_INTEGRITY_V1_STAGE2_SOURCE_AUDIT.md`

Authoritative run:
- run `36280290109`
- head `5ce9f97f62adaea2ea57c7f8e983bbde95e13330`
- artifact `10918628297`
- digest `sha256:9ac7e1fd80adee0360c099087cc7aad3b7d801045a0e6ad5d9f57fe83df37ed2`

Contract:
- candidate variants scored: **0**
- sportsbook inputs used: **0**
- target-game outcomes read: **0**

### Historical team-context authority

Leakage-safe maintained 2025 historical source rebuilt across Weeks 1-18:

- team-week rows: **1,088**
- `middle_open_rate`: **absent**
- coverage man rate: **absent**
- coverage zone rate: **absent**

Columns available:
season, week, team, success rates, pressure rates, pace, plays, dropback/pass-attempt conversion, PROE, explosive allowed, defensive pass/rush EPA.

Therefore A1 cannot be historically replayed honestly through this authority.

Disposition:
`STAGE2_A1_HISTORICAL_SOURCE_UNAVAILABLE`

### Historical slot-alignment authority

Leakage-safe maintained 2025 pregame universe:

- player-week rows: **8,688**
- WR rows: **3,231**
- pregame source: **nflverse_weekly_roster** for all rows
- alignment-like WR rows: **0**
- SWR position rows: **0**
- SWR/SLOT role rows: **0**
- weeks with alignment-like WR evidence: **none**

The builder intentionally refuses non-week-tagged depth snapshots to avoid future leakage. No honest historical slot alignment survives in the maintained walk-forward authority.

Disposition:
`STAGE2_B1_HISTORICAL_SOURCE_UNAVAILABLE`

## Preserved 2026 evidence audit

### Week 1

Long-retention artifact `10124274040` still exists and preserves exact current-role identity. It proves pregame alignment information existed:
- LWR: 63
- SWR: 61
- RWR: 57

However that artifact does not preserve the upstream PlayerForm / TeamContext / rule-input state required to replay A0B0 versus B1 exactly. It contains role identity plus downstream sportsbook/research snapshots, not the complete receiving-rule state.

Therefore it is useful provenance evidence but **not a valid Stage-2 accuracy A/B authority**.

### Week 2

The original exact paid-origin Full Slate artifact `10523345092` has expired.

Later long-retention derivative artifacts preserve exact priced / graded outputs, but not the upstream `alignment_position` + `team_context_v3.middle_open_rate` state needed to recompute the four frozen cells without reconstruction drift.

Therefore Week 2 also cannot be used for an honest semantic-repair A/B.

## Scientific conclusion

The deterministic defects remain confirmed.

What is **not** established:
- that A1 improves actual receiving prediction accuracy;
- that B1 improves actual receiving prediction accuracy;
- that A1B1 should be promoted.

The frozen plan explicitly forbids inventing historical slot alignment or rebuilding missing pregame semantic fields from current state after outcomes.

Therefore retrospective Stage 2 fails closed on **source availability**, not on the semantic hypothesis.

## Production disposition

No Stage-3 production integration is authorized.

Do not:
- promote A1 merely because unit semantics are obviously wrong;
- promote B1 merely because SLOT labels were lost;
- tune the middle-open threshold;
- change slot multipliers;
- infer historical slot roles from target-week results;
- retrofit Week-1/Week-2 pregame state from present-day depth charts.

## Next valid evidence

The already-frozen Week-3 Stage-1 artifact is a legitimate prospective lock because all four cells were materialized before Week-3 outcomes:

- A0B0 current production
- A1B0 middle-open unit repair
- A0B1 slot-alignment repair
- A1B1 combined repair

After Week-3 games are final, actual targets/receptions/receiving yards may be attached to those **unchanged pregame projections** and graded under the original Stage-2 directional gates.

One week cannot by itself guarantee promotion; if sample/support is insufficient, continue prospective capture under the same frozen semantics.
