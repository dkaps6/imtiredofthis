# QB-Receiver Pair State V1 — Source / Support Result

**STATUS: COMPLETE / QB_RECEIVER_PAIR_STATE_SOURCE_READY**

Frozen plan:
`docs/research/QB_RECEIVER_PAIR_STATE_V1_PLAN.md`

Authority:
- branch: `research-qb-receiver-pair-state-v1`
- run: `37626968980` — **SUCCESS**
- source SHA: `fc84e78bc50e9714f5b5beb1471f618c5d0191e4`
- artifact: `11483954236`
- digest: `sha256:414c24ceb6406227c46654ddb8652b89feae85e3aa62ad2d58327e4c4430a065`
- captured: `2026-10-07T13:15:21.242612+00:00`

Final disposition:

`QB_RECEIVER_PAIR_STATE_SOURCE_READY`

## Historical source support

Regular-season public PBP is present and pair-identifiable across every audited season.

2022:
- PBP rows: 47,157
- official pass attempts: 18,069
- target events: 17,306
- distinct passers: 104
- distinct receivers: 508
- distinct passer-receiver pairs: 991
- joint passer+receiver ID coverage: 100%

2023:
- PBP rows: 47,399
- official pass attempts: 18,315
- target events: 17,483
- distinct pairs: 918
- joint ID coverage: 100%

2024:
- PBP rows: 47,274
- official pass attempts: 17,811
- target events: 17,013
- distinct pairs: 883
- joint ID coverage: 100%

2025:
- PBP rows: 46,452
- official pass attempts: 17,439
- target events: 16,609
- distinct pairs: 914
- joint ID coverage: 100%

## Live 2026 Week-5 readiness

Only Weeks 1-4 were read.

- PBP rows: 11,155
- official pass attempts: 4,216
- target events: 4,003
- max source week: 4
- distinct passers: 54
- distinct receivers: 366
- distinct passer-receiver pairs: 460
- joint pair-ID coverage: 100%

All 30 scheduled Week-5 teams have a current-primary-passer proxy derived strictly from Weeks 1-4 attempts.

Live current-primary-passer receiver pairs:
- 329 total
- 329 with >=1 pair target
- 198 with >=5 pair targets
- 136 with >=10 pair targets
- 89 with >=15 pair targets
- 259 with >=2 pair games
- 175 with >=3 pair games
- position identity coverage: 100%

Receivers who have already taken targets from multiple 2026 passers:
- 74

That last number is especially relevant: the pair object is not redundant with receiver identity. A meaningful set of receivers have already played in more than one passer environment.

## Interpretation

The public source can represent exact passer x receiver state historically and live with clean stable identities.

This is materially different from:
- receiver-only history;
- QB-only history;
- team receiving-weapon aggregates;
- WR/TE room entitlement;
- generic matchup multipliers.

The source result says nothing yet about predictive value.

The next authorized scientific question is whether strictly-prior pair-specific efficiency contains incremental information beyond the same receiver's own strictly-prior efficiency.

No production change is authorized.
No sportsbook data was used.
No Week-5 outcome was read.
