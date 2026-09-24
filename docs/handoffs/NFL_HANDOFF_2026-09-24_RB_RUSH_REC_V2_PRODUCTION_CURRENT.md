# NFL HANDOFF — 2026-09-24 — RB RUSH+RECEIVING V2 PRODUCTION CURRENT

GitHub is canonical over chat memory.

## Immediate state

A real model-quality improvement was found, independently backtested, integrated,
production-certified, and merged into the stable Full Slate production path.

PR #628 is **MERGED/CLOSED**.

- merge SHA: `e26fcedade9a94a9634f6ba74558775968573818`
- Repo CI: run `36010438820` = SUCCESS
- preserved paid Full Slate replay: run `36010438890` = SUCCESS
- replay artifact: `10812077491`
- replay digest: `sha256:074a0332127a908443c8e9d5b952592dc14d0faf6d147e8f9d63d9567ad15150`
- replay passed certified pricing, final-board quarantine, and strict repository audits.

Do not restart this work, reopen PR #628, or repeat the historical test absent a
concrete production defect or genuinely new prospective evidence.

The candidate is:

`RB_RUSH_REC_CONSERVATION_V2`

Scope:
- RB only;
- non-Week-1 only;
- rush+receiving yards only.

Formula:

`rush_rec_yards = final standalone rush_yards projection + final standalone rec_yards projection`

At draw level:

`rush_rec_draw[i] = final-mean-aligned rush_draw[i] + final-mean-aligned rec_draw[i]`

No fitted coefficient, blend, router, threshold, sportsbook field or Week-2
outcome enters the formula.

FB is explicitly excluded. The historical qualification population was RB only,
and the Week-2 integration sample was 32 RB / 0 FB.

## Why this lane was opened

The preserved Week-2 production board exposed a structural contradiction:

- 33 RBs had all three markets;
- 32/33 had final `rush_rec_yards` below
  `rush_yards + rec_yards`;
- mean gap = **-9.9631 yd**;
- max absolute gap = **29.2915 yd**.

Week 1, where P3 conservation already existed, conserved the identity essentially
exactly.

This was identified from pregame production projections before using Week-2
outcomes to define the candidate.

## Historical no-fit qualification

Plan:
`docs/research/RB_RUSH_REC_CONSERVATION_V2_MEAN_PLAN.md`

Result:
`docs/research/RB_RUSH_REC_CONSERVATION_V2_MEAN_RESULT.md`

Run:
- `36005675177`
- artifact `10809812602`
- disposition `RB_RUSH_REC_CONSERVATION_V2_MEAN_QUALIFIED`

Authority:
- exact preserved PR #549 production-order trace;
- 2,787 complete RB player-games;
- 2024-2025;
- actual rush+receiving identity max gap = 0;
- fitted parameters = 0;
- sportsbook inputs = 0;
- 2026 outcomes = 0.

Pooled:
- MAE **27.6853 -> 25.5718** (+2.1135 yd);
- RMSE **39.8437 -> 36.4030**;
- bias **+14.3388 -> +9.3891**;
- p90 AE **64.7742 -> 57.2604**;
- 30+ yard misses **841 -> 768**;
- candidate closer **59.13%**.

2024:
- MAE **28.2918 -> 25.9442**.

2025:
- MAE **27.0783 -> 25.1991**.

All frozen gates passed.

## Draw-level integration

Plan:
`docs/research/RB_RUSH_REC_CONSERVATION_V2_INTEGRATION_PLAN.md`

Initial result:
`docs/research/RB_RUSH_REC_CONSERVATION_V2_INTEGRATION_RESULT.md`

After the RB-only scope correction, integration was rerun:

- head `ab9a74a5649030aa980be2c277106592229548fd`
- run `36009227527`
- artifact `10812095478`
- digest `sha256:30fcf3205e18eadb9159e674ccdf41f7495d43ba23e15146e67065c5adb21f57`
- PASS.

Preserved Week-2 paid-origin authority:
- run `35282021679`
- artifact `10523345092`.

Exact A/B:
- V5 baseline rows: 3,178;
- V2 candidate rows: 3,178;
- applied offers: 104;
- applied player-games: 32 RB;
- non-combo model-proj gap: 0.0;
- non-combo fair-prob gap: 0.0;
- pathwise identity gap: 0.0;
- sportsbook inputs added upstream: 0;
- Week-1 rows changed: 0.

Week-2 observational confirmation, not a tuning gate:
- baseline MAE 30.3620;
- candidate MAE 28.9014;
- gain 1.4606 yd;
- baseline signed underprojection bias +20.1339;
- candidate bias +9.8670;
- candidate closer 16/32.

## Production certification

Promotion plan:
`docs/production/RB_RUSH_REC_CONSERVATION_V2_PRODUCTION_PROMOTION_PLAN.md`

Scope amendment:
`docs/production/RB_RUSH_REC_CONSERVATION_V2_SCOPE_AMENDMENT.md`

Final result:
`docs/production/RB_RUSH_REC_CONSERVATION_V2_PRODUCTION_RESULT.md`

Final stable-entrypoint certification:
- head `03bd101c0c4a846ac41002ca37d95ea61525fc12`
- run `36009823313`
- artifact `10811878970`
- digest `sha256:c88751876bd42838f312f385377be439e12e46b07e96c0c1d34dc3fd8039a04f`
- PASS.

The final certification proved:
- focused RB V2 unit tests green;
- explicit FB no-op;
- stable `run_pricing_with_full_roster_universe_v3.main` points to V6;
- protected V5 baseline green;
- V6 candidate green;
- existing downstream Full Slate certification checks green;
- exact V5-vs-V6 protections green.

The canonical Full Slate workflow now also requires the V2 production audit on
live pricing runs.

## Production code

New:
- `scripts/modeling/rb_rush_rec_conservation_v2.py`
- `scripts/run_pricing_with_full_roster_universe_v6_production.py`
- `tests/test_rb_rush_rec_conservation_v2.py`

Modified:
- `scripts/run_pricing_v2.py` — research-gated V2 seam;
- `scripts/run_pricing_with_full_roster_universe_v3.py` — stable public authority -> V6;
- `.github/workflows/full-slate.yml` — V2 lineage fail-closed gate;
- V2 integration/production certification workflows.

## Important closed/source-blocked side lanes from the same session

### Public coach/beat intent V1B automation

The manually verified source family still exists and remains conceptually
interesting, but generic unauthenticated search-engine HTML inside GitHub Actions
was not a reliable scalable retrieval transport.

Result:
`docs/migrations/QB_FIRST_DOWN_PUBLIC_INTENT_SOURCE_V1B_RESULT.md`

Disposition:
`RETRIEVAL_AUTOMATION_NOT_QUALIFIED`.

This is an operational retrieval failure, not a scientific rejection of
pregame public intent. Do not cycle through more HTML search engines. A future
restart requires a materially different search/source connector. Prospective
current-week manual/shadow capture remains scientifically distinct.

### Player practice trajectory source audit

The maintained injury feed has weekly status rows, but not dense
Wednesday/Thursday/Friday observations.

Result:
`docs/research/PLAYER_PRACTICE_TRAJECTORY_V1_SOURCE_AUDIT_RESULT.md`

Disposition:
`PRACTICE_TRAJECTORY_SOURCE_NOT_DENSE`.

2023-2025 skill-position player-weeks with practice status: 4,820.
Multi-date trajectory coverage was effectively 0%.
Do not relabel the existing weekly DNP/LIMITED flag as a new trajectory feature.

## Standing protections

- M96 retrospective RB rushing router family remains CLOSED.
- This V2 is not an M96 reopening; it fixes an exact combined-market construction
  identity using independently qualified component projections.
- QB pass yards remains frozen/prospective.
- Week-1 P3/R22/R26 remains unchanged.
- No global SD rescale.
- No sportsbook lines upstream.
- No paid OddsAPI pull without explicit user approval.
- TE Width V2 provider-drift work remains a separate unresolved lane; do not
  confuse its source archaeology with this completed RB combined-market result.

## Immediate next operational action

Production promotion is complete on `main`.

Return to actual model-improvement/new-information research rather than
scoreboard work. Preserve these stopping rules:

- do not reopen V2 without a concrete defect or genuinely new evidence;
- do not reopen M96 retrospective RB routing;
- keep QB pass yards prospective;
- keep the public-intent and practice-trajectory source lanes parked unless a
  materially new data source becomes available;
- TE Width V2 provider-drift archaeology remains separate and should stay
  time-boxed rather than becoming another infrastructure loop.
