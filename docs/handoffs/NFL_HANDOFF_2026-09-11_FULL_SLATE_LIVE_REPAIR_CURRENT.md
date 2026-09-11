# NFL Production Handoff — 2026-09-11 — Week 1 Full Slate Live Repair

## Active priority

Finish the Week-1 live Full Slate production incident. The QB/WR shared-opportunity / public-pregame-intent V1B science lane is temporarily parked until production is repaired.

This is a mechanical/data-production incident. Do **not** redesign or retune model science while repairing it.

## Frozen production science

Production authorities remain unchanged:

- QB mean: M89/M90 / `QB_PASS_SYNTHESIS_V1`
- QB distribution: `C2_QB_MEAN_NEUTRAL_DISTRIBUTION_V1`
- WR: M38 WR1 hierarchy + `WR_R15_PRODUCTION_MODEL_V1`
- TE: `TE_R5P_PRODUCTION_MODEL_V1`
- RB rushing: `RB_P3_SYNTHESIS_V1` for the qualified Week-1 route
- RB receptions/opportunity: R26
- RB receiving-yard tail/distribution: R22 using frozen R19 assets
- M108 scientific regression baseline: previously 26/26 PASS

Do not change production science while fixing this incident.

## Original failed live run

- Original main SHA: `0c6fcefea0d38d0975dea7fd04edd43216b2cbab`
- Failed Full Slate run: `34602328548`
- Failed job: `103272747139`
- Football construction Steps 1-24 passed.
- Failure occurred at Step 25: `Fetch and gate player props and game odds after football eligibility`.
- Preserved GitHub artifact ID: `10265557165`
- Artifact name: `run_34602328548`
- Artifact SHA256: `08727a37cf174a5be9767855a8a6aa3854c0f56cc5e650099ec3dced127e12b1`
- Prior runtime download path: `/mnt/data/run_34602328548.zip`

Important: repeated calls to the GitHub decoded job-log endpoint for job `103272747139` have not surfaced usable decoded log text in the current chat/tool response. Do **not** claim an exact historical exception unless it is actually retrieved and seen.

## Roster-universe finding

The preserved live `roles_ourlads.csv` contained 28 teams. That is legitimate for the remaining live slate.

The four absent teams were:

- LAR
- NE
- SEA
- SF

Those teams had already played and were not required in the remaining-slate sportsbook/event universe.

Correct invariant:

> Every team represented by the current sportsbook/event universe must exist in the current roster snapshot. Teams outside the active event universe may legitimately be absent. A missing current-event team must still fail closed.

## Mechanical root cause repaired

Production repair commit:

- `ea0af251f943fe6a540e82e36c38ab5a6295fb7d`
- `Fix live roster validation for remaining-slate event scope`

Repair behavior:

- `_build_roster_index()` now accepts `required_teams`.
- Live repair derives the required team universe from current sportsbook events.
- Missing current-event teams still fail closed.
- Absent non-event teams are allowed.
- Callers without an event scope retain the historical all-32 validation behavior.
- Invalid required team codes fail closed.
- No model science, pricing semantics, projection logic, or promotion rules were changed.

Regression cases added:

A. 28-team roster + every current-event team present -> PASS  
B. One current-event team missing -> FAIL CLOSED  
C. Full 32-team roster -> PASS  
D. Missing non-event teams -> ignored

## Verification completed

### Repo CI

- Run: `34621082108`
- Job: `103334928263`
- SHA: `ea0af251f943fe6a540e82e36c38ab5a6295fb7d`
- Result: SUCCESS
- Compile production modules: PASS
- Static repo audit: PASS
- Unit tests: PASS

### Non-paid Full Slate push run

- Run: `34621082187`
- Job: `103334928676`
- SHA: `ea0af251f943fe6a540e82e36c38ab5a6295fb7d`
- Overall result: SUCCESS
- Steps 1-24: PASS
- Steps 25-31: SKIPPED because live sportsbook mode was not enabled

This confirms the football pipeline remained healthy after the roster repair. It is **not** the preserved-artifact replay and it is **not** the paid live validation run.

## Current status

`roster bug fixed -> regression CI green -> football pipeline green -> preserved-artifact identity replay pending -> paid live run not triggered`

## Separate player-identity issue still pending

A large number of player names appeared uncategorized/unmatched in the failed live run. Do not assume those rows were caused by the 28-team roster issue.

After replaying the preserved artifact against the repaired code, classify every remaining identity failure into:

- unmatched sportsbook names;
- unresolved canonical names;
- Tier-1 unmatched report;
- role-chain depth/fallback failures;
- alias/name-normalization defects;
- genuine roster defects.

Add aliases only when positively confirmed. Do not guess.

Historical precedent from M95L was surgical and verified only:

- Jeff Wilson -> Jeffery Wilson
- Kenny -> Kenneth Gainwell
- Chris Rodriguez Jr. -> Chris Rodriguez
- Kenneth Walker III -> Kenneth Walker
- Chris Brooks -> Christopher Brooks

The same philosophy applies here.

## Exact next execution order

1. Retrieve GitHub artifact `10265557165` / `run_34602328548` and preserve the original archive unchanged.
2. Replay the exact failed artifact against the repaired production code.
3. Verify the false all-32 roster assertion is gone and the event-scoped roster gate passes.
4. Prove the active event-team universe is fully covered by the roster snapshot.
5. Enumerate and classify every remaining player-identity failure.
6. Apply only positively verified, surgical identity repairs.
7. Run targeted tests plus the broader regression suite; M108 must remain 26/26 PASS or better.
8. Review any identity-only diff carefully and merge only mechanical/data repairs.
9. Only then execute exactly one paid live Full Slate run.
10. Validate production output: nonempty edges, real sportsbook lines/prices, selected-book semantics, duplicate collapse, provenance, status codes, and no unresolved current-event team/player identity failures.
11. Update the decision log/ledger with exact commit/run/artifact lineage.

## Guardrails

Do **not** alter:

- M107 thresholds;
- model weights;
- PlayerForm science;
- role-chain science;
- pricing semantics;
- promotion rules;
- sportsbook comparison logic;
- projections;
- any other scientific behavior.

Architecture remains football-first:

`GAME STATE -> TEAM OPPORTUNITY -> POSITION/ROOM POOL -> PLAYER ENTITLEMENT -> PLAYER EFFICIENCY -> JOINT MONTE CARLO -> PLAYER PROJECTIONS/DISTRIBUTIONS -> FAIR PROBABILITIES -> SPORTSBOOK COMPARISON`

Vegas cannot teach upstream football projections which direction to move.

## Paid-run rule

Do **not** trigger another paid OddsAPI Full Slate merely to discover the next bug. The preserved artifact must be replayed and the identity audit completed first.

## Parked science lane

The separate QB/WR shared-opportunity / first-down pass-propensity / public pregame-intent V1B work remains preserved at:

- `docs/handoffs/NFL_HANDOFF_2026-09-11_QB_WR_SHARED_OPPORTUNITY_CURRENT.md`

Resume that lane only after the live production incident is closed. It has made no production changes.

## Resume rule for the next chat

GitHub is canonical; chat memory is secondary.

Read `CURRENT_NFL_RESEARCH_HANDOFF.md`, then this file. Verify current `main` before writing anything. Continue from the artifact replay / identity-audit step rather than restarting research or redesigning the model.
