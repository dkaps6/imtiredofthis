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
- Preserved archive SHA256 was re-verified during the repair replay.

Important: do not claim an exact historical exception from the original failed job unless the decoded log is actually retrieved and seen. The preserved artifacts themselves are the authority for the replay findings below.

## Roster-universe finding — closed

The preserved live `roles_ourlads.csv` contained 28 teams. That is legitimate for the remaining live slate.

The four absent teams were LAR, NE, SEA and SF; they had already played and were not required in the remaining sportsbook/event universe.

Correct invariant:

> Every team represented by the current sportsbook/event universe must exist in the current roster snapshot. Teams outside the active event universe may legitimately be absent. A missing current-event team must still fail closed.

Original roster-scope repair:

- commit `ea0af251f943fe6a540e82e36c38ab5a6295fb7d`
- PR #518 `Fix live roster validation for remaining-slate event scope`

Behavior:

- `_build_roster_index()` accepts `required_teams`.
- Required teams come from current sportsbook events.
- Missing current-event teams fail closed.
- Absent non-event teams are allowed.
- Callers without an event scope retain historical all-32 validation.
- Invalid required team codes fail closed.
- No model science, pricing semantics, projection logic or promotion rules changed.

Regression cases cover 28-team remaining-slate PASS, missing current-event team FAIL CLOSED, full-32 PASS, and missing non-event teams ignored.

## Preserved-artifact replay — COMPLETE

The exact preserved live artifact from run `34602328548` was replayed against the repaired production identity boundary without another OddsAPI fetch.

Replay result:

- event-scoped 28-team roster gate: PASS
- active sportsbook/event team universe covered by Ourlads: PASS
- unresolved modeled-core identities: **0**
- modeled-core markets checked:
  - `player_pass_yds`
  - `player_rush_yds`
  - `player_reception_yds`
  - `player_receptions`

The large apparent unmatched population was not a broad modeled-player failure.

Exhaustive non-core classification:

- 84 unresolved anytime-TD/non-core labels total
- 56 were synthetic defense labels
- 28 were player names
- 27 of those 28 players were absent from the current authoritative roster/slate and remain correctly quarantined/non-modeled
- the one confirmed current-player provider alias was `Zonovan Knight -> Bam Knight`

No speculative aliases were added for the other names.

Detailed replay paper trail:

- `docs/production/WEEK1_LIVE_PRESERVED_ARTIFACT_REPLAY_2026_09_11.md`

## Second mechanical defect found by replay — future same-opponent rematches

The old live odds event gate used only the unordered team pair. Because some Week-1 opponents meet again later in the season, four future rematches survived the pair-only filter in the preserved artifact and generated placeholder sportsbook rows.

Observed separation in the preserved replay:

- legitimate Week-1 provider events were approximately 17.0-24.33 hours from the repository's date-level UTC schedule anchor
- leaked future rematches were more than 1,173 hours away

This is sportsbook boundary plumbing only; it does not change any football projection/model science.

### PR #519 — matchup + kickoff scoping

- merged main SHA `c54746444b9c64bacb170649ee89832e4abda206`
- title: `Harden Week 1 live event scope after preserved replay`

Repair:

- `run_live_odds_gate.py` now requires canonical matchup **and** bounded authoritative kickoff proximity
- tolerance: 36 hours, intentionally wide enough for the date-level UTC schedule representation while far below later-season rematches
- active-pair sportsbook rows with invalid `commence_time` fail closed
- regression explicitly supplies the same two teams in Week 1 and a future rematch and proves only the Week-1 event survives

Repo CI passed this repair, including 211 unit tests.

## Third mechanical defect found in review — conflicting mirrored kickoff anchors

PR #519 review correctly identified that mirrored `team_week_map` rows could theoretically carry two different parseable kickoff anchors for the same matchup. Accepting an event close to either anchor would violate the required-schedule consistency contract.

### PR #521 — fail closed on schedule-anchor disagreement

- merged main SHA `7ccf12603da3da456e61586e4dd27062fa33b84e`
- title: `Fail closed on conflicting Week 1 kickoff anchors`

Repair:

- mirrored schedule rows for one matchup must resolve to exactly one distinct `kickoff_utc`
- multiple distinct parseable anchors now raise before sportsbook scoping
- regression proves conflicting mirrored KC/DEN anchors fail closed

Repo CI passed before merge. No review comments or unresolved review threads remained.

## Verified Knight alias — correct boundary

The preserved sportsbook artifact used `Zonovan Knight`; the current Arizona roster identity is `Bam Knight`.

Verified provider-name bridge:

- `Zonovan Knight -> Bam Knight`
- current team: ARI
- position: RB
- stable GSIS person ID: `00-0037157`
- verified through Arizona Cardinals documentation

The live canonical-name bridge remains in `data/manual_name_overrides.csv`.

A first attempt also added a stable historical alias row to `data/player_identity_aliases.csv`. The no-paid integration run correctly rejected that row because nflreadpy historical roster identity for GSIS `00-0037157` is already `Bam Knight`, so declaring `historical_name=Zonovan Knight` violated the identity-history contract.

The validator was **not weakened**. The bad stable alias row was removed and the verified live provider-name bridge was retained.

### PR #522 — correct Knight alias boundary

- merged main SHA `de7bd83e8ff8eb23535efbc53f92d8b79182934d`
- title: `Keep Knight alias at live canonical-name boundary`

Repo CI passed before merge.

## Verification lineage

### Original roster-scope repair CI

- run `34621082108`
- job `103334928263`
- SHA `ea0af251f943fe6a540e82e36c38ab5a6295fb7d`
- SUCCESS

### Original roster-scope no-paid Full Slate

- run `34621082187`
- job `103334928676`
- SHA `ea0af251f943fe6a540e82e36c38ab5a6295fb7d`
- SUCCESS
- Steps 1-24 PASS
- sportsbook Steps 25-31 SKIPPED

### Intermediate no-paid run that caught the invalid stable Knight alias

- run `34645962275`
- job `103416720917`
- SHA `7ccf12603da3da456e61586e4dd27062fa33b84e`
- FAILED at Step 15 PlayerForm
- exact failure: stable identity alias expected `historical_name=Zonovan Knight`, but the historical source for GSIS `00-0037157` already used `Bam Knight`
- this was a correct fail-closed validation result
- no sportsbook fetch was performed

### Final Repo CI after correcting Knight alias boundary

- post-merge Repo CI run `34646336751`
- main SHA `de7bd83e8ff8eb23535efbc53f92d8b79182934d`
- SUCCESS

### Final no-paid Full Slate integration gate

- run `34646336732`
- job `103417908187`
- main SHA `de7bd83e8ff8eb23535efbc53f92d8b79182934d`
- **SUCCESS**
- Steps 1-24 PASS
- PlayerForm Step 15 PASS
- strict repository/readiness audits PASS
- artifacts uploaded
- sportsbook/live-odds Steps 25-31 SKIPPED because live mode was disabled
- no OddsAPI credits used

## Current status

`preserved artifact verified -> event-scoped roster repair PASS -> modeled-core identity replay 0 unresolved -> exhaustive non-core classification complete -> future-rematch leak repaired -> conflicting schedule-anchor gate repaired -> verified Knight live alias repaired at correct boundary -> Repo CI green -> no-paid Full Slate green -> M108 exact 26/26 regression invocation still must be located/verified -> paid live run NOT triggered`

## Remaining pre-paid gate — M108

The active handoff inherited the explicit requirement:

> targeted tests plus the broader regression suite; **M108 must remain 26/26 PASS or better**.

The current GitHub repository has been searched across code, commits, branches, PRs, issues, workflow runs and continuity/handoff documents for literal `M108`, `Migration 108`, `26/26`, and `scientific regression` lineage. The handoff preserves the fact that the baseline was previously 26/26 PASS, but the exact workflow/script/run that generated that label has not yet been recovered from GitHub under the `M108` name.

Do **not** silently substitute ordinary Repo CI or the no-paid Full Slate for this explicit scientific gate. Do **not** trigger the paid Full Slate until the exact M108 regression lineage is identified and verified, or the canonical source of that gate is recovered.

If the exact M108 lineage cannot be recovered from GitHub, ask the user for a pointer to the chat/file/run that named M108 rather than guessing.

## Exact next execution order

1. Recover the exact M108 26/26 regression invocation/run lineage.
2. Re-run or otherwise verify the exact M108 gate on the repaired current production head; require 26/26 PASS or better.
3. If M108 is green, verify current `main` again.
4. Execute exactly one paid live Full Slate run.
5. Validate production output: nonempty edges, real sportsbook lines/prices, selected-book semantics, duplicate collapse, provenance, status codes, correct event-week scope, and zero unresolved current-event team/core-player identity failures.
6. Update this handoff and the decision/continuity ledger with exact paid-run commit/run/artifact lineage.
7. Only after the incident is closed, resume the parked QB/WR shared-opportunity / public-pregame-intent V1B lane.

## Guardrails

Do **not** alter:

- M107 thresholds
- model weights
- PlayerForm science
- role-chain science
- pricing semantics
- promotion rules
- sportsbook comparison logic
- projections
- any other scientific behavior

Architecture remains football-first:

`GAME STATE -> TEAM OPPORTUNITY -> POSITION/ROOM POOL -> PLAYER ENTITLEMENT -> PLAYER EFFICIENCY -> JOINT MONTE CARLO -> PLAYER PROJECTIONS/DISTRIBUTIONS -> FAIR PROBABILITIES -> SPORTSBOOK COMPARISON`

Vegas cannot teach upstream football projections which direction to move.

## Paid-run rule

No paid Full Slate has been triggered during this repair sequence after the preserved failed artifact was recovered. Do **not** spend another OddsAPI Full Slate merely to discover the next bug. M108 is the remaining explicit gate.

## Parked science lane

The separate QB/WR shared-opportunity / first-down pass-propensity / public pregame-intent V1B work remains preserved at:

- `docs/handoffs/NFL_HANDOFF_2026-09-11_QB_WR_SHARED_OPPORTUNITY_CURRENT.md`

Resume that lane only after the live production incident is closed. It has made no production changes.

## Resume rule for the next chat

GitHub is canonical; chat memory is secondary.

Read `CURRENT_NFL_RESEARCH_HANDOFF.md`, then this file. Verify current `main` before writing anything. Do **not** restart the preserved-artifact replay or identity audit; those are complete. Resume from the exact M108 gate-recovery checkpoint above.
