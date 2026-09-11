# NFL Production Handoff — 2026-09-11 — Week 1 Full Slate Live Repair

## Active checkpoint

GitHub is canonical; chat memory is secondary.

The preserved-artifact replay and live player-identity audit are complete. The remaining-slate roster defect, future-rematch event defect, conflicting schedule-anchor defect, and the verified Knight provider-name bridge have all been repaired and validated. Repo CI and the no-live-odds Full Slate integration gate are green.

Do **not** restart those completed investigations and do **not** redesign or retune model science.

Current main at this correction: `292cd883fcaeba8b6eb0d80810be1d7c145938a9` (root continuity correction only, parent production state `9f7b875a4ebf7a0a4ed61ad7f87e7575cdff2e3f`).

## Frozen production science

Production authorities remain unchanged:

- QB mean: M89/M90 / `QB_PASS_SYNTHESIS_V1`
- QB distribution: `C2_QB_MEAN_NEUTRAL_DISTRIBUTION_V1`
- WR: M38 WR1 hierarchy + `WR_R15_PRODUCTION_MODEL_V1`
- TE: `TE_R5P_PRODUCTION_MODEL_V1`
- RB rushing: `RB_P3_SYNTHESIS_V1` for the qualified Week-1 route
- RB receptions/opportunity: R26
- RB receiving-yard tail/distribution: R22 using frozen R19 assets

## IMPORTANT CORRECTION — M108 / 26-of-26

Earlier continuity notes carried an explicit claim that `M108` was a scientific regression gate that had previously passed `26/26`, and the prior version of this handoff treated recovery of that invocation as a blocker.

That was not sufficiently grounded in canonical GitHub lineage.

The repository was searched across code, handoffs/continuity docs, commits, PRs, issues, branches, and workflow history for `M108`, `Migration 108`, `26/26`, and related regression wording. The inherited statement is present in the handoff, but **no authoritative M108 workflow, script, PR, commit, branch, or run has been recovered that proves a canonical repository test named M108 produced a 26-of-26 result**.

Therefore:

- do not describe `M108 26/26` as a verified repository gate;
- do not fabricate or reconstruct an M108 test by name;
- do not hold the production repair indefinitely on that chat-memory label;
- if concrete M108 evidence is found later, preserve it as historical lineage and evaluate what it actually was;
- use the actual named repository validation below as the authoritative pre-live gate.

This correction changes documentation/lineage only. It does not change model science, thresholds, weights, projection logic, pricing semantics, or sportsbook comparison behavior.

## Original failed live run

- original main SHA `0c6fcefea0d38d0975dea7fd04edd43216b2cbab`
- failed Full Slate run `34602328548`
- failed job `103272747139`
- Steps 1-24 passed; failure occurred at Step 25, `Fetch and gate player props and game odds after football eligibility`
- preserved artifact ID `10265557165`
- artifact name `run_34602328548`
- artifact SHA256 `08727a37cf174a5be9767855a8a6aa3854c0f56cc5e650099ec3dced127e12b1`

Do not claim an exact historical exception from that job unless the decoded log itself is retrieved. The preserved artifact is the authority for the replay findings.

## Repair 1 — remaining-slate roster scope — CLOSED

The preserved `roles_ourlads.csv` had 28 teams. LAR, NE, SEA and SF had already played and were legitimately outside the remaining event universe.

Correct invariant: every team represented by the current event universe must exist in the current roster snapshot; non-event teams may be absent; a missing current-event team fails closed.

- PR #518 `Fix live roster validation for remaining-slate event scope`
- commit `ea0af251f943fe6a540e82e36c38ab5a6295fb7d`

Regression coverage:

- 28-team roster + all current-event teams present -> PASS
- one current-event team missing -> FAIL CLOSED
- full 32-team roster -> PASS
- missing non-event teams -> ignored

## Preserved-artifact replay / identity audit — COMPLETE

Exact preserved run artifact replayed without another live-odds fetch.

Results:

- event-scoped roster gate PASS
- current event-team universe covered by Ourlads PASS
- unresolved modeled-core identities: **0**
- modeled-core markets checked: `player_pass_yds`, `player_rush_yds`, `player_reception_yds`, `player_receptions`

The apparent unmatched population was primarily non-core:

- 84 unresolved anytime-TD/non-core labels
- 56 synthetic defense labels
- 28 player names
- 27/28 player names absent from current authoritative roster/slate and correctly quarantined
- one confirmed current provider alias: `Zonovan Knight -> Bam Knight`

No speculative aliases were added.

Detailed replay record: `docs/production/WEEK1_LIVE_PRESERVED_ARTIFACT_REPLAY_2026_09_11.md`.

## Repair 2 — future same-opponent rematches — CLOSED

Pair-only event matching allowed four later-season rematches to survive the Week-1 gate.

- PR #519 `Harden Week 1 live event scope after preserved replay`
- merged SHA `c54746444b9c64bacb170649ee89832e4abda206`

`run_live_odds_gate.py` now requires canonical matchup plus bounded authoritative kickoff proximity. Tolerance is 36 hours. Invalid `commence_time` on an active pair fails closed. Regression proves a later rematch cannot survive the Week-1 scope.

## Repair 3 — conflicting mirrored kickoff anchors — CLOSED

- PR #521 `Fail closed on conflicting Week 1 kickoff anchors`
- merged SHA `7ccf12603da3da456e61586e4dd27062fa33b84e`

Mirrored schedule rows for one matchup must resolve to one distinct `kickoff_utc`; conflicting parseable anchors now fail closed.

## Repair 4 — Knight live provider-name bridge — CLOSED

Verified bridge:

- `Zonovan Knight -> Bam Knight`
- ARI, RB
- GSIS `00-0037157`

The bridge belongs in `data/manual_name_overrides.csv`.

An attempted stable historical alias was correctly rejected by PlayerForm because historical nflreadpy identity for that GSIS ID already uses `Bam Knight`. The validator was not weakened; the invalid historical alias was removed.

- PR #522 `Keep Knight alias at live canonical-name boundary`
- merged SHA `de7bd83e8ff8eb23535efbc53f92d8b79182934d`

## Authoritative validation lineage

- roster-scope repair Repo CI: run `34621082108`, job `103334928263`, SHA `ea0af251...`, SUCCESS
- roster-scope no-live Full Slate: run `34621082187`, job `103334928676`, SHA `ea0af251...`, SUCCESS; Steps 1-24 PASS, live-odds Steps 25-31 skipped
- fail-closed Knight historical-alias run: `34645962275`, job `103416720917`, SHA `7ccf1260...`, failed correctly at PlayerForm Step 15 before any live fetch
- final Repo CI after Knight boundary correction: `34646336751`, SHA `de7bd83e...`, SUCCESS
- final no-live Full Slate integration: run `34646336732`, job `103417908187`, SHA `de7bd83e...`, SUCCESS; Steps 1-24 PASS; PlayerForm PASS; strict repository/readiness audits PASS; artifacts uploaded; live-odds Steps 25-31 skipped
- handoff-only main SHA `9f7b875a4ebf7a0a4ed61ad7f87e7575cdff2e3f`: Full Slate run `34646857105` SUCCESS and Repo CI `34646857098` SUCCESS

## Current status

`artifact replay COMPLETE -> event-scoped roster repair PASS -> modeled-core identity replay 0 unresolved -> non-core classification COMPLETE -> future-rematch repair PASS -> conflicting-anchor repair PASS -> Knight boundary PASS -> Repo CI GREEN -> no-live Full Slate GREEN -> inherited M108 label audited and NOT VERIFIED AS A CANONICAL GITHUB GATE -> ready for one controlled live Full Slate`

## Exact next execution order

1. Read `AGENTS.md`, `CURRENT_NFL_RESEARCH_HANDOFF.md`, then this file.
2. Verify current `main`. If it advanced, inspect only the intervening diff/commits; do not restart completed history.
3. Do not repeat a broad M108 search unless genuinely new evidence appears.
4. Run **exactly one** controlled live Full Slate from current repaired main with live odds enabled.
5. Audit the output before declaring closure:
   - nonempty priced output/edges where markets are available
   - real lines/prices and no placeholder or future-rematch contamination
   - selected-book semantics correct
   - duplicate collapse correct
   - provenance/status codes correct
   - event/week scope correct
   - every current event team covered by roster authority
   - zero unresolved current-event modeled-core identities
   - QB passing-yards rows use promoted M89/M90 synthesis
   - Week-1 RB rushing-yards rows use qualified `RB_P3_SYNTHESIS_V1`
   - no live-market inputs leak upstream into football projections
6. If that run fails mechanically, preserve exact run/job/artifact/log lineage and repair only the demonstrated plumbing defect. Do not retune science or immediately spend another live fetch just to find the next bug.
7. If it passes, update this handoff and `NFL_MASTER_CONTINUITY_RECORD.md` with exact final lineage and mark the Week-1 incident CLOSED.
8. Only then resume the parked QB/WR shared-opportunity / first-down pass-propensity / public-pregame-intent V1B lane.

## Guardrails

Do not alter M107 thresholds, model weights, PlayerForm science, role-chain science, pricing semantics, promotion rules, sportsbook comparison logic, projections, or any other scientific behavior while closing this incident.

Architecture remains football-first:

`GAME STATE -> TEAM OPPORTUNITY -> POSITION/ROOM POOL -> PLAYER ENTITLEMENT -> PLAYER EFFICIENCY -> JOINT MONTE CARLO -> PLAYER PROJECTIONS/DISTRIBUTIONS -> FAIR PROBABILITIES -> SPORTSBOOK COMPARISON`

## Parked science lane

Preserved at `docs/handoffs/NFL_HANDOFF_2026-09-11_QB_WR_SHARED_OPPORTUNITY_CURRENT.md`. Resume only after the live incident is closed.

## Resume rule for the next chat

Do not restart the preserved-artifact replay, roster audit, identity audit, rematch investigation, Knight alias investigation, or broad M108 search. Verify current main and continue from the **one controlled live Full Slate** checkpoint.