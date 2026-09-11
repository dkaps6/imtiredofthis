# NFL Production Handoff — 2026-09-11 — Week 1 Full Slate Live Repair

## Active checkpoint

GitHub is canonical; chat memory is secondary.

The Week-1 live Full Slate incident has advanced past the original failure. The original Step-25 live fetch/gate incident is repaired. One controlled paid live Full Slate has now been executed from repaired `main`. That paid run passed football Steps 1-24 and live sportsbook Steps 25-28, then failed mechanically at Step 29 `Classify live Full Slate data quality`.

**Do not restart completed investigations. Do not redesign or retune model science. Do not spend another paid live-odds call to discover the next bug.**

Production state audited immediately before this handoff update: `be061eaf23372f080db3911d3b4919120c744c53` (`Dispatch one controlled Week 1 live Full Slate`). This handoff update is documentation-only and may advance `main`; always inspect only the intervening diff before continuing.

## Integrity verdict — model methodology was NOT changed by this incident repair

A direct Git compare from the original failed production SHA `0c6fcefea0d38d0975dea7fd04edd43216b2cbab` through the paid-run SHA `be061eaf23372f080db3911d3b4919120c744c53` shows changes only in:

- `.github/workflows/dispatch-full-slate-live-odds-once.yml`
- `CURRENT_NFL_RESEARCH_HANDOFF.md`
- `data/manual_name_overrides.csv`
- this handoff
- preserved-artifact replay documentation
- `scripts/repair_live_prop_identity_v1.py`
- `scripts/run_live_odds_gate.py`
- live-odds / roster-scope regression tests

No QB, WR, RB, TE model-science implementation, promoted-model authority, projection-weight, distribution, pricing-science, or football-opportunity methodology file was changed in that repair sequence.

Even more importantly, the final successful no-live Full Slate baseline was run `34649759674` on SHA `1f7255805647e978adddb2cf4aecf2883f0dda48`. The paid live-dispatch SHA `be061eaf23372f080db3911d3b4919120c744c53` differs from that baseline by exactly **one file and one line**: the authorized marker in `.github/workflows/dispatch-full-slate-live-odds-once.yml`. Therefore the paid Step-29 failure was not introduced by a model-methodology change between the green baseline and the live run.

## Why we had a green Full Slate before and still failed now

The prior green Full Slate was a **no-live-odds integration run**. It successfully executed football Steps 1-24, then intentionally skipped live sportsbook Steps 25-31 because live odds were disabled.

The controlled paid run enabled live odds and therefore exercised code paths that the green no-live run never traversed. It successfully passed Steps 25, 26, 27 and 28, then exposed a stale Step-29 certification invariant. This is a deeper live-path validation failure, not evidence that the football model regressed.

## Frozen production science

Production authorities remain unchanged:

- QB mean: M89/M90 / `QB_PASS_SYNTHESIS_V1`
- QB distribution: `C2_QB_MEAN_NEUTRAL_DISTRIBUTION_V1`
- WR: M38 WR1 hierarchy + `WR_R15_PRODUCTION_MODEL_V1`
- TE: `TE_R5P_PRODUCTION_MODEL_V1`
- RB rushing: `RB_P3_SYNTHESIS_V1` for the qualified Week-1 route
- RB receptions/opportunity: R26
- RB receiving-yard tail/distribution: R22 using frozen R19 assets

Architecture remains football-first:

`GAME STATE -> TEAM OPPORTUNITY -> POSITION/ROOM POOL -> PLAYER ENTITLEMENT -> PLAYER EFFICIENCY -> JOINT MONTE CARLO -> PLAYER PROJECTIONS/DISTRIBUTIONS -> FAIR PROBABILITIES -> SPORTSBOOK COMPARISON`

Vegas/live sportsbook inputs remain downstream only and must not teach upstream football projections.

## M108 / M107 continuity correction

Earlier continuity notes carried an unsupported claim that `M108` was a scientific regression gate that had passed `26/26`, and also referenced M107 thresholds as if a canonical migration lineage had been recovered.

Repository searches did not recover an authoritative M108 workflow/script/result/branch/commit proving such a gate. Do not fabricate or reconstruct M108 by name and do not reinterpret generic Repo CI counts as `26/26`.

This is a documentation-lineage correction only. It does not change model science.

## Original failed live run

- original main SHA `0c6fcefea0d38d0975dea7fd04edd43216b2cbab`
- failed Full Slate run `34602328548`
- failed job `103272747139`
- Steps 1-24 passed
- original failure occurred at Step 25 `Fetch and gate player props and game odds after football eligibility`
- preserved artifact ID `10265557165`
- artifact name `run_34602328548`
- artifact SHA256 `08727a37cf174a5be9767855a8a6aa3854c0f56cc5e650099ec3dced127e12b1`

The exact historical exception was not preserved; do not invent one.

## Completed repair lineage

### Repair 1 — remaining-slate roster scope — CLOSED

The preserved `roles_ourlads.csv` had 28 teams because LAR, NE, SEA and SF had already played. Correct live invariant: every team represented by the current event universe must exist in the current roster snapshot; non-event teams may be absent; a missing current-event team fails closed.

- PR #518
- merged SHA `ea0af251f943fe6a540e82e36c38ab5a6295fb7d`

### Preserved-artifact replay / modeled-core identity audit — COMPLETE

Core markets audited:

- `player_pass_yds`
- `player_rush_yds`
- `player_reception_yds`
- `player_receptions`

Result: **0 unresolved modeled-core player identities**.

The apparent unmatched population was primarily non-core anytime-TD / defense-label material. One current provable provider alias was confirmed: `Zonovan Knight -> Bam Knight`, ARI RB, GSIS `00-0037157`.

Detailed replay: `docs/production/WEEK1_LIVE_PRESERVED_ARTIFACT_REPLAY_2026_09_11.md`.

### Repair 2 — future same-opponent rematches — CLOSED

Pair-only event matching could admit later-season rematches. Live event scope now requires canonical matchup plus bounded authoritative kickoff proximity (36 hours).

- PR #519
- merged SHA `c54746444b9c64bacb170649ee89832e4abda206`

### Repair 3 — conflicting mirrored kickoff anchors — CLOSED

Mirrored schedule rows for one matchup must resolve to one distinct `kickoff_utc`; conflicting anchors fail closed.

- PR #521
- merged SHA `7ccf12603da3da456e61586e4dd27062fa33b84e`

### Repair 4 — Knight live provider-name bridge — CLOSED

The verified provider bridge belongs only at the live canonical-name boundary:

- `data/manual_name_overrides.csv`: `Zonovan Knight -> Bam Knight`

An attempted stable historical alias was correctly rejected because the historical source for GSIS `00-0037157` already uses Bam Knight. The validator was not weakened; the bad historical alias was removed.

- PR #522
- merged SHA `de7bd83e8ff8eb23535efbc53f92d8b79182934d`

## Last green no-live Full Slate

- run `34649759674`
- job `103428976657`
- SHA `1f7255805647e978adddb2cf4aecf2883f0dda48`
- conclusion: SUCCESS
- football Steps 1-24: PASS
- live sportsbook Steps 25-31: SKIPPED because live odds were disabled
- strict repository audits/artifact generation: PASS

This run proved the football/model stack was still healthy but did **not** exercise Step 29.

## Controlled paid live Full Slate — EXECUTED ONCE

Authorized dispatch commit:

- SHA `be061eaf23372f080db3911d3b4919120c744c53`
- commit message `Dispatch one controlled Week 1 live Full Slate`
- only change from the immediately preceding green baseline: one-line authorized marker in `.github/workflows/dispatch-full-slate-live-odds-once.yml`

Paid live run:

- run ID `34650067599`
- job ID `103429972300`
- SHA `be061eaf23372f080db3911d3b4919120c744c53`
- overall conclusion: FAILURE at Step 29

Step results:

- Steps 1-24 football/model build: **SUCCESS**
- Step 25 fetch/gate player props and game odds: **SUCCESS**
- Step 26 validate/compact live sportsbook boundary: **SUCCESS**
- Step 27 validate live opponent map: **SUCCESS**
- Step 28 audit live player identity semantics: **SUCCESS**
- Step 29 classify live Full Slate data quality: **FAILURE**
- Steps 30-32: skipped after Step 29 failure
- artifact build/upload still completed

This proves the original Step-25 production incident is repaired.

## Paid artifact — zero-credit replay authority

- artifact ID `10283817522`
- artifact SHA256 `a70c90023632059476cc688070fb21ad63b9ebd14aa4db102844b9e342188359`

Important observed live status:

- active sportsbook events: 14
- full Week-1 scheduled games: 16
- live event-team universe: 28 teams
- absent live teams: LAR, NE, SEA, SF — the four teams that had already played
- actual prop rows: 894
- production compact rows: 782
- production compact unique players: 344
- unresolved modeled-core identities: 0
- strict market identity failures: 0
- sportsbook/current-roster identity mismatches: 0
- future-rematch contamination: blocked by Step-26 live boundary validation

Use the saved paid artifact for all Step-29 and downstream debugging. **Do not pay for another fetch to discover bugs.**

## Current demonstrated failure — stale Step-29 all-32 roster invariant

`scripts/validate_full_slate_data_quality_v1.py` currently does two different things:

1. correctly requires the authoritative Week-1 schedule to contain all 32 scheduled teams; and
2. incorrectly requires the **current live Ourlads roster snapshot** to equal that same 32-team scheduled set.

The relevant stale logic is effectively:

`if role_teams != scheduled_teams: raise RuntimeError("current Ourlads roster team set does not match scheduled teams")`

For the paid live artifact:

- scheduled teams = 32
- current live roster teams = 28
- schedule minus roster = `LAR, NE, SEA, SF`
- those four teams had already played and are intentionally outside the 14 remaining live sportsbook events
- current live sportsbook/event teams = 28 and are all represented by the roster snapshot

Therefore Step 29 is applying the old all-32 roster rule even though the upstream live roster gate was already correctly repaired to event scope.

This is a **mechanical certification-scope defect**, not a provider-fetch failure, not an identity failure, and not a model-science failure.

## Exact next execution order

1. Verify current `main` and inspect only commits after the production state recorded above.
2. Do **not** rerun broad historical investigations, alias research, M108 searches, or paid live odds.
3. Patch only the Step-29 current-roster certification invariant so that:
   - the full schedule must still prove 32 teams / 16 games;
   - every current live sportsbook/event team must exist in `roles_ourlads.csv`;
   - non-event teams may be absent;
   - a missing current-event team fails closed;
   - a full 32-team roster remains valid;
   - future/off-slate events cannot expand the required roster universe.
4. Add focused regression coverage for the cases above.
5. Run targeted tests and Repo CI/no-paid Full Slate.
6. Replay the saved paid artifact through Step 29 locally/offline with the patched validator. Zero OddsAPI credits.
7. If Step 29 passes, continue replaying downstream Steps 30-31 offline wherever possible and exhaustively audit:
   - real lines/prices
   - nonempty priced output/edges where markets exist
   - selected-book semantics
   - deterministic duplicate collapse
   - provenance/status codes
   - active-week event scope
   - zero unresolved current-event modeled-core identities
   - QB passing-yard rows use M89/M90 synthesis
   - Week-1 RB rushing rows use `RB_P3_SYNTHESIS_V1`
   - no sportsbook input leaks upstream into football projections
8. Only after the saved paid artifact and repository validations are clean should any decision about another live run be made.
9. When fully closed, update this handoff and `NFL_MASTER_CONTINUITY_RECORD.md`, then resume the parked QB/WR research lane.

## Guardrails

Do not alter:

- QB/WR/RB/TE model science
- model weights or thresholds
- PlayerForm science
- role-chain science
- projection methodology
- football opportunity logic
- promoted model authorities
- pricing semantics
- sportsbook comparison semantics except a separately demonstrated downstream mechanical bug
- promotion rules except a narrowly demonstrated mechanical validation-scope repair

Do not weaken truth/integrity gates. The fix must preserve fail-closed behavior for missing **current-event** teams.

## Current status

`original Step-25 incident FIXED -> remaining-event roster scope PASS -> modeled-core identities 0 unresolved -> rematch scope PASS -> kickoff-anchor fail-close PASS -> Knight live alias boundary PASS -> no-live football/model stack GREEN -> one paid live run executed -> live Steps 25-28 PASS -> Step 29 exposed stale all-32 roster certification rule -> MODEL SCIENCE INTEGRITY CHECK PASS -> next action is surgical Step-29 event-scope repair + zero-credit paid-artifact replay`

## Parked science lane

Preserved at `docs/handoffs/NFL_HANDOFF_2026-09-11_QB_WR_SHARED_OPPORTUNITY_CURRENT.md`. Resume only after the live incident is closed.

## Resume rule for the next chat

Do not restart the preserved-artifact replay, roster investigation, identity audit, rematch investigation, Knight alias investigation, M108 search, or paid live fetch. Verify current `main`, read this exact checkpoint, and continue with the **Step-29 event-scoped certification repair using the saved paid artifact**.
