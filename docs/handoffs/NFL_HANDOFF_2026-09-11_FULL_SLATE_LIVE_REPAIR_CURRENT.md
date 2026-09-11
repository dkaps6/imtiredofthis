# NFL Production Handoff — 2026-09-11 — Week 1 Full Slate Live Repair

## Active checkpoint

GitHub is canonical; chat memory is secondary.

The Week-1 live Full Slate incident has advanced past the original Step-25 failure and past the controlled paid-run Step-29 failure. The original live fetch/gate incident is repaired. One controlled paid live Full Slate was executed from repaired `main`; it passed football Steps 1-24 and live sportsbook Steps 25-28, then failed mechanically at Step 29 because a downstream validator still required the current live roster to contain all 32 Week-1 teams even though four teams had already played.

PR #523 now contains the mechanical/current-availability certification repair. The exact preserved paid artifact from run `34650067599` has been replayed through Steps 29-31 and strict repository audits with **zero certification blockers and zero additional OddsAPI acquisition**. PR #523 remains **OPEN / UNMERGED** pending final integrity reconciliation against current `main`.

**Do not restart completed investigations. Do not redesign or retune model science. Do not spend another paid live-odds call to discover bugs. Do not merge PR #523 blindly without first reconciling it to current `main`.**

Current `main` immediately before this handoff correction was `e4c1a620e9d1720d60f3f2d9267bb731ea016f8f` (`Reconcile Full Slate handoff with PR 523 replay`). This handoff update is documentation-only and advances `main`; always inspect only the intervening diff before continuing.

## Integrity verdict — the paid failure was NOT caused by model-methodology degradation

The key distinction is execution coverage, not a change in football science.

The final successful no-live Full Slate baseline was run `34649759674` on SHA `1f7255805647e978adddb2cf4aecf2883f0dda48`. The paid live-dispatch SHA `be061eaf23372f080db3911d3b4919120c744c53` differed from that immediately preceding green baseline by exactly **one file / one authorized live-dispatch marker line** in `.github/workflows/dispatch-full-slate-live-odds-once.yml`. No QB, WR, RB, TE, projection, distribution, entitlement, simulation, pricing-science, or fitted-model implementation changed between that green baseline and the controlled paid run.

Therefore the Step-29 failure cannot be attributed to a scientific-model mutation introduced between the green baseline and the paid live run. What changed was that the paid run exercised the live sportsbook / late-certification path that the no-live run intentionally skipped.

A broader compare across the original live incident repair through the paid-run SHA likewise shows operational live-odds scope/identity/workflow/documentation changes, not changes to the frozen QB/WR/RB/TE production authorities.

This conclusion is deliberately scoped: it proves the integrity of the incident interval and the current PR #523 repair. It is **not** a claim that no methodology file has ever changed anywhere in the repository’s full historical lifetime.

## Why a prior green Full Slate could still be followed by this failure

The prior green Full Slate was a **no-live-odds integration run**. It successfully executed football Steps 1-24, then intentionally skipped live sportsbook Steps 25-31 because live odds were disabled.

The controlled paid run enabled the live path. It successfully passed Steps 25, 26, 27 and 28, then exposed a stale Step-29 certification invariant that the no-live run never executed.

So:

- green no-live Full Slate = football/model stack and non-live integration healthy;
- paid live Full Slate = additionally exercises live acquisition, boundary validation, identity semantics, data-quality classification, pricing/certification and downstream lineage;
- the paid failure was a newly exercised **mechanical certification-scope defect**, not evidence that the football model regressed.

## Frozen production science — unchanged

Production authorities remain:

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

Canonical correction: `docs/production/WEEK1_LIVE_M107_M108_PROVENANCE_CORRECTION_2026_09_11.md`.

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

## Completed repair lineage before the controlled paid run

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

This run proved the football/model stack was healthy but did **not** exercise the live sportsbook/certification path.

## Controlled paid live Full Slate — EXECUTED ONCE

Authorized dispatch commit:

- SHA `be061eaf23372f080db3911d3b4919120c744c53`
- commit message `Dispatch one controlled Week 1 live Full Slate`
- no model-science change from the immediately preceding green no-live baseline

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

The Step-29 error was the stale full-32 current-roster invariant. The paid artifact showed 405 current roster rows across the correct 28 remaining event teams while the authoritative full Week-1 schedule still contained all 32 teams / 16 games. The missing four live-roster teams were LAR, NE, SEA and SF, which had already played.

This proves the original Step-25 production incident was repaired and the next failure was downstream mechanical certification scope.

## Paid artifact — zero-credit replay authority

- source paid run: `34650067599`
- artifact ID `10283817522`
- artifact name: `run_34650067599`
- artifact SHA256 `a70c90023632059476cc688070fb21ad63b9ebd14aa4db102844b9e342188359`

Observed live status:

- active sportsbook events: 14
- full Week-1 scheduled games: 16
- live event-team universe: 28 teams
- absent live teams: LAR, NE, SEA, SF — already played
- actual prop rows: 894
- production compact rows: 782
- production compact unique players: 344
- unresolved modeled-core identities: 0
- strict market identity failures: 0
- sportsbook/current-roster identity mismatches: 0
- future-rematch contamination: blocked by live boundary validation

Use this saved artifact for downstream verification. **Do not pay for another fetch to discover bugs.**

## PR #523 — current-availability certification repair — OPEN / UNMERGED

PR #523: `Scope Step 29 roster quality gate to live events`

- branch: `repair/week1-step29-event-roster-scope`
- currently audited head: `723e304291ee1dcf40f5a15df50585cbffcd4471`
- state at this handoff: **OPEN, NOT MERGED; GitHub currently reports `mergeable=false`**
- comparison to current handoff-era `main`: branch is `ahead_by=16`, `behind_by=2`, status `diverged`, merge base `be061eaf23372f080db3911d3b4919120c744c53`
- do not merge until the branch is explicitly reconciled/rebased against current `main` and revalidated

PR #523 changed only replay/certification/lineage plumbing and tests:

- `.github/workflows/replay-paid-full-slate-artifact-once.yml`
- `scripts/audit_market_model_lineage_v1.py`
- `scripts/audit_market_model_lineage_v2.py`
- `scripts/operations/apply_current_availability_downstream_certification_seam_v1.py`
- `scripts/stamp_qb_c2_pricing_lineage_v1.py`
- `scripts/validate_full_slate_data_quality_v1.py`
- `scripts/validate_full_slate_mechanical_certification_v1.py`
- `tests/test_full_slate_data_quality_scope.py`
- `tests/test_qb_c2_lineage_current_scope.py`

It does **not** change core model-generation/science files, fitted model assets, weights, learned parameters, football feature-generation formulas, simulation formulas, entitlement math, or sportsbook-to-football isolation.

### What PR #523 actually changes

The repair replaces stale hard-coded full-Week-1 cardinality assertions in downstream certification with the explicit certified current-availability authority where appropriate:

- Step-29 current roster: require every **current live event team**; preserve full 32-team schedule validation; allow already-played non-event teams to be absent; fail closed if a current-event team is missing.
- Injury scope: certify nflverse only when current official-report rows positively cover all 32 scheduled teams; incomplete scope remains a blocker.
- QB C2 lineage coverage: reconcile to current eligible-team authority instead of hard-coded 32 while preserving C2 version, selector and mean-neutral contract.
- WR-R15 certification: reconcile current WR1 anchor count to current eligible teams while preserving M38 WR1 anchor and R15 WR2+ redistribution contracts.
- R22 certification: reconcile current adapted-RB count to the actual R22 audit rather than a frozen historical player count.
- target-pool / certified-stack counts: reconcile players/teams/games to the certified current football universe instead of historical hard-coded cardinalities.

The downstream transformer contains protected science fragments and raises if those protected scientific contracts are changed by the transform.

### Pre-existing operational current-availability seams

The offline replay also invokes current-availability operations seams that already existed before PR #523. They convert stale full-32/current-player coverage assertions to the certified current-player universe without changing model parameters or formulas.

One explicit operational QB availability rule should remain visible for integrity review: the current-role seam uses `primary_qb = max(is_starter, is_qb1)` for current QB row authority. That is an operational current-player selection rule, not a fitted scientific model change, and it predates PR #523. Do not hide or reinterpret it as learned QB science.

The existing seams explicitly preserve:

- M89/M90 QB mean authority
- C2 states / selector / mean-neutral distribution contract
- R26 protected contracts
- P3 rushing authority
- M38/R15 entitlement logic
- TE-R5P contracts
- R22 source assets/parameters
- sportsbook isolation from football-generation inputs

## Exact paid-artifact replay after PR #523 — PASS, ZERO NEW ODDSAPI ACQUISITION

Repo CI on PR head:

- run `34655125555`
- head `723e304291ee1dcf40f5a15df50585cbffcd4471`
- compile: PASS
- static repo audit: PASS
- unit tests: PASS

Offline replay:

- run `34655125574`
- job `103445740378`
- head `723e304291ee1dcf40f5a15df50585cbffcd4471`
- downloaded exact paid artifact `run_34650067599`
- replay environment `FETCH_LIVE_ODDS=false`
- replay environment `FULL_SLATE_SOURCE_RUN_ID=34650067599`
- no OddsAPI acquisition command exists in the replay workflow
- conclusion: SUCCESS

Replay evidence artifact:

- artifact name `offline_replay_34655125574`
- artifact ID `10285470181`
- SHA256 `aa4d7a06f8f0fc521a0b19066c9ec691e3185d187c6ea91359ebdea99ebd75a9`

### Replay results

Step 29 / data quality:

- certification blockers: **0**
- schedule: 32 teams / 16 games certified
- current live roster: 405 rows / 28 live-event teams
- sportsbook/current-roster mismatches: 0
- nflverse injury evidence: 167 rows / 32 of 32 scheduled teams -> `CERTIFIED_REPORT_SCOPE`
- direct WR/CB unavailable state remained explicitly gated off rather than silently synthesized

Pricing / football universe:

- production compact rows: 782
- exact pricing book-line rows: 1,524
- priced side rows: 3,048
- priced players: 344
- quarantined rows: 126
- football players: 405
- football teams: 28
- canonical current games: 14
- sportsbook rows used to define football player universe: 0
- sportsbook inputs used to generate football distributions: false
- provider event IDs installed only post-simulation for lookup

Promoted-model lineage:

- QB football rows: 28
- C2 selected QB rows: 26
- C2 raw-mean preservation gap: floating-point noise only (~`5.68e-14`)
- WR-R15 current WR1 anchors: 28; M38 anchor preserved
- WR-R15 current WR2+ rows: 129
- R22 current adapted RB keys: 82; receiving mean preserved
- R26 receptions pricing lineage: PASS
- P3 rush+reception conservation: PASS
- TE-R5P active and team-pool/non-TE conservation preserved

Final mechanical certification:

- `disposition = PAID_FULL_SLATE_REPLAY_MECHANICAL_EXECUTION_CERTIFIED`
- `certification_scope = MECHANICAL_EXECUTION_DATA_IDENTITY_PRICING_AND_COMPONENT_ROUTING_ONLY`
- `does_not_certify_all_market_science = true`
- `certification_blockers = 0`
- `source_run = 34650067599`
- `odds_api_refetched = false`
- strict repository audit: PASS
- 2026 production-readiness audit: PASS

The replay is strong evidence that the preserved live data, identity boundary, pricing plumbing, current-availability certification and promoted-model routing now execute coherently. It is **not** a claim that every still-open market-specific research lane has acquired scientific certification.

## Concurrency / coordination note

During PR #523 replay, another repository lane had already made part of the QB lineage certification current-scope aware. An exact-anchor transformer check failed rather than overwriting that concurrent repair. The PR was then re-read against its current head and reconciled so the already-correct QB lineage implementation remained canonical and the downstream transformer touched only the still-stale assertions.

This was a coordination/stale-context issue during repair work, not evidence of model-science degradation. Continue to re-read current `main` and the current PR head before every write or merge.

## Guardrails

Do not alter:

- QB/WR/RB/TE model science
- model weights or scientific thresholds
- PlayerForm science
- projection methodology
- football opportunity logic
- promoted model authorities
- fitted model assets or parameters
- sportsbook isolation from upstream football generation
- pricing semantics except a separately demonstrated downstream mechanical bug
- sportsbook comparison semantics except a separately demonstrated downstream mechanical bug
- promotion rules except a narrowly demonstrated mechanical validation-scope repair

Do not weaken truth/integrity gates. Current-availability repairs must remain fail-closed for missing current-event teams or incomplete required source scope.

## Exact next execution order

1. Treat current `main` plus this handoff as canonical; do not resume from stale chat state.
2. Do **not** make another paid OddsAPI call for debugging. The exact paid artifact already replays successfully through the repaired mechanical chain.
3. PR #523 currently diverges from `main` and GitHub reports `mergeable=false`; reconcile/rebase it before any merge attempt.
4. After reconciliation, review PR #523 specifically for methodology integrity. The required verdict is not merely “tests green”; verify again that only availability/certification/lineage cardinalities are changed and the protected scientific authorities above remain untouched.
5. Re-run Repo CI and the zero-credit paid-artifact replay on the reconciled PR head.
6. If those checks remain clean, merge PR #523 as a mechanical production repair. Do not bundle unrelated research or science changes into the merge.
7. After merge, run Repo CI and a no-paid/offline verification from merged `main`. Reuse the preserved paid artifact where needed; no paid fetch is necessary merely to prove the patch.
8. A future fresh paid live run, if desired, should be a deliberate production-validation decision after merge—not a debugging mechanism.
9. When the production incident is formally closed, update `NFL_MASTER_CONTINUITY_RECORD.md` and then resume the parked QB/WR research lane.

## Current status

`original Step-25 incident FIXED -> remaining-event roster scope PASS -> modeled-core identities 0 unresolved -> rematch scope PASS -> kickoff-anchor fail-close PASS -> Knight live alias boundary PASS -> green no-live football/model stack (live Steps 25-31 intentionally skipped) -> exact pre-live diff proves no science mutation -> one paid live run executed -> live Steps 25-28 PASS -> Step 29 exposed stale all-32 certification rule -> PR #523 repairs current-availability certification chain without changing frozen model methodology -> exact paid artifact replay passes Steps 29-31 + strict audits with 0 blockers and no refetch -> PR #523 remains OPEN/UNMERGED and is now diverged from current main; reconcile + revalidate before merge`

## Parked science lane

Preserved at `docs/handoffs/NFL_HANDOFF_2026-09-11_QB_WR_SHARED_OPPORTUNITY_CURRENT.md`. Resume only after the live incident is closed.

## Resume rule for the next chat

Do not restart the preserved-artifact replay, roster investigation, identity audit, rematch investigation, Knight alias investigation, M108 search, or paid live fetch. Verify current `main`, read this exact checkpoint, reconcile PR #523 against current `main`, rerun its zero-credit validation, and continue from the **merge-integrity decision**, not from the original Step-29 diagnosis.
