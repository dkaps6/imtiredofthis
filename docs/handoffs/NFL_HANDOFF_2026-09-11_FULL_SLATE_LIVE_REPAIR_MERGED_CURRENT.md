# NFL Production Handoff — 2026-09-11 — Week 1 Full Slate Live Repair — MERGED CURRENT

## Canonical checkpoint

GitHub is canonical; chat memory is secondary.

The Week-1 live Full Slate mechanical incident is now repaired and merged. Do **not** restart the preserved-artifact replay, player-identity audit, rematch audit, Knight alias audit, or broad M107/M108 search. Do **not** redesign or retune production model science while continuing from this checkpoint.

Production repair merge:

- PR #523 `Scope Step 29 roster quality gate to live events`
- validated PR head: `723e304291ee1dcf40f5a15df50585cbffcd4471`
- merged to `main`: `f84c6242da02b1804b4b9675c3a3a3e679838e10`
- merge status: SUCCESS

Post-merge validation on the exact merge SHA:

- Repo CI run `34656484043`: **SUCCESS**
- no-live Full Slate run `34656483980`, job `103449935779`: **SUCCESS**
- football/model Steps 1-24: PASS
- live sportsbook Steps 25-31: skipped in this post-merge push run because live acquisition was disabled
- strict repository audit: PASS
- no paid OddsAPI fetch was performed by these post-merge checks

## Integrity verdict — model methodology remains intact

The paid-run failure was not caused by degradation or mutation of the football model.

The last green no-live baseline before the paid run was SHA `1f7255805647e978adddb2cf4aecf2883f0dda48`. The controlled paid-run SHA `be061eaf23372f080db3911d3b4919120c744c53` differed from that baseline only by the authorized live-dispatch workflow marker. No QB, WR, RB, TE, projection, distribution, entitlement, simulation, pricing-science, fitted-model asset, or learned parameter changed between those two runs.

PR #523 itself changes only workflow/replay, certification, lineage-stamping, current-availability governance seams, and tests. It does **not** change core football model-generation formulas, fitted assets, weights, learned parameters, opportunity/entitlement formulas, simulation formulas, or sportsbook-to-football isolation.

Frozen production authorities remain:

- QB mean: M89/M90 / `QB_PASS_SYNTHESIS_V1`
- QB distribution: `C2_QB_MEAN_NEUTRAL_DISTRIBUTION_V1`
- WR: M38 WR1 hierarchy + `WR_R15_PRODUCTION_MODEL_V1`
- TE: `TE_R5P_PRODUCTION_MODEL_V1`
- RB rushing: `RB_P3_SYNTHESIS_V1` for the qualified Week-1 route
- RB receptions/opportunity: R26
- RB receiving-yard tail/distribution: R22 using frozen R19 assets

Architecture remains football-first:

`GAME STATE -> TEAM OPPORTUNITY -> POSITION/ROOM POOL -> PLAYER ENTITLEMENT -> PLAYER EFFICIENCY -> JOINT MONTE CARLO -> PLAYER PROJECTIONS/DISTRIBUTIONS -> FAIR PROBABILITIES -> SPORTSBOOK COMPARISON`

Sportsbook/live-market inputs remain downstream only and may not teach upstream football projections.

## Why the prior green Full Slate did not contradict the paid failure

The earlier green Full Slate was a no-live integration run. It executed football/model Steps 1-24, then intentionally skipped live sportsbook Steps 25-31.

The controlled paid run exercised the live path for the first time in this repair sequence and produced:

- paid run `34650067599`
- job `103429972300`
- SHA `be061eaf23372f080db3911d3b4919120c744c53`
- Steps 1-24: PASS
- Step 25 live fetch/gate: PASS
- Step 26 sportsbook boundary: PASS
- Step 27 opponent map: PASS
- Step 28 player-identity semantics: PASS
- Step 29 data-quality classifier: FAIL on a stale all-32 current-roster assertion

That failure was a newly exercised downstream certification-scope defect, not a football-model regression.

## Preserved paid artifact authority

The exact paid artifact from run `34650067599` remains the authority for zero-credit downstream replay:

- artifact ID `10283817522`
- artifact name `run_34650067599`
- SHA256 `a70c90023632059476cc688070fb21ad63b9ebd14aa4db102844b9e342188359`
- active sportsbook events: 14
- full Week-1 scheduled games: 16
- current live event-team universe: 28 teams
- already-played teams absent from the live roster/event universe: LAR, NE, SEA, SF
- production compact prop rows: 782
- production compact unique players: 344
- unresolved modeled-core identities: 0
- sportsbook/current-roster mismatches: 0

Do not spend another paid live-odds call merely to rediscover mechanical bugs.

## What PR #523 repaired

The repair is scope/certification plumbing only:

1. Step-29 roster quality now requires every **current live-event team** to be covered by Ourlads while preserving the authoritative 32-team Week-1 schedule. Missing a current live-event team still fails closed; a full 32-team roster remains valid.
2. Injury scope is certified only when evidence proves the full scheduled-team scope; incomplete scope remains a blocker.
3. Downstream QB C2, WR-R15, R22, target-pool, and certified-stack coverage audits reconcile to the explicit current eligible football universe instead of stale historical hard-coded cardinalities.
4. Protected science/version fragments remain guarded; the availability transformer raises if those protected scientific contracts change.
5. Pricing provenance is stamped to the real paid source run rather than an obsolete historical run ID.

No production model science was retuned.

## Exact zero-credit replay — PASS

Before merge, the exact paid artifact was replayed on the validated PR head with no OddsAPI acquisition:

- Repo CI run `34655125555`: PASS
- offline replay run `34655125574`
- replay job `103445740378`
- replay head `723e304291ee1dcf40f5a15df50585cbffcd4471`
- `FETCH_LIVE_ODDS=false`
- `FULL_SLATE_SOURCE_RUN_ID=34650067599`
- conclusion: **SUCCESS**
- evidence artifact `offline_replay_34655125574`
- evidence artifact ID `10285470181`
- evidence SHA256 `aa4d7a06f8f0fc521a0b19066c9ec691e3185d187c6ea91359ebdea99ebd75a9`

Replay certification results:

- Step 29 certification blockers: **0**
- schedule: 32 teams / 16 games certified
- current roster/live event scope: 405 roster rows / 28 teams
- injury evidence: 167 rows / 32 of 32 scheduled teams -> `CERTIFIED_REPORT_SCOPE`
- exact pricing book-line rows: 1,524
- priced side rows: 3,048
- priced players: 344
- football players: 405
- football teams: 28
- canonical current games: 14
- sportsbook rows used to define football player universe: 0
- WR-R15 current WR1 anchors: 28
- R22 current adapted RBs: 82
- strict repository/readiness audits: PASS
- durable provenance: `source_run=34650067599`
- `odds_api_refetched=false`

This replay proved the repaired Steps 29-31 and downstream certification against the exact sportsbook artifact acquired by the paid live run.

## Current production conclusion

The original Step-25 incident is CLOSED. The stale Step-29/current-availability downstream certification defect is also CLOSED and merged.

The combined evidence is:

- merged current football/model tree passes no-live Steps 1-24;
- the controlled paid run passed live Steps 25-28;
- the exact artifact from that paid run passes repaired Steps 29-31 and strict audits with zero blockers;
- post-merge Repo CI and no-live Full Slate are green.

A second paid live run is **not required for debugging or mechanical certification**. A future paid fetch should be performed only when a genuinely fresh sportsbook snapshot is desired for production output/decision use, not to prove this already-closed repair.

## M107 / M108 correction

Do not claim `M108 26/26` or M107 thresholds as verified canonical GitHub gates unless concrete repository lineage is recovered. The prior inherited label was unsupported by canonical GitHub evidence. This correction is documentation/provenance only and does not change model science.

## Exact next step

1. Verify current `main` and inspect only commits after merge SHA `f84c6242da02b1804b4b9675c3a3a3e679838e10`; do not restart completed repair history.
2. Treat the Week-1 live mechanical incident as closed unless a new concrete production failure appears.
3. If fresh live market output is needed, make that a deliberate production-data acquisition decision; do not use paid acquisition as a debugging probe.
4. Resume the parked QB/WR shared-opportunity / first-down pass-propensity / public-pregame-intent V1B research lane only after confirming no newer production incident supersedes this checkpoint.

Parked science handoff:

- `docs/handoffs/NFL_HANDOFF_2026-09-11_QB_WR_SHARED_OPPORTUNITY_CURRENT.md`

Historical detailed live-repair handoff retained at:

- `docs/handoffs/NFL_HANDOFF_2026-09-11_FULL_SLATE_LIVE_REPAIR_CURRENT.md`
