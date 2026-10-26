# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## ARCHIVED CHECKPOINT (SUPERSEDED) — preserved verbatim from the prior active root checkpoint

**Model:** PRISM — Player Role & Individual Situation Model

### READ THIS DEEP HANDOFF NEXT

docs/handoffs/NFL_HANDOFF_2026-10-26_V4_IDENTITY_REAUDIT_BLOCKERS_PLAYER_FIRST_CONTINUITY.md

Restart prompt:

docs/handoffs/NEXT_CHAT_PROMPT_2026-10-26_V4_IDENTITY_REAUDIT_BLOCKERS_PLAYER_FIRST_CONTINUITY.md

Displaced checkpoints are archives only, not resume points:
- docs/handoffs/NFL_HANDOFF_2026-10-26_V4_ARCHIVED_TOP_IDENTITY_PRIORITIES_DONE_REAUDIT_PENDING.md
- docs/handoffs/NFL_HANDOFF_2026-10-26_V3_ARCHIVED_TOP_IDENTITY_CARD_ACCEPTED_SUBSTRATE.md
- docs/handoffs/NFL_HANDOFF_2026-10-26_V2_ARCHIVED_TOP_WEEK7_FORENSIC_PLAYER_FIRST_TRANSMISSION.md

Do not recursively read older dated handoffs.

### ABSOLUTE ASSIGNMENT

PRISM is an **INDIVIDUAL-PLAYER PROJECTION MODEL**.

> **THE PLAYER SHOULD HAVE THE RULES. THE POSITION SHOULD NOT HAVE THE PLAYER.**

> **ANOMALIES ARE QUESTIONS, NOT TRENDS.**

Position MAE/RMSE is downstream QA only. Identity/state/opponent work is additive. Do not scrap/rebuild PRISM or reopen frozen science without new causal evidence.

### CONTROL ROOM

- Claude = executor / research lead
- GPT-5.6 = independent auditor / project control
- shared lane = Issue #535

Before substantive work: read AGENTS.md; read only this TOP checkpoint; read the V4 deep handoff; fetch newest #535; reconcile live main / active research branch / PRs / exact-head Actions.

### LIVE STATE AT V4 HANDOFF

Pre-handoff main:
a6f296234ecf9db50d30b82e43587e2c5e9b8482

This V4 handoff is docs-only and advances main from that parent. Re-fetch the live SHA before acting.

Active research branch:
research-prism-player-first-identity-environment-2026-10-26 @ aa70663cae1b65b32687b2809c57326d66093949

Pre-handoff compare:
- diverged
- ahead 30
- behind 2
- merge base c4de3d4044e828e39d80aa7db19af0de2caf652a

The V4 docs commit makes the research branch one additional docs commit behind unless rebased/ported.

Newest #535 comment observed:
6426366490

There was **NO Claude response after 6426366490** at the moment this checkpoint was written.

Exact-head pre-handoff Actions:
- Full Slate 40392535113 — SUCCESS
- Archive Market Track Record 40393072600 — SUCCESS
- Repo CI 40392535296 — FAILURE at unit tests; known stale no-credit FETCH_LIVE_ODDS exact-string debt remains the expected cause unless a later exact-head audit proves otherwise. **Do not weaken the live-odds guard.**

**No paid OddsAPI call is authorized. Betting remains CLOSED.**

### GOVERNING PLAYER IDENTITY DISPOSITIONS

Original descriptive substrate disposition from GPT 6425282770:

PLAYER_IDENTITY_CARD_V1_DESCRIPTIVE_SUBSTRATE_ACCEPTED__NO_PREDICTIVE_CLEARANCE

That remains true.

Claude later presented revised target aa70663c in 6426050333 after:
- injury-context resolver;
- fail-closed PARTIAL_GAME source blocking;
- week-matched team-share work;
- threshold sensitivity sweep;
- explicit model-facing opportunity universe;
- evidence_authority / population-fallback visibility.

GPT re-audited the exact implementation in 6426307861 and 6426366490.

Current revised-target disposition:

PLAYER_IDENTITY_CARD_V1_REVISED_TARGET_BLOCKED_ON_NO_BASELINE_CONTAMINATION_AND_TRADE_SHARE_SEMANTICS

The revised target is **NOT cleared**.

### BLOCKER A — NO_BASELINE INJURY CONTAMINATION

Current aa70663c ordering can mark an early/sparse player NO_BASELINE and then treat NO_BASELINE as usable before the injury resolver can withhold a known limiting injury or return-from-injury state.

Required:
- baseline insufficiency cannot override independent source-qualified contamination;
- known limiting injury / qualified return can force WITHHELD even with fewer than MIN_BASELINE_WEEKS;
- NO_BASELINE means insufficient baseline and no independent contamination.

Required tests:
1. first/second participation week + limiting injury => withheld;
2. injury-explained return + insufficient baseline => withheld;
3. sparse player with no contamination => remains NO_BASELINE.

### BLOCKER B — TEAM SHARE IS NOT TRADE-SAFE

The revised week-matched share still uses target-week team_now against a player current-season opportunity frame that can contain old-team opportunities.

Required semantic:
**target-team-tenure share**.

Only prior player-weeks where roster authority says player team == target-week team may enter current-team numerator/denominator.

Newly acquired player with no prior accepted target-team week => NaN / explicit no-current-team-evidence.

Apply to:
- team_share_current;
- red-zone share;
- inside-5 share;
- the all-weeks current-team comparison fields.

Add an in-season trade regression test proving old-team workload cannot leak into new-team share.

### MATERIALITY CORRECTION

The reported 5,422 / 10,960 = 49.5% position_prior_only rate across opportunity-eligible card-weeks is **not yet accepted as current-production severity**.

Required outcome-blind decomposition:
- by target_week;
- by opportunity_kind;
- by roster_position × opportunity_kind;
- Week-7 slice;
- split position_prior_only into no_history vs participation_only_zero_opportunity;
- identify position-kind rows that actually enter production allocation;
- show production prior value / whether fallback can move finite allocation;
- preferably quantify implied fallback opportunity mass entering the finite room.

Do not delete rare-use rows or hard-code away unusual players.

### REVISED TARGET DETAILS

Research head:
aa70663cae1b65b32687b2809c57326d66093949

Claude-reported green runs:
- card 40391557775
- sweep 40391557902
- 68 tests under pandas 1.5.3 / numpy 1.26.4

Threshold sweep remains accepted only as descriptive uncertainty:
- ratios 0.30 / 0.40 / 0.50 / 0.60 / 0.70
- spread 362 cards = 13.66%
- MODERATELY_THRESHOLD_SENSITIVE
- no winner
- no outcomes inspected
- do not choose 0.50 simply because it is current

Injury resolver direction is accepted as conservative descriptive resolution:
- UNRESOLVED 1,358 -> 1,078
- RETURN_FROM_INJURY 213
- INJURY_LIMITED 67
- explained abnormal weeks remain WITHHELD
- PARTIAL_GAME remains SOURCE BLOCKED for 2026

### NEXT EXACT ACTION

First check Issue #535 for a Claude response after 6426366490.

If Claude has responded:
1. audit the exact new head and exact diff;
2. verify Blocker A and all three regression tests;
3. verify target-team-tenure semantics and trade regression;
4. verify the requested authority/materiality decomposition;
5. verify branch/main reconciliation and preserved audit lineage;
6. do not clear predictive use unless all are correct.

If Claude has not responded:
- do not redo Walker / JSN / Kupp / Barner / zero-history audits;
- do not score the card;
- keep project control focused on the assigned blockers.

### ENVIRONMENT TRANSMISSION REMAINS OPEN

The Identity Card does not solve opponent/environment transmission.

Durable audits:
- Walker: docs/research/PRISM_WEEK7_WALKER_ENVIRONMENT_TRANSMISSION_DECOMPOSITION_V1.md @ cca416bd
- JSN/Kupp/Barner: docs/research/PRISM_WEEK7_SEA_RECEIVERS_VS_KC_ENVIRONMENT_TRANSMISSION_V1.md @ 87dcfd9
- zero-history allocation: docs/research/PRISM_WEEK7_ZERO_HISTORY_POSITION_PRIOR_ALLOCATION_AUDIT_V1.md @ aef0692

Core Walker result: opponent-caused rushing opportunity/efficiency/final-mean/sigma movement was 0 for the exact Week-7 SEA case because rich rush-defense context was stranded behind coarse generic thresholds.

Receiving has coarse opponent opportunity transmission and finite-room coupling, but rich efficiency context is still largely stranded.

Generic pass share remains hardcoded at 0.57; lead/trail does not change the generic split.

### STAGE-2

Week-7 prospective Stage-2 readiness is permanently fail-closed by chronology:

PRISM_FUTURE_STAGE2_TRANSLATION_NOT_READY__FAIL_CLOSED

Do not weaken/filter the chronology guard, reconstruct Week 7 postgame, or call Week 7 prospectively ready.

Next Stage-2 target must be genuinely future and frozen before target-week rows exist.

### WEEK-7 BETTING FORENSIC

Accepted: primary ticket failure was construction/threshold/exposure.

Still in force:
- alt direction conflicts canonical main-line direction => PASS;
- cushion != safety;
- use mean + SD + exact threshold + P(clear) + price;
- do not call ~41% Safe;
- expose repeated-player/same-game concentration;
- ATD uncertified/separate;
- no universal anchor.

This does not close environment transmission.

### SCIENCE BOUNDARIES TO PRESERVE

- QB M89/M90 + C2 active/frozen
- WR-R15 active
- TE-R5P active
- RB P3/R22 Week-1-only, not Week-7 authority
- Stage-2 qualified research, not production-wired
- direct WR/CB source gated/unavailable
- ATD uncertified
- RB rushing Stage-1 SOURCE BLOCKED
- RB/TE team-context reset candidates PREDICTIVE CLOSED
- failed SLOT carry candidate stays closed
- no aggregate-MAE optimization
- no Week-7 fitting
- no sportsbook upstream

The next chat MUST re-fetch #535 and live GitHub before asserting this remains current.
