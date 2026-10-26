# ARCHIVED HANDOFF CHECKPOINT — 2026-10-26 — V4 — IDENTITY BLOCKERS REQUIRED

**THIS IS NOT THE ACTIVE CHECKPOINT. DO NOT RESUME FROM THIS FILE.**

The active checkpoint is the single TOP CHECKPOINT in root
`CURRENT_NFL_RESEARCH_HANDOFF.md`. Archived per the AGENTS.md continuity rule:
supersede in place, keep the displaced history here, never leave two TOP
checkpoints to disambiguate.

Superseded because its required work is DONE. It names Blocker A, Blocker B
and the fallback materiality decomposition as outstanding and points at branch
head `aa70663c`; all three are complete and the target is now `febb1acf`.
Resuming from this file would redo finished work and audit a stale commit.

Retained verbatim apart from the demoted heading.

---

## ARCHIVED CHECKPOINT (SUPERSEDED) — 2026-10-26 — V4 — IDENTITY CARD REVISED TARGET BLOCKED / TWO PLAYER-FIRST CORRECTNESS FIXES REQUIRED / NO PREDICTIVE CLEARANCE

**Model:** PRISM — Player Role & Individual Situation Model

### READ THIS DEEP HANDOFF NEXT

`docs/handoffs/NFL_HANDOFF_2026-10-26_V4_PLAYER_FIRST_IDENTITY_REAUDIT_BLOCKERS_CONTINUITY.md`

Restart prompt:
`docs/handoffs/NEXT_CHAT_PROMPT_2026-10-26_V4_PLAYER_FIRST_IDENTITY_REAUDIT_BLOCKERS_CONTINUITY.md`

Displaced active checkpoint archived at:
`docs/handoffs/NFL_HANDOFF_2026-10-26_V4_ARCHIVED_TOP_AUDIT_PRIORITIES_DONE_REAUDIT_PENDING.md`

Do not recursively read older handoffs.

### ABSOLUTE ASSIGNMENT

PRISM is an **INDIVIDUAL-PLAYER PROJECTION MODEL**.

> **THE PLAYER SHOULD HAVE THE RULES. THE POSITION SHOULD NOT HAVE THE PLAYER.**

> **ANOMALIES ARE QUESTIONS, NOT TRENDS.**

Position MAE/RMSE is downstream QA only. Identity/state/opponent work is additive; do not scrap/rebuild validated PRISM science.

### CONTROL ROOM

- Claude = executor / research lead
- GPT-5.6 = independent auditor / project control
- shared lane = Issue #535

Before substantive work: read `AGENTS.md`; read only this TOP checkpoint; read the V4 deep handoff; fetch newest #535; reconcile live `main` / PRs / Actions.

### LIVE STATE AT HANDOFF

Pre-handoff `main`: `dc6add89347cd6c177dfac655b755425e393e909`. The V4 handoff commit is a direct child; re-fetch `main` before acting.

Active research branch:
`research-prism-player-first-identity-environment-2026-10-26 @ aa70663cae1b65b32687b2809c57326d66093949`

Exact-head main before handoff:
- Full Slate `40392535113` SUCCESS
- artifact `12685836060`
- digest `sha256:b607ad03cec823b75f080a6fc53852b462e63c41ff3793b688f01122de359da4`
- Repo CI `40392535296` failed only the known stale exact-string no-credit workflow assertion: 660 passed / 1 skipped / 1 failed.
- **Do not weaken the live-odds guard.**
- No paid OddsAPI call authorized.
- Betting CLOSED.

### GOVERNING #535 STATE

Newest governing GPT audits at handoff:
- `6426307861`
- `6426366490`

Prior important Claude response:
- `6426050333` revised fixed target `aa70663c`.

Prior descriptive-substrate acceptance:
- `6425282770`.

Current disposition:

`PLAYER_IDENTITY_CARD_V1_REVISED_TARGET_BLOCKED_ON_NO_BASELINE_CONTAMINATION_AND_TRADE_SHARE_SEMANTICS`

The Identity Card descriptive substrate remains accepted. The revised target is **not** cleared for predictive/scoring use.

### BLOCKER A — NO_BASELINE MUST NOT OVERRIDE KNOWN CONTAMINATION

A sparse-history player can currently receive `NO_BASELINE`, bypass the injury resolver, and become USABLE even when a source-qualified limiting injury/return state exists.

Required:
- same-week limiting injury can force WITHHELD despite insufficient baseline;
- return-from-injury can force WITHHELD despite insufficient baseline;
- NO_BASELINE means insufficient baseline **and no independent contamination signal**;
- add explicit sparse-player tests.

### BLOCKER B — CURRENT-TEAM SHARE MUST BE TRADE-SAFE

Current implementation can mix old-team player opportunity in the numerator with the target/new team's denominator.

Required semantic:
**target-team-tenure share**.

Only use strictly-prior player-weeks where weekly roster authority says player team == target-week team. Numerator and denominator must use those exact same accepted target-team weeks. If no prior accepted target-team week exists, return NaN / explicit no-current-team-evidence. Apply the same rule to RZ, inside-5 and all-weeks current-team comparison fields. Add a synthetic in-season trade regression test.

### REQUIRED FALLBACK MATERIALITY DECOMPOSITION

The raw 5,422 / 10,960 = 49.5% `position_prior_only` opportunity-eligible card-row rate is **not yet an accepted live-production severity rate**.

Before interpreting severity, decompose outcome-blind by:
- target_week;
- opportunity_kind;
- roster_position × opportunity_kind;
- Week-7 slice;
- no_history vs participation_only_zero_opportunity;
- production prior value / whether fallback can move finite allocation;
- implied fallback opportunity mass entering the finite room where possible.

### ACCEPTED REVISED-WORK COMPONENTS

Keep:
- source-qualified injury resolver;
- `PARTIAL_GAME` SOURCE BLOCKED for 2026;
- explained abnormal weeks remain WITHHELD;
- threshold sweep accepted as descriptive uncertainty only;
- `MODERATELY_THRESHOLD_SENSITIVE`;
- 362 / 2,651 = 13.66% usable-evidence sensitivity over 0.30-0.70;
- do not choose a threshold winner;
- complete roster durable card + explicit model-facing QB/RB/FB/HB/WR/TE universe;
- evidence_authority / population-fallback visibility;
- player-level identity bridge;
- 99.80% coverage on opportunity-eligible cards with current-season opportunity.

### OPPONENT / ENVIRONMENT TRANSMISSION REMAINS OPEN

Identity-card work does not resolve:
- Walker rushing opponent transmission = zero in exact W7 generic path;
- rich rush-defense context stranded;
- JSN/Kupp direct KC movement zero;
- Barner coarse coverage-threshold lift;
- finite-room cross-player compression;
- generic `pass_share = 0.57`;
- Stage-2 not production-wired.

Durable docs:
- `docs/research/PRISM_WEEK7_WALKER_ENVIRONMENT_TRANSMISSION_DECOMPOSITION_V1.md`
- `docs/research/PRISM_WEEK7_SEA_RECEIVERS_VS_KC_ENVIRONMENT_TRANSMISSION_V1.md`
- `docs/research/PRISM_WEEK7_ZERO_HISTORY_POSITION_PRIOR_ALLOCATION_AUDIT_V1.md`

### STAGE-2

Week-7 readiness is:
`PRISM_FUTURE_STAGE2_TRANSLATION_NOT_READY__FAIL_CLOSED`

Reason:
`target/future stat rows already present; refuse audit: weeks=[7]`

Do not weaken the guard or reconstruct a postgame Week-7 Stage-2 shadow.

### BETTING CONSTRUCTION — SEPARATE

Week-7 ticket failure was primarily construction:
- ~41-47% naive probability on "safe" tickets;
- 25/57 unique yardage legs bought-down OVER against canonical UNDER direction;
- SD/P(clear) mishandled;
- repeated-player concentration;
- uncertified ATD.

Until evidence-backed Construction V2:
direction conflict => PASS; cushion != safety; require mean + SD + exact threshold + P(clear) + price + exposure; do not label ~41% "Safe"; surface concentration; ATD separate.

### EXACT NEXT ACTIONS

1. re-fetch #535 for Claude work after `6426366490`;
2. if present, independently audit exact diff/code/artifacts/runs;
3. otherwise fix/assign BLOCKER A and BLOCKER B;
4. produce the required authority/materiality decomposition;
5. rebase/port the research lane onto current main without losing the accepted forensic docs;
6. require a new fixed head + exact green runs;
7. no predictive scoring until independent re-audit clears it;
8. after clearance, freeze the first predictive use prospectively before scoring;
9. keep opponent/environment transmission and betting construction as separate lanes.

No paid OddsAPI. No Week-7 fitting. No sportsbook upstream. No aggregate-MAE optimization. No post-hoc subgroup rescue. No reopening closed science without new causal evidence.

The next chat MUST re-fetch #535 and live GitHub before asserting that this remains current.
