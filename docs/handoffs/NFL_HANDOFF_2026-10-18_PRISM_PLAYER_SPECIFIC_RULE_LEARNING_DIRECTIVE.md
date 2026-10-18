# NFL PRISM ELITE CONTINUITY HANDOFF — 2026-10-18
## PLAYER-SPECIFIC RULE LEARNING DIRECTIVE

Repository: `dkaps6/imtiredofthis`

Model: **PRISM — Player Role & Individual Situation Model**

This handoff exists so the next ChatGPT session can continue as the same research lead without asking the user to restate philosophy, completed work, stopping rules, or current state.

---

## 1. THE MOST IMPORTANT NEW DIRECTIVE

The user explicitly reframed the next major PRISM architecture:

> **The player should have the rules. The position should not have the player.**

PRISM must move beyond assuming that all RBs, WRs, TEs, or QBs respond to the same features with the same weights.

The scientific object is the **named player in his current football situation**.

A player's position remains useful as:
- a prior;
- a sparse-data fallback;
- an archetype/role family;
- a calibration/QA grouping.

It must **not** permanently dictate the player's projection rule set.

The intended hierarchy is:

`LEAGUE PRIOR -> ROLE/ARCHETYPE PRIOR -> INDIVIDUAL PLAYER EVIDENCE -> CURRENT PLAYER STATE`

As same-player evidence becomes sufficient and stable, the player's own learned sensitivities should be allowed to dominate the broader prior.

This directive is already recorded canonically in Issue #535 comment **6245756869**:
`PRISM NEXT-PHASE DIRECTIVE — PLAYER-SPECIFIC RULE LEARNING`.

### What this means scientifically

Do not ask only:
- "Does this lower RB MAE?"
- "Does this mechanism help WRs?"
- "What is the best TE-wide weight?"

Instead ask:
- "For this named player, which mechanisms have historically mattered?"
- "Which features are stable predictive drivers for him?"
- "Which mechanisms should be active for him this week?"
- "How strongly should his own evidence override the role/archetype prior?"
- "Does the answer differ by market: carries, targets, receptions, yards, touchdowns?"

A signal may be:
- strong for one subset of players;
- irrelevant for another subset;
- actively harmful for another subset.

A flat pooled position metric does **not** automatically mean the mechanism contains no player-level value.

Position-level MAE/RMSE remains the scoreboard / downstream QA. It is not the modeling target.

### Anti-overfit requirement

Do **not** fit an unconstrained bespoke model for every player from tiny samples.

Individualization must be hierarchical and earned.

Potential architecture:
1. player-specific mechanism eligibility/gating;
2. player-specific feature weights or sensitivities;
3. shrinkage toward role/archetype/position priors when individual evidence is thin;
4. market-specific personalization;
5. rolling strictly-prior learning only;
6. minimum-support and stability requirements before deviation from prior;
7. equal-player / player-cluster / named-player OOS evaluation;
8. no sportsbook information upstream in football projection learning.

Stars must not be allowed to drag pooled MAE lower while many individual players worsen.

---

## 2. NON-NEGOTIABLE PRISM PHILOSOPHY

PRISM is an **INDIVIDUAL PLAYER PROJECTION MODEL**.

Canonical reasoning order:

`PLAYER IDENTITY/HISTORY -> CURRENT ROLE/STATE/TRAJECTORY -> TEAMMATE COMPETITION -> FINITE OPPORTUNITY ALLOCATION -> OPPORTUNITY TYPE/QUALITY -> TEAM/GAME ENVIRONMENT -> PLAYER-CONDITIONED OPPONENT MATCHUP -> INDIVIDUAL EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`

The prior "Construct NFL Tickets" J.K. Dobbins discussion is binding conceptual context.

For a named player, understand:
- his history;
- his current role and trajectory;
- teammate competition;
- finite team opportunity;
- opportunity type/quality;
- scoring-area role where relevant;
- team/game context;
- opponent interaction with **his** actual role;
- his individual efficiency;
- projection reliability;
- the exact requested market.

Carries, targets, receptions, yards and TDs are different football questions.

A correct final projection can hide incorrect internal component reasoning. Component cancellation has already been proven in replay diagnostics.

---

## 3. LIVE REPOSITORY STATE AT HANDOFF

Before this handoff write:

- `main`: `3bfa7fea55e84cf3284e4e4cebb90f85b1a0955e`
  - message: `Docs: finalize PRISM next-chat continuity prompt`
- research branch:
  `research-individual-opportunity-roadmap-2026-10-08`
- branch head:
  `cfbf688447b4747230d237fb20f2d2e213775d63`
- branch head message:
  `Research: record RB short-yardage carry-type V1 close`
- PR #672:
  - OPEN
  - DRAFT
  - head = research branch above
  - exact head = `cfbf688447b4747230d237fb20f2d2e213775d63`
  - base = `main`
  - GitHub currently reports mergeable = false
- Issue #535 is the primary research ledger.
- Latest directive comment at handoff:
  **6245756869**
- No commit-status checks or pull-request-triggered workflow runs were returned for exact branch head `cfbf688...`; the scientific run itself completed earlier and is recorded below.

Always reconcile live state again before writing because another session/tool may have advanced GitHub after this handoff.

---

## 4. MOST RECENT COMPLETED SCIENCE

### A. RB short-yardage carry-type identity V1 — CLOSED

Result:
`docs/research/PRISM_RB_SHORT_YARDAGE_CARRY_TYPE_IDENTITY_V1_RESULT.md`

Canonical signal run:
- run `39097257642` — SUCCESS
- exact scored head `c366ef479b66ab4ee99907964748430002904351`
- job `117586172786`
- artifact `12125077660`
- digest `sha256:56c5eb81a7f90f0c401baa53883852026771eb859e6d1617dbea6f08bd091bc3`
- result-doc commit `cfbf688447b4747230d237fb20f2d2e213775d63`
- Issue #535 comment `6245671847`

Disposition:
`RB_SHORT_YARDAGE_CARRY_TYPE_IDENTITY_CLOSED`

Frozen question:
conditional on the named RB receiving a carry, does his own strictly-prior same-team 3rd/4th-and-1/2 carry-type history improve over the strictly-prior team-RB carry-type baseline?

2024 development passed both frozen development gates.

2025 confirmation:
- carry Brier gain +0.00055954
- carry log-loss gain +0.00279814
- conditional-game Brier gain +0.00687777
- conditional-game log-loss gain +0.01992709
- carry player-cluster CI [+0.00003286,+0.00121715]
- carry game-cluster CI [+0.00022467,+0.00091296]
- carry equal-player CI [+0.00042213,+0.00713377]
- conditional-game game-cluster CI [+0.00035029,+0.01363806]
- **conditional-game player-cluster CI [-0.00139111,+0.01590455] — FAIL**

Ten of eleven frozen gates passed. The single player-stability failure is binding.

Do NOT rescue with:
- different shrinkage;
- support floors;
- recent windows;
- second-down/goal-line variants;
- lead-back carveouts;
- opponent interactions;
- yardage/TD translation;
- sportsbook;
- 2026 outcomes.

Important conceptual lesson:
this is a perfect example of why pooled/average improvement is not enough. The player-level stability gate mattered.

### B. Player high-value conversion identity V1 — CLOSED ALL

Issue #535 comment `6245259482`.

Source run:
- `39093918886`
- exact head `5b517c1ac71a1f67d24cddf510c0f7ef5fe191b9`
- disposition `PRISM_PLAYER_HIGH_VALUE_CONVERSION_IDENTITY_SOURCE_READY_PARTIAL`

Signal run:
- `39094077341`
- exact head `659455070c5a9b6d61ae5f23141672fede1ed30f`
- disposition `PRISM_PLAYER_HIGH_VALUE_CONVERSION_IDENTITY_CLOSED_ALL`

RB rush inside 10:
- source ready;
- direct same-player conversion identity worsened prediction.

TE receiving red zone:
- source ready;
- direct same-player conversion identity worsened prediction.

RB receiving red zone:
- source blocked under frozen support floor; do not lower gate or pool.

Do NOT encode a generic persistent "goal-line finisher" or "red-zone converter" scalar.

This does **not** mean TDs are player-agnostic. It means player-specific TD value must be constructed from current role, teammate competition, finite opportunity, opportunity type/quality, team scoring creation and valid player-conditioned context — not generic historical conversion memory.

Generic ATD remains uncertified.

---

## 5. IMPORTANT RECENT CLOSURES / BLOCKS — DO NOT REOPEN

### QB identity / pass-catcher opportunity quality

Same-QB target-share adjustment:
- source ready;
- predictive signal FAILED CLOSED;
- run `39067914161`.

Target-depth by QB regime:
- source ready;
- signal CLOSED.

Target-CPOE by QB regime:
- source ready;
- signal CLOSED;
- run `39088048507`.

Do not attempt a third cosmetic QB-pair target-quality variant.

### RB NGS rushing efficiency identity V1

Run `39068580334`.

Disposition:
`PRISM_RB_NGS_RUSHING_EFFICIENCY_IDENTITY_SOURCE_BLOCKED`.

Provider/source quality was acceptable but frozen same-player support was insufficient.

Do not lower gates or pool seasons after seeing support.

### Week-7 Stage-2 market-translation readiness

Reprobe run `39093471632`.

Disposition:
`PRISM_FUTURE_STAGE2_TRANSLATION_NOT_READY__FAIL_CLOSED`.

Week-6 prior-state sources were incomplete for most target-week teams. Only DEN/SEA were present at that point.

This is readiness-only. No Week-6 grading, no scientific lock, no fit, no sportsbook/OddsAPI, no production mutation.

Recheck only when complete legal Week-6 prior-state data exist.

### Other protected closures

Do not reopen:
- raw recent-share replacement;
- Room Competition State;
- RB/TE depth-role transition;
- generic defense-vs-position;
- M95A/M95B generic RB run-defense routes;
- RB run-location matchup V1;
- FMT RB/WR/TE integration candidates;
- WR target-depth / receiver CPOE / catchability / early-read target-quality chain;
- RB YPR/YPT/YAC/xYAC/YACOE receiving-efficiency mean transforms;
- TE target-quality efficiency V1;
- player mechanism-instability trust V1;
- generic efficiency-volatility/tail-stability V1;
- generic same-player high-value TD-conversion persistence;
- RB short-yardage carry-type identity V1.

Do not relabel old closed mechanisms as "new" just because they are now discussed in a player-specific framework.

A previously closed mechanism can only be reconsidered if the new hypothesis is materially different, legally sourced, frozen before scoring, and genuinely player-conditioned rather than a parameter rescue.

---

## 6. EXISTING QUALIFIED / PROTECTED SCIENCE

### Extended Week-4/5 player-first replay

Run `39060246734` — SUCCESS:
- full Week-4 authority preserved = 873 rows;
- supported Week-4 player markets = 483;
- all Week-5 rows = 396;
- casebook = 80;
- matched controls = 80;
- no Week-6 outcomes / OddsAPI / fit / production mutation.

Weeks 1-3 legitimately feed Week 4.
Weeks 1-4 feed Week 5.
2025 same-player history is identity context where source-valid.

Do not rerun.

### Component cancellation

Run `39061107622`.

Small final error can hide large opposing component errors.

Therefore do not treat a correct projection as proof the named-player reasoning was correct.

### Finite-allocation architecture

Conservation stays.

The problem is not conservation itself. The target is **jointly coherent named-player entitlement before conservation**.

Qualified Stage-2 cells:
- RB rushing;
- RB receiving;
- TE receiving.

QB exact Stage-2 CLOSED.
WR Stage-2 source-blocked.

### Player target-share trajectory

WR/TE player target-share trajectory is a completed player-share success.
Run `37638269235`.
Disposition:
`PLAYER_TARGET_SHARE_TRAJECTORY_SIGNAL_CONFIRMED`.

Do not replace it with generic recent-share logic.

### Reliability

WR opportunity-difficulty trust qualified.
WR efficiency-difficulty trust qualified.
WR dual reliability historical integration qualified.
Weeks 7-10 WR dual prospective protocol remains protected.

TE allocation-difficulty trust qualified.
TE efficiency-trust V1 CLOSED.

No analogous newly qualified generic RB named-player trust state should be assumed without anti-retest.

---

## 7. WEEK-6 / LIVE PROTECTION RULES

- Week-6 locks remain immutable.
- Do not grade Week 6 early.
- Do not backfill Week-6 WR dual composite.
- Weeks 7-10 WR dual prospective work remains protected.
- No paid OddsAPI without explicit user authorization.
- Sportsbook information stays downstream only.
- No production mutation from research unless separately authorized.
- Do not call source readiness predictive proof.
- Fail closed instead of weakening gates.
- Freeze one exact mechanism before scoring.

---

## 8. THE NEXT MAJOR RESEARCH FRONTIER

Unless GitHub contains a newer canonical instruction after Issue #535 comment `6245756869`, the next research phase is:

# PLAYER-SPECIFIC RULE / WEIGHT LEARNING

This should be treated as an **architecture problem**, not "let's rerun every failed feature per player."

### Research question

Can PRISM determine, using only strictly-prior legal football evidence, **which mechanisms matter for each named player's projection and how much**, while shrinking safely toward broader priors when individual evidence is insufficient?

### Recommended first architecture investigation

Start source/lineage/anti-retest only. Do not immediately fit a huge model.

Map the existing PRISM inputs/mechanisms into a candidate personalization taxonomy, for example:
- player identity/history;
- role/usage trajectory;
- teammate competition;
- allocation/entitlement;
- opportunity type/quality;
- team/game environment;
- player-conditioned matchup;
- efficiency;
- reliability.

For each market, determine:
1. which mechanisms already have player-level repeated history;
2. which can be scored strictly prior on an untouched historical target;
3. which are source-blocked;
4. which already failed in a way that forbids simple personalization rescue;
5. what sample exists per named player;
6. what shrinkage hierarchy is scientifically defensible.

### Do not begin with a free coefficient per player

A safer initial family is hierarchical:
- global/market prior;
- role/archetype prior;
- player deviation from prior only when support/stability is sufficient;
- current-state override where football semantics require it.

Possible outputs are not necessarily one continuous weight vector.
Player-specific **mechanism eligibility** may be more robust than fully free weights.

Example:
- one player may earn a stable workload/teammate-sensitive adjustment;
- another may stay mostly on role prior because his individual evidence is sparse;
- another may earn a matchup-sensitive adjustment only for a specific market.

### Required evaluation

A candidate cannot qualify solely on pooled position MAE.

Require player-aware evidence such as:
- equal-player average change;
- player-cluster CIs;
- share of named players improved;
- median named-player change;
- support/stability by player;
- tail-harm analysis;
- current-state consistency;
- market-specific OOS performance.

Also retain aggregate market/position metrics as guardrails.

### Key design principle

Player individuality does **not** mean ignoring population information.

The architecture should answer:
"How much evidence has this player earned to deviate from his role/archetype prior for this exact market?"

That is the bridge between football realism and statistical discipline.

---

## 9. EXACT NEXT-CHAT STARTUP ORDER

1. Read `AGENTS.md` in full.
2. Read ONLY the newest TOP checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`.
3. Read this handoff:
   `docs/handoffs/NFL_HANDOFF_2026-10-18_PRISM_PLAYER_SPECIFIC_RULE_LEARNING_DIRECTIVE.md`.
4. Read newest relevant Issue #535 comments, starting with:
   - `6245259482` high-value conversion close;
   - `6245671847` RB short-yardage close;
   - `6245756869` player-specific-rule directive;
   then any comments newer than `6245756869`.
5. Reconcile:
   - live `main`;
   - research branch `research-individual-opportunity-roadmap-2026-10-08`;
   - PR #672;
   - exact-head Actions/status.
6. If no newer science supersedes this handoff, begin the player-specific architecture **with anti-retest/source mapping first**.
7. Do not ask the user to restate the project.
8. Continue autonomously.

---

## 10. USER EXPECTATION / WORKING STYLE

The user wants the research lead to keep thinking and moving, not wait to be told every next idea.

They explicitly said they are glad they keep thinking outside the box and want the assistant to do the same.

That means:
- do not get trapped in one modeling lane;
- periodically challenge the current abstraction;
- search for structurally different ways to solve named-player projection error;
- still preserve scientific stopping rules;
- do not "innovate" by retuning a failed mechanism after looking at results.

The balance is:
**creative architecture + strict experimental discipline**.

The user does not want long planning monologues. Do the work, give concise substantive updates, and checkpoint meaningful science in GitHub.

---

## 11. FINAL ONE-SENTENCE CONTINUITY

PRISM has finished the current direct player-identity mechanism sequence without a new promotion, and the next major step is to research a hierarchical, market-specific architecture where **individual named players can earn their own mechanism eligibility/weights instead of inheriting one fixed position-wide rule set**, while preserving strict-prior evidence, shrinkage, finite opportunity, player-level OOS gates, protected Week-6/WR prospective locks, and sportsbook separation.
