# NFL PRISM TRUE FINAL ELITE CONTINUITY — NO RETEACH / INDIVIDUAL PLAYER FIRST / RETURN-STATE FRONTIER

Date: 2026-10-18

Repository: `dkaps6/imtiredofthis`

Model: **PRISM — Player Role & Individual Situation Model**

Purpose: this handoff exists because the conversation hit the limit while the user was trying to ensure continuity. The next chat must behave as the same assistant continuing, but with the assignment fully internalized and without forcing the user to re-teach foundational rules.

---

## 0. READ THIS AS AN OPERATING CONTRACT

The user has repeatedly explained the same central requirement:

> **PRISM is an INDIVIDUAL-PLAYER PROJECTION MODEL.**

The objective is not to make a positional average look good.

The objective is:

> **project this named player correctly, in this game, in his actual current football state, for this exact market.**

Position-level MAE/RMSE is only downstream QA/reporting. It is a consequence of getting individual players right. It is not the optimization target and must never replace player-level football reasoning.

Permanent rule:

> **The player should have the rules. The position should not have the player.**

Correct causal order:

`UNDERSTAND THE PLAYER -> EXPLAIN ABNORMAL OBSERVATIONS -> ESTABLISH CURRENT ROLE/STATE -> MODEL FINITE OPPORTUNITY/TEAMMATE COMPETITION -> MODEL OPPORTUNITY TYPE/QUALITY -> MODEL EFFICIENCY/CONTEXT -> PROJECT THE EXACT MARKET -> EVALUATE AFTERWARD`

Forbidden inversion:

`CHASE A POSITION METRIC -> TUNE PLAYER RULES TO LOWER THE AVERAGE`

Do not make the user teach this again.

---

## 1. PLAYER-FIRST REASONING IS THE PRIMARY SCIENTIFIC OBJECT

Every player projection must begin with the named player.

Before projecting, answer:
- Who is this player?
- What is his historical identity and normal role?
- What is true about his role now?
- Which recent observations are healthy/comparable?
- Which observations are contaminated by injury, early exit, DNP, role transition, QB/starter change, teammate absence, coaching change or unusual game state?
- How much of the team's finite opportunity belongs to him now?
- What type/quality of opportunity does he get?
- What matchup/environment is he entering?
- What does all of that imply for this exact market?

Targets, receptions, receiving yards, carries, rushing yards, passing yards, passing TDs, rushing TDs and receiving TDs are separate market-specific football questions.

Do not infer one automatically from another.

A research mechanism is not ready merely because it might improve a summary metric. It must answer:

> **Why should this specific player's projection change?**

If that cannot be explained at named-player grain in football terms, do not promote it.

---

## 2. PERMANENT ANOMALY RULE

> **ANOMALIES ARE QUESTIONS, NOT TRENDS.**

Before any abrupt stat/usage discontinuity becomes role, trajectory, efficiency, reliability or betting evidence, determine why it happened.

Mandatory context reconciliation when source-valid:
- injury/practice/game designation;
- active/inactive/DNP state;
- offensive snaps;
- routes only if actual route data exist;
- carries/targets/receptions;
- early exit / partial game;
- return from injury;
- QB/starter changes;
- coaching changes;
- teammate availability;
- plausible game-state effects.

State labels may include:
- `HEALTHY_FULL`
- `INJURY_LIMITED`
- `PARTIAL_GAME`
- `DNP_INACTIVE`
- `RETURN_FROM_INJURY`
- `UNRESOLVED_CONTEXT`

Rules:
- DNP is not zero performance.
- Injury-shortened participation is not an ordinary full-game observation.
- Return-from-injury is a distinct current-role state.
- If the reason cannot be resolved, fail closed as uncertain.
- Do not turn unexplained rows into affirmative regression/growth evidence.

The required question is not only `what happened?` but `why did this look different?`.

---

## 3. JALEN COKER — PERMANENT FAILURE CASE

This must remain the canonical example of what PRISM must never do again.

2026 sequence:
- W1: 9 targets / 8 receptions / 138 yards / 2 TD;
- W2: 9 / 8 / 66;
- W3: 4 / 2 / 18 — injury-affected early exit, did not return;
- W4: DNP;
- W5: bye;
- W6: 12 / 6 / 97.

On 2026-10-18:
- market approximately 55.5 receiving yards;
- prior PRISM/chat projection approximately 38 yards;
- prior reasoning invoked regression;
- user reported roughly 60 yards in the first half.

The important failure was not a position metric.

It was a **bad individual-player projection caused by false causal-state interpretation**.

Reading `138 -> 66 -> 18 -> DNP` as ordinary regression was wrong.

Correct interpretation:
- healthy strong role;
- healthy strong role;
- injury-shortened game;
- DNP;
- bye;
- return with 12 targets / 97 yards.

There was no valid healthy-game evidence of role collapse.

A low projection/Under therefore required some other source-supported named-player football reason.

This is not a post-hoc excuse. This context must be incorporated **before** projection.

---

## 4. WHAT THE ASSISTANT DID WRONG — DO NOT REPEAT

The user explicitly lost confidence because the assistant repeatedly drifted away from the assignment.

Failures to avoid:
- treating position-level MAE/RMSE as though it were the goal;
- leading with aggregate positional performance rather than the named player;
- treating player context as post-hoc explanation rather than upstream evidence;
- using contaminated injury/DNP rows as regression evidence;
- briefly promoting QB passing-TD backlog research without user authorization;
- switching lanes because another idea looked interesting;
- letting technical research mechanics obscure the question `do we understand this player?`;
- stopping or pausing work when the user actually asked for continuity / handoff / continued execution.

When the user says `keep going`, continue the exact canonical active lane unless he explicitly changes direction.

Do not ask him to re-explain PRISM's objective.

Do not ask whether player context matters.

Do not ask whether MAE is the goal.

Those answers are already canonical.

---

## 5. QUEUE GOVERNANCE

Betting:
- Sunday betting is LOCKED/CLOSED unless the user explicitly reopens it.
- Do not rebuild today's tickets or parlays unless explicitly asked.

Odds:
- No paid OddsAPI without explicit user authorization.
- Sportsbook is downstream decision/comparison information only.

Research:
- Do not switch research lanes on your own.
- QB passing TD remains BACKLOG ONLY unless explicitly promoted.
- Preserve closed science and anti-retest rules.
- Never weaken byte identity, artifact digest, parity, source-lineage or chronology gates.
- Never claim a workflow is running unless a fresh Actions read shows `queued` or `in_progress`.

---

## 6. PRESERVE QUALIFIED / CLOSED SCIENCE

Do not restart or rescue completed families.

Preserve:
- RB receiving Stage-2 player-specific authority: **QUALIFIED**;
- RB rushing Stage-2 hard authority: **CLOSED**;
- Stage-1 direct RB/TE player-authority bridges: **CLOSED**;
- WR signed opportunity bias: **QUALIFIED**;
- TE signed opportunity bias: **QUALIFIED**;
- WR signed efficiency bias: **CLOSED**;
- TE signed efficiency bias: **CLOSED**.

WR exact authority:
- WR-R15 exact recovery run `39110742338`: **FAILED CLOSED**;
- rebuilt bytes did not match frozen original authority;
- never weaken byte identity;
- never replace the expected digest;
- never score WR bridge from rebuilt bytes.

TE signed opportunity:
- TE-R5P source authority itself is independently preserved;
- frozen bridge required combined WR/TE `READY_ALL`;
- WR fail-close prevented that state;
- no frozen TE-only execution path existed;
- disposition:
  `PRISM_TE_SIGNED_OPPORTUNITY_CONSERVED_BRIDGE_EXECUTION_BLOCKED_BY_FROZEN_COMBINED_SOURCE_GATE`;
- no TE-only rescue.

WR/TE opportunity uncertainty:
- WR confirmed;
- TE confirmed;
- exact historical specialist-distribution integration is source-blocked;
- prospective shadow only if frozen before future target outcomes.

WR/TE component-dominance routing:
- run `39153337388`;
- disposition `PRISM_WR_TE_PLAYER_COMPONENT_DOMINANCE_ROUTING_V1_CLOSED_ALL`;
- do not rescue with alternate dominance ratios, entropy/Gini, windows, pooling or subgroup tricks.

---

## 7. PLAYER-STATE / INJURY LINEAGE

### Observation Health-State V1
- SOURCE READY.
- Run `39154145666`.
- Establishes source-valid injury / no-snap / participation states.

### PlayerForm Participation-State Bridge V1
- SOURCE BLOCKED.
- Run `39160797272`.
- Frozen provider identity reconciliation gate failed.
- Do not weaken it.

### PlayerForm Participation-State Membership V2
- SOURCE READY.
- Run `39160972259`.
- Stable player identity -> current PlayerForm history membership only.

### Comparable-Observation Receiving Opportunity Identity V1
- CLOSED.
- Freeze `43e1c81a177e47118415d02fc072758e22f50825`.
- Run `39161342922`.
- Exact complete-deletion action did not qualify.
- This does NOT mean injury context is irrelevant.
- Do not rescue with partial weighting / reweighting variants of the same action.

### Receiving Opportunity Earning / Participation Exposure V1
- SOURCE READY.
- Freeze `42ff90db6202b783f7e97b73d0b3d629e9566bb3`.
- Run `39161711741`.
- Artifact `12149181200`.
- Digest `sha256:6cae427b667db4a120a1308747d00b82b6770fe0adbc46d3efc68fce3e1db240`.
- Separates participation exposure from target earning.
- Snaps are snaps, never routes.

### Injury-Partial Receiving Exposure Decomposition V1
- CLOSED.
- Freeze `653eb5e4e46a49f0b575c119259d2b11eb845380`.
- Run `39162113968`.
- Disposition `PRISM_INJURY_PARTIAL_RECEIVING_EXPOSURE_DECOMPOSITION_V1_CLOSED`.
- Do not rescue with another snap multiplier/window/position coefficient/reweighting.

### Return-from-Injury Role Restoration V1
- **SOURCE READY**.
- Freeze `3eafb971c0a842750fa5af31a77a8b17adf890b0`.
- Canonical source run `39162437467` — SUCCESS.
- Exact source-run head `4e5edc98c95f25800f6f66633443aa79379a2944`.
- Job `117778377817` — SUCCESS.
- Artifact `12149417234`.
- Digest `sha256:7d69111925b7ad48f16b07150f5e5fbfac5e592bd01430d03efd907ca778c51e`.
- Result commit on research branch `a69386569a441176254ecc559cc7264612e34c26`.
- Disposition `PRISM_RETURN_FROM_INJURY_ROLE_RESTORATION_V1_SOURCE_READY`.

2024:
- 309 episodes / 240 named players;
- restored: 162 episodes / 146 players;
- still reduced: 46 / 42;
- no return: 101 / 100.

2025:
- 342 episodes / 257 named players;
- restored: 187 / 154;
- still reduced: 44 / 38;
- no return: 111 / 111.

No target/carry/reception/yard/TD predictive relationship has yet been scored from return state.

---

## 8. LIVE GITHUB STATE AT THIS HANDOFF

Reconciled immediately before this handoff package:

Main:
- branch `main`;
- head before this package: `47efd7e18e430940d0f55baef1bfb9c08be61f17`;
- message: `Docs: promote final conversation-limit PRISM player-first checkpoint`.

Research branch:
- `research-individual-opportunity-roadmap-2026-10-08`;
- head `a69386569a441176254ecc559cc7264612e34c26`;
- message: `Research: record return from injury role restoration source`.

PR #672:
- OPEN;
- DRAFT;
- head `a69386569a441176254ecc559cc7264612e34c26`;
- base SHA observed `81d0b56fbba5ac0372891d6e59420057a8709634`.

Exact research-head Actions:
- run `39178553484` — unrelated legacy pass-catcher QB-regime source workflow — completed/failure;
- run `39178552821` — unrelated legacy Stage-2 market-translation source workflow — completed/failure;
- no legitimate active PRISM research workflow was queued/in_progress at this research head.

Always re-read live state before any status claim.

---

## 9. EXACT NEXT RESEARCH ACTION

Read/reconcile in order:
1. `AGENTS.md`;
2. ONLY newest TOP checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`;
3. this handoff;
4. newest material Issue #535 comments;
5. live `main`;
6. live research branch;
7. PR #672;
8. exact-head Actions/status.

Then continue the latest legitimate player-first frontier.

### Freeze ONE predictive return-state mechanism BEFORE scoring.

Question:

> **Once this named player has source-certified `RETURN_PARTICIPATION_RESTORED`, does his next current-role/opportunity state behave more like his own pre-injury role than like a continued-decline state?**

This is a named-player current-role/state question.

Required boundaries:
- named player is the primary unit;
- strictly-prior same-player history;
- same-player pre-injury role anchor;
- source-certified return participation state;
- teammate competition / finite opportunity preserved;
- participation restoration separate from target earning;
- target earning separate from efficiency;
- no automatic yards multiplier;
- no position coefficient as player identity;
- snaps remain snaps and must never be called routes without route data;
- do not reuse the closed exposure×earning arithmetic;
- do not rescue closed deletion/reweighting families;
- anti-retest before freeze;
- freeze formula, chronology, support, player-level qualification gates and failure disposition BEFORE scoring;
- no 2026 outcomes in historical mechanism construction;
- no sportsbook/OddsAPI predictors upstream;
- no paid OddsAPI;
- no production mutation.

If source/semantics cannot support this cleanly:
- document the exact blocker;
- move only to the next genuinely distinct named-player CURRENT ROLE / STATE mechanism;
- do not manufacture a result.

---

## 10. COMMUNICATION STANDARD FOR THE NEXT CHAT

Do not lead with position averages.

Do not say:
- `we need to lower WR MAE`;
- `TE MAE improved` as the scientific objective;
- `the position is solved`.

Lead with:
- the named player;
- what happened;
- why it happened;
- current state;
- comparable history;
- teammate / finite-opportunity context;
- football mechanism;
- exact market;
- projection and reliability.

Aggregate metrics may appear afterward as QA guardrails only.

Permanent causal standard:

> **Get the players right first. Aggregate statistics follow.**

---

## 11. TRUST / CONTINUITY REQUIREMENT

The user has made clear that repeatedly re-teaching the assignment is unacceptable.

The next chat must demonstrate continuity through behavior.

Do not ask the user:
- what PRISM's goal is;
- whether player context matters;
- whether MAE is the goal;
- whether injury/DNP context should matter;
- whether QB passing TD should be active;
- whether betting should resume.

Those answers are already canonical.

If uncertain, read GitHub.

When the user says `keep going`, continue the exact active canonical lane.

When the user asks for a handoff, actually write the handoff to GitHub and provide the reusable prompt. Do not stop because the user is frustrated; complete the requested continuity work.

---

## 12. ONE-SENTENCE CONTINUITY STATE

**PRISM exists to project individual named players correctly; the Coker failure proved causal player-state interpretation must precede trend/model arithmetic, position averages are downstream-only, Return-from-Injury Role Restoration V1 is source-ready, and the exact next move is to freeze one named-player return-state predictive mechanism without drifting into betting, aggregate optimization, QB-TD backlog work, or any closed-family rescue.**
