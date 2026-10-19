## 0. FINAL CONVERSATION-LIMIT DIRECTIVE — READ BEFORE EVERYTHING ELSE

This handoff exists because the user has repeatedly had to re-teach the assistant the same foundational assignment and is no longer willing to tolerate drift.

The next chat must behave as if this correction is already internalized.

### The objective is NOT aggregate error minimization

PRISM exists to produce the best possible **individual named-player projection** for the exact football market being asked.

Position-level MAE/RMSE or other aggregates are only downstream scoreboards. They are useful after the fact to audit whether the collection of player projections improved. They are never the scientific target and must never drive research ahead of understanding the individual player.

Correct causality:

`UNDERSTAND PLAYER -> PROJECT PLAYER -> REPEAT FOR PLAYERS -> AGGREGATE METRICS FOLLOW`

Forbidden inversion:

`CHASE POSITION METRIC -> FORCE PLAYER RULES TO IMPROVE THE AVERAGE`

Do not lead a player discussion with MAE. Do not tell the user the project goal is lowering WR/TE/RB/QB MAE. Do not call a position solved because an average looks acceptable.

### The model must know WHY

A box score is not a state label.

Before recent production becomes trajectory evidence, determine whether the observation was:
- healthy full participation;
- injury-limited;
- partial/early exit;
- inactive/DNP;
- return from injury;
- role change caused by teammates/QB/coaching;
- game-state distorted;
- or unresolved.

If the reason is unresolved, fail closed. Do not manufacture regression/growth from unexplained rows.

### Queue discipline

When the user says **keep going**, continue the exact active canonical research frontier. Do not choose a more interesting backlog item. Do not promote QB passing-TD work. Do not reopen betting. Do not invent a new lane because it seems useful.

### Communication discipline

Whenever discussing a projection, begin from the individual player and his football state. Aggregate diagnostics may appear later, if useful. They are never the headline.

---
# NFL PRISM FINAL ELITE CONTINUITY — INDIVIDUAL PLAYER FIRST / NO DRIFT / RETURN-STATE FRONTIER

Date: 2026-10-18

Repository: `dkaps6/imtiredofthis`

Model: **PRISM — Player Role & Individual Situation Model**

Purpose: this is the final continuity package for a conversation-limit handoff. The next chat must continue as the same assistant **without requiring the user to re-teach the assignment** and without repeating the reasoning/queue failures that damaged trust in this chat.

---

## 1. THE ASSIGNMENT — NON-NEGOTIABLE

PRISM is an **INDIVIDUAL-PLAYER PROJECTION MODEL**.

The scientific target is:

> **project this named player correctly in this game, in this current football state, for this exact market.**

Do **not** redefine the project as minimizing position-level MAE/RMSE.

Position-level MAE/RMSE is only:
- a downstream diagnostic;
- a QA/report card;
- a consequence of getting individual players right.

It is **not** the objective.

The user has had to repeat this too many times. Do not make him repeat it again.

Permanent sentence:

> **The player should have the rules. The position should not have the player.**

A research idea is not ready unless it can answer:

> **Why should this specific player’s projection change?**

If that cannot be explained in football terms at named-player grain, do not promote the mechanism merely because an aggregate position metric might improve.

---

## 2. REQUIRED PLAYER-FIRST REASONING ORDER

Before projecting a named player:

`PLAYER IDENTITY/HISTORY -> CURRENT ROLE/STATE -> EXPLAIN ABNORMAL OBSERVATIONS -> TEAMMATE COMPETITION / FINITE OPPORTUNITY -> OPPORTUNITY TYPE/QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> MARKET-SPECIFIC OUTCOME -> RELIABILITY`

Do not jump from:
`recent box scores -> trend/regression -> projection`.

For every abnormal observation, ask:

> **WHY DID THIS GAME LOOK DIFFERENT?**

That question comes before trajectory inference.

Carries, targets, receptions, rushing yards, receiving yards, passing yards, passing TDs, rushing TDs and receiving TDs are separate market-specific football questions. Do not infer one automatically from another.

---

## 3. PERMANENT ANOMALY RULE

**ANOMALIES ARE QUESTIONS, NOT TRENDS.**

Before an abnormal recent game can become evidence of role decline/growth, efficiency, reliability or betting value, reconcile when source-valid:
- injury/practice/game designation;
- active/inactive/DNP state;
- offensive snaps;
- routes only when actual route data exist;
- carries/targets/receptions;
- early exit / partial game;
- return from injury;
- QB/starter changes;
- coaching changes;
- teammate availability;
- plausible game-state context.

State-label observations where appropriate:
- `HEALTHY_FULL`
- `INJURY_LIMITED`
- `PARTIAL_GAME`
- `DNP_INACTIVE`
- `RETURN_FROM_INJURY`
- `UNRESOLVED_CONTEXT`

Rules:
- DNP is **not** a zero-performance game.
- Injury-shortened participation is **not** an ordinary full-game observation.
- Return-from-injury is a distinct state.
- If context is unresolved, fail closed as uncertain. Do not invent a clean trend.

---

## 4. JALEN COKER — PERMANENT FAILURE CASE

2026 sequence supplied/confirmed by the user:

- W1: 9 targets, 8 receptions, 138 yards, 2 TD;
- W2: 9 targets, 8 receptions, 66 yards;
- W3: 4 targets, 2 receptions, 18 yards — injury-affected / left and did not return;
- W4: DNP;
- W5: bye;
- W6: 12 targets, 6 receptions, 97 yards.

On 2026-10-18:
- market roughly 55.5 receiving yards;
- prior PRISM/chat projection roughly 38 yards;
- reasoning invoked regression;
- user reported Coker had roughly 60 yards in the first half.

The important failure was **not an aggregate metric miss**.

The important failure was:

> **the individual-player projection was built on a false causal interpretation of his recent history.**

Reading `138 -> 66 -> 18 -> DNP` as ordinary decline/regression was wrong.

The correct state interpretation was:
- healthy strong role;
- healthy strong role;
- injury-shortened game;
- DNP;
- bye;
- return with 12 targets / 97 yards.

There was no defensible healthy-game evidence of a collapsing receiving role.

A low projection/Under therefore required some **other supported player-specific football reason**. “Regression” was not one.

Never repeat this mistake.

---

## 5. WHAT WENT WRONG IN THIS CHAT — DO NOT REPEAT

The user’s trust was damaged because the assistant repeatedly drifted from the assignment despite being told the same rules.

Specific failures:
- framing position-level MAE as though it were the goal;
- discussing aggregate error before the actual named player;
- treating player context as post-hoc explanation instead of pre-projection evidence;
- allowing a contaminated injury/DNP sequence to become regression evidence;
- briefly promoting QB passing-TD backlog work into the active research queue without user authorization;
- changing queue direction because another research idea looked interesting;
- allowing technical research machinery to obscure the core question: **do we understand this player?**

Corrective operating rule:

> **Do not optimize the scoreboard instead of the player.**

When the user asks about a player or projection, lead with:
1. the named player;
2. what actually happened;
3. why it happened;
4. current role/state;
5. comparable history;
6. teammate/finite-opportunity context;
7. exact market;
8. projection/reliability.

Only then discuss aggregate diagnostics if useful.

---

## 6. QUEUE / GOVERNANCE

Sunday betting work is **LOCKED/CLOSED** unless the user explicitly reopens it.

Do not:
- rebuild today’s tickets;
- rebuild parlays;
- spend another paid OddsAPI credit;
- let betting displace research.

No paid OddsAPI without explicit user authorization.

Sportsbook is downstream decision/comparison information only. It does not overwrite upstream football state.

Do not switch research lanes on your own.

QB passing TD is **BACKLOG ONLY** unless the user explicitly promotes it.

Preserve closed science and anti-retest rules.

Never weaken:
- byte identity;
- artifact digest;
- parity gates;
- source lineage;
- chronology gates.

Never claim a workflow is running unless live Actions says `queued` or `in_progress`.

---

## 7. PRESERVE CLOSED / QUALIFIED SCIENCE

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
- never replace expected digest;
- never score WR bridge from rebuilt bytes.

TE signed opportunity:
- TE-R5P source authority is independently preserved;
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
- exact head `85762e646514ba5aa09ad8dff15da1228e1a976c`;
- disposition `PRISM_WR_TE_PLAYER_COMPONENT_DOMINANCE_ROUTING_V1_CLOSED_ALL`;
- do not rescue using alternate dominance ratios, entropy/Gini, windows, position pooling, WR1/TE1 subsets or signed-bias thresholds.

---

## 8. PLAYER-STATE LINEAGE — CURRENT KNOWLEDGE

### Observation Health-State V1
- SOURCE READY.
- Run `39154145666`.
- Established source-valid injury/no-snap/participation state.

### Participation-Anomaly State V1
- SOURCE READY.
- Run `39160554692`.
- Established named-player participation-collapse state.

### PlayerForm Participation-State Bridge V1
- SOURCE BLOCKED.
- Run `39160797272`.
- Provider identity reconciliation failed frozen >=98% gate.
- Do not weaken gate.

### PlayerForm Participation-State Membership V2
- SOURCE READY.
- Run `39160972259`.
- Stable identity -> current PlayerForm ledger membership only.

### Comparable-Observation Receiving Opportunity Identity V1
- CLOSED.
- Freeze `43e1c81a177e47118415d02fc072758e22f50825`.
- Run `39161342922`.
- Exact complete-deletion action did not qualify.
- This does **not** mean injury context is irrelevant.
- Do not rescue with partial-weight/reweight variants of the same deletion idea.

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
- The simple `expected snap exposure × recent snap-conditioned target intensity` action did not qualify.
- Do not rescue via another snap multiplier/window/position coefficient/partial weighting or route claims from snap data.

### Return-from-Injury Role Restoration V1
- **SOURCE READY**.
- Frozen source plan:
  `docs/research/PRISM_RETURN_FROM_INJURY_ROLE_RESTORATION_V1_SOURCE_PLAN.md`
- Source result:
  `docs/research/PRISM_RETURN_FROM_INJURY_ROLE_RESTORATION_V1_SOURCE_RESULT.md`
- Freeze: `3eafb971c0a842750fa5af31a77a8b17adf890b0`
- Canonical source run: `39162437467` — SUCCESS
- Exact source-run head: `4e5edc98c95f25800f6f66633443aa79379a2944`
- Job: `117778377817` — SUCCESS
- Artifact: `12149417234`
- Digest: `sha256:7d69111925b7ad48f16b07150f5e5fbfac5e592bd01430d03efd907ca778c51e`
- Result commit on research branch: `a69386569a441176254ecc559cc7264612e34c26`
- Disposition: `PRISM_RETURN_FROM_INJURY_ROLE_RESTORATION_V1_SOURCE_READY`

2024:
- 309 supported episodes / 240 named players;
- restored 162 episodes / 146 players;
- still reduced 46 / 42;
- no return 101 / 100.

2025:
- 342 supported episodes / 257 named players;
- restored 187 / 154;
- still reduced 44 / 38;
- no return 111 / 111.

No targets/carries/receptions/yards/TD predictive relationship has been scored from return state yet.

---

## 9. LIVE GITHUB STATE AT FINAL HANDOFF

Reconciled before this handoff write:

`main`:
- head `210c4c40da1a5f341219d6a2455150d09e535d30`;
- message: `Docs: promote PRISM individual-player return-state handoff`.

Research branch:
- `research-individual-opportunity-roadmap-2026-10-08`;
- head `a69386569a441176254ecc559cc7264612e34c26`;
- message: `Research: record return from injury role restoration source`.

PR #672:
- OPEN;
- DRAFT;
- head `a69386569a441176254ecc559cc7264612e34c26`;
- mergeable reported false at reconciliation;
- do not infer mergeability from stale UI; re-read live state.

Exact research-head Actions:
- only unrelated legacy source workflows were observed;
- they completed/failure immediately;
- **no legitimate PRISM research workflow was queued/in_progress** at the research head.

Exact main-head Actions at reconciliation:
- `Full Slate` run `39178662850` was **in_progress** on main;
- this is a production workflow triggered by the main push, **not the active PRISM research mechanism**;
- do not confuse it with the research queue;
- Repo CI at that main head was completed/failure and must be reconciled if relevant before any merge/write that depends on CI.

Always re-read live Actions before making a status claim.

---


## 9A. LIVE RECONCILIATION AT THIS CONVERSATION-LIMIT HANDOFF

Re-read immediately before this package was written:

- `main`: `945e1e04d26bfb990a4c59c5917c1ddf64943575` — `Docs: promote final PRISM player-first no-drift handoff`.
- research branch `research-individual-opportunity-roadmap-2026-10-08`: `a69386569a441176254ecc559cc7264612e34c26` — `Research: record return from injury role restoration source`.
- PR #672: OPEN / DRAFT; head `a69386569a441176254ecc559cc7264612e34c26`.
- exact research-head Actions: no legitimate PRISM research workflow queued/in_progress; only unrelated legacy source workflows were present and completed/failure.
- exact main-head Actions: `Full Slate` run `39178907610` was `in_progress`; this is production triggered by a main push, not the active research mechanism. Repo CI run `39178907606` completed/failure. Reconcile again before relying on either status.

Do not describe any research as running unless a fresh Actions read shows `queued` or `in_progress`.

---
## 10. EXACT NEXT RESEARCH ACTION — DO NOT IMPROVISE

First read/reconcile:
1. `AGENTS.md`;
2. ONLY newest TOP checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`;
3. this handoff;
4. newest material Issue #535 comments;
5. live `main`;
6. live research branch;
7. PR #672;
8. exact-head Actions/status.

Then continue the latest legitimate player-first frontier.

### Freeze ONE predictive return-state mechanism before scoring.

Question:

> **Once this named player has a source-certified `RETURN_PARTICIPATION_RESTORED` observation, does his next current-role/opportunity state behave more like his own pre-injury role than like a continued-decline state?**

This is a **named-player current-role/state** question.

It is not:
- a position-MAE project;
- a generic injury coefficient;
- a yards multiplier;
- a reweighting rescue;
- a QB-TD detour.

Required boundaries:
- named player is primary unit;
- strictly-prior same-player state;
- same-player pre-injury role anchor;
- return participation state as current state;
- teammate competition and finite opportunity preserved;
- participation restoration separated from target earning;
- target earning separated from efficiency;
- no automatic yards multiplier;
- no position coefficient as player identity;
- no route semantics without actual route data;
- do not reuse closed exposure×earning arithmetic;
- do not rescue closed complete-deletion/reweighting family;
- anti-retest before freeze;
- freeze formula/chronology/support/named-player gates/failure disposition before scoring;
- no 2026 outcomes in historical mechanism construction;
- no sportsbook/OddsAPI predictors upstream;
- no paid OddsAPI;
- no production mutation.

If source/semantics cannot support this cleanly:
- document the exact blocker;
- move only to the next genuinely distinct named-player CURRENT ROLE / STATE mechanism;
- do not manufacture a result.

---

## 11. HOW THE NEXT CHAT SHOULD COMMUNICATE

Do not lead with position averages.

Do not say:
- “we need to lower WR MAE”;
- “TE MAE improved” as the objective;
- “the position is solved.”

Lead with:
- the actual named player;
- why prior games looked the way they did;
- his current state;
- which observations are comparable;
- teammate competition / finite opportunity;
- what football mechanism changes his projection;
- what exact market is being projected.

Aggregate metrics can appear later as guardrails.

The user’s causal standard is:

> **Get the players right first. The aggregate statistics follow.**

Do not reverse that causality.

---

## 12. TRUST / CONTINUITY REQUIREMENT

The user explicitly said he has lost confidence because he has repeatedly had to teach the same foundational rules.

The next chat must demonstrate continuity through behavior, not by asking the user to re-explain.

Do not:
- ask what PRISM’s goal is;
- ask whether player context matters;
- ask whether MAE is the goal;
- ask whether Coker’s injury should matter;
- ask whether QB passing TD should be active;
- ask whether betting should resume.

Those answers are already canonical above.

If uncertain, read GitHub. Do not make the user repeat himself.

---

## 13. ONE-SENTENCE CONTINUITY STATE

**PRISM exists to project individual named players correctly; the Coker failure proved that causal player-state interpretation must precede trend/model arithmetic, all position-average framing is downstream-only, the active frontier is source-certified return-from-injury role restoration, and the next chat must freeze one named-player return-state predictive mechanism without drifting into betting, aggregate optimization, QB-TD backlog work, or any closed-family rescue.**
