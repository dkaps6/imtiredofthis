# NFL PRISM ELITE CONTINUITY HANDOFF — INDIVIDUAL PLAYER FIRST / COKER CAUSAL-STATE FAILURE / RETURN-FROM-INJURY FRONTIER

Date: 2026-10-18

Repository: `dkaps6/imtiredofthis`

Model: **PRISM — Player Role & Individual Situation Model**

Purpose: this handoff exists so the next chat can continue without making the user re-teach the assignment, the reasoning standard, the research boundaries, or the current GitHub state.

---

## 1. READ THIS AS A CORRECTION TO HOW THE ASSISTANT MUST THINK

The user has repeatedly and explicitly defined PRISM as an **INDIVIDUAL-PLAYER PROJECTION MODEL**.

This is not a position-MAE project.

Do not drift back to:
- “how low can we get WR MAE?”;
- “how low can we get TE MAE?”;
- “position X is solved because average error improved”;
- treating aggregate error as the scientific objective.

The correct framing is:

> **MAE/RMSE is a downstream report card that improves when named-player projections improve. It is not the end goal.**

The unit of truth is:

> **this named player, in this game, in this current football state, for this exact market.**

The user should **not** have to explain this again.

If a research idea cannot be stated as:
> “why should this specific player’s projection change?”

then it is not ready to be a PRISM mechanism.

Position/family aggregates may still be used as:
- priors;
- shrinkage;
- sparse-player fallback;
- QA;
- dependence-aware guardrails.

They must never become the player’s identity or the optimization objective.

Permanent operating sentence:

> **The player should have the rules. The position should not have the player.**

---

## 2. PLAYER-FIRST REASONING ORDER — REQUIRED BEFORE ANY PROJECTION

For every named player, reason in this order:

`PLAYER IDENTITY/HISTORY -> CURRENT ROLE/STATE -> EXPLAIN ABNORMAL OBSERVATIONS -> TEAMMATE COMPETITION / FINITE OPPORTUNITY -> OPPORTUNITY TYPE/QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> MARKET-SPECIFIC OUTCOME -> RELIABILITY`

Do not skip directly from recent stat line to trend to projection.

A correct player projection requires understanding:
- what this specific player has historically been;
- what his current role actually is;
- whether recent observations are comparable;
- whether injury, DNP, partial participation, return-from-injury, QB change, teammate availability, coaching or game-state explains an apparent change;
- what finite team opportunity remains for him;
- what kind of opportunity he gets;
- what this opponent changes;
- what exact market is being projected.

Carries, targets, receptions, yards, passing TDs, rushing TDs and receiving TDs are **different football questions**.

Do not infer one automatically from another.

---

## 3. PERMANENT ANOMALY RULE — THIS IS NON-NEGOTIABLE

**ANOMALIES ARE QUESTIONS, NOT TRENDS.**

Before an abnormal recent observation can become:
- decline evidence;
- growth evidence;
- role evidence;
- efficiency evidence;
- reliability evidence;
- betting evidence;

first ask:

> **WHY DID THIS GAME LOOK DIFFERENT?**

Mandatory investigation when source-valid:
- injury/practice/game designation;
- active/inactive/DNP state;
- offensive snaps;
- routes only if actual route data exist;
- carries/targets/receptions;
- early exit or partial-game state;
- return from injury;
- QB/starter changes;
- coaching changes;
- teammate availability;
- plausible game-state context.

State-label abnormal observations before trajectory logic, e.g.:
- `HEALTHY_FULL`;
- `INJURY_LIMITED`;
- `PARTIAL_GAME`;
- `DNP_INACTIVE`;
- `RETURN_FROM_INJURY`;
- `UNRESOLVED_CONTEXT`.

A DNP is **not** a zero-performance game.

An injury-shortened game is **not** an ordinary full-game observation.

If context cannot be resolved, fail closed:
- mark the trajectory/state uncertain;
- do not use the abnormal observation as affirmative evidence of decline/growth;
- do not let PRISM or the assistant invent a clean trend from contaminated rows.

---

## 4. THE JALEN COKER FAILURE — PERMANENT CASE STUDY

This is the concrete failure that triggered the current reset.

2026 Coker receiving sequence supplied by the user:

- Week 1 vs CHI: **9 targets, 8 receptions, 138 yards, 2 TD**;
- Week 2 at ATL: **9 targets, 8 receptions, 66 yards**;
- Week 3 at CLE: **4 targets, 2 receptions, 18 yards — injury-affected / left and did not return**;
- Week 4 vs DET: **DNP**;
- Week 5: **bye**;
- Week 6 at PHI: **12 targets, 6 receptions, 97 yards**.

On 2026-10-18:
- market was roughly **55.5 receiving yards**;
- PRISM/chat reasoning had him roughly **38 yards**;
- user reported he had already reached roughly **60 yards in the first half**.

The important error is **not** “our MAE was bad.”

Do not frame it that way.

The important error was:

> **PRISM/chat gave a bad individual-player projection because it misunderstood the causal meaning of the player’s recent history.**

The prior reasoning effectively read:
`138 -> 66 -> 18 -> DNP`
as decline/regression.

That was wrong.

The correct player-state interpretation was:
- strong healthy usage;
- strong healthy usage;
- injury-shortened game;
- DNP;
- bye;
- return with **12 targets / 97 yards**.

There was no defensible healthy-game evidence of a collapsing receiving role.

Before making an Under / low projection, PRISM needed a **different supported football reason**.

This failure must remain permanently visible because it illustrates what “know the player” means.

---

## 5. ASSISTANT FAILURE / TRUST REQUIREMENT

The user is frustrated because the same foundational instruction has had to be repeated across chats.

Do not make the user teach this again.

Specific failures to avoid:
- reverting to position-average optimization;
- talking as though MAE is the goal;
- treating player context as a post-hoc explanation instead of pre-projection evidence;
- switching research lanes because another idea looks interesting;
- promoting a backlog item without explicit user direction;
- trusting a model output when the player-state inputs are obviously contaminated;
- presenting “regression” without explaining abnormal games first.

The next chat should behave as though these rules are already internalized.

Do not respond to the user with a lecture about aggregate metrics.

When discussing a projection, explain:
1. the player;
2. his current state;
3. the comparable history;
4. the football reasons;
5. the exact market.

---

## 6. BETTING / PRODUCTION GOVERNANCE

Sunday betting work is **LOCKED/CLOSED** unless the user explicitly reopens it.

Do not:
- rebuild today’s tickets;
- reconstruct parlays;
- spend another paid OddsAPI credit;
- let betting work displace research.

No paid OddsAPI without explicit authorization.

Sportsbook is downstream comparison/decision information only unless an explicitly market-assisted model says otherwise.

Do not alter upstream football projections because of a sportsbook discrepancy.

A large discrepancy may trigger a football audit:
> “why is PRISM different?”

It does not authorize tuning to the market.

---

## 7. QB PASSING-TD BACKLOG GOVERNANCE

The prior chat exposed that PRISM lacks a dedicated certified QB passing-TD projection layer.

A historical source-only audit was completed and found source support.

However:

> **QB passing TD is BACKLOG ONLY unless the user explicitly promotes it.**

Do not let it replace the active player-state research queue.

Earlier concurrent work briefly tried to promote QB passing-TD research into the active lane. That was explicitly corrected in Issue #535.

Do not repeat that mistake.

---

## 8. CLOSED / QUALIFIED SCIENCE THAT MUST BE PRESERVED

Do not restart, rescue or retune closed families.

Preserve:

- RB receiving Stage-2 player-specific authority: **QUALIFIED**.
- RB rushing Stage-2 hard authority: **CLOSED**.
- Stage-1 direct RB/TE player-authority bridges: **CLOSED**.
- WR signed opportunity bias: **QUALIFIED**.
- TE signed opportunity bias: **QUALIFIED**.
- WR signed efficiency bias: **CLOSED**.
- TE signed efficiency bias: **CLOSED**.

WR exact authority:
- WR-R15 exact recovery run `39110742338`: **FAILED CLOSED**.
- rebuilt bytes did not match immutable expected authority.
- never weaken byte identity;
- never replace expected digest;
- never score the WR bridge from rebuilt bytes.

TE signed-opportunity bridge:
- TE-R5P source authority itself remains independently preserved;
- but the already-frozen bridge required the combined WR/TE source recovery `READY_ALL`;
- WR fail-close prevented that state;
- no separately frozen TE-only bridge existed;
- disposition:
  `PRISM_TE_SIGNED_OPPORTUNITY_CONSERVED_BRIDGE_EXECUTION_BLOCKED_BY_FROZEN_COMBINED_SOURCE_GATE`;
- no TE-only post-result rescue.

WR/TE player opportunity uncertainty:
- WR confirmed;
- TE confirmed;
- exact historical specialist-distribution integration source-blocked;
- prospective pregame shadow remains legal only if properly frozen before target outcomes.

WR/TE component-dominance routing:
- run `39153337388`;
- exact head `85762e646514ba5aa09ad8dff15da1228e1a976c`;
- disposition:
  `PRISM_WR_TE_PLAYER_COMPONENT_DOMINANCE_ROUTING_V1_CLOSED_ALL`;
- do not rescue via alternate dominance ratios, entropy/Gini, windows, WR1/TE1 subsets, pooling or signed-bias thresholds.

---

## 9. PLAYER-STATE RESEARCH LINEAGE — WHAT HAS NOW BEEN LEARNED

### A. Observation Health-State V1 — SOURCE READY

Run:
`39154145666`

Purpose:
separate reported injury participation/no-snap state from normal observations.

No performance outcome was used to define state.

### B. Participation-Anomaly State V1 — SOURCE READY

Run:
`39160554692`

2024:
- positive-participation rows 6,577;
- baseline-supported 4,806;
- injury-corroborated collapses 139;
- unresolved collapses 182.

2025:
- positive-participation rows 6,551;
- baseline-supported 4,750;
- injury-corroborated collapses 149;
- unresolved collapses 195.

This is player-state source evidence only.

### C. PlayerForm Participation-State Bridge V1 — SOURCE BLOCKED

Run:
`39160797272`

The exact provider-reconciliation bridge failed its >=98% identity gate.

Do not weaken that gate.

### D. PlayerForm Participation-State Membership V2 — SOURCE READY

Run:
`39160972259`

This proved stable player identity -> current PlayerForm history-ledger membership.

It did not prove a predictive action.

### E. Comparable-Observation Receiving Opportunity Identity V1 — CLOSED

Freeze:
`43e1c81a177e47118415d02fc072758e22f50825`

Run:
`39161342922`

Result:
`PRISM_COMPARABLE_OBSERVATION_RECEIVING_OPPORTUNITY_IDENTITY_V1_CLOSED`

Important interpretation:
- the exact complete-exclusion action did not qualify;
- this does **not** mean injury context is irrelevant;
- it means “delete the contaminated PlayerForm observation entirely under this exact construction” is not the answer.

Do not rescue via partial weighting/reweighting variants of the same deletion idea.

### F. Receiving Opportunity Earning / Participation Exposure V1 — SOURCE READY

Freeze:
`42ff90db6202b783f7e97b73d0b3d629e9566bb3`

Run:
`39161711741`

Artifact:
`12149181200`

Digest:
`sha256:6cae427b667db4a120a1308747d00b82b6770fe0adbc46d3efc68fce3e1db240`

Disposition:
`PRISM_RECEIVING_OPPORTUNITY_EARNING_PARTICIPATION_EXPOSURE_V1_SOURCE_READY`

This source successfully separates:
- named-player participation exposure;
- target earning conditional on participation.

Snaps are **snaps**, not routes.

### G. Injury-Partial Receiving Exposure Decomposition V1 — CLOSED

Freeze:
`653eb5e4e46a49f0b575c119259d2b11eb845380`

Canonical run:
`39162113968`

Disposition:
`PRISM_INJURY_PARTIAL_RECEIVING_EXPOSURE_DECOMPOSITION_V1_CLOSED`

Question:
could next comparable target-share state be improved by:

`ordinary expected snap exposure × recent snap-conditioned target intensity`

for players with one injury-corroborated participation collapse?

It did not qualify at the primary named-player level.

Do not rescue by:
- another snap multiplier;
- another window;
- mean vs median;
- position coefficients;
- WR-only cherry-pick;
- partial weighting;
- route claims from snap data.

The failure means the missing state is **not** captured by this simple exposure × earning arithmetic.

It does **not** restore “injury partial = decline.”

### H. Return-from-Injury Role Restoration V1 — SOURCE READY

Frozen source plan:
`docs/research/PRISM_RETURN_FROM_INJURY_ROLE_RESTORATION_V1_SOURCE_PLAN.md`

Freeze:
`3eafb971c0a842750fa5af31a77a8b17adf890b0`

Canonical source run:
`39162437467`

Exact scored/source head:
`4e5edc98c95f25800f6f66633443aa79379a2944`

Job:
`117778377817` — SUCCESS

Artifact:
`12149417234`

Digest:
`sha256:7d69111925b7ad48f16b07150f5e5fbfac5e592bd01430d03efd907ca778c51e`

Result document committed after source run:
`docs/research/PRISM_RETURN_FROM_INJURY_ROLE_RESTORATION_V1_SOURCE_RESULT.md`

Result commit:
`a69386569a441176254ecc559cc7264612e34c26`

Disposition:
`PRISM_RETURN_FROM_INJURY_ROLE_RESTORATION_V1_SOURCE_READY`

2024:
- episodes 309;
- named players 240;
- restored 162 episodes / 146 players;
- still reduced 46 / 42;
- no return 101 / 100;
- median pre-episode snap pct 0.605;
- median restored return snap pct 0.580;
- median still-reduced return snap pct 0.165.

2025:
- episodes 342;
- named players 257;
- restored 187 / 154;
- still reduced 44 / 38;
- no return 111 / 111;
- median pre-episode snap pct 0.562;
- median restored return snap pct 0.530;
- median still-reduced return snap pct 0.210.

This is the latest legitimate active frontier.

No target/carry/reception/yard/TD relationship has yet been scored from this return state.

---

## 10. EXACT CURRENT GITHUB STATE AT HANDOFF CONSTRUCTION

Last reconciled `main` before writing this continuity package:

`7db5795959ef17e4ccc7117429d4bbaa141ffead`

Message:
`Docs: advance root handoff to exposure source-ready frontier`

Research branch:
`research-individual-opportunity-roadmap-2026-10-08`

Research branch head after recording the source result:

`a69386569a441176254ecc559cc7264612e34c26`

Message:
`Research: record return from injury role restoration source`

PR #672:
- OPEN;
- DRAFT;
- head = `a69386569a441176254ecc559cc7264612e34c26` at last reconciliation;
- do not assume mergeability from stale UI; re-read live state.

Latest source run:
`39162437467` — **SUCCESS**

Exact source-run head:
`4e5edc98c95f25800f6f66633443aa79379a2944`

At `a693865...`, the only exact-head push workflows observed were legacy unrelated source workflows that completed/failure immediately.

Do not confuse those unrelated legacy failures with the successful return-from-injury source audit.

At last reconciliation there was **no legitimate PRISM research workflow queued/in_progress**.

Always re-read live Actions before saying something is running.

---

## 11. EXACT NEXT ACTION — DO NOT IMPROVISE A DIFFERENT QUEUE

First:
1. read `AGENTS.md`;
2. read ONLY the newest TOP checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`;
3. read this handoff;
4. read newest material Issue #535 comments;
5. reconcile live main / research branch / PR #672 / exact-head Actions.

Then:

### Freeze one predictive return-state contract BEFORE scoring anything.

The next player-first question should be:

> Once this named player has a source-certified `RETURN_PARTICIPATION_RESTORED` observation, does his next-game current-role/opportunity state behave more like his own pre-injury role than like a continued decline state?

The purpose is **not** to lower position MAE.

The purpose is to improve the named player's current-state interpretation.

Required boundaries:
- primary unit = named player;
- strictly-prior player state;
- same-player pre-injury role as the player anchor;
- return participation state as a current-state object;
- teammate competition and finite opportunity preserved;
- participation restoration separated from target earning;
- target earning separated from receiving efficiency;
- no automatic direct yards multiplier;
- no position coefficient as player identity;
- no route semantics unless actual route data exist;
- do not reuse the closed simple exposure × target-intensity arithmetic;
- do not rescue the closed complete-deletion/reweighting family;
- no threshold/window search after outcomes;
- no 2026 outcomes for historical mechanism construction;
- no sportsbook/OddsAPI upstream;
- no production mutation.

Before executing:
- anti-retest the exact proposed mechanism against the closed player-state families;
- freeze formula, chronology, support, named-player gates, guardrails and failure disposition;
- only then score historical outcomes.

If the source/semantics cannot support that cleanly, document the blocker and move to the next genuinely distinct named-player current-role/state object.

Do not manufacture a result.

---

## 12. HOW TO TALK TO THE USER ABOUT RESULTS

Do not lead with:
- position MAE;
- “WR improved by X”;
- “TE average got better”;
- aggregate score optimization.

Lead with:
- which named players were helped/hurt;
- why;
- what player state the mechanism captured or missed;
- whether the football interpretation is believable;
- whether abnormal games were resolved;
- how the current role changed;
- what exact market is affected.

Aggregate metrics can appear later as guardrails/evidence.

The user’s standard is:

> **If PRISM gets the players right, the aggregate statistics will follow.**

Do not reverse that causality.

---

## 13. ONE-SENTENCE CONTINUITY STATE

**PRISM is an individual-player projection system, the Coker miss proved that unresolved injury/participation context can corrupt a named-player projection before any math matters, simple contaminated-history deletion and simple exposure×earning fixes are closed, return-from-injury participation state is now source-certified and ready for a separately frozen player-first predictive test, and the next chat must continue from that exact current-role frontier without drifting back to aggregate-MAE optimization, betting, or QB-TD backlog work.**
