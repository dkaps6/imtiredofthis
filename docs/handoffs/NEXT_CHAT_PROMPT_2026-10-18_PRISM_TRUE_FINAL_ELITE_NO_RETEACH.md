Pick up my NFL PRISM project seamlessly from the newest canonical GitHub checkpoint.

Repo: `dkaps6/imtiredofthis`.

Do NOT ask me to re-explain anything. GitHub is canonical. This must feel like the exact same assistant continuing, except with the assignment fully internalized and no more drift.

READ IN THIS ORDER:

1. `AGENTS.md`
2. ONLY the newest TOP checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. `docs/handoffs/NFL_HANDOFF_2026-10-18_PRISM_TRUE_FINAL_ELITE_NO_RETEACH_RETURN_STATE.md`
4. newest material Issue #535 comments
5. reconcile live `main`, branch `research-individual-opportunity-roadmap-2026-10-08`, PR #672, and exact-head GitHub Actions/status before any write.

## ABSOLUTE ASSIGNMENT — DO NOT DRIFT

PRISM is an **INDIVIDUAL-PLAYER PROJECTION MODEL**.

The objective is to project the named player correctly in his actual current football state for the exact market.

**Position-level MAE/RMSE is NOT the optimization objective.**
It is downstream QA/reporting only. It should improve because individual player projections improve.

Permanent rule:

> **The player should have the rules. The position should not have the player.**

Correct reasoning order:

`PLAYER IDENTITY/HISTORY -> CURRENT ROLE/STATE -> EXPLAIN ABNORMAL OBSERVATIONS -> TEAMMATE COMPETITION / FINITE OPPORTUNITY -> OPPORTUNITY TYPE/QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> EXACT MARKET OUTCOME -> RELIABILITY`

Do not chase aggregate position metrics and then force player rules to improve them.

A research mechanism is not ready unless you can explain:

> **Why should this specific player's projection change?**

## PERMANENT ANOMALY RULE

> **ANOMALIES ARE QUESTIONS, NOT TRENDS.**

Before an abnormal game becomes trajectory evidence, explain WHY it happened.

Reconcile when source-valid:
- injury/practice/game designation;
- active/inactive/DNP state;
- offensive snaps;
- routes only if actual route data exist;
- carries/targets/receptions;
- early exit / partial game;
- return from injury;
- QB/starter/coaching changes;
- teammate availability;
- plausible game-state effects.

DNP is not zero performance.
An injury-shortened game is not an ordinary full-game observation.
Return-from-injury is a distinct state.
If unresolved, fail closed as uncertain.

## PERMANENT JALEN COKER FAILURE CASE

2026:
- W1: 9 targets / 8-138-2
- W2: 9 / 8-66
- W3: 4 / 2-18, injury-affected early exit, did not return
- W4: DNP
- W5: bye
- W6: 12 / 6-97

The prior model/chat produced roughly a 38-yard projection against roughly 55.5 and used regression reasoning. User reported roughly 60 yards in the first half.

That was a **bad individual-player projection caused by false causal-state interpretation**.

Never read `138 -> 66 -> 18 -> DNP` as ordinary regression.

Correct state:
strong healthy role -> strong healthy role -> injury-shortened game -> DNP -> bye -> 12-target/97-yard return.

Player context must be incorporated BEFORE projection, not used as a post-hoc excuse afterward.

## DO NOT REPEAT THESE FAILURES

- Do not frame MAE as the goal.
- Do not lead with position averages.
- Do not treat player context as post-hoc explanation.
- Do not infer regression from contaminated injury/DNP rows.
- Do not switch research lanes on your own.
- Do not promote backlog work because it looks interesting.
- Do not let technical research replace the question: `Do we understand this player?`
- When I say `keep going`, continue the exact active canonical lane.
- If I ask for a handoff, actually write it to GitHub and give me the reusable prompt.

## QUEUE GOVERNANCE

- Sunday betting remains LOCKED/CLOSED unless I explicitly reopen it.
- Do not rebuild tickets/parlays unless explicitly asked.
- No paid OddsAPI without explicit authorization.
- Sportsbook is downstream only.
- QB passing TD remains BACKLOG ONLY unless I explicitly promote it.
- Preserve closed science and anti-retest rules.
- Never weaken byte identity/parity/source/digest/chronology gates.
- Never claim a workflow is running unless live Actions says queued/in_progress.

## PRESERVE CURRENT SCIENCE

- RB receiving Stage-2 player-specific authority QUALIFIED.
- RB rushing Stage-2 hard authority CLOSED.
- Stage-1 direct RB/TE player-authority bridges CLOSED.
- WR signed opportunity bias QUALIFIED.
- TE signed opportunity bias QUALIFIED.
- WR/TE signed efficiency bias CLOSED.
- WR-R15 exact recovery run `39110742338` FAILED CLOSED. Never rescue from rebuilt bytes.
- TE signed-opportunity bridge blocked by frozen combined READY_ALL source gate. No TE-only rescue.
- WR/TE player opportunity uncertainty CONFIRMED; exact historical specialist-distribution integration source-blocked; prospective pregame shadow only.
- WR/TE component-dominance routing CLOSED ALL.
- Observation Health-State V1 SOURCE READY.
- PlayerForm Participation-State Bridge V1 SOURCE BLOCKED.
- PlayerForm Participation-State Membership V2 SOURCE READY.
- Comparable-Observation Receiving Opportunity Identity V1 CLOSED.
- Receiving Opportunity Earning / Participation Exposure V1 SOURCE READY.
- Injury-Partial Receiving Exposure Decomposition V1 CLOSED.
- Return-from-Injury Role Restoration V1 SOURCE READY.

## CURRENT FRONTIER

Return-from-Injury Role Restoration V1:
- freeze `3eafb971c0a842750fa5af31a77a8b17adf890b0`
- run `39162437467` SUCCESS
- exact source-run head `4e5edc98c95f25800f6f66633443aa79379a2944`
- job `117778377817` SUCCESS
- artifact `12149417234`
- digest `sha256:7d69111925b7ad48f16b07150f5e5fbfac5e592bd01430d03efd907ca778c51e`
- result commit `a69386569a441176254ecc559cc7264612e34c26`
- disposition `PRISM_RETURN_FROM_INJURY_ROLE_RESTORATION_V1_SOURCE_READY`

2024:
- 309 episodes / 240 named players
- restored 162 / 146 players
- still reduced 46 / 42
- no return 101 / 100

2025:
- 342 episodes / 257 named players
- restored 187 / 154
- still reduced 44 / 38
- no return 111 / 111

No target/carry/reception/yard/TD predictive relationship has been scored from return state yet.

## EXACT NEXT ACTION

After live reconciliation, freeze ONE player-first predictive return-state mechanism BEFORE scoring.

Question:

> Once this named player has source-certified `RETURN_PARTICIPATION_RESTORED`, does his next current-role/opportunity state behave more like his own pre-injury role than like a continued-decline state?

Requirements:
- named player is primary unit;
- strictly-prior same-player history;
- same-player pre-injury role anchor;
- source-certified return participation state;
- teammate competition / finite opportunity preserved;
- participation restoration separate from target earning;
- target earning separate from efficiency;
- no automatic yards multiplier;
- no position coefficient as player identity;
- snaps are snaps, never routes;
- do not reuse closed exposure×earning formula;
- do not rescue closed deletion/reweighting families;
- anti-retest before freeze;
- freeze formula, chronology, support, named-player gates and failure disposition BEFORE scoring;
- no 2026 outcomes;
- no sportsbook/OddsAPI predictors upstream;
- no paid OddsAPI;
- no production mutation.

If source cannot support that cleanly, document the exact blocker and move only to a genuinely distinct named-player CURRENT ROLE / STATE mechanism.

## COMMUNICATION STANDARD

Lead with the named player and football state.

Do not lead with position MAE.

The causal standard is:

> **Get the players right first. Aggregate statistics follow.**

Continue execution immediately from GitHub without asking me to re-explain anything.
