# NFL HANDOFF — 2026-10-17 — PRISM PLAYER-FIRST WEEK-4/5 REPLAY CONTINUITY

**Status:** CURRENT / READ FIRST AFTER ROOT CHECKPOINT  
**Repository:** `dkaps6/imtiredofthis`  
**Model:** **PRISM — Player Role & Individual Situation Model**  
**Production branch:** `main`  
**Research branch:** `research-individual-opportunity-roadmap-2026-10-08`  
**Research PR:** #672 — OPEN / DRAFT  
**Primary issue:** #535

This handoff supersedes the narrower WR-dual-reliability-only continuation as the active working frontier. The WR reliability experiment remains protected, but it is no longer allowed to crowd out the larger PRISM player-projection objective.

## 1. NON-NEGOTIABLE MODEL OBJECT

PRISM is an **individual-player projection model**.

The scientific object is:

> the named player in his actual current football situation.

Everything moving forward must reason in this order:

`PLAYER IDENTITY/HISTORY -> CURRENT ROLE/STATE/TRAJECTORY -> TEAMMATE COMPETITION -> FINITE OPPORTUNITY ALLOCATION -> OPPORTUNITY TYPE/QUALITY -> TEAM/GAME ENVIRONMENT -> PLAYER-CONDITIONED OPPONENT MATCHUP -> INDIVIDUAL EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

Position-average MAE/RMSE is downstream QA only. It is never the primary optimization target.

Do not ask first:
- how do we lower RB MAE?;
- how do we fix WR averages?;
- what position has the biggest average miss?

Ask first:
- why did PRISM project **this player** this way?;
- what did PRISM know about his historical identity?;
- what current role/state was active?;
- what changed around him?;
- how much finite opportunity belonged to him?;
- what kind of opportunity was it?;
- how did this opponent interact with **his** route/carry/usage type?;
- how efficiently does **this player** convert those opportunities?;
- how trustworthy is the projection for this exact player?;
- what exact football question does the market ask?

Aggregate error can be computed only after that player-level reasoning.

## 2. BINDING CONTEXT FROM THE PRIOR “CONSTRUCT NFL TICKETS” CHAT

The user explicitly required this chat to reread the prior conversation beginning with the question:

> Why did you pick J.K. Dobbins to score a touchdown tonight?

That discussion is binding model context, not merely ticket-construction context.

The Dobbins example illustrated that the model must distinguish **market-specific paths to production**.

The reasoning was not:
- “RBs score touchdowns”;
- “Denver is good/bad”;
- “the line has value”;
- “Dobbins projects for X rushing yards.”

It combined football facts such as:
- lead-back/current workload state;
- recent workload trajectory;
- finite competition for carries;
- scoring-area / goal-line opportunity ownership;
- whether the offense was likely to create the relevant scoring situation;
- player-specific conversion context;
- then price/market only **after** the football case existed.

A player can be a weak rushing-yards candidate and a strong TD candidate without contradiction.

Similarly:
- carries;
- targets;
- receptions;
- receiving yards;
- rushing yards;
- touchdowns

are related but **different questions about the same named player**.

The prior ticket-order remains binding:

`FOOTBALL PROJECTION -> RELIABILITY/TRUST -> MARKET COMPARISON -> APPROVED PLAY -> ALT-LINE/CORRELATION/TICKET CONSTRUCTION`.

Ticket construction should use approved PRISM player plays, deliberate alt-line cushion, correlation controls and player-concentration controls. Safe / Moderate / High / Lottery tiers are downstream decisions. ATD remains separate/cautious unless PRISM has actually certified the relevant TD pathway.

## 3. THIS PLAYER-FIRST DIRECTION WAS ALREADY ON THE DOCKET

This is **not** a new redesign.

The prior Week-4 handoff and root operating contract already said the user’s core concern was no longer aggregate averages, but whether the model understood the specific:
- player;
- role;
- offense;
- defense;
- injuries;
- teammate hierarchy;
- game environment.

The recent research did contain legitimate player-first pieces:
- active-role participation;
- finite opportunity allocation;
- named-player reliability;
- player-conditioned opponent-boundary rules.

But the work became too narrowly centered on the frozen WR reliability protocol. The current correction is to restore the broader player-first objective, not discard validated work.

## 4. WEEK-4/5 REPLAY IS NOW THE ACTIVE DIAGNOSTIC LANE

Frozen protocol:

`docs/research/PRISM_WEEK4_5_PLAYER_FIRST_FORENSIC_REPLAY_V1_PROTOCOL.md`

Freeze commit:
`b4bc97a484a9c71676bdd6f6694983a3b822f33b`

Disposition:
`PRISM_WEEK4_5_PLAYER_FIRST_FORENSIC_REPLAY_V1_FROZEN`.

The replay begins at **Week 4**.

Why:
- Week 4 has a preserved genuine pregame board and complete postmortem;
- Week 5 has the same, plus enough current-season history for mechanisms requiring four completed current-season games where eligible;
- there is no scientific reason to sit idle waiting for Week 6 merely to inspect how PRISM handled already-completed players.

Important nuance:
- the **exact frozen WR dual-reliability** calculation requires >=4 prior current-season same-player games, so Week 5 is the first week eligible for that exact rule;
- that does **not** block a broader Week-4 PRISM player replay;
- Week 4 may use legitimate player identity/history, prior-season/career context and Weeks 1-3 current-season state where source-valid;
- do not retroactively manufacture a four-current-week reliability score for Week 4.

Week-6 prospective locks/shadows remain untouched.

## 5. PRESERVED WEEK-4 / WEEK-5 AUTHORITIES

### Week 4

Pregame board:
- run `36935917903`;
- artifact `11197776900`;
- digest `sha256:3f058570037ca016a5cbf1fa79e6e6abc4d845384de4dbdfaf477c8f0a3160a8`;
- 873 player-market rows;
- 16 games.

Postmortem:
- run `37482840038`;
- artifact `11421298874`;
- expanded failure-mode run `37485600879`;
- artifact `11423535484`.

### Week 5

Pregame Full Slate:
- run `37860592615`;
- artifact `11585973062`;
- digest `sha256:bf52aa4bc30bf6e8cc6029a0a86c0f33cdfaa09f587c38a138f23cef6b7436f4`;
- production behavior authority `24704e5f1ef872ec86c36446d2c73f6d377c824c`.

Settled postmortem:
- run `38395384228`;
- artifact `11812200405`.

Player failure casebook:
- run `38397361891`;
- artifact `11812403368`.

No sportsbook refetch is authorized for this replay.

## 6. WEEK-5 NAMED-PLAYER FORENSIC REPLAY — COMPLETE

Result:

`docs/research/PRISM_2026_WEEK5_PLAYER_STATE_FORENSIC_REPLAY_V1_RESULT.md`

Certified execution:
- run `39039061242`;
- exact head `a2cf1adbc4ffa37f72a3635d73a0894e5d07b239`;
- artifact `12103281893`;
- artifact digest `sha256:ddd80921006625ce1381167fcc6530c5ecedf9b20e06f0e36d765e2a247aa02a`.

Disposition:
`PLAYER_STATE_FORENSIC_REPLAY_COMPLETE__NO_PROMOTION`.

Key findings:
- 318 strict-prior same-player/market Week-5 replay rows;
- 166 unique players;
- 30/40 frozen top-failure cases had all four prior same-market weeks;
- 9 repeated same-player + same-market top-40 cases between Weeks 4 and 5;
- **7/9 (77.78%) changed failure component state** between opportunity / efficiency / mixed;
- among 23 direct skill-player major misses (rush yards / receiving yards / receptions), **22/23 realized opportunity exceeded predicted opportunity**;
- among the 8 opportunity-dominant direct-skill cases, **8/8 realized opportunity exceeded prediction**;
- among those 8, **6/8 had strictly-prior raw current player share above the production-rule share**, median difference +0.07263.

Interpretation:
- the same named player does **not** fail for the same reason every week;
- “this player is volatile” is insufficient;
- PRISM must determine **which state of this player is active this week**;
- many large misses showed current-role evidence above production-rule opportunity, but the blunt fix “replace entitlement with raw recent share” remains CLOSED.

Protected closed family:
- Opportunity Authority Priority / raw recent-share replacement remains CLOSED;
- do not rescue it with RB-only / TE-only exceptions, alternate Bayes weights or retuned strengths.

## 7. WEEK-5 OPPONENT / INJURY FORENSIC LESSONS

Among the 23 direct skill major misses:
- relevant opponent defensive EPA existed in the parity player-state trace for **23/23**;
- the relevant rules-layer efficiency multiplier was exactly 1.00 in **21/23**.

This does **not** authorize generic defense-vs-position.
It shows that useful opponent state existed but was usually not transmitted at this preserved rules field.

Future matchup science must remain:
- player-conditioned;
- role/opportunity-subtype specific;
- source-parity valid.

Historical row-level injury/teammate reconstruction was incomplete in the preserved failure-case artifacts:
- injury_status non-null in 0/40;
- injury_designation non-null in 0/40.

Do not invent historical injury narratives from blanks.

Future pregame traces should preserve row-level:
- availability;
- room state;
- opponent state;
- opportunity;
- efficiency;
- reliability

for the exact named player.

## 8. ROOM COMPETITION STATE V1 — SOURCE AUDIT COMPLETE / BLOCKED

Frozen source plan:

`docs/research/PRISM_ROOM_COMPETITION_STATE_V1_SOURCE_PLAN.md`

Certified source audit:
- run `39039540742`;
- exact head `73542d0b0eed3fd212352e0d51c7d550decd04e2`;
- artifact `12103282541`;
- digest `sha256:98bbfb4d75727e5d49985f1ba0e4195e1bdacb53ac7cbd3a66c7495c401f64f0`.

Disposition:
`PRISM_ROOM_COMPETITION_STATE_SOURCE_BLOCKED`.

This source-only lane measured competitive room removed/remaining using:
- official inactive identity;
- strictly-prior snaps;
- finite room-state construction.

It **did not** transfer vacated opportunity to a successor.

Observed coverage:
- RB 2024: active snap 94.27%, inactive snap 78.37%, finite room 68.58%;
- RB 2025: active 93.61%, inactive 76.73%, finite room 60.16%;
- WR 2024: active 94.47%, inactive 79.42%, finite room 73.78%;
- WR 2025: active 94.80%, inactive 83.33%, finite room 83.26%;
- TE 2024: active 93.85%, inactive 72.41%, finite room 66.61%;
- TE 2025: active 94.91%, inactive 83.56%, finite room 81.17%.

Frozen coverage gates were not met, so this lane fails closed.

Do not:
- lower gates;
- infer missing inactivity from target-game snaps;
- use sportsbook listings for availability;
- convert this into the old vacancy-transfer formula;
- treat raw recent share as entitlement.

Role Elasticity V1 remains SOURCE_BLOCKED.

## 9. PROTECTED PRISM SCIENCE — DO NOT RESTART

Stage 1 active-role:
`PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED_ALL`.

Stage 2 conditional allocation:
- RB rushing QUALIFIED;
- RB receiving QUALIFIED;
- TE receiving QUALIFIED.

QB exact Stage-2 V1:
CLOSED.

WR Stage-2:
source-blocked / not scored.

Parent denominator persistence:
`PRISM_PARENT_DENOMINATOR_PERSISTENCE_PARITY_PASS`
- run `38991915503`;
- artifact `12085194208`.

Opportunity-quality closures:
- direct team high-value creation CLOSED — run `38999998874`;
- scoring-territory drive creation CLOSED — run `39000352640`.

Opponent/environment:
- boundary audited;
- generic DvP and simple usage×matchup formulas remain closed;
- no generic opponent candidate is authorized;
- richer matchup must be genuinely player-conditioned.

Reliability:
- WR opportunity trust QUALIFIED;
- WR efficiency trust QUALIFIED;
- WR dual trust historically supported / prospective confirmation still required;
- TE allocation trust QUALIFIED;
- TE efficiency trust CLOSED;
- generic efficiency-tail stability CLOSED;
- no equivalent new RB named-player trust state has been qualified.

## 10. WR DUAL RELIABILITY PROSPECTIVE PROTOCOL REMAINS PROTECTED — BUT IT IS NOT THE WHOLE PROJECT

Mechanics:
- run `39001244942` SUCCESS;
- job `117304434689`;
- exact execution head `0fbfd104cbea803d8afbbcc9c56b23414b309c88`;
- artifact `12088664427`;
- digest `sha256:384850e42c28b53171a8f3062998cdf910dcc1714e36d1abea0a1e4acbee7418`;
- 4/4 tests PASS;
- strict repository audit PASS.

Frozen prospective horizon:
2026 Weeks 7-10.

Exact fixed formula:
`dual_fragility_score = (opportunity_fragility_score + efficiency_fragility_score) / 2`.

No Week-6 dual backfill.

The prospective mechanics remain valid and should be executed when chronology allows, but **do not stop broader PRISM research while waiting**.

## 11. WEEK-6 PROSPECTIVE EVIDENCE REMAINS IMMUTABLE

Week-6 Sunday/Monday Stage-2 shadow:
- 13 games;
- 248 player-cell rows;
- SEA-DEN excluded;
- zero Week-6 outcomes read at lock.

Week-6 trust locks:
- WR opportunity run `38823520954`, artifact `12012884262`;
- WR efficiency run `38826804491`, artifact `12015596169`;
- TE trust run `38822346088`, artifact `12013018536`.

Do not:
- grade early;
- alter locks;
- manufacture a Week-6 WR dual composite;
- use Week-6 outcomes to tune the Week-4/5 forensic lane.

## 12. EXACT ACTIVE NEXT ACTION

The active task is **execution of the frozen Week-4/5 player-first forensic replay**, not more debate about whether it is allowed.

Start with the preserved Week-4 and Week-5 pregame authorities.

Build:
1. all-row Week-4/5 player-market replay table;
2. player-level situation fields;
3. named-player attribution table;
4. detailed largest/informative miss casebook;
5. matched correctly projected controls with similar position/market/role context;
6. repeated-mechanism counts only **after** player attribution;
7. list of pregame signals that existed but were not transmitted;
8. list of source-blocked/missing player-conditioned signals;
9. candidate football mechanisms ranked by importance and independent testability;
10. aggregate MAE/RMSE only as downstream QA.

For every player, ask:
- identity/history;
- current role/state/trajectory;
- teammate competition;
- finite opportunity;
- opportunity subtype/quality;
- team/game environment;
- player-conditioned matchup;
- efficiency;
- reliability;
- exact market outcome.

Do not merely label misses opportunity-vs-efficiency and stop there.

Week-5 narrower forensic evidence is already available and should seed the combined replay without predetermining attribution.

Week-4 is especially important because it allows examination of the model with only Weeks 1-3 current-season evidence plus legitimate historical identity/playbook context.

## 13. SCIENTIFIC BOUNDARY FOR THE REPLAY

Week-4/5 outcomes are already known, so this lane is **forensic diagnostic evidence**, not preregistered prospective promotion evidence.

It may identify:
- repeated player-level failure mechanisms;
- pregame information that existed but was dropped;
- source gaps;
- new football hypotheses.

It may NOT:
- fit a coefficient from Week-4/5 outcomes;
- tune thresholds on those outcomes;
- promote a subgroup because it looked good retrospectively;
- invent a DvP multiplier;
- change production directly.

Any new mechanism found here must be frozen separately and tested on independent historical/prospective evidence.

## 14. HARD GUARDRAILS

- GitHub is canonical over chat memory.
- Do not recursively read old handoffs unless this handoff explicitly points to one.
- Do not restart closed research.
- Do not weaken frozen gates.
- Do not spend a paid OddsAPI credit without explicit user authorization.
- Sportsbook is downstream only.
- No sportsbook variable may construct the independent football projection.
- Do not use 2026 outcomes as predictors in a purported pregame mechanism.
- Preserve exact source/provenance boundaries.
- No silent fallback.
- No production mutation from forensic work.

## 15. LIVE GITHUB STATE AT HANDOFF PREPARATION

Research branch before handoff-document writes:
`b4bc97a484a9c71676bdd6f6694983a3b822f33b`.

PR #672:
OPEN / DRAFT.

Main base observed from PR metadata:
`5ce513f23a48952bca69616eccee66f06a38a87d`.

Important: reconcile live branch/main/PR/Actions before any new write because concurrent work has advanced this branch multiple times during the session.

## 16. WHAT THE NEXT CHAT MUST NOT DO

Do **not**:
- ask the user to re-explain PRISM;
- reduce the task to position averages;
- treat reliability as the only active frontier;
- say “we must wait for Week 6” before using existing Week-4/5 evidence;
- backfill Week-4 with an invalid four-current-game reliability score;
- invent injury/teammate facts not preserved in the evidence;
- reopen raw recent-share replacement;
- reopen generic defense-vs-position;
- reopen the two closed team high-value persistence families;
- touch paid OddsAPI;
- stop after writing a plan.

The next chat should **continue executing** the Week-4/5 player-first replay immediately.

