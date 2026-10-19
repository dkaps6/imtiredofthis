# NFL PRISM ELITE HANDOFF — 2026-10-19
## PLAYER-FIRST / ADDITIVE MODEL / TEAM-CONTEXT FRONTIER / NO RETEACH

Repository: `dkaps6/imtiredofthis`

Research branch:
`research-individual-opportunity-roadmap-2026-10-08`

Primary research PR:
`#672`

Canonical issue:
`#535`

This handoff exists so the next conversation can continue as the same assistant,
with the assignment fully internalized and without making the user teach the
project again.

---

# 1. ABSOLUTE ASSIGNMENT — THIS OVERRIDES ANY TEMPTATION TO DRIFT

PRISM is an **INDIVIDUAL-PLAYER PROJECTION MODEL**.

The objective is:

> Project the named player correctly in his actual current football state for
> the exact market.

Position-level MAE/RMSE is **NOT** the optimization objective.

MAE/RMSE is downstream QA/reporting. It should improve because named-player
projections improve.

Permanent rule:

> **THE PLAYER SHOULD HAVE THE RULES. THE POSITION SHOULD NOT HAVE THE PLAYER.**

Correct reasoning order:

`PLAYER IDENTITY/HISTORY
-> CURRENT ROLE/STATE
-> EXPLAIN ABNORMAL OBSERVATIONS
-> TEAMMATE COMPETITION / FINITE OPPORTUNITY
-> OPPORTUNITY TYPE/QUALITY
-> OPPONENT/ENVIRONMENT
-> EFFICIENCY
-> EXACT MARKET OUTCOME
-> RELIABILITY`

A mechanism is not ready unless the assistant can answer:

> **WHY SHOULD THIS SPECIFIC PLAYER'S PROJECTION CHANGE?**

Do not lead with position averages.
Do not optimize a position aggregate and then force player rules around it.

---

# 2. THIS WORK IS ADDITIVE — DO NOT SCRAP THE MODEL

The user explicitly does **NOT** want the existing PRISM model thrown away.

A large amount of useful science has already been found and frozen.

The current research program is an **additional player-state / arbitration
layer on top of the existing projection stack**.

Do not restart the model.
Do not rewrite qualified components merely because a new player-state object is
being studied.
Do not discard prior production logic that survived its own frozen gates.

The architecture is:

1. preserve the existing player baselines / entitlements / qualified
   opportunity science;
2. add higher-resolution named-player current-state evidence only where it is
   separately source-valid and predictively qualified;
3. preserve teammate competition and finite opportunity;
4. translate opportunity to efficiency / matchup / market outcomes only after
   the upstream state is established;
5. use aggregate metrics only as downstream QA.

New science must be **additive and governed**, not a wholesale replacement.

---

# 3. PERMANENT ANOMALY RULE

> **ANOMALIES ARE QUESTIONS, NOT TRENDS.**

Before an abnormal observation becomes trajectory evidence, explain why it
happened.

When source-valid, reconcile:
- injury/practice/game designation;
- active/inactive/DNP state;
- offensive snaps;
- routes only when actual route data exist;
- carries/targets/receptions;
- early exit / partial game;
- return from injury;
- QB/starter/coaching changes;
- teammate availability;
- team change / roster context;
- plausible game-state effects.

DNP is not zero performance.

An injury-shortened game is not an ordinary full-game observation.

Return-from-injury is a distinct current state.

Missing source membership is not automatically zero performance/opportunity.

If an abnormal observation cannot be resolved cleanly, fail closed as
uncertain.

---

# 4. PERMANENT JALEN COKER FAILURE CASE

2026:
- W1: 9 targets / 8-138-2
- W2: 9 / 8-66
- W3: 4 / 2-18, injury-affected early exit, did not return
- W4: DNP
- W5: bye
- W6: 12 / 6-97

The prior model/chat produced roughly a 38-yard projection against roughly
55.5 and used regression logic. The user reported roughly 60 yards in the
first half.

That was a **bad individual-player projection caused by false causal-state
interpretation**.

Never read:

`138 -> 66 -> 18 -> DNP`

as ordinary regression.

Correct state:

`strong healthy role
-> strong healthy role
-> injury-shortened game
-> DNP
-> bye
-> strong-return role`

Player context must affect the projection **before** the projection is made.
It is not a post-hoc excuse.

This case is an example of the reasoning standard, not a reason to make all of
PRISM an injury model.

---

# 5. CORE SCIENCE THAT MUST REMAIN PRESERVED

Preserve the existing validated / frozen science and anti-retest boundaries.

Important inherited state includes:

- RB receiving Stage-2 player-specific authority — **QUALIFIED**.
- RB rushing Stage-2 hard authority family — preserve canonical disposition;
  do not retune under the current lane.
- Stage-1 direct RB/TE player-authority bridges — **CLOSED**.
- WR signed opportunity bias — **QUALIFIED**.
- TE signed opportunity bias — **QUALIFIED**.
- WR/TE signed efficiency bias — **CLOSED**.
- WR-R15 exact recovery run `39110742338` — **FAILED CLOSED**.
  Never rescue from rebuilt bytes.
- TE signed-opportunity bridge blocked by frozen combined READY_ALL source gate.
  No TE-only rescue.
- WR/TE player opportunity uncertainty — **CONFIRMED**.
- exact historical specialist-distribution integration — source blocked;
  prospective pregame shadow only.
- WR/TE component-dominance routing — **CLOSED ALL**.
- Observation Health-State V1 — **SOURCE READY**.
- PlayerForm Participation-State Bridge V1 — **SOURCE BLOCKED**.
- PlayerForm Participation-State Membership V2 — **SOURCE READY**.
- Comparable-Observation Receiving Opportunity Identity V1 — **CLOSED**.
- Receiving Opportunity Earning / Participation Exposure V1 — **SOURCE READY**
  as a source object, subject to its PlayerForm-row membership semantics.
- Injury-Partial Receiving Exposure Decomposition V1 — **CLOSED**.
- Return-from-Injury Role Restoration V1 — **SOURCE READY**.
- same-QB target allocation V1 — **CLOSED**.
- same-QB target-depth V1 — **CLOSED**.
- same-QB target-CPOE V1 — **CLOSED**.
- RB/TE discrete depth-role transition — **SOURCE BLOCKED**.
- playcaller regime transition V1 — **SOURCE BLOCKED**.
- restored-return injury-report persistence V1 — **CLOSED**.
- return-state receiving opportunity V1 — **SOURCE BLOCKED**.

Do not reopen a closed family merely by changing:
- windows;
- coefficients;
- subsets;
- positions;
- thresholds;
- blending;
- downstream market;
- one convenient season.

A new mechanism must be **genuinely new information**, not a renamed rescue.

---

# 6. QUALIFIED RETURN-STATE SCIENCE

## Return-State Next-Role Persistence V1

Disposition:

`PRISM_RETURN_STATE_NEXT_ROLE_PERSISTENCE_V1_QUALIFIED_STATE_ONLY`

Authority:
- freeze:
  `c25d5f3d19cab3f68a0c0ad0482c79bf53a6730e`
- run:
  `39179843306` — SUCCESS
- exact scored head:
  `57c962a935f25a6fbad521c3aad43dc2db3e4bb9`
- artifact:
  `12156195287`
- digest:
  `sha256:8a6ac90e3209f1ec8f53d859a2f7b2a64b84b682f033667512d99b16ddee3988`

2024:
- 84 scored episodes;
- 79 named players;
- equal-player gain +0.336361;
- median +0.300000;
- improved/tied/worsened:
  86.08% / 0.00% / 13.92%.

2025 independent confirmation:
- 105 episodes;
- 95 named players;
- equal-player gain +0.299474;
- median +0.255000;
- improved/tied/worsened:
  85.26% / 0.00% / 14.74%;
- player-bootstrap 95% CI:
  [+0.234841, +0.365316].

Meaning:

> once this named player's participation is source-certified restored, his own
> pre-injury participation role is materially more faithful for the next clean
> participation observation than carrying the interruption forward as
> continued decline.

This is **participation-state only**.

It does not automatically restore:
- targets;
- carries;
- routes;
- receptions;
- efficiency;
- yards;
- touchdowns.

Snaps are snaps, never routes.

Participation restoration is separate from target earning.
Target earning is separate from efficiency.

---

# 7. RETURN-STATE DOWNSTREAM BOUNDARIES

## Receiving opportunity bridge

Canonical corrected disposition:

`PRISM_RETURN_STATE_RECEIVING_OPPORTUNITY_V1_SOURCE_BLOCKED`

Authoritative chronology freeze:
`930907dd91c1b4139f2aa6be52db23a730c7a96b`

Canonical corrected run:
`39180575868` — SUCCESS

Exact head:
`af3ca4469ae3fad59b219a36a79adbbb1b3ed640`

Artifact:
`12156510164`

Digest:
`sha256:ff57c3b2f3b700444ffb5fcc89f9ffa98242861711c7f143ec948d6ad3a03bf6`

The old run `39180266614` is **NONCANONICAL** because its chronology could
skip reported no-snap / unresolved observations.

Exact blocker:
the immutable PlayerForm-derived source can contain
`playerform_weekly_row_present=false` while carrying a filled numeric
`targets=0`.

PRISM may not interpret a missing player target row as true zero target earning.

The corrected >=3-row strictly-prior exact target support was only:
- 2024: 68.71%;
- 2025: 69.94%;

versus the frozen 95% gate.

Finite team target opportunity itself passed cleanly.

Do not lower the target-membership gate.

## Injury-report persistence

Disposition:

`PRISM_RESTORED_RETURN_INJURY_REPORT_STATE_PERSISTENCE_V1_CLOSED`

Authority:
- freeze:
  `513a5e5ee29b909aeaff387396352e1a10d83633`
- run:
  `39180544730` SUCCESS
- artifact:
  `12155682815`
- digest:
  `sha256:0e868c6424f76abd423e4aff6cdebdeec6d5c4690f212759dfd4c67cc292fec8`

2024 had inadequate no-report subgroup support.

2025:
- no-report return equal-player clean-next: 66.07%;
- report-present: 63.03%;
- gain +0.030462;
- player-cluster 95% CI:
  [-0.169569, +0.221526].

Do not rescue with body part, practice status, designation subsets,
position/family cuts, alternate lookahead or changed return thresholds.

The injury-specific lane is finished except for the already-qualified
participation-state rule.

---

# 8. PLAYCALLER REGIME SOURCE — BLOCKED

Disposition:

`PRISM_OFFENSIVE_PLAYCALLER_REGIME_TRANSITION_V1_SOURCE_BLOCKED`

Authority:
- freeze:
  `533e628c99359fbfddbb9f4d787a96d8dbfbc67c`
- run:
  `39181153541` SUCCESS
- exact head:
  `f62d72df3a8b9ca0ce2491d58ac800504006d053`
- artifact:
  `12155618757`
- digest:
  `sha256:0bfcebc29d7ddeaf07c565afc6f1ac010d1974473a9475d966fb32ccac8ab5ec`

The playcaller identity object itself was reconstructable.

2024:
- caller coverage 100%;
- 19 caller-change events;
- 484 transition player-weeks / 445 named players;
- >=3 strictly-prior same-team positive-participation support:
  8.47%;
- frozen minimum:
  40%.

2025:
- 18 change events;
- 438 transition player-weeks / 382 players;
- prior-role support:
  42.24%;
- source gates passed.

2026 prospective caller/effective-week parity passed.

The blocker was 2024 same-player role-history support.

Do not:
- lower the 40% gate;
- use only 2025;
- exclude offseason transitions post hoc;
- substitute a position/team prior for player identity;
- merge playcaller and QB identity as a rescue.

No skill-player predictive outcome was scored.

---

# 9. CURRENT CENTRAL SCIENCE — PLAYER TEAM-CONTEXT TRANSITION SOURCE READY

This is the important current frontier.

Disposition:

`PRISM_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_READY`

Canonical freeze:

`bd254ec9435cd0a17e2c7d82d78532e67c2cd23e`

Canonical source run:
`39182076127` — SUCCESS

Exact source head:
`6c047e4df67702affde52493f7d1b90296751968`

Job:
`117836769489` — SUCCESS

Artifact:
`12156204035`

Digest:
`sha256:abc2a5b9d3ea2a4d683e56626ea2d33495fa1e7cad2c36c56c01071ccc0339c8`

Result doc:
`docs/research/PRISM_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_RESULT.md`

Important lineage:
the earlier looser freeze
`fc63601c34760d042adcc00fff12976496622574`
and run `39181783189` are noncanonical diagnostics only.

The later `bd254ec9...` contract is authoritative because it additionally
requires:
- zero unresolved multi-team stable-ID player-weeks;
- stronger transition/family support;
- >=4 previous-team roster observations;
- current same-family roster-competition metadata.

## 2024

- admitted player-weeks: 11,599;
- players: 867;
- scheduled team-week roster coverage: 100%;
- stable-ID coverage: 100%;
- transition events: 329;
- transition players: 287;
- offseason / in-season:
  216 / 113;
- previous-team >=4 roster observations:
  268 / 329 = 81.46%;
- previous-team >=8:
  75.38%;
- same-family competitor metadata:
  100%.

By family:
- RB: 90 events / 77 players;
- WR: 161 / 141;
- TE: 78 / 69.

## 2025

- admitted player-weeks: 11,732;
- players: 839;
- roster coverage: 100%;
- stable-ID coverage: 99.92%;
- transition events: 351;
- transition players: 271;
- offseason / in-season:
  202 / 149;
- previous-team >=4:
  281 / 351 = 80.06%;
- previous-team >=8:
  68.66%;
- competitor metadata:
  100%.

By family:
- RB: 105 events / 73 players;
- WR: 159 / 129;
- TE: 87 / 69.

## 2026 prospective parity

- latest complete observed roster week: 6;
- complete team coverage: 100%;
- stable-ID coverage: 99.85%;
- transitions: 257;
- players: 240;
- offseason / in-season:
  216 / 41;
- parity gates failed:
  NONE.

## Scientific meaning

PRISM may now source-certify:

> this named player's current team / competition environment is not the same
> environment that generated his immediately prior roster history.

This is a real current-context boundary.

It does **NOT** mean:
- new team = boost;
- new team = penalty;
- old role persists automatically;
- old role resets automatically;
- vacated opportunity transfers automatically;
- position-average transfer is valid.

Those remain predictive questions.

---

# 10. FIRST TEAM-CONTEXT PREDICTIVE SEAM — RB CARRY IDENTITY V1

Frozen contract:

`PRISM_RB_TEAM_CONTEXT_CARRY_IDENTITY_PERSISTENCE_V1`

Freeze:
`668bd3266b6e6c131875ef245ce037b1a9925be7`

Question:

> For this exact RB in his first source-certified in-season new-team context,
> is his own strictly-prior old-team carry-share identity more predictive of
> his first new-team carry share than the canonical current-team RB rushing
> entitlement?

Important:
this was a legitimate player-first identity-vs-current-context question.

It preserved:
- the named RB as the primary unit;
- old-team strictly-prior carry identity;
- canonical new-team RB rushing entitlement as baseline;
- finite current-team RB + residual room mass;
- teammate competition;
- 2024 development / 2025 confirmation;
- no yards/efficiency/TD translation;
- no 2026 outcomes;
- no sportsbook/OddsAPI.

## CANONICAL RESULT — SOURCE BLOCKED BEFORE PREDICTIVE SCORING

Canonical run:
`39182998402` — SUCCESS as a fail-closed workflow.

Exact head:
`5a51ad27e38e6a87455dc78e31c1a2b2646d2868`

Disposition:

`PRISM_RB_TEAM_CONTEXT_CARRY_IDENTITY_PERSISTENCE_V1_SOURCE_BLOCKED`

Exact source blocker:

- stable-count linkage:
  **0 / 39 = 0.00%**
- source exclusion:
  `target_count_identity_missing: 39`
- eligible scored events:
  **0**
- named players scored:
  **0**
- outcomes compared:
  **0**

This is crucial:

> **THE FOOTBALL HYPOTHESIS DID NOT FAIL A PREDICTIVE TEST.**
>
> **THE REQUIRED EXACT IDENTITY BRIDGE TO THE CARRY-COUNT AUTHORITY FAILED
> BEFORE ANY PLAYER OUTCOME WAS SCORED.**

Do not describe this as:
- old-team carry identity was bad;
- canonical entitlement beat old-team identity;
- the candidate worsened MAE;
- the team-context idea failed.

None of those were tested.

The result is source-blocked because the team-context parent's exact
`stable_id` could not be linked to the canonical carry-count universe's
identity key under the frozen exact-match rule.

Name-only matching was correctly forbidden.

The workflow was repaired to fail closed cleanly rather than fabricate rows.

Latest exact research head at this result:
`5a51ad27e38e6a87455dc78e31c1a2b2646d2868`.

---

# 11. EXACT NEXT ACTION AFTER THIS HANDOFF

Stay inside the **team-context transition lane**.

Do not abandon the source-ready team-context boundary merely because the first
RB market bridge lacked an exact count-ID join.

Do not immediately switch to a random backlog idea.

The next legal action is source/lineage only:

## PRISM RB Team-Context Count Identity Bridge — source audit first

Question:

> Is there a canonical, leakage-safe, deterministic identifier crosswalk that
> maps the team-context transition `stable_id` to the historical carry-count
> authority's player identity for 2024/2025, without names, fuzzy matching or
> target-game outcomes?

Requirements:

1. inspect the exact identity semantics in:
   - canonical team-context transition artifact;
   - canonical RB rushing entitlement artifact;
   - canonical historical carry-count builder/universe;
2. identify whether both ultimately carry a shared provider-level stable ID
   (for example GSIS/player ID) that was lost only in an intermediate frame;
3. no name-only matching;
4. no fuzzy matching;
5. no manual one-off player map derived from outcomes;
6. no target-game usage to infer identity;
7. no lowering the frozen >=99% exact-link gate;
8. no predictive scoring during this source audit;
9. preserve the frozen RB V1 formula and gates if an exact bridge is genuinely
   source-valid;
10. if no exact deterministic bridge exists, document
    `SOURCE_BLOCKED` and do not manufacture one.

If a clean exact-ID bridge passes:
- only then may the already-frozen RB team-context carry V1 be executed on the
  exact frozen contract;
- do not change history windows / coefficients / subgroups after seeing data.

If the exact bridge remains blocked:
- preserve
  `PRISM_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_READY`;
- close only the RB carry-count linkage path;
- remain in the team-context state family and select the next **distinct
  market-specific seam that already has an exact shared stable identity and a
  qualified canonical baseline**;
- freeze that next mechanism before scoring.

Do not jump to WR/TE or another market merely to find better numbers.
Choose the next seam only if source identity and football semantics support it.

---

# 12. QUEUE GOVERNANCE

- Sunday betting remains LOCKED/CLOSED unless the user explicitly reopens it.
- Do not rebuild tickets/parlays unless explicitly asked.
- No paid OddsAPI without explicit authorization.
- Sportsbook is downstream only.
- QB passing TD remains BACKLOG ONLY unless explicitly promoted.
- Preserve closed science and anti-retest rules.
- Never weaken byte identity / digest / parity / chronology gates.
- Never claim a workflow is running unless a fresh live Actions read shows
  queued/in_progress.
- Ignore unrelated legacy workflow failures on push unless they are the active
  canonical lane.
- No production mutation from source-only or research qualification work unless
  a separately frozen integration authority is later passed.

---

# 13. GITHUB / LIVE-STATE RESTART PROTOCOL

The next chat must not rely on this handoff alone for live status.

Read in this order:

1. `AGENTS.md`
2. ONLY the newest TOP checkpoint in
   `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. this handoff
4. newest material Issue #535 comments
5. reconcile live:
   - `main`
   - branch
     `research-individual-opportunity-roadmap-2026-10-08`
   - PR #672
   - exact-head GitHub Actions/status.

Do not recursively reread every old handoff unless the newest canonical docs
explicitly require it.

At handoff creation, the important live research state was:

- research branch head:
  `5a51ad27e38e6a87455dc78e31c1a2b2646d2868`
- PR #672:
  open / draft;
- current central source result:
  `PRISM_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_READY`;
- current market attempt:
  `PRISM_RB_TEAM_CONTEXT_CARRY_IDENTITY_PERSISTENCE_V1_SOURCE_BLOCKED`;
- exact blocker:
  stable carry-count identity linkage 0/39.

Always reconcile because branch/main may advance after this document is written.

---

# 14. COMMUNICATION STANDARD

Lead with the named player and football state.

Never lead with:
- position MAE;
- generic regression;
- position-average rules.

When explaining research, state:
1. what exact named-player football state is being represented;
2. why it might change this player's exact market;
3. whether the source is valid before scoring;
4. whether the predictive relationship actually passed;
5. what the result does **not** authorize.

When the user says **keep going**:
- continue the exact canonical lane;
- do not switch research families on your own;
- do not promote backlog work because it looks interesting.

When a run is source blocked:
- distinguish source failure from predictive failure.

When a signal closes:
- preserve the rest of PRISM.

When a signal qualifies:
- do not automatically mutate production;
- require the separately frozen downstream integration/authority step.

---

# 15. FINAL CONTINUITY PRINCIPLE

The project is not trying to build the smallest position-level MAE model.

It is trying to build a football model that understands:

> **WHO IS THIS PLAYER, WHAT STATE IS HE IN NOW, WHY IS THAT STATE DIFFERENT,
> HOW MUCH OPPORTUNITY DOES HE OWN RELATIVE TO TEAMMATES, WHAT KIND OF
> OPPORTUNITY IS IT, AND WHAT SHOULD THAT MEAN FOR THIS EXACT MARKET?**

The existing PRISM stack already contains valuable science.

The new state-arbitration work sits **on top of it**.

Do not scrap the model.

Do not erase qualified work.

Do not use a blocked bridge as proof that the underlying football state is
worthless.

Get the players right first.

Aggregate statistics follow.
