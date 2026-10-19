Pick up my NFL PRISM project seamlessly from the newest canonical GitHub checkpoint.

Repo: `dkaps6/imtiredofthis`.

Do NOT ask me to re-explain anything. GitHub is canonical. This must feel like
the exact same assistant continuing, except with the assignment completely
internalized and no drift.

READ IN THIS ORDER:

1. `AGENTS.md`
2. ONLY the newest TOP checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. `docs/handoffs/NFL_HANDOFF_2026-10-19_PRISM_ELITE_ADDITIVE_TEAM_CONTEXT_IDENTITY_BRIDGE.md`
4. newest material Issue #535 comments
5. reconcile live:
   - `main`
   - branch `research-individual-opportunity-roadmap-2026-10-08`
   - PR #672
   - exact-head GitHub Actions/status

Do not recursively reread old handoffs unless the newest canonical checkpoint
explicitly requires one.

## ABSOLUTE ASSIGNMENT — DO NOT DRIFT

PRISM is an **INDIVIDUAL-PLAYER PROJECTION MODEL**.

The objective is to project the named player correctly in his actual current
football state for the exact market.

Position-level MAE/RMSE is NOT the optimization objective.
It is downstream QA/reporting only.

Permanent rule:

**THE PLAYER SHOULD HAVE THE RULES. THE POSITION SHOULD NOT HAVE THE PLAYER.**

Correct reasoning order:

PLAYER IDENTITY/HISTORY
-> CURRENT ROLE/STATE
-> EXPLAIN ABNORMAL OBSERVATIONS
-> TEAMMATE COMPETITION / FINITE OPPORTUNITY
-> OPPORTUNITY TYPE/QUALITY
-> OPPONENT/ENVIRONMENT
-> EFFICIENCY
-> EXACT MARKET OUTCOME
-> RELIABILITY

A mechanism is not ready unless you can explain:

**WHY SHOULD THIS SPECIFIC PLAYER'S PROJECTION CHANGE?**

Do not lead with position averages.
Do not chase aggregate MAE and force player rules around it.

## THIS IS ADDITIVE — DO NOT SCRAP THE MODEL

Do NOT rebuild PRISM from scratch.

We have already found a large amount of qualified / useful science.

The current player-state research is a higher-resolution layer **on top of the
existing model**.

Preserve all existing qualified:
- player baselines;
- opportunity/entitlement science;
- player-specific authority;
- uncertainty/distribution science;
- valid matchup/environment science;
- finite-opportunity architecture;
- existing production logic that survived frozen gates.

A new current-state rule may override or arbitrate a specific upstream state
only if separately source-valid and qualified.

It must not erase the rest of the model.

## PERMANENT ANOMALY RULE

**ANOMALIES ARE QUESTIONS, NOT TRENDS.**

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
- team/roster-context change;
- plausible game-state effects.

DNP is not zero performance.
An injury-shortened game is not an ordinary full-game observation.
Return-from-injury is a distinct state.
Missing source membership is not automatically zero opportunity.

If unresolved, fail closed as uncertain.

## PERMANENT JALEN COKER FAILURE CASE

2026:
- W1: 9 targets / 8-138-2
- W2: 9 / 8-66
- W3: 4 / 2-18, injury-affected early exit, did not return
- W4: DNP
- W5: bye
- W6: 12 / 6-97

The prior roughly 38-yard projection against roughly 55.5 was a BAD
INDIVIDUAL-PLAYER PROJECTION caused by false causal-state interpretation.

Never read `138 -> 66 -> 18 -> DNP` as ordinary regression.

Correct state:
strong healthy role -> strong healthy role -> injury-shortened game -> DNP ->
bye -> strong return.

Player context belongs BEFORE projection, not as a post-hoc excuse.

## PRESERVE CURRENT SCIENCE / ANTI-RETEST

Do not reopen closed families via new windows, subsets, coefficients, thresholds,
position cuts, blends or downstream markets.

Important inherited state includes:
- RB receiving Stage-2 player-specific authority QUALIFIED.
- Stage-1 direct RB/TE player-authority bridges CLOSED.
- WR signed opportunity bias QUALIFIED.
- TE signed opportunity bias QUALIFIED.
- WR/TE signed efficiency bias CLOSED.
- WR-R15 exact recovery run `39110742338` FAILED CLOSED.
- TE signed-opportunity bridge blocked by frozen combined READY_ALL source gate.
- WR/TE player opportunity uncertainty CONFIRMED.
- WR/TE component-dominance routing CLOSED ALL.
- Observation Health-State V1 SOURCE READY.
- PlayerForm Participation-State Bridge V1 SOURCE BLOCKED.
- PlayerForm Participation-State Membership V2 SOURCE READY.
- Comparable-Observation Receiving Opportunity Identity V1 CLOSED.
- Receiving Opportunity Earning / Participation Exposure V1 SOURCE READY as a
  source object, subject to exact row-membership semantics.
- Injury-Partial Receiving Exposure Decomposition V1 CLOSED.
- Return-from-Injury Role Restoration V1 SOURCE READY.
- same-QB target allocation CLOSED.
- same-QB target depth CLOSED.
- same-QB target CPOE CLOSED.
- RB/TE depth-role transition SOURCE BLOCKED.
- offensive playcaller regime transition SOURCE BLOCKED.
- restored-return injury-report persistence CLOSED.
- return-state receiving-opportunity bridge SOURCE BLOCKED.

## QUALIFIED RETURN-STATE SCIENCE

Return-State Next-Role Persistence V1 is:

`PRISM_RETURN_STATE_NEXT_ROLE_PERSISTENCE_V1_QUALIFIED_STATE_ONLY`

Authority:
- freeze `c25d5f3d19cab3f68a0c0ad0482c79bf53a6730e`
- run `39179843306` SUCCESS
- artifact `12156195287`
- digest
  `sha256:8a6ac90e3209f1ec8f53d859a2f7b2a64b84b682f033667512d99b16ddee3988`

2025 confirmation:
- 105 episodes / 95 players;
- equal-player gain +0.299474;
- 85.26% improved;
- player-bootstrap 95% CI [+0.234841,+0.365316].

Meaning:
once participation is source-certified restored, the player's own pre-injury
participation role is more faithful than carrying the interruption forward as
continued decline.

This does NOT automatically restore targets/carries/routes/efficiency/yards/TDs.

Snaps are snaps, never routes.

## CURRENT CENTRAL FRONTIER — TEAM-CONTEXT SOURCE READY

Canonical disposition:

`PRISM_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_READY`

Freeze:
`bd254ec9435cd0a17e2c7d82d78532e67c2cd23e`

Canonical run:
`39182076127` SUCCESS

Exact head:
`6c047e4df67702affde52493f7d1b90296751968`

Artifact:
`12156204035`

Digest:
`sha256:abc2a5b9d3ea2a4d683e56626ea2d33495fa1e7cad2c36c56c01071ccc0339c8`

Result:
`docs/research/PRISM_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_RESULT.md`

The earlier looser freeze `fc63601c...` is noncanonical.

Meaning:

PRISM can source-certify before kickoff that this exact named RB/WR/TE moved
from one NFL team context to another, with:
- exact stable identity;
- prior-team chronology;
- current same-family competitor metadata;
- historical 2024/2025 support;
- 2026 semantic parity.

This is a **real current-context boundary**.

It is NOT itself a boost/penalty or projection adjustment.

## FIRST PREDICTIVE TEAM-CONTEXT SEAM — RB CARRY IDENTITY

Frozen contract:
`PRISM_RB_TEAM_CONTEXT_CARRY_IDENTITY_PERSISTENCE_V1`

Freeze:
`668bd3266b6e6c131875ef245ce037b1a9925be7`

Question:

For this exact RB in his first source-certified in-season new-team context, is
his own strictly-prior old-team carry-share identity more predictive of his
first new-team carry share than the canonical current-team RB rushing
entitlement?

Canonical run:
`39182998402` SUCCESS as a fail-closed workflow.

Exact head:
`5a51ad27e38e6a87455dc78e31c1a2b2646d2868`

Disposition:

`PRISM_RB_TEAM_CONTEXT_CARRY_IDENTITY_PERSISTENCE_V1_SOURCE_BLOCKED`

Exact blocker:
- stable count linkage 0/39;
- source exclusion:
  `target_count_identity_missing: 39`;
- eligible scored events 0;
- named players scored 0.

CRITICAL:

**THIS WAS A SOURCE-LINKAGE FAILURE, NOT A PREDICTIVE FAILURE.**

No player outcome was scored.

Do NOT say:
- old-team carry identity was worse;
- the canonical current-team entitlement won;
- the team-context idea failed;
- the candidate hurt MAE.

None of that was tested.

The exact stable-ID bridge between the team-context parent and the historical
carry-count authority was unavailable under the frozen exact-match rule.

Name matching is forbidden.

## EXACT NEXT ACTION

Stay in the TEAM-CONTEXT lane.

Do not abandon `PRISM_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_READY`.

Do not switch to a random backlog topic.

FIRST, perform a source/lineage audit only:

> Is there a canonical, leakage-safe, deterministic identifier crosswalk from
> the team-context transition `stable_id` to the historical carry-count
> authority's player identity for 2024/2025?

Requirements:
- inspect exact ID semantics in the team-context artifact, RB rushing entitlement
  artifact and carry-count builder;
- find a shared provider-level stable ID only if it genuinely exists;
- no names;
- no fuzzy matching;
- no outcome-derived manual mapping;
- no target-game usage to infer identity;
- do not lower the frozen >=99% exact-link gate;
- score ZERO predictive outcomes during the bridge audit;
- do not change the frozen RB V1 formula.

If a valid exact bridge exists:
- source-certify it first;
- then execute the already-frozen RB V1 without changing gates/windows/formula.

If no exact bridge exists:
- document SOURCE_BLOCKED;
- preserve the team-context source state;
- close only the RB carry-linkage path;
- select the next distinct TEAM-CONTEXT market seam only if it already has an
  exact shared stable identity and qualified canonical baseline;
- freeze that next mechanism BEFORE scoring.

Do not choose a seam because it gives prettier numbers.

Ask:
**WHY SHOULD THIS SPECIFIC PLAYER'S PROJECTION CHANGE?**

## QUEUE GOVERNANCE

- Betting remains LOCKED/CLOSED unless I explicitly reopen it.
- Do not rebuild tickets/parlays unless explicitly asked.
- No paid OddsAPI without explicit authorization.
- Sportsbook is downstream only.
- QB passing TD remains BACKLOG ONLY unless I explicitly promote it.
- Preserve closed science and anti-retest rules.
- Never weaken source/digest/chronology/parity/byte-identity gates.
- Never claim a workflow is running unless live Actions says queued/in_progress.
- Ignore unrelated legacy push-workflow failures unless they are the active
  canonical lane.
- No production mutation from research/source qualification alone.

## COMMUNICATION STANDARD

Lead with the named player and football state.

When reporting a result, distinguish:
- source ready;
- source blocked;
- predictive qualified;
- predictive closed.

Do not confuse a missing source bridge with a bad football hypothesis.

When I say `keep going`, continue the exact active canonical lane.

Do not make me teach any of this again.

Continue execution immediately from GitHub.
