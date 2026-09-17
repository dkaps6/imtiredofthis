# RB Role Regime State V1 — Frozen Research Plan

**Status:** PLAN ONLY — NO PREDICTIVE RESULT, NO PRODUCTION CHANGE.

**Created:** 2026-09-17

**Branch:** `research-rb-role-regime-state-v1-plan`

## Motivation

Current RB production can be materially stale when a player's football role changes faster than his historical usage profile.

Examples motivating this plan:

- an incumbent back loses a long-standing split partner and becomes the clear lead back;
- a player changes teams into a materially different backfield;
- a teammate injury/vacancy creates a temporary or permanent workload expansion;
- a prior committee collapses into a concentrated room;
- a player's pass-down or goal-line role changes even when depth-chart rank does not.

This is **not** a revival of the failed Role-Order Remap V1. Depth chart is contextual evidence only and must never directly assign carries.

This plan operationalizes the already-authorized RB architecture:

`team rushing opportunity -> finite RB room -> individual entitlement -> efficiency -> joint MC`

with entitlement informed by current-role regime evidence.

## Permanent separation rules

1. Sportsbook lines are downstream audit/decision information only.
2. Market disagreement may trigger an audit but may not change the football projection.
3. Role evidence must be available before target-game kickoff.
4. Same-game partial usage is forbidden.
5. Historical usage does not automatically dominate after a verified regime change.
6. A current role state changes the **opportunity prior / entitlement uncertainty**, never directly the rushing-yard outcome.
7. Depth chart alone is insufficient workload authority.
8. No M95 tail family, R23-R27D receiving-efficiency family, or failed generic role remap may be reopened by this plan.

## Core concept

Create an explicit pregame `RB_ROLE_REGIME_STATE_V1` artifact at player-team-week grain.

The state answers:

> Is this player's expected opportunity-generating football role materially different from the regime represented by his historical sample?

It does not answer:

> How many carries will he get?

The carry/reception projection remains a finite-team allocation problem.

## Proposed state taxonomy

- `STABLE_LEAD`
  - same team / same core competition;
  - sustained lead role;
  - no material vacancy or new entrant.

- `STABLE_COMMITTEE`
  - same team;
  - multiple credible competitors remain;
  - no authoritative evidence of consolidation.

- `VACANCY_EXPANSION`
  - meaningful prior competitor departed, was traded, released, retired, or became definitively unavailable;
  - remaining player has credible lead-role evidence;
  - historical share may be stale downward.

- `NEW_TEAM_LEAD`
  - player changed teams;
  - current pregame evidence supports lead role in new room;
  - old-team workload share is not assumed transportable.

- `NEW_TEAM_COMMITTEE_OR_UNCERTAIN`
  - player changed teams but lead entitlement is not established.

- `INJURY_TEMP_EXPANSION`
  - current teammate unavailability creates a temporary workload expansion.

- `ROLE_CONTRACTION`
  - new credible competitor, demotion, or reduced role signal makes prior usage stale upward.

- `ROLE_UNCERTAIN`
  - conflicting or insufficient authoritative evidence.

- `NO_CURRENT_ROLE_EVIDENCE`
  - fail-soft state when current-role evidence cannot be established.

## Evidence families

Evidence is recorded independently. No one evidence family except definitive availability should automatically determine carry share.

### A. Current roster / transaction evidence

Pregame-valid examples:

- player changed teams;
- incumbent competitor traded/released/retired;
- meaningful RB added;
- rostered credible competitors count changed;
- transaction date relative to target kickoff.

Derived fields:

- `team_changed_since_prior_season`
- `credible_competitors_added`
- `credible_competitors_departed`
- `vacated_prior_carry_share`
- `vacated_prior_target_share`
- `returning_backfield_continuity`

### B. Current depth / role evidence

Depth chart is contextual evidence only.

Fields:

- `official_depth_rank`
- `depth_rank_changed`
- `depth_source_authority`
- `depth_freshness_days`

No formula may convert depth rank directly into a workload number.

### C. Availability / vacancy evidence

Fields:

- `definitive_unavailable_competitor_count`
- `questionable_competitor_count`
- `injury_created_vacated_share`
- `availability_state_confidence`

### D. Official qualitative football evidence

This is the structured home for the "intangible" information the user is describing.

Allowed sources:

- official team site;
- official coach/GM/player press conference/transcript;
- NFL.com reporting that directly attributes role statements;
- other separately approved authoritative football sources.

Examples of admissible statements:

- "bell-cow";
- "lead back";
- "committee";
- "third-down role";
- "goal-line role";
- "most of the carries";
- explicit coach statement that workload will expand/contract.

Every statement must preserve:

- source URL/reference;
- publication timestamp;
- quoted/normalized role phrase;
- speaker;
- speaker authority type;
- target player/team;
- available-before-kickoff flag;
- confidence;
- whether statement is direct or reporter interpretation.

Proposed normalized fields:

- `official_lead_role_signal`
- `official_committee_signal`
- `official_pass_down_role_signal`
- `official_goal_line_role_signal`
- `official_workload_expansion_signal`
- `official_workload_contraction_signal`
- `role_statement_confidence`

Qualitative evidence becomes structured **role evidence**, not an arbitrary manual yardage adjustment.

### E. Strict-prior observed usage

For Week 2 onward, completed current-season games are new pregame evidence.

Fields:

- prior-1 / prior-3 snap share;
- route participation;
- carry share;
- target share;
- two-minute participation;
- goal-line opportunity share where source supports it.

Target-game box score remains forbidden.

### F. Team / competition context

- finite team rush opportunity;
- QB rushing competition;
- number and quality of active RB competitors;
- expected RB room concentration;
- offensive-line/protection/run-context inputs where separately qualified.

## Role regime confidence

Do not use one opaque hand-set "football IQ score."

Emit separate evidence dimensions plus a transparent confidence tier:

### HIGH
Multiple authoritative sources agree, or a transaction/availability fact plus official role evidence clearly establishes the transition.

### MEDIUM
Strong structural evidence but no direct official workload statement, or one direct authoritative role signal with limited corroboration.

### LOW
Depth chart / roster implication only, weak or conflicting statements, or unresolved competition.

### ABSTAIN
Identity, timing, or source authority cannot be verified.

## History-reset / transition handling

A verified role transition should change how much historical workload evidence is trusted.

Do **not** assign fixed carry boosts.

Instead, candidate entitlement models may later test:

- lower weight on stale old-regime carry share;
- higher weight on current-room competition and recent current-season usage;
- partial-pooling priors from historical analogous transition states;
- wider uncertainty when role change is real but magnitude is unknown.

The transition mechanism must be learned/validated historically under a frozen plan. The live 2026 Gibbs/Walker cases must not be used to tune coefficients after observing outcomes.

## Market disagreement audit

Sportsbook information remains downstream.

For every live RB prop, create an audit record when model-vs-market disagreement exceeds a pre-frozen threshold.

The audit asks:

1. Is the player's current role regime correctly classified?
2. Is current roster/transaction evidence current?
3. Is a meaningful competitor absent or newly added?
4. Is strict-prior current-season usage being consumed?
5. Is team rush opportunity reasonable?
6. Is QB rush competition reasonable?
7. Is the disagreement opportunity-driven or efficiency-driven?
8. Does authoritative qualitative evidence conflict with the model's assumed role?

The audit **does not** ask the model to move toward Vegas merely because Vegas differs.

If football evidence reveals a stale/missing input, repair that football input and rerun independently. If no football defect is found, preserve the disagreement.

## Example interpretation — Jahmyr Gibbs

Pregame evidence can support a `VACANCY_EXPANSION` state when:

- historical split partner has departed;
- official depth/current room reflects reduced established competition;
- authoritative team/NFL reporting explicitly describes lead/bell-cow workload expectations;
- availability evidence confirms no equivalent replacement is active.

The model should therefore treat old split-era carry share as potentially stale, but the exact new share must still come from a validated finite-room entitlement model.

## Example interpretation — Kenneth Walker III

Pregame evidence can support `NEW_TEAM_LEAD` when:

- player changed teams;
- official current depth identifies him as lead back;
- old-team competitors and usage are not assumed to transfer;
- new-team competitor room and pass-down role are evaluated;
- completed current-season usage may enter only after those games are over.

Again, the state changes the prior/regime context, not the final carry count by fiat.

## Proposed historical evaluation

Before implementation, build a historical transition panel using only information available before target kickoff.

Required cohorts:

- same-team stable lead;
- same-team committee;
- incumbent after meaningful competitor departure;
- new-team lead candidate;
- injury-created temporary expansion;
- role contraction/new entrant;
- uncertain/conflicting role.

Primary scientific question:

> Does a role-regime-aware entitlement prior reduce carry/reception opportunity error versus the current production-equivalent baseline, especially in transition cohorts, without degrading stable-role cohorts?

Score separately:

- carry MAE/RMSE/bias;
- target/reception opportunity error;
- rushing-yard point MAE after unchanged efficiency;
- receiving-yard point MAE after unchanged efficiency;
- p75/p90 absolute error;
- 20+ carry / lead-back cohorts;
- new-team cohort;
- vacancy-expansion cohort;
- season stability;
- catastrophic misses.

## Anti-retest rules

This candidate must not be:

- simple depth-rank remapping;
- a generic RB1 multiplier;
- a market-implied workload model;
- another M95 carry-tail overlay;
- another historical YPC/YPR/YAC transform;
- an after-the-fact Gibbs/Walker patch;
- a manual player whitelist/blacklist.

## Immediate implementation sequence

1. Source audit current transaction/depth/availability inputs already in repo.
2. Define authoritative role-statement ingestion contract.
3. Build `RB_ROLE_REGIME_STATE_V1` deterministic classifier with no model coefficients.
4. Build historical leakage-safe transition panel.
5. Freeze one candidate entitlement integration before observing outcomes.
6. Run multiseason walk-forward test.
7. If it qualifies, shadow against 2026 live weeks before production promotion.

## Production boundary

This document authorizes **planning and source/feature engineering only**.

It does not authorize:
- modifying P3/R26/R22 production science;
- changing Week-2+ live projections;
- using market lines upstream;
- merging an RB model change into production;
- posting new scientific direction into Issue #535 without explicit coordination.

Disposition:

`RB_ROLE_REGIME_STATE_V1_PLAN_FROZEN_PENDING_SOURCE_ENGINEERING`
