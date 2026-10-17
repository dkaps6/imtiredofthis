# PRISM 2026 WEEK 4-5 PLAYER-FIRST FORENSIC REPLAY V1 — FROZEN PROTOCOL

Date: 2026-10-17

Status: **FROZEN BEFORE NEW REPLAY ATTRIBUTION / DIAGNOSTIC ONLY / NO PRODUCTION CHANGE**

Model:
**PRISM — Player Role & Individual Situation Model**

## 1. Governing question

The scientific object is the **named player in his actual pregame football situation**.

The replay asks, for each eligible Week-4 and Week-5 player-market projection:

> What did PRISM know before kickoff about this specific player, what did it transmit into the projection, what did it fail to transmit, and which part of the player outcome was actually wrong?

Canonical reasoning order:

`PLAYER IDENTITY/HISTORY -> CURRENT ROLE/STATE/TRAJECTORY -> TEAMMATE COMPETITION -> FINITE OPPORTUNITY ALLOCATION -> OPPORTUNITY TYPE/QUALITY -> TEAM/GAME ENVIRONMENT -> PLAYER-CONDITIONED OPPONENT MATCHUP -> INDIVIDUAL EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

This is not a position-average optimization exercise.

Aggregate MAE/RMSE may be reported only after player-level attribution.

## 2. Replay weeks

Primary replay:
- 2026 Week 4;
- 2026 Week 5.

Week 4 is included because the project already preserved a genuine pregame authority and complete postmortem.

Week 5 is included for the same reason and additionally permits any mechanism that requires four completed current-season games, where the exact same-player history requirement is satisfied.

The frozen WR dual-reliability rule requiring >=4 prior current-season games is **not** retroactively manufactured for Week 4. Week 5 may be evaluated under that exact four-prior-game requirement where eligible.

Week 6 prospective locks/shadows remain immutable and are not read, graded, altered, or backfilled by this replay.

## 3. Canonical preserved authorities

Week 4 pregame board:
- recovered run `36935917903`;
- artifact `11197776900`;
- digest `sha256:3f058570037ca016a5cbf1fa79e6e6abc4d845384de4dbdfaf477c8f0a3160a8`;
- 873 player-market rows;
- 16 games;
- no new odds acquisition permitted.

Week 4 postmortem:
- settlement run `37482840038`;
- artifact `11421298874`;
- expanded failure-mode run `37485600879`;
- artifact `11423535484`.

Week 5 pregame full-slate authority:
- run `37860592615`;
- artifact `11585973062`;
- digest `sha256:bf52aa4bc30bf6e8cc6029a0a86c0f33cdfaa09f587c38a138f23cef6b7436f4`;
- production behavior authority `24704e5f1ef872ec86c36446d2c73f6d377c824c`.

Week 5 postmortem:
- run `38395384228`;
- artifact `11812200405`;
- player failure casebook run `38397361891`;
- artifact `11812403368`.

No sportsbook refetch is allowed.

## 4. Required player-level replay record

For each eligible player-market row, preserve the frozen pregame projection and reconstruct only information that was genuinely available before kickoff.

Required fields / questions:

### A. Player identity and historical playbook
- stable player identity;
- prior-season/career role and production context available at the time;
- established versus emerging/unstable role;
- same-player historical opportunity and efficiency tendencies where source-valid.

### B. Current role/state/trajectory
- current-season participation;
- snaps/routes/dropbacks as applicable;
- carries/targets/attempts and shares;
- recent role trajectory;
- starter/committee/bellcow/hierarchy state;
- injury/availability information genuinely available pregame.

### C. Teammate competition
- relevant active/inactive teammates;
- room concentration;
- vacancy/return effects;
- finite team opportunity competition.

### D. Opportunity allocation
- team parent opportunity quantity;
- named-player allocation/share;
- whether the projection missed quantity before efficiency is considered.

### E. Opportunity type and quality
Market-specific, not generic:
- rushing opportunity vs target opportunity vs reception conversion;
- scoring-area/high-value role only where pregame source semantics are valid;
- route/target depth or carry-role subtype only where exact source authority exists.

### F. Team/game environment
- offense pass/run tendency;
- pace/play volume;
- QB/offensive environment relevant to this player's opportunity;
- game-script information that existed pregame.

### G. Player-conditioned opponent interaction
Do not use generic defense-vs-position labels.

Only record opponent information if it answers:
> how did this opponent interact with this named player's actual opportunity/role type?

If exact player-conditioned source parity is unavailable, mark it unavailable rather than inventing a multiplier.

### H. Individual efficiency
- expected and realized conversion conditional on opportunity;
- rushing efficiency, catch/YPT efficiency, QB YPA, etc., market-specific;
- distinguish opportunity miss from efficiency miss.

### I. Reliability
- same-player projection difficulty/trust where a previously qualified mechanism legally applies;
- Week 5 may additionally evaluate the exact >=4-prior-current-season-game reliability condition where eligible;
- reliability never creates over/under direction by itself.

### J. Market-specific outcome
Carries/targets/receptions/yards/TDs are separate questions.

Do not infer that a good yards projection implies a good TD projection or vice versa.

## 5. Attribution taxonomy

Every reviewed miss must receive one primary attribution and optional secondary attribution from:

- `PLAYER_IDENTITY_PRIOR_MISREAD`
- `CURRENT_ROLE_STATE_MISREAD`
- `TEAMMATE_COMPETITION_MISREAD`
- `OPPORTUNITY_QUANTITY_MISREAD`
- `OPPORTUNITY_TYPE_QUALITY_MISREAD`
- `TEAM_ENVIRONMENT_NOT_TRANSMITTED`
- `PLAYER_CONDITIONED_MATCHUP_NOT_TRANSMITTED`
- `EFFICIENCY_MISREAD`
- `RELIABILITY_OVERCONFIDENCE`
- `AVAILABILITY_IDENTITY_FAILURE`
- `ORDINARY_OUTCOME_VARIANCE`
- `SOURCE_BLOCKED_OR_UNAVAILABLE`
- `MIXED`.

A field may not be labeled a model defect merely because the realized result was surprising.

## 6. Correct-case controls

The replay must not inspect only failures.

For every failure family, include representative correctly projected players with similar position/market/role context.

Purpose:
- prevent outcome-selected storytelling;
- distinguish a real missing mechanism from normal football variance;
- identify what PRISM already understands correctly.

## 7. Evidence boundary

This replay is diagnostic.

It may:
- identify repeated player-level failure mechanisms;
- identify pregame information that existed but was dropped before final projection;
- identify source gaps;
- generate separately testable football hypotheses.

It may NOT:
- fit a coefficient from Week 4/5 outcomes;
- choose a threshold from Week 4/5 outcomes;
- create a generic defense-vs-position multiplier;
- promote a player subgroup because it looked good retrospectively;
- change production;
- alter Week-6 locks;
- spend an OddsAPI credit;
- use sportsbook information upstream.

Any proposed PRISM mechanism arising from this replay must be frozen separately and tested on independent historical/prospective evidence.

## 8. Required outputs

1. all-row Week-4/5 replay table with preserved pregame projection authority;
2. named-player attribution table;
3. detailed casebook of the largest and most informative misses;
4. matched correct-case controls;
5. repeated-mechanism counts after attribution;
6. list of pregame signals that existed upstream but were not transmitted;
7. list of genuinely missing/source-blocked player-conditioned signals;
8. candidate research questions ranked by football importance and independent testability;
9. aggregate MAE/RMSE only as downstream QA.

## 9. Existing facts that seed but do not predetermine attribution

Week 4 already established:
- current football matchup data existed upstream but much of it was not transmitted into generic RB/WR/TE projection;
- asymmetric upper-tail misses were substantial;
- Bijan was an architecture illustration of player/current-offense information being muted downstream.

Week 5 already established among the 40 largest final-model misses:
- 19 efficiency-dominant;
- 14 opportunity-dominant;
- 7 mixed opportunity/efficiency.

Those labels are starting evidence, not permission to skip the deeper player-first replay.

## 10. Ticket boundary

Downstream order remains:

`FOOTBALL PROJECTION -> RELIABILITY/TRUST -> MARKET COMPARISON -> APPROVED PLAY -> ALT-LINE/CORRELATION/TICKET CONSTRUCTION`.

The Dobbins-style question is the governing intuition:
the model must understand why a named player is likely to receive and convert the **specific opportunity relevant to the market**, rather than treating position averages or generic edge magnitude as the football explanation.

## 11. Integrity

At protocol freeze:
- new Week-4/5 outcome-driven model fits: 0;
- Week-6 outcomes read: 0;
- Week-6 locks changed: 0;
- sportsbook refetches: 0;
- OddsAPI calls: 0;
- production changes: false.

Disposition:
`PRISM_WEEK4_5_PLAYER_FIRST_FORENSIC_REPLAY_V1_FROZEN`.
