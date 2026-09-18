# Football Context Intelligence Program V1 — Master Program Map

**Status:** ACTIVE ENGINEERING PROGRAM — NO PRODUCTION SCIENCE CHANGE AUTHORIZED.

**Created:** 2026-09-17

**Canonical purpose:** preserve and coordinate the new football-information work so no valuable idea is lost, duplicated, or prematurely promoted.

This program unifies the following user-raised and repo-supported needs:

1. current player role/environment changes across seasons and teams;
2. historical matchup analogs for current player-games;
3. WR/TE vs defender exposure / eventual assignment;
4. OL vs DL / blocker-pass-rusher personnel context;
5. personnel turnover and continuity across seasons;
6. authoritative qualitative football information;
7. advanced tracking-derived geometry and protection features;
8. model-vs-market disagreement as an audit trigger, never an upstream input;
9. position-specific entitlement and efficiency experiments only after engineering is frozen.

---

# North star

Build a football-first context system capable of answering:

> Who is this player **now**, in this current role, with these teammates, coaches, blockers, quarterback, opponents and matchup conditions — and which historical football situations are actually comparable?

The system must not assume that:

- last year's player role is this year's role;
- the same team name means the same personnel;
- depth-chart rank equals workload;
- nearest defender equals coverage responsibility;
- team-level pressure equals exact blocker-rusher matchup;
- a sportsbook line is football truth;
- a historical box-score average remains the right prior after a regime change.

---

# Program architecture

```
CURRENT FOOTBALL EVIDENCE
        |
        v
PLAYER ROLE / ENVIRONMENT REGIME
        |
        +----------------------+
        |                      |
        v                      v
CURRENT PERSONNEL          HISTORICAL ADVANCED
/ MATCHUP STATE            PLAYER PROFILES
        |                      |
        +----------+-----------+
                   |
                   v
        HISTORICAL ANALOG ENGINE
                   |
                   v
        TEAM / ROOM OPPORTUNITY
                   |
                   v
          PLAYER ENTITLEMENT
                   |
                   v
             EFFICIENCY
                   |
                   v
          JOINT MONTE CARLO
                   |
                   v
        DOWNSTREAM MARKET AUDIT
```

Sportsbook information is downstream of the entire football model.

---

# Workstream A — Advanced Data Feature Contract

**Branch:** `data-frontier-advanced-feature-contract-v1`

**Status:** CI-CERTIFIED / PR #622 OPEN

Frozen fields: **47**

Sources:

- BDB 2021 route geometry;
- BDB 2023 protection geometry;
- BDB 2026 Analytics throw-window geometry.

Temporal classes:

- 13 pregame historical-derivable;
- 11 target-game post-kickoff;
- 23 retrospective-validation-only.

Permanent semantic rules:

- nearest defender = geometry only;
- BDB2026 team coverage labels != individual assignment;
- `pff_nflIdBlockedPlayer` != universal primary blocker-rusher assignment;
- raw competition data remain ephemeral;
- no predictive experiment authorized by the contract.

---

# Workstream B — Advanced Data Materialization

**Branch:** `data-frontier-advanced-feature-materializers-v1`

**Status:** ALL THREE SOURCE MATERIALIZERS PASS; FINAL CHECKPOINT / PR HANDOFF IN PROGRESS

Certified materializers:

### BDB 2021
- 78,343 route player-plays;
- 58,484 strict-prior player × route history rows;
- 12 / 12 contracted fields;
- zero target-game rows in history.

### BDB 2023
- 46,396 reconstructed protection interactions;
- 3,935 strict-prior blocker history rows;
- 13 / 13 contracted fields;
- zero target-game rows in history.

### BDB 2026
- 14,107 valid targeted-receiver throw-window plays;
- 7,161 strict-prior receiver history rows;
- 35,417 strict-prior receiver × route history rows;
- 22 / 22 contracted fields;
- zero target-game rows in history;
- zero landing/post-release fields consumed pregame.

These are engineered assets, not predictive winners yet.

---


# Workstream K — Advanced Data Signal Exploration

**Branch:** `research-advanced-data-signal-exploration-v1`

**Status:** INITIAL RECONNAISSANCE PASS — MULTIPLE PERSISTENT FAMILIES FOUND

The 47 newly materialized advanced fields are not merely support assets for the other context workstreams. They have their own dedicated research program.

Primary questions:

1. Which player-level geometry/protection traits are persistent enough to be knowable before a future game?
2. Which features are mostly play-level noise and should remain descriptive/retrospective only?
3. Which route, alignment, coverage-family and personnel strata materially change those distributions?
4. Do strict-prior player summaries preserve useful information when evaluated against later completed observations?
5. Which advanced feature families are genuinely novel versus information the current production model already captures?
6. Which feature families deserve a separately frozen position-specific predictive experiment?

Immediate non-production exploration families:

- receiver release separation persistence;
- second-defender spacing persistence;
- 2/3-yard crowding persistence;
- route-conditioned receiver spacing persistence;
- BDB2021 snap-to-throw spacing change;
- blocker snap/minimum/terminal distance persistence;
- time-to-engagement/minimum-distance persistence;
- chip/release interaction structure;
- geometry versus PFF beaten/hit/hurry/sack labels for retrospective validation;
- route × man/zone and route × detailed coverage geometry;
- player-level heterogeneity and sample-size stability;
- out-of-distribution / sparse-history rates under the frozen abstention thresholds.

Canonical initial run: `35289753485`.

Initial result:
- BDB2026 receiver release nearest spacing early/late Spearman: **0.705**
- BDB2026 receiver×route nearest spacing persistence: **0.730**
- BDB2026 strict-prior nearest spacing vs later same-family geometry: **0.482**
- BDB2021 player×route nearest throw spacing persistence: **0.593**
- BDB2023 blocker snap-separation persistence: **0.847**
- BDB2023 blocker time-to-minimum persistence: **0.575**

Result authority:
`docs/research/ADVANCED_DATA_SIGNAL_EXPLORATION_V1_RESULT.md`

The first reconnaissance therefore establishes that several newly engineered families behave like persistent football traits, while still making no yards/receptions/QB/RB predictive-lift claim.

This workstream may run descriptive, persistence, reliability, novelty and retrospective-validation analyses in parallel.

It may **not** change production or silently run a model-selection tournament. A future predictive candidate must be named, frozen and isolated by position/target mechanism.

# Workstream C — Player Role & Environment Regime

**Branch:** `research-player-role-environment-regime-v1-plan`

**Status:** CROSS-POSITION PLAN FROZEN; EVIDENCE SCHEMA V1 CI-VALIDATED; SOURCE ENGINEERING NEXT

Positions:

- QB
- RB
- WR
- TE

Core question:

> Is the current opportunity-generating football regime materially different from the historical regime represented by the player's prior sample?

Evidence families:

- team change;
- position-room turnover;
- hierarchy/depth;
- injuries/vacancies;
- coaching/play-caller changes;
- QB environment;
- OL/protection environment;
- official qualitative role evidence;
- strict-prior current-season usage;
- current personnel continuity.

Examples:

- Jahmyr Gibbs: same-team vacancy/role expansion;
- Kenneth Walker: new-team lead-role transition;
- DJ Moore: new-team primary receiver + QB/play-caller environment transition.

No fixed player boost is allowed.

Evidence contract:
- `docs/research/player_role_environment_evidence_v1.json`
- `docs/research/PLAYER_ROLE_ENVIRONMENT_EVIDENCE_V1.md`
- CI run `35290091425`: PASS

The contract preserves the existing certified current-availability authority and adds evidence families for team/room continuity, vacancy, strict-prior current-season usage, QB environment, coaching, trenches and timestamped qualitative role statements.

---

# Workstream D — RB Role Regime Specialization

**Branch:** `research-rb-role-regime-state-v1-plan`

**Status:** POSITION-SPECIFIC PLAN FROZEN; NOW SUBORDINATE TO WORKSTREAM C

This remains useful as the detailed RB specialization of the cross-position regime system.

Important non-retest rule:

The failed depth-rank remap is not reopened.

RB regime state must use a finite backfield, teammate competition, vacancy, continuity, participation and role transition evidence rather than "RB1 = carries."

---

# Workstream E — Historical Matchup Analog Engine

**Parent plan:** `FOOTBALL_MATCHUP_PERSONNEL_CONTEXT_V1`

**Status:** PLANNED / ENGINEERING NOT YET BUILT

Core question:

> Which completed historical player-games were most similar to the current target player's **pregame football state**?

Similarity dimensions:

- role regime;
- team opportunity state;
- opponent structure;
- player archetype;
- advanced strict-prior geometry history;
- personnel continuity;
- QB/play-caller environment;
- OL/front state.

Forbidden similarity inputs:

- target-game outcome;
- target-game realized coverage;
- target-game realized pressure;
- sportsbook lines;
- final score;
- final player stat.

Output:

- ranked historical analogs;
- component distances;
- evidence coverage;
- out-of-distribution state;
- abstention state.

Predictive use is a later, separately frozen experiment.

---

# Workstream F — WR / TE Defender Exposure and Assignment

**Parent plan:** `FOOTBALL_MATCHUP_PERSONNEL_CONTEXT_V1`

**Status:** PARTIALLY UNBLOCKED BY NEW BDB DATA

Historical repo authority:

- M84 = `HOLD_SOURCE_BLOCKED_NEW_INFORMATION`
- the football hypothesis was not rejected;
- exact assignment source was missing.

Three evidence levels:

### Level 1 — on-field defender context
Who is active/on field.

### Level 2 — geometry/proximity exposure
Who is nearest, how often, how close, how stable the proximity relationship is.

This is now materially engineerable from BDB tracking.

Potential fields:

- nearest-defender exposure share;
- second-defender exposure share;
- percent route frames within 1/2/3 yards;
- nearest-defender identity continuity;
- proximity switches;
- release nearest defender;
- throw nearest defender;
- crowding.

### Level 3 — true assignment/responsibility
Requires explicit labeled responsibility truth.

Until that exists:

**primary proximity defender != primary coverage defender**

Forward collection of approved current WR-CB reports is encouraged so the project can begin building its own timestamped historical archive.

---

# Workstream G — OL / DL / Pass-Rush Personnel Context

**Parent plan:** `FOOTBALL_MATCHUP_PERSONNEL_CONTEXT_V1`

**Status:** PARTIALLY UNBLOCKED BY BDB 2023 LAB; LIVE EXACT ASSIGNMENT STILL UNAVAILABLE

Historical repo authority:

- M85 = `HOLD_SOURCE_BLOCKED_NEW_INFORMATION`
- exact blocker-rusher football mechanism was not rejected;
- historical/live exact-assignment source was missing.

New BDB 2023 capability:

- blocker-target interaction geometry;
- minimum distance;
- terminal distance;
- time to minimum;
- protection-window length;
- chip/release semantics;
- blocker historical profiles.

Current personnel layer should track:

### OL
- expected LT/LG/C/RG/RT;
- starter continuity;
- backups/new starters;
- injuries;
- strict-prior snaps;
- available pass-block history.

### Defense/front
- primary edge rushers;
- interior rushers;
- rotations;
- continuity;
- injuries;
- strict-prior pressure history.

Do not fabricate one exact OL-DL assignment when it is not known.

---

# Workstream H — Personnel Continuity / Historical Sample Validity

**Status:** REQUIRED CROSS-CUTTING LAYER

Every team/player historical statistic should carry a continuity context.

Examples:

- percentage of prior opportunity represented by current teammates;
- current starter overlap;
- current OL overlap;
- current receiving-room overlap;
- current defensive-front overlap;
- QB continuity;
- play-caller continuity;
- current player same-team history;
- roster turnover.

Purpose:

> Historical team data should not silently represent a roster that no longer exists.

Potential states:

- `HIGH_CONTINUITY`
- `MODERATE_CONTINUITY`
- `LOW_CONTINUITY`
- `MAJOR_PERSONNEL_REGIME_CHANGE`
- `INSUFFICIENT_CONTINUITY_EVIDENCE`

---

# Workstream I — Qualitative Football Evidence

**Status:** REQUIRED SOURCE-ENGINEERING LAYER

Qualitative information is allowed only as structured evidence.

Examples:

- "bell cow"
- "lead back"
- "committee"
- "primary outside receiver"
- "slot role"
- "goal-line role"
- "third-down role"
- "starter"
- "limited role"
- "workload expansion"
- "workload contraction"

Each record must retain:

- source;
- publication time;
- speaker;
- authority tier;
- direct quote vs reporter interpretation;
- player/team;
- normalized concept;
- confidence;
- pre-kickoff validity;
- expiry/freshness.

No quote maps directly to yards, carries, targets or receptions.

---

# Workstream J — Model vs Market Disagreement Audit

**Status:** DESIGN FROZEN IN ROLE-REGIME PLAN; IMPLEMENTATION LATER

Vegas remains downstream.

When model-vs-market difference exceeds a frozen threshold, trigger a football audit:

1. current role state;
2. roster/transaction state;
3. current hierarchy;
4. injuries/vacancies;
5. QB environment;
6. coaching/play caller;
7. current-season strict-prior usage;
8. OL/front personnel;
9. opponent matchup context;
10. advanced historical profile;
11. analog support;
12. opportunity vs efficiency source of disagreement.

Possible disposition:

- `FOOTBALL_INPUT_DEFECT_FOUND`
- `FOOTBALL_ROLE_STATE_STALE`
- `MODEL_MECHANISM_DISAGREEMENT_NO_INPUT_DEFECT`
- `INSUFFICIENT_EVIDENCE`

Vegas itself never alters upstream football inputs.

---

# What can run in parallel

The following engineering can proceed simultaneously because none fits outcomes:

- source inventories;
- schemas/contracts;
- feature materialization;
- current roster/transaction/personnel collection;
- qualitative-evidence normalization;
- proximity/exposure materializers;
- blocker-interaction archetype builders;
- historical analog-state assembly;
- temporal leakage audits;
- coverage/abstention audits;
- forward archival pipelines.

This is encouraged.

---

# What must NOT all run at once

Predictive model experiments should be staged.

Reasons:

1. attribution — know which information actually helped;
2. avoid interaction soup;
3. preserve frozen hypotheses;
4. avoid tuning against the same holdout repeatedly;
5. protect production anchors;
6. preserve ability to reject a bad idea cleanly.

Recommended scientific sequence after engineering is frozen:

1. role/environment regime -> player opportunity/entitlement;
2. historical analog support as calibration/context;
3. receiver geometry history -> WR/TE efficiency or entitlement as separately justified;
4. trench personnel / blocker geometry -> QB/RB opportunity/efficiency;
5. proximity-exposure matchup features;
6. later true assignment features only if an honest label/source becomes available.

One candidate at a time per position/target mechanism.

---

# Master dependency graph

```
ADVANCED FEATURE CONTRACT ------> ADVANCED MATERIALIZERS ----+
                                                             |
ROLE / ENVIRONMENT EVIDENCE --------------------------------+
                                                             |
CURRENT PERSONNEL / TRANSACTIONS ----------------------------+--> ANALOG STATE ENGINE
                                                             |          |
OL / FRONT PERSONNEL ----------------------------------------+          |
                                                             |          v
PROXIMITY EXPOSURE ------------------------------------------+   POSITION-SPECIFIC
                                                                        RESEARCH
QUALITATIVE ROLE EVIDENCE -----------------------------------+          |
                                                                        v
                                                               QUALIFIED MODEL
                                                                        |
                                                                        v
                                                               MARKET DISAGREEMENT
                                                                     AUDIT
```

---

# Current "do not lose this" inventory

The following user-raised ideas are explicitly captured:

- player role changes season to season;
- team changes;
- new QB environment;
- new primary WR/RB/TE role;
- teammate departures and added opportunity;
- current-season usage superseding stale historical usage;
- established stars being more predictable than novel-role players;
- historical games similar to today's matchup;
- WR-CB matchup/assignment;
- nearest-defender tracking as exposure but not assignment;
- OL vs DL matchup;
- OL personnel turnover;
- DL/front personnel turnover;
- historical team stats becoming stale after personnel change;
- qualitative football knowledge / coach statements;
- game-script history;
- market discrepancies as "why are we different?" audits;
- opportunity and efficiency separated;
- uncertainty should widen when current role is genuinely novel.

Nothing above should be discarded merely because another lane starts first.

---

# Immediate execution order

Engineering may proceed concurrently:

### Now
1. finalize advanced materializer checkpoint / PR;
2. freeze `PLAYER_ROLE_ENVIRONMENT_EVIDENCE_V1` schema;
3. inventory current transaction/roster/depth/coaching/personnel sources;
4. build current personnel continuity schema;
5. design historical analog-state schema;
6. extend BDB route geometry into deterministic proximity-exposure summaries;
7. build BDB protection interaction archetype summaries;
8. establish forward collection design for authoritative role statements and WR-CB reports;
9. run Advanced Data Signal Exploration V1 persistence/reliability/novelty audits across all three validated labs.

### After engineering QA
10. generate coverage/abstention reports;
11. quantify historical cohort sizes / out-of-distribution frequency;
12. rank advanced-data families by persistence, temporal usability, novelty and evidence quality — not by outcome-tuned winner selection;
13. freeze first position-specific predictive candidate.

### Only after qualification
14. shadow live 2026;
15. compare against production;
16. consider promotion.

---

# Production protection

No workstream in this master program currently authorizes changes to:

- QB M89/M90/C2;
- WR M38 / WR-R15;
- TE-R5P;
- RB P3 / R26 / R22;
- Full Slate production science;
- Issue #535.

---

# Master disposition

`FOOTBALL_CONTEXT_INTELLIGENCE_PROGRAM_V1_ACTIVE_ENGINEERING`
