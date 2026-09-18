# NFL HANDOFF — 2026-09-18 — FOOTBALL CONTEXT INTELLIGENCE / CLAUDE-MAUDE CURRENT

**Repository:** `dkaps6/imtiredofthis`  
**Canonical rule:** GitHub is authoritative; chat memory is secondary.  
**Program branch:** `research-football-context-program-v1`  
**Verified branch head at handoff creation:** `732ed7a063b10e7199b5235da052fba76e2064b1`  
**Verified production main at handoff creation:** `f0dad2c6711e85104eeffedfa5f5112fd172cbf5`

> IMPORTANT: an automated GPT engineering process is still advancing `research-football-context-program-v1`. Verify the live branch head before doing anything. Read this handoff as the canonical program map, then inspect commits after the SHA above.

---

## 1. Purpose of this handoff

This file exists so Claude/Maude can rejoin the NFL Stuff project without asking the user to reconstruct prior work.

The current program is not a restart of WR/RB/QB research. It is a new cross-cutting engineering/research layer intended to answer:

> Who is this player **now**, in this role, on this team, with this quarterback, coaching staff, blockers, teammates, opponent and matchup context — and which historical football situations are genuinely comparable?

The user wants football knowledge that can materially improve player projections, not merely more statistics.

The program is therefore building a leakage-safe context layer for:
- role changes;
- team changes;
- vacancy/opportunity changes;
- current-season usage;
- personnel continuity;
- coaching/QB environment;
- historical analog state;
- WR/TE defender proximity;
- OL/DL personnel and protection context;
- advanced BDB geometry;
- structured qualitative football evidence;
- downstream model-vs-market discrepancy audits.

None of this authorizes production-model changes yet.

---

## 2. Collaboration / branch rule

GPT has an hourly engineering process operating against:

`research-football-context-program-v1`

Claude/Maude should:

1. fetch the latest head of that branch;
2. read the handoff and contracts below;
3. branch from the latest verified head;
4. work on a separate dedicated Claude/Maude branch;
5. keep commits narrow and auditable;
6. report exact SHAs, Actions run IDs, job IDs, artifacts and digests;
7. avoid direct competing pushes onto the GPT branch while the hourly process is active.

Suggested branch naming:
- `claude-football-context-qualification-v1`
- or a narrower branch reflecting the exact implementation lane.

---

## 3. Hard scope boundaries

### Do not interfere with Issue #535
Another chat/lane handles the active WR research and GPT-5.6 ↔ Claude collaboration there.

Do not:
- change frozen Issue #535 plans;
- post new direction there;
- rerun candidate experiments there;
- retune WR science there;
- take over that research.

### Do not change production science
Do not modify or retune:
- QB M89/M90;
- QB C2;
- WR M38 / WR-R15;
- TE-R5P;
- RB P3;
- RB R26;
- RB R22;
- Full Slate production science.

### No new paid odds pull
The fresh Week-2 paid Full Slate pull was already consumed. No new OddsAPI / paid sportsbook acquisition is authorized.

### Sportsbook is downstream only
Sportsbook information may:
- price;
- compare;
- trigger an audit.

It may not:
- teach football projections;
- pick feature thresholds;
- select analogs;
- create role multipliers;
- tune model weights.

### No uncontrolled predictive experiments
Engineering, QA, stability, coverage, novelty, redundancy and source work are authorized.

Predictive lift testing requires:
1. qualification;
2. a separate frozen scientific plan;
3. frozen cohorts, metrics and gates before outcomes are inspected.

---

## 4. Read first

After `AGENTS.md` and the root `CURRENT_NFL_RESEARCH_HANDOFF.md`, read:

1. `docs/research/FOOTBALL_CONTEXT_INTELLIGENCE_PROGRAM_V1.md`
2. `docs/research/FOOTBALL_CONTEXT_ENGINEERING_CHECKPOINT_2026-09-18.md`
3. `docs/research/HISTORICAL_DATA_REUSE_POLICY_V1.md`
4. `docs/research/FOOTBALL_CONTEXT_FEATURE_JOIN_CONTRACT_V1.md`
5. `docs/research/FOOTBALL_CONTEXT_SIGNAL_QUALIFICATION_V1.md`
6. `docs/research/FOOTBALL_CONTEXT_QUALIFICATION_EXECUTION_CHECKPOINT_V1.md`
7. `docs/research/ROLE_ENVIRONMENT_EVENT_LEDGER_CONTRACT_V1.md`
8. `docs/research/PERSONNEL_CONTINUITY_CONTRACT_V1.md`
9. `docs/research/HISTORICAL_ANALOG_INDEX_CONTRACT_V1.md`
10. `docs/research/DEFENDER_PROXIMITY_EXPOSURE_CONTRACT_V1.md`
11. `docs/research/OL_DL_PROTECTION_CONTEXT_CONTRACT_V1.md`

Related branches:
- `research-player-role-environment-regime-v1-plan`
- `research-advanced-data-signal-exploration-v1`
- `data-frontier-advanced-feature-contract-v1`
- `data-frontier-advanced-feature-materializers-v1`

Open PRs that must not be auto-merged:
- PR #622 — advanced feature dictionary V1
- PR #623 — advanced feature materializers V1

---

## 5. Production checkpoint

Verified production `main` at handoff creation:

`f0dad2c6711e85104eeffedfa5f5112fd172cbf5`

This includes the repaired Week-2 Full Slate operational stack and PR #621 quarantine-lineage fix.

Fresh paid Full Slate:
- run `35282021679`
- paid artifact `10523345092`
- digest `sha256:6024ed21d7032e6e6440145277d9d4b74e98f319ef286962bac2a786a8d28e3c`

Offline no-credit replay:
- run `35282629447`
- artifact `10523710840`
- digest `sha256:9556993fa0ca4d01552690c651d868b3d8dbb6f71c892cf8ff05d79da83978a5`

Production boundaries:
- QB production authority remains intact;
- WR authority remains intact;
- TE authority remains intact;
- RB P3 / R26 / R22 specialist authorities are not broadly qualified for Week 2;
- Week-2 RBs use generic calibrated ensemble/fallback behavior;
- ATD science is not certified.

Do not alter these from the context program.

---

## 6. Why the context program exists

The user observed that ordinary historical averages often miss real football transitions such as:
- RB becoming bell-cow after teammate departure;
- new-team lead-back role;
- WR moving teams and becoming primary target;
- QB environment materially changing;
- injuries opening workload;
- return-from-injury transitions;
- coaching/play-caller changes;
- OL turnover;
- front-seven turnover;
- historical team numbers representing players who are no longer on the roster.

The goal is not to apply narrative boosts.

The goal is to create structured, timestamped, strict-prior evidence that can tell the model:
- when old history is still representative;
- when old history may be stale;
- when uncertainty should widen;
- when a new regime has enough support to become established.

---

## 7. Conceptual architecture

```
GAME ENVIRONMENT
        ↓
TEAM OPPORTUNITY
        ↓
POSITION / ROOM OPPORTUNITY
        ↓
PLAYER ROLE / ENVIRONMENT REGIME
        ↓
PLAYER ENTITLEMENT
        ↓
EFFICIENCY
        ↓
JOINT MONTE CARLO
        ↓
DOWNSTREAM MARKET AUDIT
```

Equivalent context flow:

```
CURRENT FOOTBALL EVIDENCE
        ↓
PLAYER ROLE / ENVIRONMENT REGIME
        ↓
CURRENT PERSONNEL + HISTORICAL ADVANCED PROFILES
        ↓
HISTORICAL ANALOG ENGINE
        ↓
TEAM / ROOM OPPORTUNITY
        ↓
PLAYER ENTITLEMENT
        ↓
EFFICIENCY
        ↓
JOINT MONTE CARLO
```

Sportsbook remains downstream.

---

## 8. Historical data: reuse first

The repository already has a canonical position-agnostic historical player-game pipeline:
- `scripts/backtest/historical_player_logs.py`
- `scripts/backtest/historical_context.py`
- `scripts/backtest/build_historical_inputs.py`
- `scripts/player_form_v2.py`

Historical research has repeatedly materialized 2019–2024 and related cohorts.

The historical base includes ordinary player-game information such as:
- team/opponent;
- pass attempts/yards;
- carries/rush yards;
- targets/receptions/receiving yards;
- target share;
- rush share;
- team opportunity denominators;
- routes only when source-qualified;
- stable identities;
- schedule-validated games.

Frozen policy:

`HISTORICAL_DATA_REUSE_FIRST_V1_FROZEN`

Order:
1. current preserved canonical artifact;
2. existing derived historical asset;
3. deterministic rehydration from the canonical builder if exact rows are needed;
4. new source only if the information is genuinely absent.

Do not redownload ordinary game logs merely because a new research lane starts.

New-source acquisition is justified for genuinely new information such as:
- tracking geometry;
- protection geometry;
- current transaction/role context;
- coaching/play-caller changes;
- qualitative role statements;
- exact WR-CB responsibility;
- live/current OL/DL personnel.

A historical-base manifest builder now exists:

`scripts/research/build_historical_base_manifest_v1.py`

Purpose:
- season coverage;
- row counts;
- schema;
- keys;
- source versions;
- hashes;
- builder lineage.

---

## 9. Football-context branch lineage

At handoff creation the program branch head was:

`732ed7a063b10e7199b5235da052fba76e2064b1`

Recent lineage:

- `732ed7a...` — enforce strict-prior room continuity semantics
- `8e202f87...` — materialize strict-prior room continuity context
- `3a651b74...` — enforce strict-prior usage regime semantics
- `f35e751a...` — materialize strict-prior usage regime context
- `3e5cf06e...` — qualification execution-readiness checkpoint
- `00bd38e9...` — candidate-profile tests
- `1201eda9...` — candidate-profile builder
- `e178ab5a...` — strict-prior stability tests
- `57d4f466...` — strict-prior stability evidence builder
- `6c92ba67...` — qualification fail-closed tests
- `ea65974e...` — qualification inventory materializer
- `92dec401...` — freeze context signal qualification V1
- `22d49a3e...` — freeze OL/DL protection context contract
- `6dea2afb...` — freeze defender proximity exposure contract
- `e683540a...` — freeze historical analog index contract
- `a354c7a0...` — freeze personnel continuity contract
- `ee818ccd...` — freeze role/environment event ledger contract
- `ed623fba...` — football-context engineering checkpoint
- `2905f614...` — freeze feature join contract
- `af1445ee...` — test historical-base manifest builder
- `5a007750...` — add historical-base manifest builder
- `af09aa0a...` — freeze historical reuse-first policy

Verify live commits after this list.

---

## 10. Strict-prior usage regime materializer

File:

`scripts/research/build_usage_regime_context.py`

Purpose: derive player-game context using only games before the target player-game.

Current outputs include:
- prior team;
- same-team-as-prior-game;
- team-change indicator;
- career games prior;
- games with current team prior;
- prior target share;
- rolling 3-game target-share mean;
- rolling 5-game target-share mean;
- latest target-share delta versus rolling 3;
- prior rush share;
- rolling 3/5 rush-share means;
- rush-share delta versus rolling 3;
- prior route rate where real route data exists;
- rolling route-rate history;
- explicit cold-start/no-prior-game state.

Potential uses:
- team-change detection;
- new-team cold starts;
- role expansion/contraction diagnostics;
- current-role versus stale-career-history identification.

No predictive-lift claim has been made.

---

## 11. Strict-prior room continuity materializer

File:

`scripts/research/build_room_continuity_context.py`

Grain: team × season × week × position room.

Uses only completed prior games.

Current outputs include:
- prior room-game count;
- prior top-1 target-share concentration;
- prior top-2 target-share concentration;
- prior top-1 rush-share concentration;
- prior top-2 rush-share concentration;
- returning target-opportunity overlap;
- returning rushing-opportunity overlap;
- explicit no-prior-room-game state.

Semantic rule:

This is descriptive continuity, NOT an opportunity-inheritance model.

A departed 30% target share means the room changed. It does not prove one specific player receives 30%.

---

## 12. Player Role & Environment Regime program

Related branch:

`research-player-role-environment-regime-v1-plan`

Verified head at handoff creation:

`821e5267ea1dbd306081aa921b0de98c49866c7c`

Core question:

> Is the current opportunity-generating football regime materially different from the historical regime represented by the player's prior sample?

Evidence families:
1. team continuity;
2. room continuity;
3. depth/hierarchy context;
4. availability/vacancy;
5. coaching/scheme continuity;
6. QB environment;
7. OL/protection environment;
8. authoritative qualitative role evidence;
9. strict-prior current-season usage;
10. current personnel continuity.

Shared states include:
- `STABLE_SAME_ROLE`
- `EXPANDED_ROLE`
- `CONTRACTED_ROLE`
- `NEW_TEAM_ROLE_ESTABLISHED`
- `NEW_TEAM_ROLE_UNCERTAIN`
- `NEW_SYSTEM_ROLE_ESTABLISHED`
- `NEW_SYSTEM_ROLE_UNCERTAIN`
- `INJURY_CREATED_EXPANSION`
- `RETURN_FROM_INJURY_TRANSITION`
- `ROOKIE_OR_NEW_STARTER_TRANSITION`
- `CONFLICTING_ROLE_EVIDENCE`
- `INSUFFICIENT_CURRENT_ROLE_EVIDENCE`

Position-specific concepts exist for QB/RB/WR/TE.

No state maps directly to a fixed yard/target/carry boost.

---

## 13. Role/environment evidence schema

Files:
- `docs/research/player_role_environment_evidence_v1.json`
- `docs/research/PLAYER_ROLE_ENVIRONMENT_EVIDENCE_V1.md`
- `scripts/research/validate_player_role_environment_evidence_v1.py`
- `tests/test_player_role_environment_evidence_v1.py`

Approximate field count: 51.

Recent transition diagnostics include:
- prior-season target share;
- prior-season rush share;
- prior-season reception share;
- current-minus-prior target-share delta;
- current-minus-prior rush-share delta;
- current-minus-prior reception-share delta;
- cold-start role-sample flag.

These are descriptive transition diagnostics, not direct multipliers.

CI run:
`35290091425`

PASS.

---

## 14. Structured qualitative football evidence

Allowed concepts include:
- lead role;
- committee role;
- workload expansion;
- workload contraction;
- primary outside receiver;
- primary slot receiver;
- primary target;
- pass-down role;
- two-minute role;
- goal-line role;
- blocking-heavy role;
- starter confirmed;
- starter uncertain;
- limited role.

Preferred sources:

Tier A:
- official team transactions/depth/injury information;
- official coach/GM/player pressers/transcripts;
- NFL/league official.

Tier B:
- official team editorial/reporting;
- NFL.com direct-attributed reporting.

Tier C:
- separately approved high-quality reporting if direct sources unavailable.

Each record retains:
- source/reference;
- publication timestamp;
- pre-kickoff validity;
- speaker/authority;
- quote vs interpretation;
- normalized concept;
- player/team;
- confidence;
- freshness/expiry.

Never translate a quote directly into yards/carries/targets.

---

## 15. Personnel continuity

Frozen contract:

`docs/research/PERSONNEL_CONTINUITY_CONTRACT_V1.md`

Goal:

Historical team statistics should not silently represent a roster that no longer exists.

Potential context:
- current starter overlap;
- skill-room overlap;
- receiving-room overlap;
- backfield overlap;
- OL overlap;
- defensive-front overlap;
- QB continuity;
- play-caller continuity;
- departed target share;
- departed rush share;
- departed reception share;
- current-player same-team history;
- roster turnover.

Potential states:
- `HIGH_CONTINUITY`
- `MODERATE_CONTINUITY`
- `LOW_CONTINUITY`
- `MAJOR_PERSONNEL_REGIME_CHANGE`
- `INSUFFICIENT_CONTINUITY_EVIDENCE`

Vacancy is evidence that history may be stale, not proof of who inherits opportunity.

---

## 16. Historical analog index

Frozen contract:

`docs/research/HISTORICAL_ANALOG_INDEX_CONTRACT_V1.md`

This is GENERAL INFRASTRUCTURE. It is not the failed QB conditional analog architecture.

Target grain:

`season, week, game_id, player_id, market_family`

Similarity dimensions may include:
- role regime;
- team opportunity;
- opponent structure;
- player archetype;
- current-team tenure;
- advanced geometry history;
- personnel continuity;
- QB environment;
- coaching/play caller;
- OL/front state.

Forbidden neighbor-selection inputs:
- target-game result;
- target-game usage;
- target-game realized coverage;
- target-game pressure;
- final score;
- target residual;
- sportsbook line/result.

Required sequence:
1. build target pregame descriptor;
2. build strict-prior candidate pool;
3. calculate deterministic distances;
4. freeze neighbor IDs and hash;
5. only later attach historical outcomes if separately authorized.

Potential support states:
- `ANALOG_HIGH_SUPPORT`
- `ANALOG_MEDIUM_SUPPORT`
- `ANALOG_LOW_SUPPORT`
- `OUT_OF_DISTRIBUTION`
- `ABSTAIN`

Direct analog outcome replacement is not authorized.

---

## 17. Failed QB conditional analog — stay closed

Prior QB conditional analog V1 failed closed with:

`NO_ACTIONABLE_QB_CONDITIONAL_ANALOG_RELIABILITY`

Do not:
- rerun;
- rescue;
- retune;
- reinterpret;
- disguise it as this new analog infrastructure.

Any future scientific analog hypothesis requires a new frozen plan.

---

## 18. WR/TE defender proximity

Frozen contract:

`docs/research/DEFENDER_PROXIMITY_EXPOSURE_CONTRACT_V1.md`

Three evidence levels:

1. on-field defender context;
2. geometric exposure/proximity;
3. explicit coverage responsibility.

Level 2 is now materially engineerable from tracking.

Permanent semantic rule:

> nearest defender != coverage assignment

Allowed:
- nearest distance;
- second-nearest distance;
- defenders within 1/2/3 yards;
- release-window proximity;
- route-window proximity;
- throw-arrival proximity;
- nearest-defender identity continuity;
- proximity switches;
- crowding;
- route-conditioned proximity profiles.

Forbidden without explicit responsibility truth:
- "covered by";
- "assigned defender";
- "shadowed";
- "CB1 matchup."

Existing source audit:
`scripts/backtest/audit_wr_cb_source.py`

Historical result:
`NO_GO_TRUE_ASSIGNMENT`

M84 disposition:
`HOLD_SOURCE_BLOCKED_NEW_INFORMATION`

BDB unblocks proximity/exposure, not authoritative assignment.

---

## 19. OL/DL protection context

Frozen contract:

`docs/research/OL_DL_PROTECTION_CONTEXT_CONTRACT_V1.md`

Offensive context may include:
- LT/LG/C/RG/RT;
- returning starters;
- snap-weighted continuity;
- starts-together count;
- replacement flags;
- center/tackle changes;
- injuries;
- new starters;
- strict-prior individual history.

Defensive-front context may include:
- primary edge rushers;
- interior rushers;
- front continuity;
- missing high-share rushers;
- pressure contribution;
- rotation continuity;
- injuries.

BDB2023 semantic rule:

`pff_nflIdBlockedPlayer` is a blocking interaction identity.

It is NOT universal blocker-rusher responsibility.

Allowed:
- interaction count;
- repeated interaction;
- geometry;
- blocker archetype;
- time to minimum;
- protection-window context.

M85 disposition:
`HOLD_SOURCE_BLOCKED_NEW_INFORMATION`

Do not fabricate exact one-on-one responsibility.

---

## 20. Advanced Big Data Bowl sources

Three source labs are qualified.

### BDB2021 / 2018 route geometry
Slug:
`nfl-big-data-bowl-2021`

Hash:
`55de76561799514779f3fd64b57c02a52430596f4f104c2b6060f3df8fd9e1b4`

Coverage:
2018 Weeks 1–17

Rows:
- 78,343 route player-plays
- 58,484 strict-prior player×route history rows

### BDB2023 / 2021 protection geometry
Slug:
`nfl-big-data-bowl-2023`

Hash:
`1c3e1eb6fcd0cf85807c649804fa0c72421c76e973e40a742523f1dd3a929182`

Coverage:
2021 Weeks 1–8

Rows:
- 46,526 interactions
- 46,396 reconstructed
- ~99.7249% reconstruction
- 3,935 strict-prior blocker history rows

### BDB2026 Analytics / 2023 throw-window geometry
Slug:
`nfl-big-data-bowl-2026-analytics`

Hash:
`228554c6600ac4e73529e5b6309193c8a3371acca7b4bc59792188acd814fb07`

Coverage:
2023 Weeks 1–18

Rows:
- 14,107/14,108 valid targeted-receiver release geometries
- 7,161 strict-prior receiver-history rows
- 35,417 strict-prior receiver×route rows
- 93 route×coverage groups
- 24 route×man-zone groups

No same-game tracking may enter pregame features.

---

## 21. Advanced feature dictionary and materializers

PR #622:
`Data frontier: freeze advanced feature dictionary V1`

Branch:
`data-frontier-advanced-feature-contract-v1`

Verified head:
`baec3c946f4bbdc3e96e3846438feb57ee183dd5`

47 fields:
- BDB2021: 12
- BDB2023: 13
- BDB2026: 22

Temporal classes:
- 13 pregame historical-derivable
- 11 target-game post-kickoff
- 23 retrospective-validation-only

Strict prior:
`observation_end_time < target_kickoff_utc`

PR #623:
`Data frontier: materialize advanced feature layer V1`

Branch:
`data-frontier-advanced-feature-materializers-v1`

Verified head:
`47bcd58aecf453f54b3f5db06a9dbdc94b000ad2`

Canonical materialization run:
`35288075143`

Jobs:
- contract tests `105424726256`
- BDB2021 `105424726223`
- BDB2023 `105424726233`
- BDB2026 `105424726089`

Repo CI:
`35288076803`

All pass.

Checkpoint:
`docs/data_frontier/NFL_ADVANCED_FEATURE_MATERIALIZATION_V1_CHECKPOINT_2026-09-17.md`

Disposition:
`NFL_ADVANCED_FEATURE_MATERIALIZATION_V1_CERTIFIED`

Raw competition data remains ephemeral; only sanitized QA artifacts are uploaded.

---

## 22. Advanced-data signal exploration: real persistence found

Branch:

`research-advanced-data-signal-exploration-v1`

Verified head:

`00d5342eb4abedd836ea20fc22f15d27f168b39b`

Canonical successful run:

`35289753485`

Jobs:
- tests `105429807465`
- BDB2021 `105429866553`
- BDB2023 `105429866555`
- BDB2026 `105429866559`

Result:
`ADVANCED_DATA_SIGNAL_EXPLORATION_V1_INITIAL_RECONNAISSANCE_PASS_MULTIPLE_PERSISTENT_FAMILIES`

Important findings:

### BDB2026 receiver-level
Nearest release defender spacing:
- Spearman ~0.705
- Pearson ~0.881
- MAE ~0.777 yd

Second defender spacing:
- Spearman ~0.614
- Pearson ~0.821

3-yard crowding:
- Spearman ~0.322

Raw 2-yard defender count:
- ~-0.039, weak standalone

### BDB2026 receiver×route
Nearest release spacing:
- Spearman ~0.730

Second defender:
- ~0.637

### BDB2026 strict-prior future geometry
Nearest release history vs later same-family:
- ~0.482

Second:
- ~0.399

2-yard crowding:
- ~0.384

3-yard crowding:
- ~0.406

Route-conditioned nearest:
- ~0.537 Spearman
- ~0.649 Pearson
- ~1.712 yd MAE

### BDB2021
Player×route nearest throw spacing:
- ~0.593

Second:
- ~0.571

Gap:
- ~0.401

Strict-prior future nearest:
- ~0.411 overall
- ~0.492 high-support

### BDB2023 blocker geometry
Snap separation early/late:
- ~0.847

Time to minimum:
- ~0.575

Minimum distance:
- ~0.487

Strict-prior blocker-week snap:
- ~0.820 overall
- ~0.788 high-support

These establish repeatable football traits, not projection lift.

---

## 23. Advanced-data coverage / cold start

Coverage is not universal.

Approximate BDB2026 minimum-8 receiver-history coverage:
- Week 2 ~6.8%
- Week 6 ~48.4%
- Week 12 ~58.9%
- Week 18 ~68.1%

BDB2021 minimum-5 route-history:
- Week 2 ~19.1%
- Week 6 ~49.8%
- Week 14 ~61.6%
- Week 18 ~64.7%

BDB2023 minimum-10 blocker-history:
- Week 2 ~51.6%
- Week 5 ~63.5%
- Week 8 ~67.8%

Cold start and sparse-history must remain explicit.

---

## 24. Advanced-data current priority

Higher-priority future controlled candidates:
- BDB2026 nearest release-spacing history;
- BDB2026 second-defender spacing.

Research-worthy:
- route-conditioned spacing;
- 3-yard crowding;
- BDB2021 route geometry as archetype/analog context;
- BDB2023 blocker geometry as trench context.

Lower standalone priority:
- raw BDB2026 2-yard crowding count.

No receiving-yard/receptions/QB/RB predictive claim has yet been established.

---

## 25. Signal qualification system

Frozen document:

`docs/research/FOOTBALL_CONTEXT_SIGNAL_QUALIFICATION_V1.md`

Before predictive science, a candidate must establish:
1. pregame availability;
2. historical coverage;
3. identity/join integrity;
4. explicit missingness semantics;
5. strict-prior stability when appropriate;
6. a concrete football mechanism;
7. redundancy audit;
8. sample support.

Qualification states:
- `READY_FOR_FROZEN_EXPERIMENT`
- `ENGINEERING_READY_SOURCE_THIN`
- `DESCRIPTIVE_ONLY`
- `SOURCE_BLOCKED`
- `REJECTED_INTEGRITY`

Current broad default floors in the materializer:
- pregame coverage >= 0.80
- eligible rows >= 500
- stable-ID coverage >= 0.99
- duplicate published keys = 0
- canonical fanout = 0
- stability evidence required for READY

Do not tune these floors after seeing which feature passes.

---

## 26. Qualification pipeline already implemented

### Stability
`scripts/research/build_context_stability_evidence.py`

Primary current descriptive statistic:
`strict_prior_adjacent_period_spearman`

No target-game outcome modeling.

### Candidate profile
`scripts/research/build_context_candidate_profile.py`

Outputs:
- seasons;
- eligible rows;
- pregame coverage;
- stable-ID coverage;
- unknown rate;
- duplicate keys;
- fanout;
- prior support;
- attached stability;
- intended component;
- redundancy notes;
- source class;
- mechanism.

### Qualification inventory
`scripts/research/build_context_signal_qualification_inventory.py`

Fail-closed classification into the five frozen dispositions.

This is the bridge from plumbing to controlled science.

---

## 27. Immediate execution sequence

The highest-value next work is no longer more conceptual documentation.

### Step 1
Locate a still-retained canonical historical player-game artifact.

If unavailable, deterministically rehydrate from the already-qualified historical builder.

This is not new research.

### Step 2
Generate/update the historical-base manifest.

### Step 3
Materialize multi-season usage-regime context with:
`build_usage_regime_context.py`

Validate:
- no target self-use;
- duplicate keys = 0;
- stable identity coverage;
- season/week coverage;
- cold-start rate;
- position distribution.

### Step 4
Materialize multi-season room-continuity context with:
`build_room_continuity_context.py`

Validate:
- canonical grain;
- strict-prior behavior;
- explicit unknown states;
- target/rush room coverage.

### Step 5
Measure strict-prior stability for:
- target-share history;
- rush-share history;
- route-rate history where qualified;
- room concentration;
- returning opportunity overlap;
- team-tenure/regime diagnostics where appropriate.

### Step 6
Build candidate profiles.

### Step 7
Build the qualification inventory.

The goal is truthful classification, NOT forcing candidates into READY.

### Step 8
Only if one or more features qualify, freeze a separate predictive experiment before outcome inspection.

---

## 28. First role/regime questions to answer

Priority questions:
1. Can player-games with stale historical usage be identified objectively after a team change?
2. Can new-team cold starts be distinguished from established current-team roles?
3. Do target/rush/route share changes persist enough to represent regime change?
4. Does room turnover identify weeks where old player averages deserve lower confidence?
5. Does departed opportunity identify regime-change games without assuming who inherits it?
6. Should current-season share dominate stale prior-season share after a genuine transition?
7. Should uncertainty widen for novel roles even if the point mean is not immediately shifted?

Some context information may improve uncertainty/calibration rather than point means.

---

## 29. Established-star / reliability concept

The user raised the idea that established players may be more predictable.

Do not encode fame or reputation.

Potential objective proxies:
- career sample size;
- same-team tenure;
- current-role tenure;
- opportunity-share stability;
- historical variance;
- repeated high-share seasons;
- regime continuity.

This is potentially an uncertainty/reliability mechanism.

Do not manually lower variance because a player is famous.

---

## 30. Analog engineering next steps

After role/personnel context is materialized:
1. create descriptor registry;
2. define market-family descriptor subsets;
3. build strict-prior candidate pools;
4. deterministic scaling;
5. component distance accounting;
6. freeze neighbor IDs;
7. emit selection hashes;
8. report sparse/OOD rates;
9. do not attach outcomes until separately authorized.

A likely valuable use is:
- "How familiar/novel is this current football state?"

not necessarily:
- "average the yards of the nearest games."

---

## 31. Proximity engineering next steps

Useful engineering:
- receiver historical nearest spacing;
- second-defender spacing;
- route-conditioned spacing;
- local crowding;
- sample counts;
- exposure switches;
- proximity continuity;
- sparse-history states.

Key next qualification questions:
- pregame coverage;
- novelty vs existing inputs;
- support;
- route tendency mechanism that is pregame-safe.

Do not use target-game realized routes.

---

## 32. Trench engineering next steps

Start with supportable facts:
- OL starter continuity;
- OL current-starter prior snap coverage;
- OL personnel change count;
- center/tackle changes;
- front continuity;
- top-rusher continuity;
- missing high-share rusher;
- current defenders' strict-prior pass-rush history;
- BDB blocker archetypes where identity joins are valid.

Potential states:
- `TRENCH_STABLE`
- `OL_MAJOR_TURNOVER`
- `DL_MAJOR_TURNOVER`
- `BOTH_TRENCHES_CHANGED`
- `KEY_OL_ABSENCE`
- `KEY_RUSHER_ABSENCE`
- `TRENCH_UNCERTAIN`

Do not fabricate exact matchup assignment.

---

## 33. Market disagreement audit

Planned artifact:

`MODEL_MARKET_DISAGREEMENT_AUDIT_V1`

Large pre-frozen model/market differences trigger a football audit of:
- current role;
- transaction state;
- hierarchy;
- injuries/vacancy;
- QB environment;
- coaching/play caller;
- current-season usage;
- OL/front;
- opponent;
- advanced profile;
- analog support;
- opportunity vs efficiency source.

Possible dispositions:
- `FOOTBALL_INPUT_DEFECT_FOUND`
- `FOOTBALL_ROLE_STATE_STALE`
- `MODEL_MECHANISM_DISAGREEMENT_NO_INPUT_DEFECT`
- `INSUFFICIENT_EVIDENCE`

If football input is stale, repair football input and rerun independently.

If no defect exists, preserve the disagreement.

---

## 34. Known negative evidence / no-retest rules

### Depth-rank workload remap
Run:
`34063904515`

Artifact:
`9998334300`

Result:
- carry MAE ~3.483 → ~4.106
- rush-yard MAE ~20.424 → ~22.837

Conclusion:
depth-chart rank is context, not workload authority.

Do not create "RB1 = fixed carries" logic.

### QB conditional analog V1
Failed closed:
`NO_ACTIONABLE_QB_CONDITIONAL_ANALOG_RELIABILITY`

Do not rescue.

### RB receiving-efficiency transform families
R23–R27D were extensively explored/failed/closed.

Do not reopen the same transformations under new names.

### M84 WR-CB
Source blocked. Proximity is not assignment.

### M85 blocker-rusher
Source blocked. Interaction is not universal assignment.

---

## 35. Week-2 Full Slate under-bias diagnostic

Fresh board unique player-market model vs median line:
- under ~272
- over ~156
- ~63.6% under

Approximate market split:
- pass O18 / U12
- receiving yards O61 / U84
- receptions O48 / U95
- rush+receiving O2 / U31
- rushing O27 / U50

This is an audit trigger only.

Possible causes:
- market over shading;
- low model means;
- stale current-role information;
- Week-2 RB fallback limitations;
- distribution/calibration.

Do not force projections upward solely because the board is under-heavy.

---

## 36. What may proceed in parallel

Engineering can proceed concurrently:
- source inventory;
- event/role evidence;
- transaction/personnel extraction;
- strict-prior feature materialization;
- proximity summaries;
- blocker summaries;
- analog-state construction;
- continuity;
- leakage audits;
- coverage;
- abstention;
- identity QA;
- redundancy;
- qualitative normalization;
- forward archival pipelines.

This is encouraged.

---

## 37. What must be serialized

Predictive science should be staged one mechanism at a time.

Recommended eventual sequence:
1. role/environment → opportunity/entitlement;
2. analog support → uncertainty/context;
3. receiver geometry → separately justified WR/TE mechanism;
4. trench/personnel geometry → separately justified QB/RB mechanism;
5. proximity matchup features;
6. true assignment only if an honest source eventually exists.

Reason:
- attribution;
- holdout protection;
- no feature soup;
- clean failures;
- production protection.

---

## 38. Recommended Claude/Maude lane now

Highest-value contribution:

### Build the first real historical qualification dataset for role/regime and continuity.

From the latest context branch head, on a separate branch:

1. locate/reuse preserved canonical history;
2. deterministically rehydrate only if necessary;
3. produce historical-base manifest;
4. materialize usage-regime context;
5. materialize room-continuity context;
6. run strict-prior QA;
7. run stability evidence;
8. build candidate profiles;
9. build qualification inventory;
10. document every candidate's exact disposition.

Do not optimize for making features pass.

Optimize for identifying what is truly:
- stable;
- novel;
- pregame-knowable;
- sufficiently covered;
- mechanistically useful.

---

## 39. Safe alternative Claude/Maude lanes

If the role/regime execution is already being handled when you inspect GitHub:

### Alternative A — advanced-data novelty/redundancy qualification
Focus on:
- receiver nearest release spacing;
- second-defender spacing;
- route-conditioned spacing;
- blocker snap geometry;
- time-to-minimum;
- interaction diversity.

Determine whether these contain information not nearly reconstructed by existing canonical inputs.

Do not fit target outcomes.

### Alternative B — personnel continuity materializer
Implement strict-prior:
- current roster overlap;
- position-room overlap;
- OL starter continuity;
- front-seven continuity;
- departed opportunity;
- current-team tenure;
- QB continuity.

No inheritance assumptions.

---

## 40. Do-not-lose inventory

The master program must preserve:
- player role changes;
- team changes;
- new QB;
- new primary RB/WR/TE;
- teammate departures;
- vacated opportunity;
- injuries;
- return from injury;
- rookies/new starters;
- coaching/play caller;
- current-season usage;
- stale prior-season usage;
- established-player reliability;
- similar historical games;
- WR-CB context;
- nearest defender as exposure;
- route geometry;
- crowding;
- OL/DL;
- blocker geometry;
- roster turnover;
- historical team sample validity;
- qualitative role statements;
- game-script context;
- opportunity versus efficiency;
- uncertainty for novel roles;
- market-gap audits;
- analog OOD/abstention;
- cold-start coverage;
- advanced-data coverage limits.

Do not discard one simply because another lane is implemented first.

---

## 41. Scientific standard for future candidates

Before any predictive experiment, freeze:
- candidate fields;
- mechanism;
- target component;
- cohort;
- training period;
- holdout;
- missingness;
- cold-start behavior;
- baseline;
- metrics;
- tail metrics;
- replication requirement;
- pass/fail gates;
- no-retest rule.

Where appropriate evaluate:
- MAE;
- RMSE;
- bias;
- correlation;
- median absolute error;
- p75/p90 errors;
- large-miss frequency;
- player-level behavior;
- role/regime cohorts;
- early/late season;
- high/low support;
- temporal replication;
- calibration;
- mechanism coherence.

Do not accept a tiny global improvement that hides severe transition-regime regressions.

---

## 42. Permanent leakage rule

For target kickoff T:

all pregame information must satisfy:

`information_timestamp < T`

Historical reconstruction must use completed prior games only.

Never leak:
- target-game snaps;
- target-game routes;
- target-game touches;
- target-game tracking;
- postgame depth chart;
- postgame role statements;
- future roster;
- future-season information.

---

## 43. Identity discipline

Stable player IDs are canonical whenever supported.

Names are audit helpers only.

Every materialization should report:
- stable-ID coverage;
- unmatched count;
- ambiguous count;
- duplicate keys;
- fanout;
- source version;
- temporal cutoff;
- row counts by season/week/position.

No fuzzy name match may silently become football truth.

---

## 44. Current program dispositions

Master:
`FOOTBALL_CONTEXT_INTELLIGENCE_PROGRAM_V1_ACTIVE_ENGINEERING`

Advanced feature materialization:
`NFL_ADVANCED_FEATURE_MATERIALIZATION_V1_CERTIFIED`

Advanced signal reconnaissance:
`ADVANCED_DATA_SIGNAL_EXPLORATION_V1_INITIAL_RECONNAISSANCE_PASS_MULTIPLE_PERSISTENT_FAMILIES`

Historical reuse:
`HISTORICAL_DATA_REUSE_FIRST_V1_FROZEN`

Historical analog contract:
`HISTORICAL_ANALOG_INDEX_CONTRACT_V1_FROZEN`

Defender proximity:
`DEFENDER_PROXIMITY_EXPOSURE_CONTRACT_V1_FROZEN`

OL/DL:
`OL_DL_PROTECTION_CONTEXT_CONTRACT_V1_FROZEN`

Qualification:
`FOOTBALL_CONTEXT_SIGNAL_QUALIFICATION_V1_FROZEN`

No context feature has been promoted to production.

That is intentional.

---

## 45. Expected checkpoint format

At each meaningful Claude/Maude checkpoint report:

### Git
- branch;
- base SHA;
- resulting SHA;
- files changed.

### Execution
- workflow run;
- job;
- artifact;
- digest.

### Data
- source;
- seasons;
- rows;
- identities;
- duplicate/fanout;
- leakage;
- unknown/missing rates.

### Interpretation
- what was learned;
- stability;
- novelty/redundancy;
- support;
- exact qualification disposition.

### Boundaries
Confirm:
- no production change;
- no market teacher;
- no paid pull;
- no Issue #535 interference;
- no failed-family rescue.

### Next action
State the exact next executable task.

---

## 46. Final directive

Do not restart research.

Do not redo BDB discovery.

Do not repeatedly reacquire ordinary historical game logs.

Do not reopen failed families.

Pick up from current GitHub state.

The immediate objective is:

> Turn the newly engineered football-context concepts into actual historical, leakage-safe, identity-safe qualification evidence and determine which pieces contain enough stable, novel, pregame information to deserve a controlled predictive experiment.

The ultimate objective is:

> Find real football knowledge that materially improves individual-player pregame projection accuracy while preserving scientific discipline and production reliability.

Parallelize engineering.

Serialize predictive science.

Preserve failures.

Keep GitHub canonical.
