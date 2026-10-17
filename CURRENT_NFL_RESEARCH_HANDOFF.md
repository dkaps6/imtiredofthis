# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## TOP CHECKPOINT — 2026-10-16 — PRISM WEEK-6 STAGE-2 SHADOW LOCKED / OPPORTUNITY QUALITY SOURCE AUDIT NEXT

**Model:** PRISM — Player Role & Individual Situation Model

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-10-16_PRISM_WEEK6_SHADOW_LOCKED_OPPORTUNITY_QUALITY_NEXT.md`

Canonical player-first order:

`PLAYER IDENTITY -> CURRENT ROLE/STATE -> TEAMMATE COMPETITION -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

### Historical Stage-2 milestone

`PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_PARTIAL`

Qualified:
- RB rushing;
- RB receiving;
- TE receiving.

Closed:
- QB passing exact V1.

WR receiving remains source-blocked/not scored.

### Week-6 forward shadow

`PRISM_WEEK6_SUNMON_STAGE2_SHADOW_LOCKED`

Canonical lock:
- run `38990758368` — SUCCESS;
- exact head `67677542e3e0230930d71a67b082fa73ead39399`;
- artifact `12085545101`;
- digest `sha256:2f1c9b65b6db2e746d9ab8419fa361e0d6c5706cf8b493fbfe64d5d7189d1b61`;
- result `docs/research/PRISM_WEEK6_SUNMON_STAGE2_SHADOW_V1_LOCK_RESULT.md`.

Population:
- 13 Sunday/Monday Week-6 games;
- SEA-DEN excluded because it already played before the Stage-2 freeze;
- 248 locked player-cell rows;
- RB rushing / RB receiving / TE receiving.

At lock:
- Week-6 target outcomes read 0;
- sportsbook upstream 0;
- OddsAPI calls 0;
- production mutation false.

Do not alter or grade the shadow until the included games settle.

### Exact next action

Continue PRISM research without waiting for outcomes.

The next architectural layer is **opportunity quality/topology**, but do not rescue the closed simple high-value-share persistence V1.

Freeze a source/anti-retest audit for a distinct question:

Can PRISM separately model:
1. **how many high-value opportunities the team creates**, and
2. **which named player receives those opportunities conditional on current role/competition**,

without assuming recent player red-zone share simply persists?

Use the already-qualified high-value PBP source as a parent where appropriate.

No sportsbook.
No OddsAPI.
No 2026 outcomes.
No production mutation.

---

## TOP CHECKPOINT — 2026-10-16 — PRISM STAGE-2 BRIDGE QUALIFIED PARTIAL / WEEK-6 PROSPECTIVE SHADOW NEXT

**Model name:** PRISM — Player Role & Individual Situation Model

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-10-16_PRISM_STAGE2_BRIDGE_QUALIFIED_PARTIAL_CURRENT.md`

Player-first order remains canonical:

`PLAYER IDENTITY -> CURRENT ROLE/STATE -> TEAMMATE COMPETITION -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

### New canonical Stage-2 milestone

`PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_PARTIAL`

Frozen before scoring:
- commit `87400a9c9c1db465c9c4bd70b0cb241dba27c6c9`;
- contract `docs/research/PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_V1_MECHANISM_FREEZE.md`.

Canonical scored authority:
- run `38957575645` — SUCCESS;
- job `117164201650`;
- exact head `eba5787c0b06aa5f975008965df7d023b3525ba7`;
- artifact `12070774378`;
- digest `sha256:e07edbad1bf97a4917ba95c58156c8941448e7b2668bd0eae2e9bcad5d4945b0`;
- result `docs/research/PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_V1_RESULT.md`.

Qualified:
- `RB_RUSHING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED`;
- `RB_RECEIVING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED`;
- `TE_RECEIVING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED`.

Closed:
- `QB_PASSING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_CLOSED`.
  QB improved CRPS but worsened expected attempts and room allocation and failed 2025 dependence-aware confirmation. Do not rescue the exact V1 mechanism.

Not scored:
- WR receiving, because 2025 WR-R15 independent OOS authority remains unavailable/forbidden.

### Exact next action

Do not production-promote historical Stage 2.

Freeze an outcome-blind 2026 Week-6 prospective PRISM shadow for the three qualified cells only:
- RB rushing;
- RB receiving;
- TE receiving.

Use the exact Stage-1 p_active semantics and exact qualified Stage-2 bridge with current football-only pregame inputs. Preserve canonical entitlement and team opportunity volume. QB and WR remain baseline.

No sportsbook upstream.
No new OddsAPI call.
No Week-6 outcomes at lock time.
No production mutation.

Before launching the shadow, reconcile whether existing preserved Week-6 football artifacts contain every required input. If not, build a no-live-odds football-only source path rather than spending another paid credit.

Preserve all terminal anti-retest lanes, WR/TE trust locks, and current validated production science.

---

## TOP CHECKPOINT — 2026-10-16 — ACTIVE-ROLE MECHANISM QUALIFIED ALL / STAGE 2 NEXT

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-10-16_PLAYER_FIRST_ACTIVE_ROLE_MECHANISM_QUALIFIED_CURRENT.md`

Player Projection First Principle remains canonical:

`PLAYER IDENTITY -> CURRENT ROLE/STATE -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

Position MAE/RMSE remains downstream QA.

### New canonical scientific milestone

`PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED_ALL`

Frozen-before-scoring authority:
- freeze commit `16df6e95388a5dd7d682a619ea0b6f49d50f37b4`;
- contract `docs/research/PLAYER_ACTIVE_ROLE_PARTICIPATION_V1_MECHANISM_FREEZE.md`.

Canonical scored authority:
- run `38907459996` — SUCCESS;
- job `116994537005`;
- exact scored head `1c2c92ca498e8dce1eee19d39b2f276dfd37d8f7`;
- artifact `12049120604`;
- digest `sha256:41b1acd3f1b8d77abb53980db160ece32f8fc40351939d4827248b9caad0c64b`;
- result `docs/research/PLAYER_ACTIVE_ROLE_PARTICIPATION_V1_MECHANISM_RESULT.md`.

Per-position:
- `QB_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED`
- `RB_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED`
- `WR_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED`
- `TE_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED`

2025 independent-confirmation Brier improvements:
- QB +0.092720;
- RB +0.038602;
- WR +0.015566;
- TE +0.006296.

Every frozen 2025 player-cluster, game-cluster and equal-player confirmation gate passed.

Named-player 2025 share improved:
- QB 88.41%;
- RB 89.47%;
- WR 82.22%;
- TE 88.89%.

Stage 1 is qualified:
`P(any offensive snap | named player's strictly-prior same-team participation state)`.

This does NOT authorize automatic player removal/zeroing or production mutation.

### Exact next action

Keep the player model two-stage.

1. Stage 1 active-role / zero-state participation — **QUALIFIED**.
2. Stage 2 conditional opportunity allocation — **source/lineage reconciliation frozen next**.

The existing allocation roadmap has been reconciled. Do not launch another
generic recent-share allocator.

Frozen next source plan:
`docs/research/PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_V1_SOURCE_PLAN.md`.

Audit 2024/2025 historical as-of/OOS-compatible canonical entitlement authority
separately for QB passing, RB rushing, RB receiving, WR targets and TE targets.
The bridge must preserve existing validated conditional-allocation science and
fail a cell closed if its canonical historical parent cannot be proven.

Only source-ready cells may advance to a separately frozen active-set bridge
mechanism and scoring run.

No sportsbook.
No OddsAPI.
No 2026 outcomes.
No production mutation.

Preserve WR/TE trust qualifications and frozen Week-6 locks, all validated production science and all terminal anti-retest lanes in Issue #535.

Week-6 guardrail remains: exactly one paid acquisition has already been used.

Canonical model status remains:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.

---

## TOP CHECKPOINT — 2026-10-16 — ACTIVE-ROLE SOURCE READY ALL / NEXT MECHANISM MUST BE FROZEN

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-10-16_PLAYER_FIRST_ACTIVE_ROLE_SOURCE_READY_CURRENT.md`

**Restart prompt:**

`docs/handoffs/NEXT_CHAT_PROMPT_2026-10-16_PLAYER_FIRST_ACTIVE_ROLE_SOURCE_READY.md`

Player Projection First Principle remains canonical:

`PLAYER IDENTITY -> CURRENT ROLE/STATE -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

Position MAE/RMSE remains downstream QA, not the primary modeling objective.

### Latest canonical scientific state

1. `PLAYER_ROLE_ELASTICITY_SOURCE_BLOCKED`
   - run `38847501580` SUCCESS;
   - artifact `12024335417`;
   - only frozen blocker was <90% strictly-prior same-team opportunity-history coverage;
   - do not lower the gate.

2. `PLAYER_HIGH_VALUE_SCORING_OPPORTUNITY_SOURCE_READY`
   - run `38848080453` SUCCESS;
   - artifact `12023852453`;
   - named-player scoring-area topology is source-qualified.

3. `PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_CLOSED`
   - run `38848538570` SUCCESS;
   - artifact `12024037594`;
   - simple last-4 high-value share is not a robust replacement for ordinary player usage;
   - do not rescue with alternate bands/windows/subsets/TD outcomes.

4. **CURRENT FRONTIER:**
   `PLAYER_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY_ALL`.

Canonical corrected source authority:
- run `38895205962` — SUCCESS
- job `116953914726`
- exact head `28ebf8d7159fdd1fec8c2e4a0fe1c9e2c691537c`
- artifact `12044155086`
- digest `sha256:a42322579aee682497223c7f53bce0b14a96dc6f466558e811770e681a035dad`
- result `docs/research/PLAYER_ACTIVE_ROLE_PARTICIPATION_V1_SOURCE_RESULT.md`.

Corrected head excludes bye roster weeks.

Per-position source dispositions:
- `QB_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY`
- `RB_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY`
- `WR_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY`
- `TE_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY`.

No active-role predictive result exists yet.

### Exact next action

Freeze a separate active-role mechanism **before scoring anything**.

Question:

> Does this named player's own strictly-prior participation history forecast
> whether he records any offensive snap in the target game better than a
> strictly-prior position-level participation baseline?

Keep the player model two-stage:

1. active-role / zero-state participation;
2. conditional opportunity allocation given participation.

Do not collapse them.

Use true pregame universe:
weekly roster ACT + stable identity - exact official inactives.
Target-game zero offensive snaps are valid outcomes.
Do not use target-game stats/snaps to define pregame membership.

Adjudicate QB/RB/WR/TE separately.
Freeze exact baseline/candidate/probability mapping/support/inference gates before scoring.
Require player-level heterogeneity + dependence-aware player/game inference.

No sportsbook.
No OddsAPI.
No 2026 outcomes.
No production mutation.

### Why this matters

SEA-DEN reinforced the user's core argument:
manual J.K. Dobbins TD reasoning used **his** current role, usage and likely
scoring-area opportunity, while most model-backed ticket legs failed.
That game is motivation, not training proof.

The model must learn player identity/current role instead of merely fitting a
position average.

### Protected continuity

- WR opportunity-difficulty trust remains qualified; Week-6 WR lock remains frozen.
- TE trust remains qualified; Week-6 TE lock remains frozen.
- current generic ATD remains uncertified.
- exactly one paid Week-6 OddsAPI acquisition was used; do not pull again without explicit authorization.
- preserve all terminal anti-retest lanes in Issue #535 and the detailed handoff.
- canonical model status remains `PLAYER_INDIVIDUALIZATION_PARTIAL`.

---

## TOP CHECKPOINT — 2026-10-15 — PLAYER ACTIVE-ROLE PARTICIPATION SOURCE AUDIT ACTIVE

Player Projection First Principle remains canonical.

### Previous frontier closed cleanly

`PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_CLOSED`
- run `38848538570` SUCCESS;
- artifact `12024037594`;
- exact high-value-share persistence V1 is terminal;
- scoring-area usage remains valid descriptive player identity, but recent
  high-value share alone is not a robust next-game entitlement replacement.

### New Stage-1 player-opportunity frontier

The W1-W4 full-roster replay exposed a distinct pregame zero-state problem:
610 / 1,837 player-weeks (33.21%) had no weekly-stat row while the model still
assigned material opportunity to many rostered players.

By position:
- QB 7.81%;
- RB 30.15%;
- WR 31.55%;
- TE 45.31%.

This is now framed explicitly as a two-stage player question:

1. **active-role participation:** will this named player have a real offensive
   role at all?
2. **conditional opportunity allocation:** if yes, how much team/room
   opportunity belongs to him?

Do not combine the two stages before Stage 1 is source-qualified.

Frozen source contract:
`docs/research/PLAYER_ACTIVE_ROLE_PARTICIPATION_V1_SOURCE_AUDIT.md`
freeze commit `78b2dc54452c8a84f0866883ae80dcf397212b6f`.

Implementation:
`scripts/research/audit_player_active_role_participation_v1.py`
commit `30bb39fe9f93b904fdfff08ca7f23c2cd2fe22c3`.

Safeguards:
`tests/test_player_active_role_participation_v1.py`
initial commit `20a159a5eca070abad42cce606b7b4f534961aef`.

Workflow:
`.github/workflows/research-player-active-role-participation-source-v1.yml`
commit `0122f493fde6223bca704810d9147c2526c5b2f7`.

Trigger head:
`ae845ac6279e91bee9a13f69dffb61042417bd8c`.

### Exact source design

Historical seasons:
- 2024
- 2025.

Pregame player universe:
- weekly roster authority;
- QB/RB/WR/TE family;
- roster ACT where status is available;
- stable GSIS identity;
- exact M78 official game-day inactives removed;
- player remains in universe even if target-game offensive snaps are zero.

Official inactive authority:
- 1,088 team-weeks;
- exact SHA-256
  `d39aaf0feea101f3e0d2721ebd4118ef33fb1a4d3c76670e2a4f17734e37b609`.

Participation source:
- historical PFR offensive snap counts;
- missing player row is treated as zero only when the entire team-week snap
  source is proven valid.

Source audit adjudicates QB/RB/WR/TE separately.

It fits no model and makes no scientific prediction.

### Anti-retest protection

Do not relabel/reopen:
- Availability -> Opportunity rule-order gap;
- M78/M79 generic inactive QB identity;
- RB Vacancy;
- role-elasticity teammate-availability V1;
- RB carry/snap allocation;
- RB receiving-room share;
- WR/TE target-share trajectory;
- WR-R15 / TE-R5P snap entitlement;
- route participation/YPRR historical source block.

This active-role probability is a separate upstream zero-state layer.

Current disposition:
`PLAYER_ACTIVE_ROLE_PARTICIPATION_SOURCE_AUDIT_PENDING`.

No sportsbook.
No OddsAPI.
No 2026 outcomes.
No production mutation.

Canonical status:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.

---

## TOP CHECKPOINT — 2026-10-15 — PLAYER HIGH-VALUE OPPORTUNITY SOURCE READY / SIMPLE PERSISTENCE CLOSED

Player-first principle remains canonical:
`PLAYER IDENTITY -> CURRENT ROLE -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

### Role-elasticity source audit

`PLAYER_ROLE_ELASTICITY_SOURCE_BLOCKED`
- run `38847501580` SUCCESS
- artifact `12024335417`
- digest `sha256:1a22efad89da4593124d366d21176d8c70d0d5f2cc0e7498a6216fab57353186`
- result `docs/research/PLAYER_ROLE_ELASTICITY_TEAMMATE_AVAILABILITY_V1_SOURCE_RESULT.md`.

RB/WR/TE all had strong official-inactive identity, snap and multi-room-state support,
but the exact frozen >=90% strictly-prior same-team opportunity-history gate failed:
RB 88.68/88.36%, WR 88.67/89.26%, TE 85.59/84.88% (2024/2025).
Do not lower the gate or infer missing history from target-game outcomes.

### High-value scoring-opportunity source

`PLAYER_HIGH_VALUE_SCORING_OPPORTUNITY_SOURCE_READY`
- run `38848080453` SUCCESS
- exact head `1448fcce54f803d9737a7b3f7d9ff451b5189add`
- artifact `12023852453`
- digest `sha256:1d8a22bf6240b4d4443ab7432d3224b63eaef97439639e7b47f536aed174adda`
- result `docs/research/PLAYER_HIGH_VALUE_SCORING_OPPORTUNITY_IDENTITY_V1_SOURCE_RESULT.md`.

2022-2025 PBP supports named-player RB inside-20/10/5 rushing topology and
RB/WR/TE red-zone target topology with all frozen source gates passed.
Optional end-zone-target source also passed.

### High-value opportunity persistence V1

Frozen question:
does the named player's recent high-value share forecast next-game high-value
share better than his ordinary overall usage share?

Canonical result:
`PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_CLOSED`
- run `38848538570` SUCCESS
- exact scored head `89b92b56d9b9786a3cc8802a3d2b645caeca9ba5`
- artifact `12024037594`
- digest `sha256:2b02f5a6c7a8a05962517f72a9d62ba287d06ffe5ced37abfd70b623a5120afe`
- result `docs/research/PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_V1_RESULT.md`.

Disposition:
- RB rushing: CLOSED; 2025 favorable but clustered CIs cross zero and 2022 worsened.
- RB receiving: favorable all four seasons and pooled-supported, but 2025
  independent player/game clustered CIs cross zero; exact V1 CLOSED.
- WR receiving: worse in all four seasons; CLOSED.
- TE receiving: worse in all four seasons; CLOSED.

Do not rescue by changing yardline bands, last-4/min-3 history, support floors,
lead-back/WR1/TE1 subsets, pooling positions, or adding TD outcomes.

Interpretation:
the Dobbins manual reasoning is not reduced to a red-zone-share coefficient.
It used current workload, trajectory, lead-back status, teammate competition,
scoring-area role and team scoring-situation creation together. High-value share
alone is insufficient.

Current generic ATD remains execution-capable but not dedicated-science certified.

No paid OddsAPI.
No sportsbook upstream.
No production mutation.
Canonical status remains:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.

---

## TOP CHECKPOINT — 2026-10-15 — HIGH-VALUE SCORING OPPORTUNITY IDENTITY SOURCE AUDIT ACTIVE

Player-first frontier has advanced.

### Role-elasticity V1 closed at source gate

Canonical source audit:
- run `38847501580` SUCCESS;
- job `116801249742`;
- exact head `fb516aea9e8af21563af71a6f1506edd154d29af`;
- artifact `12024335417`;
- digest `sha256:1a22efad89da4593124d366d21176d8c70d0d5f2cc0e7498a6216fab57353186`;
- result `docs/research/PLAYER_ROLE_ELASTICITY_TEAMMATE_AVAILABILITY_V1_SOURCE_RESULT.md`.

Disposition:
`PLAYER_ROLE_ELASTICITY_SOURCE_BLOCKED`.

Every frozen identity/event/snap/multi-room-state/chronology gate passed.
The only blocker was the frozen >=90% strictly-prior same-team opportunity-history gate:
- RB 88.68% / 88.36%;
- WR 88.67% / 89.26%;
- TE 85.59% / 84.88%
for 2024 / 2025.

Do not lower the gate. Do not run the predictive role-elasticity mechanism from this exact V1.

### Current production ATD gap reconciled

Current production is sportsbook-independent upstream but dedicated ATD science is explicitly uncertified.

Current generic logic:
- player historical offensive TD/game rate, position fallback if needed;
- blended `rz_tgt_share`;
- blended `rz_carry_share`;
- `rz_share = max(rz_tgt_share, rz_carry_share)`;
- generic `0.75 + rz_share` multiplier;
- small team-win-probability script modifier;
- joint-MC Bernoulli draw.

Canonical validator:
`scripts/validate_anytime_td_model_status_v1.py`

Scientific status:
`DEDICATED_ATD_WALK_FORWARD_CERTIFICATION_REQUIRED`.

The validator's own required frontier is:
`football-only team TD opportunity -> red-zone/end-zone pool -> player entitlement -> probability calibration`.

### Canonical high-value opportunity frontier

Frozen plan:
`docs/research/PLAYER_HIGH_VALUE_SCORING_OPPORTUNITY_IDENTITY_V1_PLAN.md`.

Primary question:
does the named player's strictly-prior high-value opportunity identity forecast which player receives the team's next scoring-area rushing/receiving opportunity?

Channels remain separate:
- RB rushing;
- RB receiving;
- WR receiving;
- TE receiving.

The first mechanism predicts high-value **opportunity entitlement**, not touchdown probability.

No TD outcome selects the player-entitlement mechanism.

Source audit:
- historical 2022-2025 regular-season PBP;
- weekly-roster / GSIS player identity;
- no sportsbook;
- no OddsAPI;
- no 2026 outcomes;
- no predictive model in source stage;
- no production mutation.

Canonical workflow:
`.github/workflows/research-player-high-value-opportunity-source-audit-v1.yml`.

Current trigger head:
`6d32ffe994559decc09066f293ec6ae7965bc87f`.

Current disposition:
`PLAYER_HIGH_VALUE_SCORING_OPPORTUNITY_SOURCE_AUDIT_PENDING`.

A separately created RB-only high-value topology audit was explicitly superseded **before scientific adjudication** by this broader canonical contract. Do not compare results and choose the favorable one.

### Protected state

- Player Projection First Principle remains non-negotiable.
- Position MAE/RMSE remains downstream QA, not the primary scientific unit.
- RB-PD2 Weeks 4/5/6 remain permanent prospective non-observations.
- WR dual reliability prospective confirmation begins Week 7 with the frozen 50/50 formula.
- Week-6 WR/TE/QB locks remain immutable; do not use target-week outcomes to retune them.
- no additional Week-6 OddsAPI pull.

Canonical model status:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.

---

## TOP CHECKPOINT — 2026-10-15 — PLAYER ROLE-ELASTICITY SOURCE AUDIT FROZEN / IMPLEMENTED / RESULT PENDING

Player-first continuation after the canonical first-principle update.

### Why this lane

The new question is **named-player role elasticity under teammate/room change**:

> when a player's competitive room changes before kickoff, does this exact player
> historically absorb or lose opportunity in a repeatable way?

This is broader and more player-specific than a universal vacancy-transfer rule.

### Anti-retest protection

Do NOT reopen:
- QB M78/M79 generic official-inactive identity;
- generic injury burden;
- RB Vacancy Opportunity V1 deterministic vacated-share -> snap-weight transfer;
- M96 retrospective RB router/threshold work;
- WR simple vacated-target / higher-usage absence formulas;
- situational-target residual V1;
- postgame inference of absence from zero participation.

RB Vacancy V1 Week-3 remains `WEEK3_OBSERVATIONAL_MIXED`:
- rush-attempt MAE improved `4.0279 -> 3.8549`;
- rush-yard MAE improved `28.1177 -> 26.6182`;
- DEN worsened in both markets;
- PIT improved in both.
That mixed team behavior motivates successor/player identity research; it does not authorize retuning the old formula.

### Frozen source audit

Contract:
`docs/research/PLAYER_ROLE_ELASTICITY_TEAMMATE_AVAILABILITY_V1_SOURCE_AUDIT.md`

Freeze commit:
`5c7881abff1275abc3d7e4eb328ea6ae6a7aa5bd`

Positions adjudicated separately:
- RB/FB
- WR
- TE.

Pregame absence authority:
- committed corrected M78 official-inactive snapshot;
- 1,088 team-weeks / 544 per season for 2024 and 2025;
- CSV SHA-256 `d39aaf0feea101f3e0d2721ebd4118ef33fb1a4d3c76670e2a4f17734e37b609`;
- official inactive identity is treated only as a fixed pre-kickoff fact.

Other source inputs:
- weekly roster identity;
- nflverse weekly player usage;
- PFR offensive snap participation;
- strictly-prior rows only.

Source audit reads:
- target-game outcomes 0;
- sportsbook 0;
- OddsAPI 0;
- predictive models 0;
- projection mutations 0;
- production mutations 0.

Implementation:
- `scripts/research/audit_player_role_elasticity_teammate_availability_v1.py`
- commit `871762e47fbbeef7135bd0f913f000abe487d8a5`.

Safeguards:
- `tests/test_player_role_elasticity_teammate_availability_v1.py`
- commit `9ceafbbb97051cf29112eed58aa86e20209c903f`.

Workflow:
- `.github/workflows/research-player-role-elasticity-source-audit-v1.yml`
- commit `edfc09cb3a151372cfffd43e306b236c91453426`.
- mechanical trigger-only follow-up head `fb516aea9e8af21563af71a6f1506edd154d29af`.

### Current disposition

`PLAYER_ROLE_ELASTICITY_SOURCE_AUDIT_PENDING_CANONICAL_ACTION`

No scientific/source-ready disposition has been claimed yet because no authoritative run is visible through the current GitHub Actions connector.

Do not convert implementation into a PASS.
Do not attach target-game outcomes until source readiness is genuinely adjudicated.

If a position becomes SOURCE_READY, the next step is a separately frozen named-player mechanism test comparing that player's own role under different room states, with opportunity and efficiency separated and position-average MAE/RMSE only as guardrails.

### Other protected current state

- RB-PD2 Weeks 4/5/6 remain permanent prospective non-observations; Week 3 remains 1/8 weeks, 46/400 rows.
- WR dual-reliability historical support remains valid; prospective confirmation starts Week 7.
- WR/TE/QB Week-6 prospective locks remain immutable; do not use Week-6 outcomes to alter them.
- exactly one Week-6 paid OddsAPI acquisition remains the hard limit.

Canonical model status:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.

---

## TOP CHECKPOINT — 2026-10-15 — PLAYER PROJECTION FIRST PRINCIPLE CANONICAL

The repository-wide modeling objective is now explicit in `AGENTS.md` and:
`docs/research/PLAYER_IDENTITY_ROLE_STATE_ARCHITECTURE_V1.md`.

Non-negotiable interpretation:
- this is a **named-player projection model**;
- position-level MAE/RMSE is a downstream diagnostic / guardrail, not the primary modeling objective;
- historical data should build the named player's identity, role, usage trajectory, situational opportunity, efficiency tendencies, teammate-allocation behavior and reliability;
- current-week state determines which parts of that historical identity are active now;
- teammate competition determines opportunity allocation;
- opponent/defensive/game context is the weekly obstacle/environment the player faces, not the player's identity;
- different markets map to different parts of the player state (for example, rushing yards and anytime TD can legitimately have different conclusions);
- improving individual-player projections should improve aggregate position metrics as a consequence;
- do not chase a cleaner position average while losing player-level football meaning;
- do not declare a position solved from aggregate accuracy alone.

Conceptual sequence:
`PLAYER IDENTITY -> CURRENT ROLE -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

This is additive. Preserve all validated/promoted science and terminal closed-lineage stop rules. Do not restart the model.

Current research work continues from the newest frozen player-level frontier below under this first principle.

---

## TOP CHECKPOINT — 2026-10-15 — WR DUAL TRUST SUPPORTED / QB WEEK-6 INTENT LEDGER FROZEN

### WR player reliability

Historical WR dual-reliability integration:
- disposition `WR_DUAL_RELIABILITY_TRUST_HISTORICAL_INTEGRATION_SUPPORTED__PROSPECTIVE_REQUIRED`;
- run `38827426902` SUCCESS;
- job `116738041310`;
- exact head `ca4da46fe07341900263dacc8602a7aef002ed03`;
- artifact `12015229072`;
- digest `sha256:2b27705844beaf9c35cf181e6fc5b7efff2c029df0255f035ba8c88bf24722c2`;
- result `docs/research/WR_DUAL_RELIABILITY_TRUST_INTEGRATION_V1_RESULT.md`.

Frozen no-fit formula:
`dual_fragility = 0.5 * opportunity_fragility + 0.5 * efficiency_fragility`.

Pooled:
- opportunity-only association +0.28853;
- efficiency-only +0.30057;
- dual +0.32746;
- dual-minus-opportunity +0.03893;
- player-cluster gain 95% CI [+0.02314,+0.05494];
- team-week gain 95% CI [+0.02427,+0.05342];
- DUAL_HIGH_TRUST final abs error 13.73 yd vs DUAL_LOW_TRUST 31.49 yd;
- every frozen gate passed.

This is historical integration support only. The parent seasons were already exposed.

Prospective dual confirmation:
- frozen plan `docs/research/WR_DUAL_RELIABILITY_TRUST_PROSPECTIVE_CONFIRMATION_V1_PLAN.md`;
- starts Week 7, NOT Week 6;
- fixed Weeks 7-10;
- exact 50/50 formula, no fitted weights;
- no retroactive Week-6 composite;
- Week-6 opportunity and efficiency locks remain separate and immutable.

### QB prospective pass-intent source lane

Unchanged V1L Thursday Week-6 collector:
- run `38827689707` SUCCESS;
- job `116738872975`;
- exact head `6c694209323a5af646e0b2d80c175d3fec281dd4`;
- artifact `12016115756`;
- digest `sha256:c741cbd9a94c17d0996c2a00e789710da0964cfe5c6a2e0ec3fcf0ac380c3b01`;
- 28 scheduled team-weeks;
- 15 teams with retained sources;
- 38 retained first-party/timestamp-safe candidate sources;
- candidate-source coverage 53.57%;
- outcomes/residuals/sportsbook/OddsAPI/predictive fits all zero.

Frozen semantic ledger:
- `docs/research/QB_TEAM_PASS_INTENT_WEEK6_PROSPECTIVE_STRUCTURED_DECISIONS_V1L.md`;
- commit `9e1b0c8b065958bced83b96ce8dbc165eca743d9`;
- EXPLICIT 4 / NO_EXPLICIT 13 / INSUFFICIENT 21.

Precision-safe explicit tags:
- CAR: PROTECTION_ADJUSTMENT;
- NYG: TEMPO_UP;
- NYG: PERSONNEL_USAGE_CHANGE;
- DAL: RUN_VOLUME_UP.

Do not infer pass/run direction from NO_EXPLICIT or INSUFFICIENT rows.
Week 6 alone does not authorize a predictive QB candidate or projection mutation.
The exact collector may continue source-only snapshots before remaining team kickoffs under the unchanged frozen rules.

### Immediate research action

1. Preserve WR/QB milestones above.
2. Audit only existing Week-5 and Week-6 RB-PD2 actions/artifacts for valid **pregame** prospective locks.
3. Week 4 is permanently non-observation: no valid RB-PD2 lock existed and none may be reconstructed.
4. Do not create a retrospective RB-PD2 lock from replay.
5. Do not pull OddsAPI.
6. Do not read/use Week-6 outcomes for WR/TE/QB prospective locks.
7. RB retrospective M96 architecture remains stopped; only already-frozen prospective evidence may advance it.

Canonical status remains:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.

Exactly one Week-6 paid OddsAPI acquisition remains the hard limit. No additional paid pull without explicit authorization.

---

## TOP CHECKPOINT — 2026-10-15 — WR EFFICIENCY TRUST QUALIFIED + WEEK-6 LOCK FROZEN

Canonical historical result:
- `PASS_CATCHER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_PARTIAL`
- run `38826418315` SUCCESS
- job `116734822024`
- exact head `c15f70f3bacf9d0d7f301d334419841ad6d9d492`
- artifact `12015303274`
- digest `sha256:cbab5ac5b424861d7dfd1fc8e965b6ad579936eab5e53820019392e25f2b6cdf`
- result `docs/research/PASS_CATCHER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_V1_RESULT.md`.

WR:
- `WR_PLAYER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`;
- 3,224 rows / 211 WRs;
- pooled partial rank association +0.16238 after controlling opportunity difficulty;
- pooled player-cluster 95% CI [+0.12437,+0.19757];
- pooled team-week 95% CI [+0.12851,+0.19455];
- 2024 confirmation player/team-week CIs strictly positive;
- HIGH_EFFICIENCY_TRUST final abs error 15.27 yd vs LOW 31.07 yd;
- every frozen WR gate passed.

TE:
- `NO_ACTIONABLE_TE_PLAYER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY`;
- direction favorable but 2025 confirmation player/team-week clustered CIs crossed zero;
- exact V1 is CLOSED; no rescue;
- existing TE opportunity trust remains qualified and frozen.

Week-6 WR efficiency trust prospective authority:
- `WEEK6_WR_EFFICIENCY_TRUST_PREGAME_LOCK_FROZEN`
- run `38826804491` SUCCESS
- job `116736044870`
- exact head `4f7443ed21273040c91dd29593a47654acad61be`
- artifact `12015596169`
- digest `sha256:6fc91bee57067219c69f6817950afa87ac07588a391275fe9ffe0c04f58b05dd`
- row digest `sha256:0bf72b516548fed6ba6c48d464674f011ef357e09388695b28ca5dd3ae9cf20f`
- result `docs/research/WR_PLAYER_EFFICIENCY_DIFFICULTY_TRUST_WEEK6_PREGAME_LOCK_V1_RESULT.md`.

Frozen Week-6 WR efficiency cohort:
- 152 WRs;
- 141 supported / 11 unsupported;
- HIGH_EFFICIENCY_TRUST 35 / MIDDLE 70 / LOW_EFFICIENCY_TRUST 36;
- 717 reconstructed W1-W5 rows;
- history max week 5;
- Week-6 outcomes read 0;
- sportsbook 0 / OddsAPI 0 / models fit 0;
- projection/distribution changes 0;
- production unchanged.

Interpretation:
- WR now has **two separate qualified same-player reliability axes** around unchanged WR-R15: opportunity difficulty and efficiency difficulty.
- Do NOT combine them yet.
- HIGH/LOW on either axis is reliability only, not over/under or bet/fade.
- Week-6 locks are immutable until target-week settlement.
- no production promotion yet.

Immediate research rule:
- do not read or use Week-6 outcomes;
- do not reopen generic efficiency volatility, WR mean integration, TE efficiency trust, or other closed lanes;
- anti-retest the remaining RB/QB/player-reliability frontier before opening a new historical experiment;
- a prospective scoring/comparison contract may be frozen now, but no Week-6 scoring until settlement.

Canonical status:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.

Exactly one Week-6 paid OddsAPI acquisition remains the hard limit. No additional paid pull without explicit authorization.

---

## TOP CHECKPOINT — 2026-10-15 — SEA-DEN TICKET CONTINUITY + WR/TE TRUST LOCKS FROZEN

Authoritative detailed handoff:
- `docs/handoffs/NFL_HANDOFF_2026-10-15_WEEK6_SEA_DEN_TICKETS_WR_TE_TRUST_CONTINUITY_CURRENT.md`

Restart prompt:
- `docs/handoffs/NEXT_CHAT_PROMPT_2026-10-15_WEEK6_SEA_DEN_TICKETS_WR_TE_TRUST_CONTINUITY.md`

### Immediate user-facing lane

Tonight is **SEA @ DEN**.

Use:
- `docs/research/MODEL_BACKED_ALT_LINE_PARLAY_CONSTRUCTION_V1.md`
- the frozen Week-6 live board
- current sportsbook lines only when explicitly reconciled.

Ticket tiers:
- Core / safe
- Core + juice / moderate
- Aggressive
- High-ceiling / lottery.

The user specifically asked to resume the actual GitHub ticket-construction process for the lottery ticket rather than reconstruct one from chat memory.

ATD is allowed tonight only as an explicitly **judgment-only / non-certified** optional layer using football role, red-zone/goal-line context, game environment and current offered price. Dedicated ATD science remains uncertified.

### Week-6 paid odds guardrail

Exactly one paid Week-6 acquisition:
- run `38791377571`
- artifact `11999923880`
- digest `sha256:0916cd1cbec93f9a2529190b3df7c39172eee011a31eb9df429a35cdb9b41f61`.

Final green replay:
- run `38792341891` SUCCESS
- artifact `12000113891`
- no refetch.

**Do not spend another Week-6 OddsAPI credit.**

### WR player trust/selectivity — QUALIFIED

- disposition `WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`
- run `38823063793` SUCCESS
- job `116723999254`
- exact scored head `35a10d09ba92fd1218d37e812768fda5d2f4023e`
- artifact `12013153664`
- digest `sha256:12fd0de1dc3ebceee4d356669d59f30d96776d8c0800d1fb3ac6a293af457b7c`
- result `docs/research/WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_SELECTIVITY_V1_RESULT.md`.

Exact final WR-R15 parity:
- 3,224 rows
- 211 WRs
- pooled trust association +0.28853
- HIGH_TRUST abs error 14.61 yd
- overall 23.25 yd
- LOW_TRUST 30.47 yd
- positive pooled + 2024 player/game clustered CIs.

Interpretation: same-player strictly-prior opportunity difficulty is a qualified reliability selector for the **unchanged final WR-R15 receiving-yard projection**. It is not an over/under rule and does not authorize mean/width mutation.

### Week-6 WR trust shadow — FROZEN PROSPECTIVELY

- status `WEEK6_WR_TRUST_PREGAME_LOCK_FROZEN`
- run `38823520954` SUCCESS
- job `116725474887`
- exact head `638037c82b81e3e0f4d01ce37b1db24e5cc8144b`
- artifact `12012884262`
- digest `sha256:e5bc113e507ded651ecc29080c7fdbe4fd55182d1d039d0dd14057ada4b7bb38`
- row digest `sha256:81d9e950b96d7ac9d5699eba8490fc741a5aab2c938ef29287dc261534346b8f`
- result `docs/research/WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_WEEK6_PREGAME_LOCK_V1_RESULT.md`.

Cohort:
- 152 WRs
- 141 supported
- HIGH 35 / MIDDLE 70 / LOW 36 / unsupported 11
- reconstructed 2026 W1-W5 rows 717
- Week-6 outcomes read 0
- sportsbook 0
- OddsAPI 0
- production unchanged.

### TE trust remains qualified + frozen

Historical:
- `TE_PLAYER_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`
- run `38821176218`
- artifact `12013570473`.

Week-6:
- `WEEK6_TE_TRUST_PREGAME_LOCK_FROZEN`
- run `38822346088`
- artifact `12013018536`
- 25 HIGH / 52 MIDDLE / 26 LOW / 2 unsupported
- Week-6 outcomes read 0.

### Research status / frontier

Canonical status remains:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.

Do not reopen closed QB generic mean/YPA, target-depth symmetric distribution, WR route full-stack mean integration, WR allocation-only width bridge, TE width/Dirichlet mappings, or other terminal lanes.

After tonight's betting work, anti-retest Issue #535 before opening a genuinely distinct player-level lane. A legal candidate is named-player efficiency reliability/regime state that preserves existing means on first diagnostic and remains distinct from the already-closed generic efficiency-tail volatility family.

No production mutation.
No paid OddsAPI without explicit authorization.

---

## TOP CHECKPOINT — 2026-10-15 — TE TRUST QUALIFIED / WEEK-6 TRUST SHADOW FROZEN

Authoritative detailed handoff:
- `docs/handoffs/NFL_HANDOFF_2026-10-15_TE_TRUST_WEEK6_LOCK_FROZEN_CURRENT.md`

Historical qualification:
- `TE_PLAYER_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`
- run `38821176218` SUCCESS
- artifact `12013570473`
- digest `sha256:66c7b39f08b956dc1a96bd068272d1b97ddb5ddba22ef928a7d6e02c7f463530`

Prospective Week-6 lock:
- status `WEEK6_TE_TRUST_PREGAME_LOCK_FROZEN`
- run `38822346088` SUCCESS
- job `116721639322`
- exact head `1edecb1b3f60089e95bfc4c100522baf60d560a9`
- artifact `12013018536`
- digest `sha256:98667675753bfb410c6ac6259dc097476a87e49ce29f5cfe71be6fe14a74e1f1`
- row digest `sha256:fb5e2f83822a76adf3f09a721738237158bfddd9f95a3e63a97d3e4334d8ffa5`

Week-6 TE cohort:
- 105 TE-R5P players
- 103 supported
- 25 HIGH_TRUST / 52 MIDDLE / 26 LOW_TRUST / 2 UNSUPPORTED
- 2026 history source max week = 5
- Week-6 outcomes read = 0
- sportsbook = 0
- OddsAPI = 0
- projection/distribution changes = 0
- production unchanged.

Interpretation:
- named-player TE allocation difficulty qualifies as a trust/reliability selector for the **unchanged** TE-R5P projection;
- the exact failed TE width and Dirichlet transmission mappings remain closed;
- the Week-6 lock is prospective and must not be altered using target-week outcomes.

Immediate frontier:
- continue genuinely distinct individual-player research while Week 6 remains untouched;
- anti-retest first;
- prioritize named-player reliability/mechanism state rather than position-wide means;
- keep opportunity and efficiency mechanisms separate;
- no production promotion from the Week-6 lock until prospective scoring.

Production Week-6 live board is complete and parked. Do not return to PIT-TB unless user asks.

Canonical overall status:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.

No paid OddsAPI without explicit authorization.
No production mutation.

---

## TOP CHECKPOINT — 2026-10-15 — TE TRUST SELECTIVITY QUALIFIED / WEEK-6 PROSPECTIVE LOCK

Authoritative detailed handoff:
- `docs/handoffs/NFL_HANDOFF_2026-10-15_TE_TRUST_SELECTIVITY_QUALIFIED_WEEK6_LOCK_FRONTIER_CURRENT.md`

Canonical trust result:
- `TE_PLAYER_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`
- run `38821176218` SUCCESS
- job `116717842485`
- exact scored head `270f01266bff125cfaccb53a57b378674d919901`
- artifact `12013570473`
- digest `sha256:66c7b39f08b956dc1a96bd068272d1b97ddb5ddba22ef928a7d6e02c7f463530`
- result: `docs/research/TE_PLAYER_DIFFICULTY_TRUST_SELECTIVITY_V1_RESULT.md`

Every frozen gate passed.

2025 confirmation:
- primary association `+0.11538`
- player-cluster 95% CI `[+0.04191,+0.18535]`
- game-cluster 95% CI `[+0.04679,+0.18280]`
- HIGH_TRUST standardized CRPS `0.67012`
- LOW_TRUST standardized CRPS `1.09274`
- HIGH_TRUST raw abs error `8.70 yd`
- LOW_TRUST raw abs error `23.60 yd`

Interpretation:
- named-TE strictly-prior allocation difficulty is qualified as a player-specific **trust/reliability selector** for the unchanged TE-R5P projection;
- failed TE width and joint-Dirichlet mappings remain CLOSED;
- no projection mean/width/shape mutation is authorized;
- no production promotion yet.

Exact next action:
- freeze Week-6 pregame TE trust labels from football-only authority using the exact same trailing max-8/min-4 same-player allocation-error semantics;
- no proxy target-share volatility;
- no sportsbook input;
- no Week-6 outcomes;
- HIGH_TRUST <= .25 / LOW_TRUST >= .75;
- labels only, no projection/distribution mutation;
- fail closed if exact prior allocation-error history cannot be reconstructed.

Preferred Week-6 football authority:
- no-odds Full Slate run `38790425676`
- artifact `11999618082`
- digest `sha256:d7473705d4534c3172b58843272e7d95f5fbbd9aa910238d48916a8567cde1de`

Production Week-6 board is complete and parked; do not return to PIT-TB unless user asks.

Canonical scientific status remains:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.

No paid OddsAPI without explicit authorization.
No production mutation.

---

# CURRENT NFL RESEARCH HANDOFF — READ FIRST


## TOP CHECKPOINT — 2026-10-15 — TE MARGINAL WIDTH CLOSED / PLAYER TRUST FRONTIER

Authoritative detailed handoff:
- `docs/handoffs/NFL_HANDOFF_2026-10-15_TE_MARGINAL_WIDTH_CLOSED_PLAYER_TRUST_FRONTIER_CURRENT.md`

Newest completed TE marginal experiment:
- result: `docs/research/TE_PLAYER_ALLOCATION_MARGINAL_REC_YARDS_V1_RESULT.md`
- disposition: `NO_ACTIONABLE_TE_PLAYER_ALLOCATION_MARGINAL_REC_YARDS`
- canonical run `38800926331` SUCCESS
- job `116649432539`
- exact head `e7961d5bf6abe9097a8fbf21d776fb1696a57bba`
- artifact `12003898778`
- digest `sha256:0c59e6e551e6ac025ac21e4098c5298a3a6ac3e16ecef9f7cd5e8d2bb0db73a4`

What was learned:
- named-TE allocation difficulty **does** improve the canonical TE-R5P receiving-yard marginal in both 2024 and 2025;
- pooled named-vs-canonical CRPS gain = `+0.10936`;
- player-cluster 95% CI = `[+0.01851,+0.20876]`;
- game-cluster 95% CI = `[+0.02306,+0.19571]`;
- canonical TE marginal is materially under-dispersed;
- however the named-player overlay does **not** beat the frozen pooled-TE allocation-width control in either 2024 or 2025;
- exact mapping `same-player trailing allocation MAE -> target sigma -> pregame YPT -> independent rec-yard width` is CLOSED;
- do not rescue with coefficients, caps, TE1/TE2 slices, post-result subgroups, windows, alternate YPT conversion/noise family, or pooled-season qualification.

Canonical scientific status remains:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.

Immediate legal frontier:
- **player-level trust/selectivity**, not another width law;
- test whether strictly-prior named-player TE allocation difficulty identifies when the existing canonical TE-R5P projection/distribution is reliable vs fragile;
- preserve mean and distribution exactly;
- prove incremental information beyond contemporaneous pooled TE state;
- freeze the trust score/ranking/gates before scoring;
- 2024 development + 2025 independent confirmation + player/game clustered inference;
- if qualified, freeze a Week-6 pregame trust shadow from football-only authority before reading Week-6 outcomes.

Week 6 production is parked and complete:
- final live replay run `38792341891` SUCCESS;
- main `9b70b12e2fb4bc39f08556f244370068aad9c47b`;
- exactly one paid OddsAPI acquisition was used;
- do not return to PIT-TB/tickets unless the user asks.

No paid OddsAPI without explicit authorization.
No production mutation from research.

---

## TOP CHECKPOINT — 2026-10-14 — TARGETED-ROUTE STAGE B CONFIRMED / EXACT FULL-STACK MEAN INTEGRATION CLOSED

Authoritative detailed handoff:
- `docs/handoffs/NFL_HANDOFF_2026-10-14_TARGETED_ROUTE_STAGE_B_CONFIRMED_FULL_STACK_CLOSED_CURRENT.md`

Restart prompt:
- `docs/handoffs/NEXT_CHAT_PROMPT_2026-10-14_TARGETED_ROUTE_STAGE_B_CONFIRMED_FULL_STACK_CLOSED.md`

Canonical Stage B:
- `PLAYER_TARGETED_ROUTE_MATCHUP_CONFIRMED`
- run `38641891257` SUCCESS
- job `116119186261`
- exact scored head `fe1d8ff52bd7cfa322e095a7a6f58752acb0b55f`
- artifact `11929999088`
- digest `sha256:0ff2c393e0daa0fa5163d34ce35e38eb25fb7c907cdf287019d609b804ca5128`
- WR YPT + WR catch confirmed; WR direct rec-yards residual + all TE route families closed.

Canonical WR full-stack integration:
- `NO_ACTIONABLE_WR_ROUTE_MATCHUP_FULL_STACK_INTEGRATION`
- run `38642241032` SUCCESS
- job `116120366630`
- exact scored head `bdbc132e35939133e3fb9ee4fca1e85ff1857561`
- artifact `11930768340`
- digest `sha256:a9a0198efaca76ea8967813842468894b01c5d130781f2bb5b4c71881bfc36d3`
- result: `docs/research/WR_TARGETED_ROUTE_MATCHUP_FULL_STACK_INTEGRATION_V1_RESULT.md`

Interpretation:
- named-WR route-role matchup mechanism remains scientifically confirmed;
- the exact frozen direct mean transmission into the promoted WR stack is closed;
- do not rescue via coefficient/window/support/route-family/subgroup/cap changes;
- player-first standard remains in force and canonical status remains `PLAYER_INDIVIDUALIZATION_PARTIAL`.

Next clean frontier:
- separately confirmed WR/TE player-specific opportunity uncertainty / reliability integration;
- preserve promoted means and entitlement mass;
- distribution/reliability only;
- freeze before scoring.

No paid OddsAPI without explicit authorization. No production mutation.

---

GitHub is canonical; chat memory is secondary.

---
## ACTIVE CHECKPOINT — 2026-10-14 — PLAYER-FIRST INDIVIDUALIZATION / TARGETED-ROUTE MATCHUP SOURCE READY

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-10-14_PLAYER_FIRST_TARGETED_ROUTE_SOURCE_READY_CURRENT.md`

**Compact next-chat prompt:**

`docs/handoffs/NEXT_CHAT_PROMPT_2026-10-14_PLAYER_FIRST_TARGETED_ROUTE_SOURCE_READY.md`

This supersedes all older checkpoints for immediate execution while preserving their scientific lineage.

### Current state in one screen

- Production main remains `812ac2cde79fe3846174613a2f2777e57b34b739`.
- Active research remains isolated on PR #672 / branch `research-individual-opportunity-roadmap-2026-10-08`.
- User's model standard is now explicit: this is an **individual-player projection system**. Position-wide MAE/RMSE is a guardrail, not the hypothesis grain.
- Canonical status remains `PLAYER_INDIVIDUALIZATION_PARTIAL`.
- Do not equate a failed position-level/shared candidate with exhaustion of individual-player signal.
- Confirmed player-specific uncertainty:
  - `WR_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`
    - run `38595088817`, artifact `11908832554`
    - 3,224 player-games / 211 WRs
    - pooled CRPS ~1.47% better than pooled WR uncertainty
    - player-cluster 95% CI [+0.08180,+0.30846]
  - `TE_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`
    - run `38594365393`, artifact `11909006619`
    - 2,005 player-games / 121 TEs
    - pooled CRPS ~3.28% better than pooled TE uncertainty
    - player-cluster 95% CI [+0.19727,+0.46175]
  - historical RB `RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED` remains valid.
- Closed exact player/shared candidates:
  - player-x-environment V1 = support-limited null;
  - broad 11/12/OTHER personnel-conditioned opportunity = null;
  - WR-R3 exact combined integration = closed, but its failure does **not** erase WR same-player persistence;
  - efficiency-tail volatility V1 = null;
  - receiver conversion V1 = not confirmed;
  - Q90 upper-tail hinge = development null;
  - generic matchup/coverage/target-depth/NGS/WR-CB/RB retune families remain closed/source-blocked as documented.
- High-resolution source audit says:
  - total historical routes/route participation still source-parity blocked;
  - current WR slot/wide exists but historical parity not cleared;
  - player-attributed motion not available historically;
  - nflverse participation `route` **is usable only as the primary/targeted receiver's route label**, not total routes run.
- NEWEST SOURCE BREAKTHROUGH:
  `PLAYER_TARGETED_ROUTE_MATCHUP_SOURCE_READY`
  - frozen plan: `docs/research/PLAYER_TARGETED_ROUTE_MATCHUP_V1_FREEZE.md`
  - result: `docs/research/PLAYER_TARGETED_ROUTE_MATCHUP_V1_SOURCE_RESULT.md`
  - canonical run `38637901016` SUCCESS
  - job `116105672352`
  - exact scored head `159b8213a734e3f4ee16fc86fbb51950099f3323`
  - artifact `11928258474`
  - digest `sha256:b957ff0b37f0b2d8f45232bcd44657928cecc75f5f5e6d1c292ee8e35729228c`
  - 46,666 valid WR/TE targeted-route rows
  - 554 valid WR/TE players
  - 100% route/PBP join rate on joined rows
  - 93.18% stable receiver identity coverage on taxonomy-eligible rows
  - 10 route families >=1,000 rows
  - 32/32 defenses supported by Week 8 in 2023, 2024 and 2025
  - eligible named players: 257 / 246 / 253 in 2023/24/25
  - no sportsbook, OddsAPI, 2026 outcomes or production change.
- Mechanical-only guard: WR targeted-route full-stack runs `38631681403`, `38632065221`, and `38632845699` all failed before scientific scoring. Latest failure was a pyarrow/NumPy ABI mismatch inside TE-R5P snap loading; no full-stack result JSON existed. These are NOT scientific failures. Stage B remains undispositioned.
- CRITICAL SEMANTICS:
  - `route` = route label for the primary/targeted receiver on the play;
  - NOT total routes, route participation, TPRR, YPRR, or non-target route state;
  - do not infer WR-CB assignment.

### Exact next action

Build and run **Stage B** exactly as already frozen.

For each target WR/TE:
1. same named player only;
2. trailing max 8 eligible prior player-games;
3. minimum 4 strictly-prior player-games;
4. targeted-route mix only;
5. target opponent trailing max 8 defensive games;
6. strictly-prior route-family completion rate / yards per target / receiving yards per target;
7. strictly-prior league route-family neutral;
8. derive player-specific
   `matchup_delta = weighted opponent expectation - weighted league neutral`.

Primary outcomes, WR and TE separately:
- next-game receiving yards per target;
- next-game catch rate;
- next-game receiving yards.

Frozen validation:
- 2023 development;
- 2024 independent confirmation;
- 2025 independent confirmation;
- player-cluster CI required;
- game-cluster CI required;
- show it is not merely a renamed target-depth/air-yard signal;
- position-wide aggregate alone cannot qualify the mechanism.

If Stage B passes:
- freeze a separate full-stack integration candidate beyond M38/WR-R15 or TE-R5P.

If Stage B fails:
- close the exact targeted-route interaction with no regrouping/window/support/player-filter rescue.

Parallel rule:
- player-specific WR/TE/RB uncertainty integration may continue only under separate frozen contracts;
- keep mean and distribution research separate.

No paid OddsAPI without explicit user authorization.
No production mutation from research.

### Memory-efficient start rule

Read only:
1. `AGENTS.md`
2. this newest top checkpoint
3. `docs/handoffs/NFL_HANDOFF_2026-10-14_PLAYER_FIRST_TARGETED_ROUTE_SOURCE_READY_CURRENT.md`
4. `docs/research/PLAYER_TARGETED_ROUTE_MATCHUP_V1_FREEZE.md`
5. `docs/research/PLAYER_TARGETED_ROUTE_MATCHUP_V1_SOURCE_RESULT.md`
6. WR/TE player uncertainty result docs
7. newest Issue #535 comments
8. live PR #672 / branch / Actions

Then continue execution immediately. Do not recursively reread older handoffs unless a targeted anti-retest question requires one.

## ACTIVE CHECKPOINT — 2026-10-13 — WEEK-6 REPROBE BLOCKED / EFFICIENCY VOLATILITY V1 CLOSED

This checkpoint supersedes the V2 execution-frontier checkpoint for immediate
execution while preserving its deeper continuity.

Read next:
- `docs/handoffs/NFL_HANDOFF_2026-10-13_FULL_CONTINUITY_JOINT_OPPORTUNITY_WEEK6_V2.md`
- `docs/research/WEEK6_INDIVIDUAL_OPPORTUNITY_LOCK_READINESS_RESULT.md`
- `docs/research/PLAYER_EFFICIENCY_TAIL_STABILITY_V1_RESULT.md`

Current facts:
- production main remains `812ac2cde79fe3846174613a2f2777e57b34b739`;
- research remains isolated on PR #672 / branch
  `research-individual-opportunity-roadmap-2026-10-08`;
- no paid OddsAPI acquisition was used;
- fresh Week-6 roster reprobe run `38406807851` remains FAIL-CLOSED:
  roster source week 5, exact Week-6 roster false, Week-6 injury rows 0;
- Week-5 stats/snaps remain complete with clean chronology;
- therefore no Week-6 opportunity pack or joint Dirichlet shadow was created;
- individual efficiency-tail volatility V1 is complete and NULL:
  - run `38408536257` SUCCESS
  - artifact `11819986612`
  - digest
    `sha256:4bb8bead9f3a40f6465909c2148111d0c2aec9ff0b21a6b85fdc6bb27a0bb1e5`
  - 6,280 rows / 360 players / identity coverage 100% / chronology violations 0;
  - RB rushing YPC: no three-season replication;
  - WR target YPT: full pass only in 2023; 2024/2025 fail after frozen controls;
  - TE target YPT: frozen player-support gate fails all seasons;
  - RB receiving YPT: 2023 scientific fail, 2024/2025 support fail;
- no V2 efficiency-width candidate is authorized from this exact volatility
  family;
- do not rescue with windows, pooled seasons, subgroups, support-gate changes,
  tail thresholds, or matchup combinations;
- opportunity uncertainty remains a major PASS for RB carry + WR target and is
  still awaiting a legal Week-6 prospective lock;
- QB intent remains passive/source-strict;
- direct WR/TE depth-YPT and RB run-location-YPC remain closed;
- ticket outcomes remain downstream and may not retune football science.

Immediate legal research frontier:
1. keep Week-6 exact-roster source gate fail-closed;
2. anti-retest the remaining **efficiency regime/mechanism-state** space against
   completed WR CPOE/catchability/target-depth, QB M70/M71, RB run-location,
   historical-efficiency and residual-calibration work;
3. only freeze a new experiment if it uses genuinely distinct pregame
   information or state—not another historical efficiency-volatility transform;
4. no paid OddsAPI without explicit user authorization.

---
## ACTIVE CHECKPOINT — 2026-10-13 — FULL CONTINUITY / JOINT OPPORTUNITY / WEEK-6 EXECUTION FRONTIER V2

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-10-13_FULL_CONTINUITY_JOINT_OPPORTUNITY_WEEK6_V2.md`

**Compact next-chat prompt:**

`docs/handoffs/NEXT_CHAT_PROMPT_2026-10-13_FULL_CONTINUITY_JOINT_OPPORTUNITY_WEEK6_V2.md`

This supersedes the earlier Oct-13 checkpoint for immediate execution.

### Current state in one screen

- Production main remained `812ac2cde79fe3846174613a2f2777e57b34b739` at handoff.
- Research remains isolated on PR #672 / branch
  `research-individual-opportunity-roadmap-2026-10-08`.
- Canonical Week-5 preserved board remains run `37860592615`, artifact
  `11585973062`; no second paid OddsAPI acquisition.
- Week-5 selected board: 199-151, +26.17u; raw mean layer remains unsolved
  (model MAE 18.93 vs preserved median-line MAE 17.84).
- Week-5 top-40 misses: 19 efficiency-dominant / 14 opportunity-dominant /
  7 mixed.
- Three-season player opportunity-share volatility is confirmed.
- Excess role volatility survives finite-count noise for RB carry, RB target,
  and WR target; TE target fails the current mechanism gate.
- Mean-identical Beta-Binomial historical PASS:
  - RB carry share
  - WR target share
  RB target failed closed; TE not authorized.
- Major PASS: team-conserving joint Dirichlet allocation for RB carry + WR target
  in 2023/24/25:
  - run `38401069806`
  - artifact `11815490905`
  - digest
    `sha256:2b336114dc5ef0145c0a40706266da850a0ff1d25c8935958481b996333fce29`
- Prospective joint-shadow mechanics certified:
  - run `38402018307`
  - manual runner
    `.github/workflows/research-future-week-joint-opportunity-shadow-v1.yml`
- Week-6 lock was last verified FAIL-CLOSED because exact Week-6 weekly roster
  authority was unavailable. Week-5 stats/snaps were complete. Never substitute
  Week-5 roster identity.
- FIRST next-chat action: check exact Week-6 roster authority.
  - If available: freeze the existing Week-6 opportunity pack + joint
    RB-carry/WR-target shadow before Week-6 outcomes.
  - If unavailable: remain fail-closed and start the historical
    **individual efficiency-tail uncertainty** lane after anti-retest audit and
    frozen contract.
- QB coach/OC intent stays a passive prospective side lane.
- WR/TE direct depth-YPT and RB direct run-location YPC matchup candidates are
  FAILED CLOSED.
- Ticket construction remains downstream. Watson was a genuine projection miss
  plus cross-ticket concentration; do not use ticket outcomes to retune science.
- PR-side Backtest W1-2 and Replay Paid Full Slate reds are known
  Polars/pyarrow/NumPy environment failures, not scientific invalidation.
- No paid OddsAPI without explicit user approval.

### Exact next action

1. Reconcile live main / research branch / PR #672 / Issue #535 / exact-head Actions.
2. Check exact Week-6 weekly roster authority.
3. If ready, dispatch/freeze the already-built Week-6 opportunity lock pack and
   joint RB-carry/WR-target shadow.
4. If still blocked, immediately open the historical individual
   efficiency-tail uncertainty lane:
   - anti-retest inventory first;
   - strictly-prior player-level efficiency volatility/regime state;
   - freeze exact cells/features/windows/gates before scoring;
   - no Week-5 outcome tuning.
5. Mechanically repair the two Arrow/NumPy red workflows when useful, with zero
   football-science change.
6. Keep QB V1L passive and source-strict.
7. Keep ticket construction / exposure management separate from football science.

### Memory-efficient start rule

Read only:
1. `AGENTS.md`
2. this newest top checkpoint
3. `docs/handoffs/NFL_HANDOFF_2026-10-13_FULL_CONTINUITY_JOINT_OPPORTUNITY_WEEK6_V2.md`
4. `docs/research/WEEKS1_5_CONTINUITY_WEEK5_POSTMORTEM_SYNTHESIS_V1.md`
5. `docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_HISTORICAL_RESULT.md`
6. `docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_PROSPECTIVE_SHADOW_MECHANICS_RESULT.md`
7. `docs/research/WEEK6_INDIVIDUAL_OPPORTUNITY_LOCK_READINESS_RESULT.md`
8. newest Issue #535 / PR #672 comments
9. live main / research branch / Actions

Then continue execution immediately. Do not recursively read older handoffs
unless a targeted anti-retest question requires one.

---
## ACTIVE CHECKPOINT — 2026-10-13 — WEEK-5 POSTMORTEM / JOINT OPPORTUNITY UNCERTAINTY / WEEK-6 FRONTIER

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-10-13_WEEK5_POSTMORTEM_JOINT_OPPORTUNITY_WEEK6_FRONTIER_CURRENT.md`

**Compact next-chat prompt:**

`docs/handoffs/NEXT_CHAT_PROMPT_2026-10-13_WEEK5_POSTMORTEM_JOINT_OPPORTUNITY_WEEK6_FRONTIER.md`

This supersedes all older checkpoints for immediate execution.

### Current state in one screen

- Production main remains `812ac2cde79fe3846174613a2f2777e57b34b739`; research is isolated on PR #672 / branch `research-individual-opportunity-roadmap-2026-10-08`.
- Canonical Week-5 preserved board remains run `37860592615`, artifact `11585973062`, digest `sha256:bf52aa4bc30bf6e8cc6029a0a86c0f33cdfaa09f587c38a138f23cef6b7436f4`; no second paid OddsAPI acquisition.
- Weeks 1-5 selected-board continuity:
  - W1-W4 849-818, 50.93%, -44.10u;
  - W5 199-151, 56.86%, +26.17u;
  - through W5 1,048-969, 51.96%, -17.93u.
- Week-5 raw football mean layer is NOT solved:
  - model MAE 18.9288 vs preserved median-line MAE 17.8431;
  - top-40 misses = 19 efficiency-dominant / 14 opportunity-dominant / 7 mixed.
- User-facing ticket postmortem is separate from football science:
  - Watson = genuine projection miss + cross-ticket concentration;
  - BUF-LAR core = Allen 200+ HIT, Nacua 50+ HIT, Kyren 50+ MISS at 49;
  - alt-line distance is not calibrated survival probability.
- Strongest current player-specific result:
  `PLAYER_OPPORTUNITY_SHARE_STABILITY_THREE_SEASON_CONFIRMED`.
  Individual strictly-prior share volatility predicts next-game share instability in 2023/24/25.
- V2 excess-volatility mechanism survives finite-count noise for:
  - RB carry share;
  - RB target share;
  - WR target share.
  TE target fails the frozen three-season mechanism gate.
- Mean-identical Beta-Binomial historical PASS:
  - RB carry share;
  - WR target share.
  RB target fails closed; TE not authorized.
- **Major current PASS:** joint/team-conserving Dirichlet allocation historical run `38401069806`, artifact `11815490905`, digest `sha256:2b336114dc5ef0145c0a40706266da850a0ff1d25c8935958481b996333fce29`.
  RB carry + WR target pass independently in 2023, 2024 and 2025 while preserving exact canonical means and exact finite team opportunity totals.
- Prospective joint-shadow mechanics are certified:
  - run `38402018307` SUCCESS;
  - runner `.github/workflows/research-future-week-joint-opportunity-shadow-v1.yml`;
  - research-only; no production change.
- Week-6 lock is currently correctly FAIL-CLOSED:
  - Week-5 stats/snaps complete;
  - exact Week-6 weekly roster authority unavailable;
  - roster source is still Week 5;
  - target-week injury rows 0;
  - do NOT substitute Week-5 roster identity.
- If exact Week-6 roster authority is now available, the FIRST operational move is to rerun/freeze the existing opportunity lock pack and joint RB-carry/WR-target shadow before any Week-6 outcomes.
- If Week-6 roster is still unavailable, the next non-time-gated football lane is **individual efficiency-tail uncertainty**, frozen historically before scoring. Do not tune Week-5 misses.
- QB coach/OC intent is a passive prospective side lane:
  - corrected historical transport passed;
  - historical semantic automation failed / blanket promotion closed;
  - V1L Monday refresh still retained 0 sources;
  - do not loosen source rules.
- Player-specific direct matchup transforms are closed:
  - WR/TE depth-YPT V1 failed;
  - RB run-location YPC V1 failed;
  - do not retune/rescue.
- PR-side red `Backtest Weeks 1-2 Full Board` and `Replay Paid Full Slate Artifact Once` are Arrow/NumPy compatibility failures, not scientific failures or new odds pulls. Repo CI / historical input validation / clean benchmark rebuild are green.
- No paid OddsAPI without explicit user approval.

### Exact next action

1. Reconcile live main / research branch / PR #672 / Issue #535 / exact-head Actions.
2. Check exact Week-6 weekly roster authority.
3. If available:
   - rerun the frozen Week-6 opportunity lock pack;
   - require exact Week-6 identity and complete Week-5 stats/snaps;
   - freeze raw share-volatility state;
   - freeze RB carry/snap, RB receiving, WR/TE trajectory;
   - build/freeze the joint Dirichlet shadow for RB carry + WR target only;
   - preserve run/artifact/digest before Week-6 outcomes;
   - no production promotion.
4. If unavailable:
   - remain fail-closed;
   - immediately start the next historical player-specific root cause: efficiency-tail uncertainty, after anti-retest reconciliation and frozen contract.
5. Mechanically repair the two Arrow/NumPy red PR workflows when useful, without changing football science.
6. Keep ticket construction as a separate downstream risk/portfolio process.
7. Do not reopen closed matchup, QB generic mean, M95/M96, RB-target Beta-Binomial, or TE-target uncertainty families.

### Memory-efficient start rule

Read only:
1. `AGENTS.md`
2. this newest top checkpoint
3. `docs/handoffs/NFL_HANDOFF_2026-10-13_WEEK5_POSTMORTEM_JOINT_OPPORTUNITY_WEEK6_FRONTIER_CURRENT.md`
4. `docs/research/WEEKS1_5_CONTINUITY_WEEK5_POSTMORTEM_SYNTHESIS_V1.md`
5. `docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_HISTORICAL_RESULT.md`
6. `docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_PROSPECTIVE_SHADOW_MECHANICS_RESULT.md`
7. `docs/research/WEEK6_INDIVIDUAL_OPPORTUNITY_LOCK_READINESS_RESULT.md`
8. newest Issue #535 / PR #672 comments
9. live main / research branch / Actions

Then continue execution immediately. Do not recursively load older handoffs unless a targeted anti-retest question requires one.


---
## ACTIVE CHECKPOINT — 2026-10-08 — PLAYER LANDSCAPE / OPPORTUNITY-DOMINANT CURRENT

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-10-08_PLAYER_LANDSCAPE_OPPORTUNITY_DOMINANT_CURRENT.md`

**Compact next-chat prompt:**

`docs/handoffs/NEXT_CHAT_PROMPT_2026-10-08_PLAYER_LANDSCAPE_OPPORTUNITY_DOMINANT.md`

This supersedes all older checkpoints for immediate execution.

### Current state in one screen

- The model is now proven to have a **real individual-player core**; it is not a
  position model with names attached.
- Player Landscape Transmission Audit V1 is complete:
  `INDIVIDUAL_PLAYER_CORE_CONFIRMED__LANDSCAPE_TRANSMISSION_INCOMPLETE`.
- The full 2026 W1-W4 all-player replay is complete.
- The newest decisive result is Player Output Component Decomposition V1:
  `OPPORTUNITY_DOMINANT_ACROSS_ALL_PRIMARY_MARKETS__INDIVIDUAL_ROLE_SHARE_ALLOCATION_NEXT`.
- Exact latest successful decomposition authority:
  - run `37777319154` SUCCESS
  - scientific head `cb1e511adcb58e40fc41d0949306083a304e83dd`
  - artifact `11550446909`
  - digest `sha256:7acfc453258a49f3aadf0d0ea386b13827eb715f1dfb055b96505f733320d204`
  - result doc:
    `docs/research/PLAYER_OUTPUT_COMPONENT_DECOMPOSITION_V1_RESULT.md`
    on branch `research-player-output-component-decomposition-v1`.
- Opportunity oracle beats efficiency oracle in **all 8** tested cells:
  - QB pass yards: 23.9% vs 12.7% MAE removed
  - RB rush yards: 45.1% vs 7.2%
  - RB rec yards: 45.1% vs 27.5%
  - RB receptions: 57.2% vs -2.4%
  - WR rec yards: 42.8% vs 28.2%
  - WR receptions: 54.4% vs 11.6%
  - TE rec yards: 55.0% vs 23.5%
  - TE receptions: 68.3% vs 8.1%.
- Therefore the next model-development work must stay inside **individual
  pregame opportunity / role / share allocation**, not another generic
  efficiency/matchup multiplier.
- RB receiving-room share already improved real W1-W4 individual projections:
  targets 7.60%, receptions 3.83%, rec yards 2.89%, rush+rec 1.66%.
- RB receiving-room and RB carry/snap Week-5 locks are frozen prospectively.
- WR/TE target-share trajectory is historically confirmed and Week-5 shadow is
  frozen. Exact trajectory is ineligible in 2026 W1-W4 because it requires four
  prior same-season games; do not weaken the rule.
- The universal symmetric target-depth distribution transform did not improve
  W1-W4 CRPS and remains unpromoted.
- Route participation/YPRR historical predictive testing is source-parity
  blocked; do not fabricate routes from nflverse participation labels.
- WR-R3 combined calibration was already built/run and is CLOSED:
  `NO_ACTIONABLE_WR_R3_COMBINED_CALIBRATION`.
- Coverage-v2 team man/zone is near-null; player WR-CB historical assignment is
  source blocked.
- Generic game-script/Vegas confirmation produced no actionable pregame state.
- All three simple Football Matchup Transmission V1 integration formulas failed
  closed.
- M95A/M95B generic RB role x defense remains closed.
- No paid OddsAPI without explicit user authorization.
- Week-5 prospective locks remain outcome-blind at this morning checkpoint;
  do not grade them prematurely.
- No relevant run should be assumed active; verify live Actions.

### Exact next action

1. Verify live main / active branches / Issue #535 / Actions.
2. Do **not** rerun the completed all-player replay, landscape audit, or output
   component decomposition.
3. Build a bounded next-step roadmap for **individual opportunity allocation**,
   explicitly separating:
   - participation / active-role probability;
   - team opportunity volume;
   - room share / player allocation.
4. Reconcile against all completed opportunity research before opening a new
   candidate; do not duplicate closed/source-blocked lanes.
5. WR/TE: the clean legal validation path is an exact frozen historical
   Week-5+ integration test of Target Share Trajectory V1 while preserving
   M38 / WR-R15 / TE-R5P and team target mass.
6. QB: prioritize attempt/starter/team-volume allocation, not generic YPA/mean.
7. RB: preserve M96 stop; do not back-apply Week-5 carry/snap shadow to W1-W4.
   RB receiving-room share is already retrospectively confirmed.
8. If considering participation/zero-state, first reconcile the completed
   historical-availability parity and opportunity audits. No outcome-selected
   cutoff.
9. Freeze any exact new mechanism before scoring it.
10. Keep efficiency/matchup work secondary until opportunity allocation for the
    relevant market is resolved.

### Memory-efficient start rule

Read only:
1. `AGENTS.md`
2. this newest top checkpoint
3. `docs/handoffs/NFL_HANDOFF_2026-10-08_PLAYER_LANDSCAPE_OPPORTUNITY_DOMINANT_CURRENT.md`
4. `docs/research/PLAYER_OUTPUT_COMPONENT_DECOMPOSITION_V1_RESULT.md`
   on branch `research-player-output-component-decomposition-v1`
5. `docs/research/PLAYER_LANDSCAPE_GAP_DISPOSITION_V1.md`
6. `docs/research/PLAYER_LANDSCAPE_TRANSMISSION_AUDIT_V1_RESULT.md`
7. `docs/research/ALL_PLAYER_ALL_POSITION_REPLAY_V1_RESULT.md`
8. newest Issue #535 continuity comment
9. live main / relevant branches / Actions

Then continue execution immediately. Do not recursively load older handoffs
unless a targeted anti-retest question requires one.

---
## ACTIVE CHECKPOINT — 2026-10-07 — PLAYER-CENTRIC ALL-POSITIONS CURRENT

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-10-07_PLAYER_CENTRIC_ALL_POSITIONS_CURRENT.md`

**Compact next-chat prompt:**

`docs/handoffs/NEXT_CHAT_PROMPT_2026-10-07_PLAYER_CENTRIC_ALL_POSITIONS.md`

This supersedes all older checkpoints for immediate execution.

### Current state in one screen

- The current research priority is **individual-player modeling as a complement to the validated stack**, not replacing prior team/opponent/position science.
- Player Individualization Audit V1 = `PLAYER_INDIVIDUALIZATION_PARTIAL`: real player history exists, but position shrinkage/shared weekly transforms remain material.
- Full-stack correction:
  - QB mean already strongly individualized by M89/M90; do not reopen generic QB mean.
  - WR has M38 + WR-R15 individual entitlement.
  - TE has TE-R5P individual entitlement.
  - non-Week-1 RB remains the largest individual room-allocation gap.
- RB Week-5 prospective allocation shadow is frozen:
  - run `37560824479` SUCCESS
  - 98 RB/HB/FB identities across 30 teams
  - 30/30 rooms changed
  - frozen 50/50 recent carry-share + snap-fraction allocation
  - no Week-5 outcomes / no sportsbook / no production change.
- QB-receiver pair source is live-ready, but raw pair-specific YPT is **CLOSED**; do not rescue.
- WR mechanism persistence = `WR_PLAYER_PERSISTENCE_MIXED_MECHANISM`.
- TE mechanism persistence = `TE_PLAYER_PERSISTENCE_MIXED_MECHANISM`.
- Both WR and TE show:
  - persistent individual opportunity error/difficulty;
  - persistent individual efficiency **difficulty**;
  - no stable signed efficiency bias.
- Situational third-down/red-zone/two-minute target earning is real/nonredundant source state but **CLOSED** as an explanation of opportunity residual.
- Individual target-share trajectory is **CONFIRMED** across WR and TE:
  - run `37638269235` SUCCESS
  - WR pooled rho -0.07646
  - TE pooled rho -0.09876
  - combined rho -0.08496.
- Week-5 target-share trajectory shadow is immutably frozen:
  - run `37654382316` SUCCESS
  - artifact `11497153776`
  - row digest `sha256:afbfd7f360c50fcd4850c0967be40f9a333da1bd5f835cdc676c2e88d777c1f3`
  - 277 WR/TE rows; 247 changed; M38 WR1 frozen; WR2+/TE pools and team target mass conserved.
- Individual target-depth dispersion is **CONFIRMED** as a player-specific efficiency-difficulty signal:
  - run `37655282486` SUCCESS
  - WR pooled rho +0.08499
  - TE pooled rho +0.05414
  - combined rho +0.18412.
- Branch `research-player-target-depth-distribution-shadow-v1` exists but has **no contract/code/run yet**.
- User's immediate priority: **button up all QB/RB/WR/TE player-level work before starting the all-player/all-position replay**.
- Position readiness:
  - QB: ready for this phase;
  - RB: main unfinished position;
  - WR: opportunity ready, distribution layer unfinished;
  - TE: opportunity ready, distribution layer unfinished.
- Exact trajectory V1 cannot operate in 2026 Weeks 1–4 because it requires 4 prior same-season games. Do not weaken the frozen rule after seeing outcomes.
- Target-depth dispersion can potentially replay 2026 Weeks 1–4 using strictly-prior 2025 history.
- no paid OddsAPI pull without explicit user authorization.
- no relevant run is active at checkpoint time; do not claim otherwise.

### Exact next action

1. Query live main / Issue #535 / relevant research branches / Actions.
2. Answer the user's position-status question immediately:
   - QB ready;
   - RB is the main unfinished position;
   - WR/TE opportunity ready but player-specific distribution shadow unfinished.
3. Freeze and implement `research-player-target-depth-distribution-shadow-v1` as a **mean-neutral** uncertainty/distribution candidate:
   - preserve exact receiving-yard means;
   - preserve target entitlement;
   - preserve M38 / WR-R15 / TE-R5P;
   - preserve team pass/target volume;
   - no sportsbook inputs.
4. Then resolve RB's remaining legal/open player-level scope without violating M96 or reopening closed RB mean/width families.
5. Only when QB/RB/WR/TE are explicitly buttoned up, start the all-position replay.
6. For 2026 W1–4:
   - trajectory V1 is ineligible under its exact frozen contract;
   - target-depth distribution can potentially be replayed with strictly-prior 2025 history;
   - historical Week-5+ seasons can integration-test the exact trajectory transform.
7. Preserve all prior closures / anti-retest rules.

### Memory-efficient start rule

Read only:
1. `AGENTS.md`
2. this newest top checkpoint
3. `docs/handoffs/NFL_HANDOFF_2026-10-07_PLAYER_CENTRIC_ALL_POSITIONS_CURRENT.md`
4. `docs/research/PLAYER_INDIVIDUALIZATION_AUDIT_V1_FULL_STACK_ADDENDUM.md`
5. `docs/research/RB_PLAYER_STATE_ALLOCATION_SHADOW_V1_WEEK5_LOCK.md`
6. `docs/research/PLAYER_TARGET_SHARE_TRAJECTORY_V1_RESULT.md`
7. `docs/research/PLAYER_TARGET_SHARE_TRAJECTORY_SHADOW_V1_WEEK5_LOCK.md`
8. `docs/research/PLAYER_TARGET_DEPTH_DISPERSION_V1_RESULT.md`
9. Issue #535 from comment `6029157126` onward
10. live main / relevant branches / Actions

Then work. Do not recursively load older handoffs unless a targeted anti-retest question requires one.



---
## ACTIVE CHECKPOINT — 2026-10-06 — WEEK 4 POSTMORTEM / FOOTBALL MATCHUP TRANSMISSION CURRENT

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-10-06_WEEK4_POSTMORTEM_MATCHUP_TRANSMISSION_CURRENT.md`

**Compact next-chat prompt:**

`docs/handoffs/NEXT_CHAT_PROMPT_2026-10-06_WEEK4_POSTMORTEM_MATCHUP_TRANSMISSION.md`

This supersedes all older checkpoints for immediate execution.

### Current state in one screen

- Week 4 is fully settled: **220-207, 51.52%, -3.38u**.
- Weeks 1-4 cumulative: **849-818, 50.93%, -44.10u**.
- no simple position/market/side/edge slice survives clustered BH-FDR; raw edge/fair probability remains non-validated confidence.
- failure-mode atlas found a severe asymmetric high-side miss problem: 270 rows >2 model SD, 93.33% actual-above-model, 83.70% selected UNDER, 82.59% loss rate.
- rush+receiving remains a separate center problem: RB mean error -16.60 yards, median -8.82, driven primarily by rushing.
- Opportunity State Conflict Uncertainty V1 is **NULL/CLOSED**; do not widen based on PlayerForm-vs-Bayes disagreement.
- historical Distribution Right-Tail Asymmetry V1 is **CONFIRMED** for rush_yards, rec_yards, receptions, and rush_rec_yards in both 2024 and 2025; pass_yards did not replicate.
- opponent-defender injury source-readiness V1 is CLOSED / parity not cleared.
- **primary active lane is Football Matchup Transmission V1**, based on the user's explicit football-layer concern.
- Phase A run `37504432928` SUCCESS confirmed the generic RB/WR/TE stack drops or weakly transmits meaningful pregame matchup state:
  - def_rush_epa present but unused by generic rules;
  - explosive-play rate allowed present but unused;
  - WR/TE/RB YPT allowed and outside/slot YPT dropped before TeamContext;
  - YBC/stuff run-defense fields dropped;
  - generic game-script pass share hardcoded to 0.57;
  - rules_pass_rate normally bypasses PROE/pass-tendency fallback;
  - lead/trail probabilities do not alter the simulation pass/rush split;
  - RB matchup is mainly coarse light/heavy-box YPC thresholds.
- Bijan Week-4 trace is architecture evidence only: ATL neutral pass rate 46.5%, PlayerForm rush share .6158, Bayes/rules .5421, production pass rate .57, NO box rules yielded rush-efficiency multiplier exactly 1.0. Do not tune to Bijan's outcome.
- M95A/M95B already tested generic role x defense ideas; do not create a post-hoc "bad run defense => boost RB" rule.
- no paid OddsAPI pull without explicit user authorization.

### Exact next action

1. Query live main / research branches / Issue #535 / Actions.
2. **Do not rerun Football Matchup Transmission Phase A.**
3. Continue `research-football-matchup-transmission-v1` into the already-frozen Phase B/C historical audit:
   - leakage-safe 2024 W2-18 and 2025 W2-18;
   - zero 2026 outcomes;
   - zero sportsbook inputs;
   - no coefficient/threshold fitting;
   - test predeclared defensive matchup families conditional on existing role/opportunity;
   - then test continuous usage x matchup interactions.
4. Require expected-sign replication in BOTH 2024 and 2025 + clustered support + incremental information + live semantic parity before any candidate can advance.
5. If a family passes, freeze a separate integration candidate BEFORE scoring it.
6. Secondary: write canonical RESULT docs for Right-Tail Asymmetry V1 and Matchup Transmission Phase A if still absent.
7. Preserve all closed-family anti-retest rules.

### Memory-efficient start rule

Read only:
1. `AGENTS.md`
2. this newest top checkpoint
3. `docs/handoffs/NFL_HANDOFF_2026-10-06_WEEK4_POSTMORTEM_MATCHUP_TRANSMISSION_CURRENT.md`
4. `docs/research/FOOTBALL_MATCHUP_TRANSMISSION_V1_PLAN.md`
5. Issue #535 from comment `6021335699` onward, especially `6021798748` and `6022435377`
6. live main / relevant research branches / Actions

Then immediately continue Phase B/C. Do not recursively load older handoffs unless a targeted anti-retest question requires one.


---
## ACTIVE CHECKPOINT — 2026-10-01 — WEEK 4 LIVE BOARD RECOVERED / GAME-DAY CURRENT

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-10-01_WEEK4_LIVE_BOARD_RECOVERED_CURRENT.md`

**Compact next-chat prompt:**

`docs/handoffs/NEXT_CHAT_PROMPT_2026-10-01_WEEK4_LIVE_BOARD_RECOVERED.md`

This supersedes all older checkpoints for immediate execution.

### Current state in one screen

- production main before this docs-only continuity commit: `0e6d8ab63c395b6bb9e88b0d55844083ac3221a5`;
- ONE paid Week-4 live-odds run was explicitly authorized;
- dispatcher `36934550541` SUCCESS;
- paid Full Slate `36934563481` acquired live odds and reached pricing, then failed on QB-C2 live-offer/starter identity conflict only;
- exact paid artifact `11197726111`, digest `sha256:202aa5205e71ad0acedf1910f505c2b362f564d8593a0f82dd0928fb45175aae`;
- failure identities: CHI Tyson Bagent and WAS Marcus Mariota; sportsbook is not allowed to choose upstream football starter;
- offline recovery `36935917903` SUCCESS using the same paid artifact with **no second OddsAPI fetch**;
- authoritative recovered live board artifact `11197776900`, digest `sha256:3f058570037ca016a5cbf1fa79e6e6abc4d845384de4dbdfaf477c8f0a3160a8`;
- recovered board = 3,460 side rows / 1,730 priced offers / 873 player-market rows / pricing status CURRENT;
- PR #671 merged final-board Bagent/Mariota quarantine; latest no-live main Full Slate `36936244158` SUCCESS;
- unfinished dynamic QB conflict branch `repair-week4-qb-live-starter-conflict-v1@d1003a5809bb17b02592b533eb1f42bc247fb88c` still fails verifier `36937252724` because quarantine-authority marker is missing from lineage governance;
- do **not** refetch live odds to continue today's board; use recovered artifact first;
- raw edge is not validated confidence: scalar fair-line selector and offer-level residual selector both closed null;
- GSIS RB V1 Week-4 V2 private lock remains sealed prospectively at 1/6 weeks, 2/10 vacancy team-games, 5/20 successor player-games;
- ESPN availability semantics repair #670 merged;
- RNG candidate #666 merged as validated candidate code but not silently production-activated;
- WR/CB source-blocked, CLV passive, RB-PD2 HOLD;
- Weeks 1-3 postmortem remains CLOSED at 629-611, 50.7%, -40.72u.

### Exact next action

1. Query live main / Issue #535 / #669 / Actions.
2. Pull **recovered Week-4 artifact `11197776900`** and inspect the live board/workbook.
3. Do not spend OddsAPI again unless the user explicitly reauthorizes.
4. If continuing dynamic QB plumbing, fix the quarantine-authority marker and replay exact paid artifact `11197726111`; no new sportsbook acquisition.
5. Preserve all frozen research stop rules.

### Memory-efficient start rule

Read only:
1. `AGENTS.md`
2. this newest top checkpoint
3. `docs/handoffs/NFL_HANDOFF_2026-10-01_WEEK4_LIVE_BOARD_RECOVERED_CURRENT.md`
4. Issue #535 from `5936680436` onward
5. live main / relevant PRs / branches / Actions

Then work. Do not recursively load older handoffs.


---
## ACTIVE CHECKPOINT — 2026-09-30 — WR/CB ARCHIVE BREAKTHROUGH / SOURCE INTEGRITY CURRENT

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-09-30_WR_CB_ARCHIVE_BREAKTHROUGH_CURRENT.md`

**Compact next-chat prompt:**

`docs/handoffs/NEXT_CHAT_PROMPT_2026-09-30_WR_CB_ARCHIVE_BREAKTHROUGH.md`

This supersedes all older checkpoints for immediate execution.

### Current state in one screen

- live main at checkpoint preparation: `50f6624be68fcac3af88fe40a258b43eec71402c` (verify live before editing);
- Week 3 / Weeks 1-3 is CLOSED: **W3 233-208 +2.26u; W1-3 629-611, 50.7%, -40.72u**;
- Claude's extra-DNP theory is closed: canonical 20 voids stand because snap-count participation proves the disputed players appeared;
- PR #665 draft head at checkpoint: `1a39b519d472d68a10096f2d9faa992d69e27d60`;
- #665 fail-closed repairs are integrated: provider-ID reuse guard + exact-week opponent-CB roster guard; live-page provisional source-ready **3,999**, source/model gate still CLOSED;
- historical public archive point-in-time recovery is now PROVEN:
  - 2024W1: **46 strict** verified rows + immutable lock;
  - protected 2025W14: **47 strict** verified rows + immutable lock, **no 2025 outcomes accessed**;
  - latest 2022W5 run `36773818193` SUCCESS: **41 verified pregame factual / 36 strict exact-week identity** rows, artifact `11124612896`;
- archive branch at checkpoint: `research-wr-cb-historical-snapshot-discovery-v1@ca29322430a2600eef195ca7fc2704c6e94fe13c`;
- broad Wayback scans are ACCESS-INCONCLUSIVE and can false-negative known captures; use sparse exact discovery only;
- do not restore retired `coverage_penalty()`;
- FantasyAlarm pairings are projected pregame alignments, NOT observed route responsibility; editorial grades are never football features;
- PR #663 CLV is still draft; Repo CI green; frozen Week2 replay red only because its pinned artifact expired;
- RNG candidate `30333ec2...` exact validation passed but is not production-promoted;
- PR #662 GSIS remains research-only; never upload private raw GSIS;
- no paid WR/CB data and no paid OddsAPI without explicit approval.

### Exact next action

1. Query live main / Issue #535 / #665 / archive branch / #663 / #662 / RNG Actions.
2. Freeze the successful **2022W5** archive result and immutable sanitized source lock/hash if still missing.
3. Continue sparse historical source recovery in discovery years only.
4. Keep protected 2025 outcomes untouched until source population + preregistered scientific test are frozen.
5. Prospectively capture verified 2026 WR/CB reports before kickoff if/when a real URL appears.
6. Do not rerun or retune closed Weeks 1-3 science.

### Memory-efficient start rule

Read only:
1. `AGENTS.md`
2. this newest top checkpoint
3. `docs/handoffs/NFL_HANDOFF_2026-09-30_WR_CB_ARCHIVE_BREAKTHROUGH_CURRENT.md`
4. Issue #535 from comment `5913566949` onward
5. live branches/PRs/Actions

Then work. Do not recursively load older handoffs.

---

---
## ACTIVE CHECKPOINT — 2026-09-29 — RESEARCH LEAD / WR-CB SOURCE / CLV CURRENT

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-09-29_RESEARCH_LEAD_WR_CB_CLV_CURRENT.md`

**Compact next-chat prompt:**

`docs/handoffs/NEXT_CHAT_PROMPT_2026-09-29_RESEARCH_LEAD_WR_CB_CLV.md`

This supersedes all older checkpoints for immediate execution.

### Current state in one screen

- canonical production main before these docs-only continuity commits: `deea0f68202c9ab6e85fa616f7444d0008ee10af`;
- PR #664 is MERGED: unsupported grandfathered `coverage_penalty()` heuristic retired from production;
- do **not** restore static 0.92/0.94/1.06/1.04 coverage multipliers;
- direct player-level WR-CB assignment remains fail-closed/gated off unless a validated source exists;
- free FantasyAlarm historical WR/CB archive is confirmed across 2021-2026 and is being audited in draft PR #665; older 2021/2022 vintages have incomplete slot coverage, so missing slot rows are NOT zero/no-matchup; research-only until completeness/timing/identity/semantics clear;
- PR #663 is the append-only market-snapshot/CLV architecture; still draft/open and must be made green before merge;
- only same-book <=30-minutes-before-kickoff snapshots may be labeled CLV;
- no paid OddsAPI pull solely for CLV or WR-CB research;
- Week 3 / Weeks 1-3 postmortem remains closed at **629-611, 50.7%, -40.72u**;
- projection line-cross primary hypothesis failed; Week-4+ authority-move direction is forward-only;
- RB-PD2 remains 1/8 weeks and 46/400 rows;
- RB route-volume is live-source-ready but historical/live weekly parity remains uncleared; prospective capture only;
- specialist RNG patch `30333ec2...` still requires exact frozen fingerprint validation;
- Claude owns opponent-injury propagation / cross-audit in Issue #535; read newer comments before duplicating work;
- Issue #535 comment `5901927972` contains a stale statement that the legacy team-level coverage heuristic is still active; current main and the detailed handoff supersede that statement.

### Exact next action

1. Query live main / PR #663 / PR #665 / PR #662 / RNG repair branch / Actions.
2. Finish PR #663 mechanically without changing its frozen CLV contract or spending odds credits.
3. Continue PR #665 source-readiness validation; do not fit or promote WR-CB science until its source gate clears.
4. Cross-audit Claude's opponent-injury result if posted.
5. Verify specialist RNG exact-parity repair.
6. Preserve Week-4+ forward locks/captures; do not recycle closed Weeks 1-3 science.

### Memory-efficient start rule

Read only:
1. `AGENTS.md`
2. this newest top checkpoint
3. `docs/handoffs/NFL_HANDOFF_2026-09-29_RESEARCH_LEAD_WR_CB_CLV_CURRENT.md`
4. Issue #535 from `5899500522` onward, especially `5899645764`, `5899710345`, `5899966307`, `5900122307`, `5901422664`, `5901433362`, `5901608387`, `5902638669`, plus anything newer
5. live main / PRs / branches / Actions

Then work. Do not recursively load older handoffs.

---

---
## ACTIVE CHECKPOINT — 2026-09-29 — RESEARCH LEAD / WR-CB SOURCE FRONTIER

Read this checkpoint, then:
`docs/handoffs/NFL_HANDOFF_2026-09-29_RESEARCH_LEAD_WR_CB_CURRENT.md`

Next-chat prompt:
`docs/handoffs/NEXT_CHAT_PROMPT_2026-09-29_RESEARCH_LEAD_WR_CB.md`

### Production truth to preserve

- canonical production code includes merged PR #664 at `deea0f68202c9ab6e85fa616f7444d0008ee10af`; later main descendants may be docs-only continuity commits;
- legacy `coverage_penalty()` and static 0.92/0.94/1.06/1.04 WR coverage multipliers are **removed** from production;
- direct player-level WR-CB assignment remains fail-closed/gated off when unavailable;
- do not pay for WR-CB data;
- do not restore the deleted heuristic from stale Issue wording.

### Immediate next action

Resolve PR #665 live:
`research-wr-cb-free-archive-v1`

Issue #535 contains contradictory concurrent-chat narratives:
- comment `5901608387` says a free FantasyAlarm 2021-2026 WR/CB archive may unblock true assignment research;
- comment `5901927972` says the historical machine-reproducible source gate is not yet cleared.

Do not resolve this by comment chronology. Inspect PR #665 branch, source-audit output and CI directly.

If source gate clears:
- factual pairing/alignment only;
- freeze new true-assignment science before outcome inspection;
- editorial Safe/Moderate/Risky grades are not model features;
- do not restore the old heuristic.

If source gate fails:
- close source lane cleanly;
- keep production WR-CB assignment gated off;
- spend $0.

### Parallel coordination

- Claude-owned branch exists: `research-rb-opponent-defender-injury-readiness-v1`; inspect before duplicating work.
- PR #663 = append-only downstream market snapshots / T30 CLV; query live before editing because concurrent fixes may exist.
- specialist RNG branch still requires exact frozen fingerprint validation; do not weaken gates.
- GSIS PR #662 stays research-only; raw GSIS private.
- RB-PD2 stays HOLD at 1/8 weeks and 46/400 rows.
- Weeks 1-3 postmortem remains closed: 629-611, 50.7%, -40.72u.
- no paid OddsAPI without explicit user approval.

### Concurrent-chat rule

When Issue comments disagree:
1. live `main` code is canonical for production;
2. live branch/PR code is canonical for that lane;
3. newest frozen plan/result on that branch beats stale narrative;
4. query Actions before editing.

Do not recursively read older handoffs.

---

---
## ACTIVE CHECKPOINT — 2026-09-29 — WEEK-3 POSTMORTEM COMPLETE / PROSPECTIVE RESULTS CURRENT

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-09-29_WEEK3_POSTMORTEM_COMPLETE_CURRENT.md`

**Compact next-chat prompt:**

`docs/handoffs/NEXT_CHAT_PROMPT_2026-09-29_WEEK3_POSTMORTEM_COMPLETE.md`

This supersedes all older checkpoints for immediate execution.

### Current state in one screen

- canonical main before this pointer update: `0a2d74fe0aae2454acd4de8a7c97f803f1a2b660`;
- active postmortem branch: `research-week3-postmortem-execution-v1@fb80bb6c4af6c280aee53d3013315632c69a7c9b`;
- Week 3 is fully settled; Weeks 1-3 cumulative postmortem is complete;
- cumulative production record: **629-611, 50.7%, -40.72u**;
- no tested position/market/side slice survived clustered BH-FDR correction;
- projection-authority result: `MIXED_BY_MARKET_OR_POSITION`;
- RB Vacancy Week 3: `WEEK3_OBSERVATIONAL_MIXED`;
- public intent Week 3: `WEEK3_PUBLIC_INTENT_DIRECTIONALLY_INFORMATIVE` (observational only);
- Receiving Rule Semantics Week 3: no candidate promoted;
- RB-PD2 Week 3: Observation #1 complete, **46/400 rows and 1/8 weeks**, disposition `NO_FORWARD_CONFIRMATION_INSUFFICIENT_SUPPORT`;
- latest RB-PD2 verification run `36631186678` = SUCCESS;
- no Week-3 outcome-driven production change was authorized;
- immediate unfinished Week-3 item: Availability -> Opportunity postgame closure, but only under its original frozen contract; never invent post-hoc thresholds;
- specialist RNG production-candidate repair remains a separate open mechanical lane; latest run `36504441918` failed and must be diagnosed from logs before edits;
- GSIS Lineup Detail + Formation Usage are incrementally descriptive, but temporal use requires repeated immutable snapshots; raw GSIS remains private;
- no paid OddsAPI pull without explicit user authorization.

### Exact next action

1. Read the original Availability -> Opportunity frozen contract/result and determine whether a valid postgame descriptive grade was preregistered.
2. If yes, grade it exactly as frozen with no fitting. If no, record no valid postgame grading contract rather than inventing one.
3. Create one consolidated Week-3 prospective disposition matrix.
4. Only then choose the next genuinely new science from Weeks 1-3 error, without reopening closed lanes.
5. Secondary: diagnose RNG run `36504441918` if resuming the mechanical repair lane.

### Memory-efficient start rule

Read only:
1. `AGENTS.md`
2. this newest top checkpoint
3. `docs/handoffs/NFL_HANDOFF_2026-09-29_WEEK3_POSTMORTEM_COMPLETE_CURRENT.md`
4. Issue #535 from comment `5897966236` onward, plus GSIS checkpoint `5881240970`
5. live main / relevant branches / PRs / Actions

Then work. Do not recursively load older handoffs.

---

---
## ACTIVE CHECKPOINT — 2026-09-27 — WEEK-3 LIVE POSTMORTEM / RESEARCH CURRENT

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-09-27_WEEK3_LIVE_POSTMORTEM_RESEARCH_CURRENT.md`

**Compact next-chat prompt:**

`docs/handoffs/NEXT_CHAT_PROMPT_2026-09-27_WEEK3_POSTMORTEM_RESEARCH.md`

This supersedes all older checkpoints for immediate execution.

### Current state in one screen

- canonical current main: `29e5df40a5693b3ccb7135c2bcf7c1bd3688955d`;
- docs-only continuity descendant of production/research main `d2a1366f04d38cfd1a4ad3a10a49d8f2102afafd`;
- continuity PR #658 = MERGED;
- Repo CI `36357199470` SUCCESS;
- no-live Full Slate `36357199471` SUCCESS;
- Archive Market Track Record `36357434393` SUCCESS;
- no OddsAPI acquisition occurred in those verification runs;
- canonical paid Week-3 betting board remains `36293274478` / artifact `10923570170`;
- user reports Week-3 live performance looked very poor: **freeze production and grade objectively before changing science**;
- Week-3 RB-PD2 prospective lock is VALID: 46 locked rows; Week 3 = Observation Week #1; no science PASS/FAIL before >=8 weeks and >=400 eligible player-games;
- QB M89/M90 diagnostic: remaining error mass = 42.37% opportunity, 34.01% efficiency, 23.62% residual; no candidate authorized;
- specialist non-target finite-MC path drift is CONFIRMED; only downstream materiality-vs-resampling audit is authorized next, no RNG repair yet;
- Bayesian current-state transmission mismatch is confirmed, but the first full-stack Opportunity Authority Priority V1 candidate **FAILED CLOSED**;
- do not rescue Opportunity Authority V1 with RB-only / TE-only post-hoc variants or Bayes retuning;
- frozen RB Vacancy/Public Intent, Receiving Rule Semantics, Availability -> Opportunity, Discrete Count V1, RB Rush+Receiving V2, and zero-MC closure remain unchanged;
- no fresh OddsAPI spend without explicit user authorization.

### Exact next action

When Week-3 outcomes are complete, run a brutally objective postmortem of the frozen paid board, grade every frozen Week-3 prospective authority exactly as preregistered, and do not tune thresholds/carveouts from the same outcomes.

If outcomes are not complete, preserve the frozen state. The only directly authorized no-outcome systems follow-up is the specialist RNG/path downstream materiality audit.

### Memory-efficient start rule

Read only:
1. `AGENTS.md`
2. this top checkpoint
3. `docs/handoffs/NFL_HANDOFF_2026-09-27_WEEK3_LIVE_POSTMORTEM_RESEARCH_CURRENT.md`
4. Issue #535 from comment `5857262166` onward, especially `5857330835`, `5857337113`, `5857495417`, `5857596405`, `5860608905`, `5860655436`, and `5860668958`
5. live main / relevant branches / PRs / Actions

Then work. Do not recursively load older handoffs.

---

---
## ACTIVE CHECKPOINT — 2026-09-27 — WEEK-3 PAID BOARD CERTIFIED CURRENT

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-09-27_WEEK3_PAID_BOARD_CERTIFIED_CURRENT.md`

This supersedes all older checkpoints for immediate execution.

### Current state in one screen

- current main before this docs-only continuity update: `6a1de05842a03272e51269d35a7a01ca8dd1ab12`;
- canonical paid Week-3 Full Slate `36293274478` = **SUCCESS**;
- paid artifact `10923570170`, digest `sha256:5a3d4f64592c70553e66dd51bb3bff45263d2900f4d270e370353fa60ea1c480`;
- PR #650 Ertz live-prop quarantine = MERGED;
- PR #651 verified identity-registry alias repair = MERGED;
- exact paid-snapshot identity certification `36294114610` = SUCCESS with **0 blockers**;
- PR #652 Week-3 RB workbook/market-lineage semantics repair = MERGED;
- exact paid-snapshot lineage verifier `36321385122` = SUCCESS;
- current-main Repo CI `36321552702` = SUCCESS;
- current-main no-live Full Slate `36321552693` = SUCCESS;
- **no further OddsAPI spend is needed merely to inspect/certify the preserved Week-3 paid board**;
- paid board is mechanically generated and `FULL_SLATE_DATA_QUALITY_PASS_WITH_DECLARED_LIMITATIONS`;
- Week-3 RB rush yards correctly use generic calibrated ensemble, not Week-1 P3;
- Week-3 RB rush+rec correctly reports `RB_RUSH_REC_CONSERVATION_V2`;
- ATD remains execution-capable but **not dedicated-science certified**;
- frozen RB Vacancy/Public Intent, Receiving Rule Semantics, Availability -> Opportunity, zero-MC closure, and all do-not-rescue rules remain unchanged.

### Exact next action

Use the preserved successful paid artifact / corrected workbook for Week-3 board analysis. Do not buy the same odds snapshot again just to analyze it.

Before games, preserve all frozen prospective Week-3 labels/cells unchanged. After games, grade those frozen authorities exactly as specified in the detailed handoff.

### Memory-efficient start rule

Read only:
1. `AGENTS.md`
2. this top checkpoint
3. `docs/handoffs/NFL_HANDOFF_2026-09-27_WEEK3_PAID_BOARD_CERTIFIED_CURRENT.md`
4. Issue #535 from comment `5852547875` onward
5. live main / relevant PRs / Actions state

Then work. Do not recursively load older handoffs.

---

## ACTIVE CHECKPOINT — 2026-09-26 — SYSTEMS AUDIT CLOSED / DISCRETE COUNT V1 PROMOTED CURRENT

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-09-26_SYSTEMS_AUDIT_COUNT_PROMOTED_CURRENT.md`

This supersedes all older checkpoints for immediate execution.

### Current state in one screen

- production code main before this docs-only continuity commit: `d658a176d86654e3c153d09d2fcf1e69eca4aaec`;
- PR #645 Week-3 Full Slate repair = MERGED;
- PR #646 rush-att zero-MC lineage closure = MERGED;
- PR #647 Discrete Count Mean Alignment V1 promotion = MERGED;
- post-promotion Repo CI `36286648937` = SUCCESS;
- post-promotion no-live Full Slate `36286648933` = SUCCESS;
- **no paid `fetch_live_odds=true` run has been launched; explicit user authorization remains required**;
- rush-att zero-MC is CLOSED: all 7,137 frozen blocked rows first fail at `TOP5_EXCLUDED`; no RNG/lookup/allocation plumbing bug and no repair authorized;
- Discrete Count Mean Alignment V1 is production-active for receptions + rush_att, with zero/nonfinite-MC exact no-op and non-count invariance;
- RB Vacancy/Public Intent and Receiving Rule Semantics remain frozen prospective Week-3 lanes awaiting outcomes;
- Availability -> Opportunity rule-order gap remains confirmed; do not fit vacancy coefficients from Week-3;
- RB Rush+Receiving Conservation V2 remains production-active and locked.

### Exact next action

The only current operational betting-board gate is the controlled live-odds Full Slate, and it requires explicit user authorization because it spends OddsAPI credits.

Without that authorization, preserve the frozen Week-3 prospective science until games are final and do not invent rescue variants.

### Memory-efficient start rule

Read only:
1. `AGENTS.md`
2. this top checkpoint
3. `docs/handoffs/NFL_HANDOFF_2026-09-26_SYSTEMS_AUDIT_COUNT_PROMOTED_CURRENT.md`
4. Issue #535 from comment `5851689287` onward
5. live main / relevant PRs / Actions state

Then work. Do not recursively load older handoffs.

---


## ACTIVE CHECKPOINT — 2026-09-26 — WEEK-3 FULL SLATE REPAIRED / SYSTEMS-AUDIT LOCKS CURRENT

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-09-26_WEEK3_FULL_SLATE_REPAIRED_SYSTEMS_AUDIT_CURRENT.md`

**Compact next-chat prompt:**

`docs/handoffs/NEXT_CHAT_PROMPT_2026-09-26_WEEK3_FULL_SLATE_REPAIRED.md`

This supersedes all older checkpoints for immediate execution.

### Current state in one screen

- canonical main after Week-3 repair: `4c0e9e25f0957877820cade6897cf977d265533a` plus docs-only continuity commits;
- PR #645 is MERGED;
- canonical no-live-odds Full Slate run `36283114203` = SUCCESS;
- Repo CI `36283114216` = SUCCESS;
- the earlier 30-team/32-team suspicion is CLOSED;
- real Full Slate blocker was an incomplete Bayesian authority inside LAR injury target redistribution; repaired and exact failed-artifact replay is green;
- Barion Brown unrostered non-core sportsbook quarantine fix is included;
- next actual betting-board gate is a controlled `fetch_live_odds=true` Full Slate, which consumes OddsAPI credits;
- **do not launch paid live odds without explicit user authorization**;
- RB Rush+Receiving Conservation V2 remains production-active;
- RB Vacancy V1 + DEN/PIT public-intent labels remain frozen pregame;
- Receiving Rule Semantics Week-3 A0B0/A1B0/A0B1/A1B1 cells remain frozen prospective-only;
- Availability -> Opportunity rule-order gap is confirmed;
- Discrete Count Mean Alignment V1 is research-qualified for a separate integration test;
- rush-att zero-MC ensemble transmission remains an open architecture diagnostic; next step is allocation-lineage audit, not a guessed repair.

### Memory-efficient start rule

Read only:
1. `AGENTS.md`
2. this top checkpoint
3. `docs/handoffs/NFL_HANDOFF_2026-09-26_WEEK3_FULL_SLATE_REPAIRED_SYSTEMS_AUDIT_CURRENT.md`
4. Issue #535 comments from `5850416444` onward
5. live main / active branches / Actions state

Then work. Do not recursively load older handoffs.

---

## ACTIVE CHECKPOINT — 2026-09-25 — WR ROLE-TRANSMISSION CLOSED / RB VACANCY PROSPECTIVE NEXT

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-09-25_WR_ROLE_TRANSMISSION_CLOSED_CURRENT.md`

This supersedes older checkpoints for immediate execution.

### Current state in one screen

- production behavior unchanged
- Receiver Room Targets-Per-Play V1 2024-2025 confirmation: **FAILED CLOSED**
- Offensive Regime Boundary Room History V1 Stage A: **FAILED CLOSED**
- WR Anchor / Role-Transmission Audit V1: **`WR_ANCHOR_ROLE_TRANSMISSION_NO_GAP_CLOSED`**
- authoritative WR role-transmission run: `36203881629`
- artifact: `10893281689`
- digest: `sha256:33b12fd521196838054f4ee4240d8cea8f66c2fce5b5bfc9eb902bc67d730adb`
- on 530 anchor/participation mismatch team-games, strict-prior participation leader was **-25.47 pp** worse than M38 anchor at identifying the actual top-target WR and averaged **1.498 fewer actual targets**
- no WR hierarchy candidate is authorized; do not rescue with alternate participation thresholds/windows/blends
- next active science is the **already-frozen** `research-rb-vacancy-opportunity-v1` prospective lane
- Week-2 preserved pregame evidence contained zero qualifying definitive-unavailable RB/FB events; the RB vacancy hypothesis remains open awaiting a legitimate prospective event

### Exact next action

Continue `research-rb-vacancy-opportunity-v1` without redesign:

1. inspect the current pregame 2026 availability/slate authority;
2. if at least one RB/FB is canonical `definitive_unavailable == 1` with the frozen strict-prior evidence contract, materialize and lock the no-outcome vacancy state **before kickoff**;
3. if none qualify, record `NO_EVENT`;
4. do not broaden to DOUBTFUL/QUESTIONABLE, infer absence from postgame participation, reopen M96, alter YPC, or use sportsbook information upstream.

### Memory-efficient start rule

Read only:
1. `AGENTS.md`
2. this top checkpoint
3. `docs/handoffs/NFL_HANDOFF_2026-09-25_WR_ROLE_TRANSMISSION_CLOSED_CURRENT.md`
4. latest Issue #535 comments
5. live `main` and `research-rb-vacancy-opportunity-v1` state

Then work. Do not recursively load older handoffs.



---

## ACTIVE CHECKPOINT — 2026-09-25 — RECEIVER ROOM TARGETS-PER-PLAY FAILED CLOSED / WR REGIME DIAGNOSTIC NEXT

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-09-25_RECEIVER_ROOM_FRONTIER_CURRENT.md`

**Compact next-chat prompt:**

`docs/handoffs/NEXT_CHAT_PROMPT_2026-09-25_RECEIVER_ROOM_FRONTIER.md`

This supersedes all older checkpoints for immediate execution.

### Current state in one screen

- production behavior unchanged by recent receiver research
- docs-only continuity main before this final refresh: `67b9aa349258aff78cb0b588432eab91a3f1da78`
- Receiver Targetable-Dropback team-volume science replicated across 2022-2025, but uniform player translation FAILED CLOSED
- compensation audit showed the valid team-volume signal collides with WR-room / WR1 / Q4 underallocation
- Active-Roster Receiver Room State V1 is NOT SUPPORTED / CLOSED
- WR1 current-state has real absolute-share signal, but WR1-only fixed-room candidate is NOT justified
- Room Targetable-Rate/dropback-denominator V1 FAILED CLOSED
- Receiver Room Targets-Per-Play V1 passed 2022-2023 discovery but **FAILED CLOSED** unchanged 2024-2025 confirmation

### Authoritative failed confirmation

Branch:
`research-receiver-room-targets-per-play-v1-confirm-2024-2025`

Initial run `36173485673`:
- mechanical pre-science stop only
- wrong targets-per-dropback helper import
- no scorecard/result artifact

Bounded repair:
`c1b48b90364e2d6f82b66271065cc7b359d725ca`

Authoritative corrected run:
- run `36174077739`
- job `108200327879`
- artifact `10881825507`
- digest `sha256:3956a8ed9cc2191714787ecab235831b5f9ce39c59f3e08ba260ee2918f2de9c`
- result: `docs/research/RECEIVER_ROOM_TARGETS_PER_PLAY_V1_2024_2025_CONFIRMATION_RESULT.md`
- result commit: `fb5a9196d02a69e87dd6ec80418bd01865c3600a`

Disposition:
`RECEIVER_ROOM_TARGETS_PER_PLAY_V1_2024_2025_FAILED_CLOSED`

Pooled:
- WR MAE `4.535565 -> 4.582359` — worse
- TE `2.875251 -> 2.804996`
- RB_FB `2.346042 -> 2.173209`
- macro MAE `3.252286 -> 3.186854`
- macro p90 `6.646801 -> 6.651926` — worse
- macro abs bias `0.404433 -> 0.325998`
- summed-room MAE `6.127841 -> 5.983952`
- summed-room p90 `12.704084 -> 12.049116`

2024 decisive issue:
- WR MAE `4.561046 -> 4.724467` worsened

2025:
- WR MAE `4.510085 -> 4.440251` improved

Frozen failures:
1. 2024 WR room MAE did not improve
2. pooled WR room MAE did not improve
3. pooled macro p90 nonworse gate failed

All provenance/specialist/no-sportsbook/no-fit gates passed.

### Exact next action

Do **not** build a player/full-stack integration.

Run a **diagnostic-only WR room regime-instability audit**.

Goal:
explain why WR room targets-per-play was helpful in 2022/2023/2025 but harmful in 2024, while TE/RB_FB remain directionally stable.

Diagnostic may examine strictly pregame structural variables such as:
- prior-season -> current-season WR room share/rate drift
- top-target / WR-room roster turnover
- QB change
- coordinator/play-caller change only if a reliable leakage-safe source already exists
- speed of within-season adaptation
- whether 2024 misses are broad or transition-clustered

Candidate variants scored = **0**.

No rescue:
- no exclude-2024 route
- no WR-specific multiplier
- no recency/shrinkage/window search
- no fixed57 blend
- no bias offset
- no WR1/Q4 exemption
- no 2026 fit
- no sportsbook routing

If no strong structural explanation emerges, close this room-history family and move to another architecture frontier.

### Memory-efficient start rule

Read only:
1. `AGENTS.md`
2. this top checkpoint
3. `docs/handoffs/NFL_HANDOFF_2026-09-25_RECEIVER_ROOM_FRONTIER_CURRENT.md`
4. Issue #535 from `5834428368` onward, especially `5837631919`
5. live main / relevant research state

Then work. Do not recursively load older handoffs.

---

## ACTIVE CHECKPOINT — 2026-09-24 — RUSH POOL V1 INTEGRATION FAILED CLOSED / NEW SCIENCE NEXT

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-09-24_RUSH_POOL_V1_FAILED_CLOSED_CURRENT.md`

This supersedes the prior TE-Width/new-science checkpoint for immediate execution.

### Current state in one screen

- physical production main entering this docs-only closure: `b5b1816e7d11cadca412faf841fbfe5336798c2e`
- RB Rush+Receiving Conservation V2 remains production-active/protected
- Rush Pool Evidence Guard V1's allocator-only research signal qualified, but its exact production integration **FAILED CLOSED**
- authoritative integration run: `36049144898`
- exact integration head: `808072634603c03e6c86fd3c45ea311a39b13fa6`
- result artifact: `10830277740`
- digest: `sha256:b6a0715d38984fa91b8815561b4e72679684affcc142ee773a8b7428342412a5`
- mechanics/invariance all passed: Week-1 no-op, non-rushing bit identity, team-rush identity, target identity, raw-share identity, zero sportsbook inputs, RB-V2 pathwise identity
- science failed:
  - 2025 ALL rush-att MAE `1.249158 -> 1.253482`
  - 2025 RB rush-att MAE `3.442835 -> 3.456101`
  - 2025 QB rush-att MAE `1.726906 -> 1.733759`
  - 2025 ALL rush-yard MAE `7.845055 -> 7.855806`
  - QB rush-yard MAE worsened in both seasons
- positive but insufficient: RB rush-yard MAE and RB rush+receiving MAE/p90 improved in both seasons
- PR #633 is **CLOSED / NOT MERGED**
- Repo CI `36049328629` SUCCESS
- preserved paid-artifact Full Slate replay `36049328727` SUCCESS
- production behavior remains unchanged

### Permanent closure

Do not rescue Rush Pool Evidence Guard V1 with:
- RB/QB/OTHER carveouts
- a different evidence state
- another top-N
- share thresholds
- depth rank
- injury/rookie exceptions
- Bayesian retuning
- 2026 outcome fitting

### Exact next action

Move to a genuinely different structural/new-information hypothesis.

Start read-only from current production evidence. Prefer:
- current-season role/participation information that fails to reach final opportunity;
- stale player-level room hierarchy despite correct team opportunity;
- injury/personnel transitions that fail to propagate across markets;
- cross-market contradictions created after specialist layers;
- 2026 usage states that diverge from strict-prior hierarchy without target-game leakage.

Prospective RB Vacancy V1 remains separate and unchanged.

### Memory-efficient start rule

Read only:
1. `AGENTS.md`
2. this top checkpoint
3. `docs/handoffs/NFL_HANDOFF_2026-09-24_RUSH_POOL_V1_FAILED_CLOSED_CURRENT.md`
4. newest Issue #535 comments after the V1 closure
5. live main/current research branch state

Then work. Do not load older handoffs unless explicitly needed.

---

## ACTIVE CHECKPOINT — 2026-09-24 — TE WIDTH V2 FAILED CLOSED / NEW SCIENCE NEXT

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-09-24_TE_WIDTH_V2_FAILED_CLOSED_CURRENT.md`

This supersedes the prior in-progress TE Width V2 checkpoint for immediate execution.

### Current state in one screen

- production main entering this docs-only checkpoint: `cd6cc06459fc88f2bc6d7d2c23d544ca75e9e12f`
- RB Rush+Receiving Conservation V2 remains production-active and protected
- TE-R5P Receiving-Yards Width V2 run `36040911515` completed **SUCCESS**
- exact research head: `a9c7a93ae375f75e926139a0a29675aab89d2fdc`
- result artifact `10827222034`, digest `sha256:f128cdf5b5d41c18ff1a08f1a9c23f346f911294d6e8e6afe678c8c64f526cc2`
- disposition: `TE_R5P_REC_YARDS_WIDTH_V2_FAILED_CLOSED`
- fit2024 -> test2025 CRPS worsened `11.5174 -> 11.6626`
- that direction's 80% coverage gap worsened `0.08495 -> 0.08780`
- 90% coverage improved both directions; pooled Brier/log loss improved; means stayed invariant
- frozen contract required CRPS and 80% coverage-gap improvement in both directions, so exact candidate is CLOSED
- no k search, cap rescue, subgroup rescue, global TE SD multiplier, sportsbook-conditioned width, 2026 outcome fitting, or production integration

### Exact next action

Move immediately to genuinely new Week-3 model-improvement science.

Start with a read-only structural contradiction audit of preserved/current Full Slate artifacts:
- conservation identities
- incompatible final component projections
- lost opportunity mass
- specialist overrides that break joint football identities
- current usage/hierarchy states not represented by historical priors
- injury/personnel changes that fail to propagate

Do not revive failed C1/C3 pass-receiving formulations or generic QB mean tuning. Preserve prospective RB Vacancy V1 separately.

### Memory-efficient start rule

Read only:
1. `AGENTS.md`
2. this top checkpoint
3. `docs/handoffs/NFL_HANDOFF_2026-09-24_TE_WIDTH_V2_FAILED_CLOSED_CURRENT.md`
4. newest Issue #535 comments after this disposition
5. live main / relevant active research state

Then work.

Do not load older handoffs unless this checkpoint explicitly sends you there.

---

## ACTIVE CHECKPOINT — 2026-09-24 — POST-RB-V2 / TE WIDTH V2 CURRENT

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-09-24_POST_RB_V2_TE_WIDTH_V2_CURRENT.md`

This is the newest checkpoint and supersedes the older RB-V2-open / TE-provider-drift language below for immediate execution.

### Current state in one screen

- current `main`: `28fd7b4b9a4e4f70a45787d4e09d34939d62e775`
- PR #628: **MERGED/CLOSED** — RB Rush+Receiving Conservation V2 is production-active
- PR #629: **MERGED/CLOSED** — continuity-only post-merge update
- Full Slate public pricing entrypoint on `main` routes to V6 and therefore consumes RB V2 for non-Week-1 RB `rush_rec_yards`
- RB V2 historical MAE: **27.6853 -> 25.5718**
- Week-2 observational MAE: **30.3620 -> 28.9014**
- do not reopen RB V2 absent a real defect/new evidence

### Active science now

TE-R5P Receiving-Yards Width V2 is active on:

`research-te-live-entitlement-efficiency-v1`

Current branch head:
`a9c7a93ae375f75e926139a0a29675aab89d2fdc`

Frozen validation run:
`36040911515`

At checkpoint creation it is **IN PROGRESS**.

Important: PR #549 provider/parity archaeology is **solved**. Run `35954272152` reproduced:
- exact 51,197-row baseline
- zero frozen-row `mc_proj` drift
- exact specialist trace parity

The remaining failure was a bounded validator schema defect:
`KeyError: 'game_id'`.

Mechanical repair only:
- `06bed4ddcad43f014b542fa1f3a4806ba30ccf0c`
- `a9c7a93ae375f75e926139a0a29675aab89d2fdc`

No k/gate/cohort/mean/parity/sportsbook-role change.

### Exact next action

1. Inspect run `36040911515`; do not duplicate it.
2. If science completes, obey the frozen Width V2 gates exactly.
3. If qualified, freeze a separate production-integration plan before any production edit.
4. If scientifically failed, close exact Width V2 — no rescue/search.
5. If another failure is not a bounded mechanical defect, park it and move to genuinely-new-information research.
6. Keep prospective RB Vacancy V1 intact; Week 2 had no qualifying vacancy event, so only pregame prospective 2026 capture is legitimate.

### Memory-efficient start rule

Read only:
1. `AGENTS.md`
2. this top checkpoint
3. `docs/handoffs/NFL_HANDOFF_2026-09-24_POST_RB_V2_TE_WIDTH_V2_CURRENT.md`
4. Issue #535 from comment `5815680099` onward
5. live `main`, active TE branch, and run `36040911515`

Then work.

Do not load older handoffs unless the current handoff explicitly sends you there.

---

## ACTIVE CHECKPOINT — 2026-09-24 — RB RUSH+RECEIVING V2 PRODUCTION CERTIFIED

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-09-24_RB_RUSH_REC_V2_PRODUCTION_CURRENT.md`

This is the newest model-improvement checkpoint and supersedes the TE Width V2
checkpoint as the immediate operational priority. The TE provider-drift lane
remains preserved below as a separate unresolved research lane.

### Current state in one screen

- A concrete non-Week-1 RB rush+receiving construction improvement has qualified.
- Historical no-fit test: 2,787 RB player-games, 2024-2025.
- MAE **27.6853 -> 25.5718** (+2.1135 yd).
- RMSE **39.8437 -> 36.4030**.
- p90 AE **64.7742 -> 57.2604**.
- 30+ yd misses **841 -> 768**.
- Replicated independently in 2024 and 2025.
- Exact Week-2 production A/B preserved all non-combo outputs at gap 0.0.
- Week-2 observational MAE **30.3620 -> 28.9014** (+1.4606 yd), supportive only.
- Final production scope is **RB only**. FB is explicit no-op because historical
  qualification was RB-only.
- Final stable-entrypoint certification run `36009823313` = **SUCCESS**.
- Final artifact `10811878970`, digest
  `sha256:c88751876bd42838f312f385377be439e12e46b07e96c0c1d34dc3fd8039a04f`.
- **PR #628 is MERGED/CLOSED** at main SHA `e26fcedade9a94a9634f6ba74558775968573818`.
- Repo CI `36010438820` = SUCCESS.
- Preserved paid Full Slate replay `36010438890` = SUCCESS, including certified pricing, final-board quarantine, and strict audits.
- Stable public pricing entrypoint on `main` now points to V6.
- Canonical Full Slate live pricing now has a fail-closed V2 lineage gate.
- Production promotion is complete. Do not reopen/retest V2 absent a concrete defect or genuinely new prospective evidence.
- Next priority returns to additional model-improvement/new-information research.
- Public-intent V1B automation is parked as a retrieval-transport failure.
- Historical practice-trajectory source is parked as not dense.
- M96 retrospective RB rushing remains CLOSED.
- QB pass yards remains frozen/prospective.
- No paid OddsAPI pull without explicit approval.

### Memory-efficient start rule

Read only:
1. `AGENTS.md`
2. this top checkpoint
3. `docs/handoffs/NFL_HANDOFF_2026-09-24_RB_RUSH_REC_V2_PRODUCTION_CURRENT.md`
4. newest Issue #535 comments after the V2 checkpoint
5. live PR/branch/check state

Then work.

---

## ACTIVE CHECKPOINT — 2026-09-24 — TE WIDTH V2 PROVIDER-DRIFT CHECKPOINT

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-09-24_TE_WIDTH_V2_PROVIDER_DRIFT_CURRENT.md`

This supersedes all stale PR-626-open and earlier TE-V2 status language below.

### Current state in one screen

- PR #626 is **MERGED/CLOSED**. Canonical W1/W2 record:
  **805 selected / 797 decided / 395-402 / -42.86u**.
- Scoreboard cleanup is DONE. Reopen only for a defect that materially changes
  canonical wagers/settlement or breaks the canonical replay.
- TE Live Entitlement vs Efficiency V1 is **COMPLETE / SUCCESS**:
  - run `35939588747`
  - artifact `10784323345`
  - current-snap counterfactual improves all-TE W2 target-share MAE **2.32%**
  - selected-TE target-share MAE improves **2.84%**
  - but rec-yard MAE does not improve from entitlement alone
  - perfect efficiency recovers **5.68 yd** vs perfect entitlement **4.11 yd**
  - disposition:
    `CURRENT_SNAP_ENTITLEMENT_SIGNAL_SUPPORTED_EFFICIENCY_REMAINS_PRIMARY_LIVE_GAP`
- PR #627's W3+ strict-prior 2026 snap continuation is therefore supported.
  Do not duplicate/refit it.
- TE-R5P Receiving-Yards Width V2 is the active distribution test on branch
  `research-te-live-entitlement-efficiency-v1`.
- Physical branch head at checkpoint:
  `839f7d206d47709ab8ed5576908dae205057be24`.
- V2 run #1 `35940487155`: failed before science output due provider-history
  row-universe drift.
- V2 run #2 `35944049950`: superseded diagnostic run.
- V2 run #3 `35944066618` / job `107458022324`: **FAILED CLOSED before
  science output**. Current rebuild has 51,232 rows vs frozen PR #549 51,197;
  35 extras are excluded, but frozen-row `mc_proj` still drifts by up to
  **14.0206492813 yd**.
- Therefore the problem is reproducing the exact frozen PR #549 distribution
  authority after historical provider/source drift, not the Width V2 candidate.
  Do not change V2 gates or treat this as a scientific failure.
- Preserved PR #549 compact artifact `10307242156` remains available until
  2026-09-26 and is precious; original raw-draw artifact `10306649017` is
  expired.
- Exact next action is narrowly diagnose frozen-row `mc_proj` drift / recover
  exact PR #549 draws if another surviving copy exists. Do not rerun unchanged.
- Time-box artifact archaeology. If exact replay becomes another open-ended
  infrastructure loop, document/park it and pivot to the sanctioned RB
  teammate-availability / injury-created-vacancy rushing-opportunity lane.
- QB pass yards stays frozen/prospective.
- No global SD rescale.
- No sportsbook lines upstream.
- No paid OddsAPI pull without explicit approval.

### Memory-efficient start rule

A new chat should NOT ingest the entire history. Read only:

1. `AGENTS.md`
2. this top checkpoint
3. `docs/handoffs/NFL_HANDOFF_2026-09-24_TE_WIDTH_V2_PROVIDER_DRIFT_CURRENT.md`
4. Issue #535 comments from `5805281169` onward
5. live branch/run state

Then work. Older handoffs are reference-only.

Every substantive hypothesis/test/result/disposition must be written to GitHub
and Issue #535.

---

## STANDING PROHIBITION — RETROSPECTIVE RB RUSHING RESEARCH IS CLOSED

**This is an operative rule, not a history note. Read it before proposing any RB rushing work.**

The M96 chain ran the RB opportunity/efficiency program to a pre-committed
terminal stop. Its artifacts live on unmerged research branches, so this
prohibition was previously invisible to any session working from `main` — which
is exactly how it came to be violated in Issue #535 (proposal `5754250023`,
retracted in `5754278269`). Recorded here so that cannot recur.

### Terminal disposition

`M96E_FINAL_RETROSPECTIVE_ROUTER_FAILED_STOP` / `AUTONOMOUS_RB_RESEARCH_STOP`
(run `33467630395`, job `99730679349`).

Eight of nine frozen retention checks passed. The only failure was the
predeclared materiality requirement:

- required all-RB rushing-yard MAE gain: **>= `0.150000` yards**
- observed gain: **`0.141791` yards**
- **shortfall: `0.008209` yards**

Read those carefully: `0.141791` is the gain the candidate *achieved*, not the
amount it missed by. The chain was closed by a shortfall of eight thousandths
of a yard, with a working mechanism and eight of nine gates passed -- not by a
failed or broken one.

### The continuation rule — verbatim intent

> Any further retrospective router threshold/feature variants would reuse
> exposed 2025 outcomes and risk overfitting. New RB architecture evidence must
> now come from genuinely prospective/untouched 2026 games or a separately
> justified new-data source that does not retune against the exposed historical
> outcomes.

Exactly two continuations are sanctioned:

1. **genuinely prospective / untouched 2026 evidence** — this is what the RB PD2
   forward/shadow confirmation lane produces. That lane is a sanctioned
   continuation of RB science, not incidental plumbing.
2. **a separately justified new-data source** that does not retune against
   exposed historical outcomes.

Anything else — another router variant, threshold search, feature hunt or
re-decomposition against the exposed 2025 sample — is overfitting, and is
forbidden regardless of how it is framed.

### M96A standing attribution result — do not re-derive this

M96A already performed the opportunity-vs-efficiency attribution
(run `33459376333`, job `99706110345`, artifact `9782611047`,
branch `research-rb-m96a-opportunity-efficiency-attribution`), n = 1,393
RB/FB player-games, 2025:

| Quantity | Value |
|---|---:|
| pregame M94C rush-yard MAE | **21.0312** |
| perfect actual carries, frozen efficiency | **13.3535** |
| opportunity MAE recovery | **7.6777** |
| perfect game efficiency, frozen carries | **14.3055** |
| efficiency MAE recovery | **6.7256** |
| opportunity-dominant share of games | **59.73%** |
| efficiency-dominant share of games | **40.27%** |

Routed **JOINT**: opportunity cleared the component-share gate but missed the
>= 1.0-yard recovery-margin gate by `0.048` yards.

**RB rushing yards is therefore not irreducible** — roughly seven yards of MAE
is recoverable from each factor. The binding question is *which* factor, and
that flips by workload regime:

| Actual carries | Pregame MAE | Perfect carries | Perfect efficiency | Opportunity recovery | Efficiency recovery |
|---|---:|---:|---:|---:|---:|
| 0–5 | 13.288 | 5.245 | 10.027 | **8.043** | 3.261 |
| 6–10 | 21.191 | 13.561 | 13.188 | 7.630 | **8.002** |
| 11–14 | 25.812 | 19.270 | 15.208 | 6.542 | **10.605** |
| 15–19 | 29.764 | 23.749 | 16.989 | 6.015 | **12.775** |
| 20+ | 40.005 | 28.636 | 36.409 | **11.369** | 3.596 |
| 25+ | 49.310 | 37.549 | 54.390 | **11.762** | −5.079 |

Low-volume (0–5) and high-volume (20+, 25+) games are **opportunity** problems.
The 11–19 middle is an **efficiency** problem. Any future new-data justification
should cite this table rather than re-running the attribution.

### Chain lineage and where the artifacts live

| Migration | Disposition |
|---|---|
| M96A — opportunity vs efficiency attribution | `JOINT_ADVANCE_M96B_SEPARATE_WORKLOAD_AND_EFFICIENCY_DISTRIBUTIONS` |
| M96B — modular joint workload × efficiency synthesis | `M96B_MODULAR_SYNTHESIS_COMPLETE`; M95C residual not plug-compatible with M94C |
| M96C — M94C-anchored efficiency residual | `M96C_NO_GLOBAL_WINNER_CONDITIONAL_EFFICIENCY_SIGNAL_SUPPORTED` |
| M96D — pregame conditional efficiency routing | `M96D_PRIMARY_ROUTER_FAILED` |
| M96E — role router with frozen workload-risk guard | `M96E_FINAL_RETROSPECTIVE_ROUTER_FAILED_STOP` |

Full plans and results are preserved on the research branches
`research-current-state` and `research-rb-final-qualification` under
`docs/migrations/M96A_*` through `M96E_*`, with evaluators under
`scripts/backtest/evaluate_rb_m96*`. **Do not merge those branches into `main`
to read them** — use `git show origin/research-current-state:<path>`.

---


## ACTIVE CHECKPOINT — 2026-09-23 — PR #626 INTEGRITY REPAIR + LIVE MODEL DIAGNOSIS

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-09-23_PR626_INTEGRITY_AND_MODEL_DIAGNOSIS_CURRENT.md`

This is the current operating checkpoint and supersedes the 2026-09-22 active
checkpoint below where they conflict.

### Current state in one screen

- PR #625: merged/finished. Do not reopen.
- PR #627: merged/finished. Do not duplicate.
- PR #626: **OPEN / NOT MERGE-READY**.
- Physical GitHub PR #626 head at handoff:
  `a8973fc2e398a7b842736dd0481857391d387829`.
- Six unresolved review threads remain, representing four unique blockers:
  1. all-PASS / zero-selected-bet crash (three duplicate P2 threads);
  2. P1 historical replay depends on the moving current NFL.com injury page;
  3. P1 board-provenance verifier is self-referential to an editable manifest;
  4. P2 final-board quarantine is not applied to the current production record.
- Several Codex summaries claim local fixes/commits that never advanced the
  physical GitHub PR head. They are **not canonical**. Verify remote head after
  every write.
- Latest exact-head checks on `a8973fc2...`:
  - Repo CI `35824360282` SUCCESS;
  - W1/W2 full-board/provenance `35824360270` SUCCESS;
  - preserved no-paid Full Slate replay `35824360266` FAILURE because a
    historical Week-2 replay fetched the now-current Week-3 NFL.com injury page;
  - earlier same-SHA replay `35816381724` succeeded before the source rolled.
- Claude's original W1/W2 full-board acquisition/backtest was **not duplicated**.
  GPT-5.6's later work repaired grading/decision/settlement/inference integrity
  on the same preserved boards.
- Current branch scorecard is 810 selected settlement rows / 802 decided bets /
  397-405 / -44.09u, but exact W/L/units remain provisional until #626 closes.
- Current live scientific leads remain:
  - QB pass yards encouraging;
  - TE the clearest positional weakness;
  - probability/distribution layer overconfident/too narrow;
  - rush+receiving yards has a major low-bias construction concern.
- Current-Season State Persistence V1 remains valid: role/opportunity state,
  especially RB rush share and WR/TE target share, updates quickly; early
  efficiency such as RB YPC requires heavier shrinkage.
- After #626 closes, pivot immediately to **TE receiving-yards
  entitlement-vs-efficiency**, with RB teammate-vacancy opportunity propagation
  as the strongest sanctioned parallel mean-information lane.
- No paid OddsAPI pull without explicit user approval.

### Start-of-chat read order

1. `AGENTS.md`
2. `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. `docs/handoffs/NFL_HANDOFF_2026-09-23_PR626_INTEGRITY_AND_MODEL_DIAGNOSIS_CURRENT.md`
4. newest Issue #535 comments after the 2026-09-23 handoff checkpoint
5. live PR #626 head/checks/reviews/unresolved threads

GitHub is canonical over chat memory. Do not trust local/container-only Codex
commit summaries unless the remote PR head actually contains them.

---

## ACTIVE CHECKPOINT — 2026-09-22 — CURRENT-SEASON STATE + LIVE SCOREBOARD

**Read this detailed handoff next:**

`docs/handoffs/NFL_HANDOFF_2026-09-22_CURRENT_SEASON_STATE_AND_LIVE_SCOREBOARD_CURRENT.md`

This is the current operating checkpoint and supersedes older active-priority
language below where the two conflict.

### Current state in one screen

- PR #625 is **MERGED/CLOSED**. Do not reopen it. RB PD2 is prospective
  distribution calibration, not an RB mean fix.
- Current-Season State Persistence V1 is **COMPLETE**:
  - run `35741758765` SUCCESS;
  - artifact `10699781744`;
  - 41,745 rows;
  - 9/10 production-aligned metrics improved on 2025 replication;
  - strongest live signals are RB rush share, WR target share and TE target share;
  - RB YPC is the key non-replicating early-state metric.
- PR #627 is **MERGED** at `5ecd4da5a7ba7ab314f11ae9e8cff9ef7f6308fc`.
  WR-R15/TE-R5P may consume 2026 strict-prior snaps beginning Week 3 only after
  a schedule-aware, bye-aware immediately-prior-week freshness gate passes.
  No model refit occurred.
- Claude completed the Week-1/Week-2 full-board backtest in PR #626 and preserved
  the raw boards + 866-row graded log. Canonical run `35740684289`, artifact
  `10699408992`.
- **PR #626 is NOT merge-ready.** Four Codex review threads remain, including a
  P1 grading-line/side inconsistency that can change exact W/L/units. Repair and
  regenerate/reconfirm the scorecard before canonizing the current 438-428
  headline.
- Claude's major diagnosis is still important: probability overconfidence /
  too-narrow simulated distributions, TE weakness, a rush+receiving-yard
  construction concern, systemic low bias, and encouraging early QB pass-yards
  performance.
- Next model-improvement selection should use the intersection of:
  **live weakness + historical state persistence + missing/newly restored
  pregame state + deployable data + no closed-family retest**.
- Likely first high-value diagnostic after #626 repair: TE receiving-yards
  entitlement-vs-efficiency after Week-3 current-snap activation.
- RB injury-created vacancy / teammate opportunity propagation remains a strong
  sanctioned parallel mean-information lead.
- No paid OddsAPI pull without explicit user approval.

### Start-of-chat read order

1. `AGENTS.md`
2. `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. `docs/handoffs/NFL_HANDOFF_2026-09-22_CURRENT_SEASON_STATE_AND_LIVE_SCOREBOARD_CURRENT.md`
4. latest Issue #535 comments after `5779220450`
5. live PR #626 head/review threads/checks

Verify current GitHub state before mutation. GitHub is canonical over chat memory.

---

## ACTIVE CHECKPOINT — 2026-09-22 — RB PD2 FORWARD LOCK CLOSED; PR #625 MERGED

**Current detailed handoff:**

`docs/handoffs/NFL_HANDOFF_2026-09-21_RB_PD2_FORWARD_LOCK_AND_MODEL_IMPROVEMENT_CURRENT.md`

Read that file for the implementation lineage, but this root checkpoint supersedes
its stale pre-Week-2 status.

### Final validated PR #625 state

PR #625 merged successfully into `main` at:

`2a8b4e5622df3917c672212b16a8189c2bab7921`

Production `main` immediately before that merge was:

`76e5e0452a88018b95ebc212fb0d1b21a2ca90a3`

Active branch:

`research-rb-pd2-forward-shadow-confirmation-v1`

Final **code-bearing** head before this handoff-only documentation update:

`c58860b0dee02d18a93e870d3541e8212987ae31`

PR:

`#625 — RB PD2 forward-shadow implementation validation — MERGED`

The previously open Week-2 history-freshness blocker is **closed**. The real
Week-2 outcomes were certified and the forward-history state was rebuilt and
re-pinned before any Week-3 lock can count.

### Final Week-2 certification evidence

Completed-history certification:

- run `35732450688` = **SUCCESS**
- artifact `10695992072`
- artifact name `rb-pd2-completed-2026-history-v2-week`
- digest `sha256:a84aa005dfdd35d16f891c40c1cafbad5c2779aca3dad16214624d0216e052bb`
- 108 / 108 Week-2 RB/FB football-projection rows received verified outcomes
- 93 rows from the nflverse weekly stats table
- 15 roster-confirmed verified zeroes
- 0 exclusions
- cumulative certified history = 215 rows across Weeks `[1,2]`
- sportsbook inputs used for football projection = 0

The final implementation uses the authoritative nflverse weekly player-stats
table plus weekly roster identity/verified-zero evidence only. A temporary PBP
fallback explored during debugging was removed before the final certification.
The shared GSIS suffix resolver was corrected to handle `III` / `II` / `IV`
longest-token-first, which allowed suffix variants such as Kenneth Walker III
to resolve through the normal stats path.

### Final forward-history evidence

Forward-history rebuild:

- run `35732782075` = **SUCCESS**
- artifact `10696312599`
- artifact name `rb-pd2-forward-history-v1`
- digest `sha256:1205ea152b4702e19d12379c79e29ff7fc20fd2a632a61f7c519980f64b648fb`
- 1,608 state rows
- 169 players
- 889 scoreable rows
- manifest `completed_2026_through_week = 2`
- manifest completed weeks = `[1,2]`
- zero prospective outcomes used

Full Slate now pins that exact immutable Week1+2 forward-history artifact.

### Prospective lock-integrity closure

The second late P1 is also closed. Lock assembly now:

1. writes the NPZ/JSONL lock files first;
2. records the actual persisted timestamp;
3. requires every persisted lock to retain a frozen **15-minute pre-kickoff
   upload buffer**;
4. clears the session lock and fails closed if that post-persistence gate fails.

Regression coverage proves that exactly 15 minutes before kickoff passes and
1 ms later fails.

### Exact-head validation at the final code-bearing head

At `c58860b0dee02d18a93e870d3541e8212987ae31`:

- Repo CI run `35733488365` = **SUCCESS**
- preserved paid Full Slate replay run `35733488509` = **SUCCESS**
- PR was clean/mergeable
- both prior P1 review threads were resolved
- no paid/live OddsAPI pull was triggered

A final Codex exact-head review found only one P2 documentation defect: this
root handoff still described the now-completed Week-2 blocker. This section is
the fix for that documentation defect; it does not change model science or the
validated runtime behavior above.

### What PD2 is — and what it is not

PD2 is a mean-neutral **distribution-calibration** mechanism.

Historical qualification improved pooled CRPS by +1.218% and high-difficulty
CRPS by +2.624%, with better coverage/tail calibration, but **point MAE is
unchanged by design**.

It does not fix RB carries, YPC, or rushing-yard mean accuracy.

The user explicitly wants this implementation finished, then wants the project
back on actual model-metric improvement.

### Authorized next step after PR #625 merge

The merge is complete. Do **not** reopen PR #625 or repeat its Week-2 certification.

1. preserve the final PR/artifact lineage in Issue #535 / the canonical paper trail;
2. return immediately to actual model performance / new-information research;
3. use the real Week-2 slate to build the authority-exact position/market
   scoreboard (including the downstream deduplicated betting hit-rate view);
4. prioritize sanctioned new pregame information rather than reopening exposed
   retrospective RB router variants.

The strongest already-identified RB mean-information lead remains **backfield
teammate availability / injury-created vacancy propagation into rushing
opportunity**, with prior-week snap share as a second live/free lagged
opportunity proxy.

Claude is independently running the Week-1/Week-2 full-board backtest. Do not duplicate that lane; GPT-5.6 continues the current-season state/new-information lane in parallel.

No paid OddsAPI pull is authorized without explicit user approval.

---


## ACTIVE CHECKPOINT — 2026-09-22 — CURRENT-SEASON STATE PERSISTENCE V1 COMPLETE

Branch:

`research-current-season-state-persistence-v1`

Frozen plan:

`docs/research/CURRENT_SEASON_STATE_PERSISTENCE_V1_PLAN.md`

Canonical run:

- run `35741758765` = **SUCCESS**
- head `28d79153f723765b01a447c2023b47108b14138c`
- artifact `10699781744` / `current-season-state-persistence-v1`
- digest `sha256:2d81bcb5222136ae812575b15a0d79952d27a03e68034cd312aa0f927503868a`
- frozen tests 4/4 PASS
- strict repo audit PASS
- row-level panel: 41,745 rows
- sportsbook inputs: 0
- 2026 outcomes used: 0
- production changes: 0

### Main scientific result

The user's live-season premise is supported. Current-season football state
contains substantial next-game information, but the useful signal is strongest
in **opportunity / role**, not raw early-season efficiency.

On untouched 2025 replication, the existing historical + current-season
PlayerForm four-game pseudo-prior blend improved over prior-season-only for
9 of 10 production-aligned metrics.

Most important replicated signals:

- RB rush-share MAE: `0.1486 -> 0.1166` (**21.53% improvement**), state-delta
  Spearman `+0.6256`;
- TE target-share MAE: `0.05153 -> 0.04553` (**11.66% improvement**);
- WR target-share MAE: `0.06773 -> 0.06144` (**9.30% improvement**);
- QB YPA full-season blend MAE: `1.7386 -> 1.6548` (**4.82% improvement**).

The lone non-replicating metric is RB YPC: `1.9135 -> 1.9160`. Do not chase
early-season RB efficiency noise.

### Exact Week-3 analogue: two completed current-season games

On 2025 rows with exactly two prior current-season games:

- RB rush share: `0.1288 prior -> 0.1113 blend`;
- WR target share: `0.05933 -> 0.05451`;
- TE target share: `0.03941 -> 0.03428`;
- QB YPA: prior-only beats the early blend (`1.7383 vs 1.7983`);
- RB YPC: prior-only beats the early blend (`2.0697 vs 2.1239`);
- WR/TE catch rate and raw current-only efficiency remain noisy.

Operational interpretation:

> historical prior + faster current opportunity/role updating + heavier
> efficiency shrinkage.

### Newly confirmed live 2026 source gap

The maintained nflverse snap source already has complete 2026 Weeks 1-2:

- Week 1: 1,492 rows, all 32 teams, complete offense snaps / offense pct;
- Week 2: 1,502 rows, all 32 teams, complete offense snaps / offense pct.

But WR-R15 and TE-R5P share a production snap loader hardcoded to source
seasons `2020-2025`. Therefore their strict-prior entitlement layers currently
do **not** consume available 2026 snap participation.

Disposition:

`CURRENT_2026_SNAP_SOURCE_AVAILABLE_NOT_CONSUMED`

Authorized next research action:

`WR_TE_2026_SNAP_SOURCE_CONTINUATION_V1`

This must be a frozen source-continuation/parity test only: no coefficient
refit, no sportsbook input, no production change. It must prove Week 1
invariance, strict-prior Week 2+ semantics, and quantify football-projection
impact before any production edit.

Claude is independently handling the Week-1/Week-2 full-board model/betting
grade. Do not duplicate that work. Use his result together with this state
audit to prioritize the next position/market improvement lane.

---

## ACTIVE RESEARCH CHECKPOINT — 2026-09-14

The user's explicit current priority is **WR receiving yards**. Do not skip ahead to RB until the user changes priority.

Read in this order:

1. `AGENTS.md`
2. `docs/handoffs/NFL_HANDOFF_2026-09-14_WR_RECEIVING_YARDS_CURRENT.md`
3. `NFL_MASTER_CONTINUITY_RECORD.md`
4. latest GPT-5.6 / Claude checkpoints in GitHub Issue #535

The active handoff contains the exact WR authority lineage, PR #600 benchmark state, WR1 identity correction, receiving-yard decomposition, closed QB-C2 shared-tail result, Claude independent artifact lineage, anti-retest rules, the next WR research objective, and the parked RB Weeks-2-18 checkpoint.

### Current WR interpretation

- M38 + `WR_R15_PRODUCTION_MODEL_V1` remain valid production authorities and improved football accuracy in their frozen OOS tests.
- WR receptions/opportunity are comparatively healthy.
- WR receiving-yard translation/efficiency is the active weakness, especially high-efficiency/right-tail games.
- The exact QB-C2 -> WR1 shared-tail selector has been independently tested by GPT-5.6 and Claude and is CLOSED. Do not rescue it with new percentiles/thresholds.
- Before any new WR candidate, perform an anti-retest + feature-availability audit and freeze one genuinely new leakage-safe hypothesis.
- GPT-5.6 and Claude must continue collaborating through Issue #535 and independently challenge each other's design/results.

### Open authority-exact benchmark

PR #600 remains open/mergeable at handoff creation:

- head `1167f9fdadde452deb84d3891097865da2f163d5`
- canonical run `34843204550`
- artifact `10346639168`

Use it as diagnostic evidence; do not train football projections against market lines.

## ACTIVE DRAFT PR #562 — RB-PD2 YARD-DIFFICULTY MC-WIDTH V1 — QUALIFIED 2026-09-16

Branch `research-rb-pd2-yard-difficulty-mc-width-v1`. The historical MC-distribution
checksum failure (root cause: comparing against a cross-run M95Q artifact, not
nflverse drift or a code bug) was fixed at `780087b1` -- `rebuild-distributions`
now builds its own in-job `walk_forward.py` reference. See Issue #535 comment
`5689731492`.

First end-to-end run (`35037087309`) passed every science gate but failed one
stale integrity gate (`A_parent_panel_matches_556`, an exact-row-count check
against PR #556's original frozen artifact). GPT-5.6 reviewed on Issue #535 and
approved four fixes (comment landed 2026-09-16): implement Amendment 3's crossed
player x game CRPS bootstrap, add fail-closed fresh-parent VALUE parity (not just
identity), fix mislabeled composite source provenance, and demote the old count
gate to a non-fatal disclosure. Implemented exactly as specified at `b9ad5247`,
reran once (`35039152022`, preserving `35037087309` unchanged in the paper trail).

**Final disposition: `RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED`.** All 28 gates
pass. Pooled CRPS +1.218%, high-difficulty-quartile CRPS +2.624%, both
player-clustered and crossed player x game bootstraps `p=1.0`, point-MAE
identical (mean-neutral), coverage and Brier-100 both improve, 4/4-season
robustness. Reported in full on Issue #535 comment `5690193256`.

Research qualification only -- per the frozen plan, a separate forward/shadow
confirmation is still required before any production change. None has been
started; production is untouched.

## PRODUCTION CHECKPOINT

The Week-1 Full Slate incident is already repaired/green. Do not reopen paid-live debugging for WR research.

Production handoff:

`docs/handoffs/NFL_HANDOFF_2026-09-11_FULL_SLATE_LIVE_REPAIR_MERGED_CURRENT.md`

No production-science change is authorized by the current WR diagnostic work itself.

Older research ledgers previously carried in this root file remain available in Git history and `NFL_MASTER_CONTINUITY_RECORD.md`; this root file is intentionally kept as a concise pointer to the newest canonical handoff.