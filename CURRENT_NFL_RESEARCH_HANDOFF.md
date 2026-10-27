# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## TOP CHECKPOINT — 2026-10-27 — V9 — TRANSMISSION V2 CLEARED / NAMED-PLAYER ROLE-ROUTING TRACE AUTHORIZED / WALKER NOT TO BE REDONE / NO PREDICTIVE CLEARANCE

**Model:** PRISM — Player Role & Individual Situation Model

### WHAT THE NEXT SESSION SHOULD DO FIRST

Read newest #535 **after comment 6441088661**.

The source-wide opponent/environment transmission corrections at research target **`8ed1d1778d388d54e6de9ed37ffc62d2d300e2ea` are CLEARED**. Walker's Week-7 exact case remains CLEARED and must **not** be redone.

The prior disagreement about named-player work is settled: **run/audit the compact JSN/Kupp/Barner named-player role-routing trace** because canonical role assignment is itself an untraced player-first mechanism step.

If Claude has already posted a target after 6441088661, the next GPT's first job is to **audit that exact target/code/artifacts**, not merely the summary, then post the ruling to #535.

Deep handoff:
`docs/handoffs/NFL_HANDOFF_2026-10-27_V9_SEAMLESS_CONTINUITY_NAMED_PLAYER_TRACE_AUTHORIZED.md`

Paste-ready restart prompt:
`docs/handoffs/NEXT_CHAT_PROMPT_2026-10-27_V9_NAMED_PLAYER_ROLE_ROUTING.md`

Displaced V8 checkpoint archived at:
`docs/handoffs/NFL_HANDOFF_2026-10-27_V9_ARCHIVED_TOP_V8_WALKER_GENERALIZATION_CORRECTED.md`

Do not recursively read older archived handoffs unless a specific lineage question requires one.

### ABSOLUTE ASSIGNMENT

PRISM is an **INDIVIDUAL-PLAYER PROJECTION MODEL**.

> **THE PLAYER SHOULD HAVE THE RULES. THE POSITION SHOULD NOT HAVE THE PLAYER.**

> **ANOMALIES ARE QUESTIONS, NOT TRENDS.**

Position MAE/RMSE is downstream QA only.

Required reasoning order:
1. player identity;
2. current role/state;
3. opportunity allocation;
4. opponent/environment;
5. market-specific outcome;
6. reliability.

GPT = independent auditor / project control. Claude = executor / research lead. Shared lane = Issue #535.

### LIVE STATE AT V9 FREEZE

Before V9 continuity writes:
- `main` = **`f3eef8b8a77386208f4ecebcf346df5c1888da5c`**
- research branch `research-prism-player-first-identity-environment-2026-10-26` = **`8ed1d1778d388d54e6de9ed37ffc62d2d300e2ea`**
- newest #535 = **6441088661**
- research branch compare confirmed exact identity with `8ed1d177`
- betting CLOSED
- no paid OddsAPI authorized

Always re-fetch live state before asserting this remains current.

### IMMEDIATE AUTHORIZED TASK — COMPACT NAMED-PLAYER TRACE

Question:

> **How does canonical role assignment determine which opponent-transmission path a named player is actually eligible to receive?**

Read-only JSN/Kupp/Barner trace:
1. named player / target-week team / position;
2. canonical role/alignment label;
3. opponent fields present for that role;
4. eligible matchup gates;
5. gates that actually fire;
6. pre-gate -> post-gate opportunity share;
7. finite-room/team-normalization effect;
8. shared efficiency movement such as `pass_eff_mult`;
9. role-specific efficiency signal present-but-stranded;
10. final named-player opportunity/efficiency movement attributable to opponent context.

JSN vs Kupp = same-offense/same-opponent identity-routing contrast. Barner = TE route check.

If exact roster reality makes a name inapplicable, preserve the mechanism question and document the legal substitute.

### CLEARED TRANSMISSION STATE — CORRECT STRENGTH ONLY

Walker W7 exact case:
- Seattle rush-defense context existed;
- `def_rush_epa` did not reach Walker's rushing mean;
- explosive suppression carried but unconsumed;
- RB YPT/YBC/stuff not wired;
- neither 0.60 box gate fired;
- opponent-caused W7 Walker rushing opportunity/efficiency/mean/sigma movement = **0**;
- receiving did move through zone target-share rule.

Source-wide corrected state at `8ed1d177`:
- trace stages transport separately from consumption;
- `def_rush_epa` + `explosive_play_rate_allowed` = **TRANSPORTED-BUT-STRANDED**;
- independent generic ensemble = **MC / ML / State**; Bayesian is inside MC;
- `ml_v2` and `state_v2` are opponent-blind;
- MC can transmit opponent effects where a canonical rule exists and fires;
- box gate is a real RB opponent -> YPC -> MC-mean path;
- therefore correct claim is **outside the box gate**, generic RB rushing mean has no opponent run-defense transmission;
- gate firing frequency = **NOT ESTABLISHED**;
- receiving YPT has shared opponent pressure movement through `pass_eff_mult`;
- missing piece is role-specific opponent efficiency/conversion from `ypt_allowed_*`;
- direct generic-skill volatility multiplier is pressure-driven, but sigma also moves indirectly and promoted QB C2 has separate opponent-aware distribution logic;
- WR3+ / lower perimeter WRs can receive no canonical WR target multiplier;
- 108/88/63-style counts are literal source-reference inventory, not causal transmission counts;
- certified TeamContext dataflow = 17 fields populated / 13 read.

### MATERIALITY / ZERO-HISTORY — DO NOT REOPEN

A+B CLOSED.

Production-faithful W7 trace:
- 396 football-universe rows / 28 teams
- true zero-history / `position_prior_only` = **34**, across 21 teams
- `current_games=0` = **69**
- all fallback posteriors nonzero
- zero-history rush top-five = **13 rows / 11 teams**
- final rush allocation mass = **1.824446**
- zero-history target entitlement = **1.264686 / 26.6 = 4.7545%**
- **17** players >= .05
- priced-board fallback 13 players is sportsbook availability only and cannot bound football allocator population.

Retracted — never restate:
- 30.4% allocation-moving
- 1,730 zero-prior
- SEA 1.42 production prior mass
- “at most 13 could enter allocator”
- 36× framing

### IDENTITY CARD

Accepted descriptive substrate; no predictive promotion.

- 35,196 complete-roster cards
- 10,960 model-facing cards
- 3,053 players / 32 teams
- A CLOSED: injury contamination cannot be bypassed by NO_BASELINE
- B CLOSED: `TARGET_TEAM_TENURE_SHARE_V1`, participation-only tenure authority
- `PARTIAL_GAME` source-blocked for 2026
- threshold sensitivity **13.66%**, unresolved by design
- team shares descriptive/provisional, not predictive role authority

### DURABILITY

Forward explanatory-state archive is on `main` since `00a9ae21`.

Week 7 explanatory state recovered to git.

Residual gap:
- Weeks 1–6 explanatory state unavailable;
- current preservation retained opponent/environment column names but not values.

Future fix:
preserve opponent values + `rules_*_mult` so gate firing frequency can be measured on future runs.

### STILL OPEN

- hardcoded generic `pass_share = 0.57`
- lead/trail not altering generic pass/rush split
- future gate-frequency measurement once values are preserved
- Inside-5 TD design audit / future legal prospective test
- betting construction V2 spec
- orphan grading-tool port
- Week-7 grading after grader restoration

### FAIL-CLOSED / DESIGN-ONLY

Stage-2 Week 7 remains chronology fail-closed. No postgame shadow.

Inside-5 TD preregistration remains design-only:
- Week 7 generated hypothesis and cannot test it
- `MIN_I5_SUPPORT=10`
- inside-5 vs RZ correlation 0.890
- clip not main issue
- no scoring/promotion yet

### BETTING

Betting CLOSED.

Future construction rules remain:
- main-line direction conflict => PASS
- cushion != safety
- use mean + SD + exact threshold + P(clear) + price/EV
- probability/dependency-aware tiers
- surface repeated-player/same-game concentration
- no arbitrary exposure cap invented from W7
- ATD separate/uncertified

### NON-NEGOTIABLE BOUNDARIES

- no paid OddsAPI without explicit user approval
- no sportsbook upstream
- no Week-7 fitting
- no post-hoc subgroup rescue
- no named-player exception
- no aggregate-positional-MAE chase
- no reopening A/B
- no weakening `FETCH_LIVE_ODDS` guard to make CI green
- no wholesale merge of orphan `claude/nice-cori-4qkwqs`
- no frequency claim from source wiring alone
- Week 7 identifies mechanisms; it does not score fixes

### EXACT NEXT QUEUE

1. Re-fetch #535 after **6441088661**.
2. Audit Claude's compact JSN/Kupp/Barner role-routing target if posted; otherwise wait for/coordinate that exact authorized read-only executor item.
3. Fix future trace preservation to retain opponent values + multipliers.
4. Trace hardcoded pass-share / lead-trail game-script path.
5. Audit/freeze Inside-5 TD design before any legal test.
6. Betting-layer V2 spec.
7. Port orphan grading tooling file-by-file.
8. Grade Week 7.

After the next substantive milestone, supersede this root checkpoint in place and archive V9.
