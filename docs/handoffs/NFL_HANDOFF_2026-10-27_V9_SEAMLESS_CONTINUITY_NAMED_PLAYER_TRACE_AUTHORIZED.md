# NFL PRISM V9 — SEAMLESS CONTINUITY HANDOFF
**Date:** 2026-10-27  
**Repo:** `dkaps6/imtiredofthis`  
**Project:** PRISM — Player Role & Individual Situation Model  
**Purpose:** exact cross-chat continuation after conversation cap; GitHub is canonical.

## 0. READ THIS BEFORE DOING ANYTHING

This handoff supersedes all prior chat memory when they conflict.

Permanent doctrine:

> **THE PLAYER SHOULD HAVE THE RULES. THE POSITION SHOULD NOT HAVE THE PLAYER.**

> **ANOMALIES ARE QUESTIONS, NOT TRENDS.**

PRISM is an **individual named-player projection model**. Position MAE/RMSE is downstream QA only, never the modeling objective.

Required reasoning order:
1. player identity;
2. current role/state;
3. opportunity allocation;
4. opponent/environment;
5. market-specific outcome;
6. reliability.

GPT = independent auditor / project control. Claude = executor / research lead. Shared async lane = **GitHub Issue #535**.

Do not ask the user to re-explain any of this.

---

## 1. EXACT CANONICAL RESTART ORDER

Read in this order only:

1. `AGENTS.md`
2. root `CURRENT_NFL_RESEARCH_HANDOFF.md` — **single active V9 TOP checkpoint only**
3. this file:
   `docs/handoffs/NFL_HANDOFF_2026-10-27_V9_SEAMLESS_CONTINUITY_NAMED_PLAYER_TRACE_AUTHORIZED.md`
4. newest comments in Issue **#535**, especially everything after **comment 6441088661**
5. live `main`, research branch, PRs and Actions

Do **not** recursively read archived V2–V8 handoffs unless a specific lineage question requires one.

The displaced V8 active TOP is archived at:
`docs/handoffs/NFL_HANDOFF_2026-10-27_V9_ARCHIVED_TOP_V8_WALKER_GENERALIZATION_CORRECTED.md`

The restart prompt the user can paste into the next chat is:
`docs/handoffs/NEXT_CHAT_PROMPT_2026-10-27_V9_NAMED_PLAYER_ROLE_ROUTING.md`

---

## 2. LIVE CONTROL STATE AT THIS HANDOFF

Verified immediately before writing V9:

- `main` = **`f3eef8b8a77386208f4ecebcf346df5c1888da5c`** before the V9 continuity writes.
- Research branch:
  `research-prism-player-first-identity-environment-2026-10-26`
  = **`8ed1d1778d388d54e6de9ed37ffc62d2d300e2ea`**
  and comparison showed that exact target was identical to the live research-branch head.
- Newest #535 at freeze: **6441088661**.
- That comment independently re-audited `8ed1d177`, **CLEARED** all six source-generalization corrections, preserved Walker exact-case clearance, and **AUTHORIZED** the compact JSN/Kupp/Barner named-player role-routing trace.
- Betting remains **CLOSED**.
- No new paid OddsAPI pull is authorized.
- Week 7 may identify a missing mechanism; it may **not** score or tune the fix.

Always re-fetch #535 before acting because Claude may have posted the named-player trace after this handoff was written.

---

## 3. THE PRECISE CURRENT STOPPING POINT

The prior disagreement over whether named-player WR/TE work was redundant is **settled**.

Comment **6441088661** rules:

> **DO THE COMPACT JSN / KUPP / BARNER TRACE.**

Why it is not redundant:
source archaeology tells us what *can* route, but a named-player trace answers the new player-first mechanism question:

> **How does canonical role assignment determine which opponent-transmission path a named player is actually eligible to receive?**

### The next executor item

Strictly read-only compact trace for:
- **JSN**
- **Kupp**
- **Barner**

The exact report should include only:

1. named player / target-week team / position;
2. canonical role/alignment label actually assigned;
3. opponent fields present for that role;
4. which canonical matchup gates are eligible;
5. which gates actually fire;
6. pre-gate -> post-gate opportunity share;
7. finite-room / team-normalization effect;
8. any shared efficiency movement, especially pressure `pass_eff_mult`;
9. role-specific efficiency signal present-but-stranded;
10. final named-player opportunity/efficiency movement attributable to opponent context.

### Why these three

**JSN vs Kupp** is the key same-offense / same-opponent contrast:
- isolate player identity / role routing from defensive environment;
- determine whether canonical role labels send them through different target-opportunity gates;
- see whether one receives a multiplier while the other does not;
- see whether finite-room conservation transfers movement between teammates;
- identify any role-specific opponent conversion signal upstream that remains stranded.

**Barner** checks the TE-specific opportunity path against the same architecture.

If exact current-week roster/player reality makes a name inapplicable, preserve the mechanism question rather than forcing the name. Use the nearest legal pregame same-offense role contrast and document why.

### GPT's next job

If Claude has posted a new target after **6441088661**:
- audit the **exact code/artifacts/target**, not just Claude's prose;
- determine whether the player-first role-routing claim is actually supported;
- post the ruling back to #535.

If Claude has **not** posted it yet:
- do not invent results;
- do not reopen already-cleared Walker/materiality work;
- the authorized executor item remains the compact named-player trace above.

---

## 4. WALKER / OPPONENT-ENVIRONMENT TRANSMISSION — WHAT IS ACTUALLY CLEARED

### Walker Week-7 exact case — CLEARED

Do **not** redo Walker.

Canonical original decomposition:
- `docs/research/PRISM_WEEK7_WALKER_ENVIRONMENT_TRANSMISSION_DECOMPOSITION_V1.md`
- commit `cca416bd2cdad8f142a5a063923149344c7e5c5d`

For Walker's frozen W7 generic rushing path:
- Seattle's rich rush-defense context existed pregame;
- `def_rush_epa` did **not** reach Walker's rushing mean;
- `explosive_play_rate_allowed` was carried but not consumed by the generic rule;
- RB YPT/YBC/stuff signals were not wired into generic player projection;
- neither 0.60 light/heavy-box gate fired;
- therefore opponent-caused Walker W7 rushing movement was exactly:
  - opportunity = **0**
  - efficiency = **0**
  - final mean = **0**
  - sigma = **0**
- Walker receiving **did** move through the zone-coverage target-share rule.

This is outcome-blind architecture evidence, not a claim about what the correct Seattle adjustment should have been.

### Source-wide transmission trace — corrected V2 state CLEARED

Exact research target:
**`8ed1d1778d388d54e6de9ed37ffc62d2d300e2ea`**

Files audited:
- `scripts/research/trace_opponent_environment_transmission_v1.py`
- `docs/research/PRISM_OPPONENT_ENVIRONMENT_TRANSMISSION_TRACE_V1.md`
- `tests/test_trace_opponent_environment_transmission_v1.py`
- `data/research/opponent_transmission/opponent_environment_transmission_v1_status.json`

The six corrections are **CLEARED**:

1. **Transport != consumption.**
   The trace now separates:
   present on player row -> transported/bridged -> referenced by consumer -> read by rule.
   `def_rush_epa` and `explosive_play_rate_allowed` are correctly **TRANSPORTED-BUT-STRANDED**.

2. **Ensemble = 3 independent generic components, not 4.**
   `COMPONENTS = ("mc_proj", "ml_proj", "state_proj")`.
   Bayesian v2 is inside MC.
   `ml_v2` and `state_v2` are opponent-blind; MC can express opponent effects where a canonical rule exists and fires.

3. **RB rushing generalization narrowed correctly.**
   The light/heavy-box route is a real opponent -> YPC -> MC-mean channel:
   - light box >= .60 => ×1.07
   - heavy box >= .60 => ×0.94
   Therefore the correct source statement is:
   > **Outside the box gate, the generic RB rushing mean has no opponent run-defense transmission.**
   Do not call the gate dead. Gate firing frequency is **NOT ESTABLISHED**.

4. **Receiving efficiency is not opponent-blind.**
   Shared pressure mismatch changes `pass_eff_mult`, which changes `rules_ypt`.
   The gap is absence of **role-specific** opponent efficiency/conversion channels using `ypt_allowed_wr/slot/te/rb/outside`.

5. **Sigma wording narrowed.**
   Pressure is the direct explicit generic-skill volatility multiplier.
   Sigma can also move indirectly through targets and YPT.
   Promoted QB C2 has a separate opponent-aware distribution route and must not be overwritten by generic-skill claims.

6. **Population-frequency language removed.**
   Source wiring cannot prove that a threshold is common, rare, reachable, or ordinary.
   Gate firing frequency remains unmeasured.

Also accepted:
- `_wr_role_labels()` labels slots plus only the top two perimeter WRs by target share; **WR3+ / lower perimeter WRs can get no canonical WR target multiplier**.
- literal-reference inventory counts are not causal transmission counts.
- certified dataflow statement: `context_bridge_v3` populates **17** TeamContext fields and rules read **13**.
- `def_rush_epa` and `explosive_play_rate_allowed` are the critical transported-but-stranded examples.
- `rush_stuff_rate`, `yards_before_contact_per_rb_rush`, and `ypt_allowed_*` are built/stored but not consumed by the generic player projection.
- an orphaned rush-EPA adjustment exists in `pricing_v2`, but it is not canonical, not validated on the promoted stack, and **not endorsed**.

---

## 5. MATERIALITY / ZERO-HISTORY POSITION-PRIOR ENTITLEMENT — CURRENT CLEAN STATE

Do not reopen Blockers A/B. They are closed.

The static DEFAULTS materiality interpretation was retracted and replaced by a production-faithful Week-7 trace.

Durable Week-7 explanatory state:
`data/research/week7_opportunity_trace/`

Paid source artifact:
- run `39849371821`
- head `4dc178b59da8708dbea978b8e54e753b1e2f75c5`
- artifact `12456298985`
- digest `sha256:d1a1fb48e5bb920c5291f35b7457b5870801b689751deb2d742f98c555603a85`

Five-part reconciled report:
1. football universe = **396 rows / 28 teams**
2. `position_prior_only` = true zero-history = **34 players**, across **21 teams**
3. `current_games=0` = **69**
4. all fallback posteriors are nonzero:
   - QB rush 0.111763 / target 0.000706
   - RB rush 0.359568 / target 0.079174
   - TE rush 0.002982 / target 0.129391
   - WR rush 0.008382 / target 0.165842
5. exactly-zero rush fallback rows = **0**
6. exactly-zero target fallback rows = **0**
7. zero-history rush top-five admission = **13 rows / 11 teams**
8. final rush allocation mass = **1.824446**
9. zero-history target entitlement = **1.264686 / 26.600000 = 4.7545%**
10. players with entitlement >= .05 = **17**
11. priced-board fallback = **13 players / 60 priced rows**, but that is **sportsbook availability only** and must never bound the football universe.

### RETRACTED — DO NOT REPEAT AS PRODUCTION FACT

Never restate:
- "30.4% allocation-moving"
- "1,730 zero-prior cards"
- "SEA 1.42 production prior mass"
- "at most 13 players could have entered an allocator"
- "36× larger"

Those were invalid interpretations:
- static DEFAULTS != actual production posterior;
- priced sportsbook board != football allocation universe.

Methodological lesson:
> **Where production records its own final route field, read it rather than re-deriving it.**

The target entitlement path is an example: `entitlement_tgt_share` is authoritative; four attempted re-derivations did not reproduce it.

---

## 6. IDENTITY CARD — ACCEPTED DESCRIPTIVE SUBSTRATE, NO PREDICTIVE PROMOTION

Identity Card state:
- 35,196 complete-roster cards
- 10,960 model-facing opportunity cards
- 3,053 players
- 32 teams
- seasons 2025 + 2026, target weeks 1–7

Blocker A CLOSED:
- contamination resolves before baseline sufficiency;
- `contaminated = abnormal | returned_from_absence | (~enough & has_limiting)`;
- 151 player-weeks moved from usable into withheld states;
- CI requires `no_baseline_with_limiting_designation = 0`.

Blocker B CLOSED:
- `TARGET_TEAM_TENURE_SHARE_V1`;
- numerator, accepted weeks and denominator all target-team aligned;
- no trustworthy tenure => NaN;
- tenure only from participation authority;
- weaker opportunity-derived tenure was intentionally removed because it biases shares upward;
- GPT accepted that stricter semantic.

Still true:
- `PARTIAL_GAME` source-blocked for 2026;
- team shares descriptive/provisional, not predictive role authority;
- threshold sweep `MODERATELY_THRESHOLD_SENSITIVE` = **13.66%**;
- do not select a threshold from Week 7;
- unresolved player-weeks remain and must not become invented trends.

---

## 7. FORWARD ARCHIVE / DURABILITY STATE

On `main` since `00a9ae21`:
`archive-market-track-record-v1.yml` preserves pre-sportsbook explanatory state on every Full Slate.

It reuses the same extractor/denylist, and is deliberately soft-failing after the paid-raw-input archive step so enrichment failure never jeopardizes paid-input preservation.

Residual gap:
- Week 7 explanatory state recovered and committed;
- Weeks 1–6 explanatory state was not preserved and those artifacts are gone;
- current Week-7 preservation filter retained opponent/environment column **names** but not the values.

Required future fix:
- preserve opponent/environment values + `rules_*_mult` so gate firing frequency/reachability can be measured from future live runs.
- do not infer frequency from source code.

---

## 8. GAME SCRIPT / STAGE-2 — STILL OPEN OR FAIL-CLOSED

Still untraced/open:
- generic game script hardcodes `pass_share = 0.57`;
- lead/trail state does not currently alter the generic pass/rush split.

Stage-2:
`PRISM_FUTURE_STAGE2_TRANSLATION_NOT_READY__FAIL_CLOSED`

Do not:
- weaken the chronology guard;
- reconstruct a postgame Week-7 Stage-2 "shadow";
- call Week 7 prospectively ready;
- use Week 7 as retrospective substitute evidence.

Stage-2 research may be qualified historically, but is not production-wired.

---

## 9. INSIDE-5 TD OPPORTUNITY QUALITY — DESIGN ONLY

Preregistration:
`docs/research/PRISM_INSIDE5_TD_OPPORTUNITY_QUALITY_PREREGISTRATION_V1.md`

State:
- design-only;
- unscored;
- unpromoted;
- Week 7 generated the hypothesis and therefore cannot validate it;
- `MIN_I5_SUPPORT = 10` frozen before any legal test.

Important negative evidence:
- inside-5 share vs red-zone share correlation = **0.890**
- largely redundant signal;
- current 1.35 clip is not the main compression issue;
- this is a narrow role-quality refinement hypothesis, not evidence that production is broadly blind to goal-line usage.

First legal scoring window must be prospective Week 8+ or explicitly uninspected historical data, after design audit.

---

## 10. WEEK-7 BETTING FORENSIC / CONSTRUCTION LANE

Betting remains CLOSED while science work is active.

Week-7 failure was ruled primarily a **construction** problem; do not use it to declare projection science solved or failed.

Future betting-layer V2 direction already established:
- proposed alt direction conflicting with canonical main-line edge => **PASS** by default;
- raw yard cushion != safety;
- use model mean + SD + exact threshold + `P(clear)`;
- require price/implied probability and EV when authority is valid;
- make ticket tiers probability-aware / dependency-aware;
- surface repeated-player and same-game exposure;
- do not invent universal exposure caps from Week 7;
- ATD remains separate/uncertified;
- sportsbook information stays downstream of independent football projection.

Durable Week-7 ticket evidence exists under:
`docs/research/evidence/2026-10-25_week7_ticket_ledger/`

Do not reconstruct tickets from chat memory.

---

## 11. PROMOTED / CLOSED / SOURCE-BLOCKED SCIENCE — PRESERVE

Do not casually reopen or overwrite:
- QB M89/M90 + C2 active;
- WR-R15 active;
- TE-R5P active;
- RB P3/R22 Week-1-only and not active Week 7;
- Stage-2 qualified research but not production-wired;
- direct WR/CB unavailable/gated;
- ATD uncertified;
- RB rushing Stage-1 source reset SOURCE BLOCKED;
- RB/TE team-context reset candidates PREDICTIVE CLOSED.

No aggregate-MAE optimization as a substitute for individual-player correctness.

---

## 12. REPO DEBT / OPERATING HAZARDS

Carry forward:
- one known stale no-credit `FETCH_LIVE_ODDS` test failure on main. **Do not weaken the live-odds guard just to make CI green.**
- orphan branch `claude/nice-cori-4qkwqs` has no merge base; port useful tooling file-by-file only, never wholesale merge.
- orphan tooling still needed for Week-7 grading.
- `requirements.txt` leaves `pyarrow` unbounded while workflows pin it individually.
- some tests write status JSON into canonical `data/`; treat as residue, do not commit.
- cloud environment can be blocked from direct Actions artifact blob download; runner-based artifact work may be required.
- old handoff directory contains near-duplicate timeout-era V4 files; use the active V9 root/prompt, not filename guessing.

---

## 13. CURRENT PRIORITY QUEUE

### Immediate
1. Re-fetch #535 after **6441088661**.
2. If Claude has posted the compact JSN/Kupp/Barner trace, **audit that exact target immediately**.
3. If not, the authorized executor task is the compact named-player role-routing trace described in §3.

### Then
4. Fix future explanatory-state preservation to retain opponent values + `rules_*_mult`.
5. Trace hardcoded `pass_share = 0.57` and lead/trail game-script transmission.
6. Audit/freeze Inside-5 TD design before any legal prospective scoring.
7. Betting-layer construction V2 spec.
8. Port orphan Week-6/Week-7 grading tooling file-by-file.
9. Grade Week 7 after graders are restored.

Do not skip directly to betting.

---

## 14. NON-NEGOTIABLE BOUNDARIES

- no paid OddsAPI without explicit user approval;
- no sportsbook variable upstream of independent football projection;
- no Week-7 fitting;
- no post-hoc subgroup rescue;
- no invented named-player exception;
- no threshold/coefficient/sign chosen because of a Week-7 result;
- no reopening A/B;
- no reopening cleared science without new causal evidence;
- no recursive old-handoff archaeology as a substitute for the current root;
- no position-average MAE chase as the objective;
- no silently turning a population prior into named-player truth;
- no claim that a source-code gate is common/rare without measured frequency;
- betting CLOSED until explicitly reopened.

---

## 15. HOW THE NEXT GPT SHOULD BEHAVE

The user wants the next chat to feel like the same assistant continuing, not a new analyst learning the project.

Therefore:
- do not ask for a recap;
- do not ask the user to restate PRISM doctrine;
- treat GitHub as canonical;
- independently inspect code/results before clearing Claude's work;
- communicate with Claude through #535 when useful;
- preserve disagreements rather than smoothing them over;
- correct semantic overclaims even when the underlying direction is right;
- keep user updated concisely during long audits;
- after the next substantive milestone, supersede the root handoff again with exactly one TOP checkpoint.

**Exact resume sentence:**  
The role-generalization corrections at `8ed1d177` are CLEARED and the compact JSN/Kupp/Barner named-player role-routing trace is AUTHORIZED; re-fetch #535 after comment `6441088661` and audit Claude's next exact target if present.
