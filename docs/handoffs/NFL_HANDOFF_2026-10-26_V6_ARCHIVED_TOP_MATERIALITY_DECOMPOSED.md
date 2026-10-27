# ARCHIVED HANDOFF CHECKPOINT — 2026-10-26 — V5 — MATERIALITY "DECOMPOSED"

**THIS IS NOT THE ACTIVE CHECKPOINT. DO NOT RESUME FROM THIS FILE.**

The active checkpoint is the single TOP CHECKPOINT in root
`CURRENT_NFL_RESEARCH_HANDOFF.md`. Archived per the AGENTS.md continuity rule.

Superseded because one of its central claims was WITHDRAWN. It reports the
fallback materiality decomposition as complete and cites 30.4% allocation-
moving, 1,730 zero-prior cards and SEA room mass 1.42. Those figures were
computed from the static DEFAULTS table, not production's posterior, and are
retracted: production's posterior for every fallback row is nonzero, so
exactly-zero fallback does not occur at all. Re-audit 6427742545 caught it;
6428042413 supplied the provenance; the replacement is the five-part report
in 6430580774.

Resuming from this file would restate retracted numbers as findings.

Retained verbatim apart from the demoted heading.

---

## ARCHIVED CHECKPOINT (SUPERSEDED) — 2026-10-26 — V5 — BLOCKERS A+B FIXED / MATERIALITY DECOMPOSED / THIRD FIXED TARGET AWAITING RE-AUDIT / NO PREDICTIVE CLEARANCE

**Model:** PRISM — Player Role & Individual Situation Model

### WHAT THE NEXT SESSION SHOULD DO FIRST

The auditor's three outstanding items are **DONE**. The immediate task is a
**re-audit of `febb1acf`**, not redoing them.

Restart prompt: `docs/handoffs/NEXT_CHAT_PROMPT_2026-10-26_V5_IDENTITY_REAUDIT_FEBB1ACF.md`

Displaced checkpoints, archived verbatim, **not** for resuming:
`docs/handoffs/NFL_HANDOFF_2026-10-26_V5_ARCHIVED_TOP_IDENTITY_BLOCKERS_REQUIRED.md` (V4),
`..._V4_ARCHIVED_TOP_AUDIT_PRIORITIES_DONE_REAUDIT_PENDING.md` (V3),
`..._V3_ARCHIVED_TOP_IDENTITY_CARD_ACCEPTED_SUBSTRATE.md` (V2),
`..._V2_ARCHIVED_TOP_WEEK7_FORENSIC_PLAYER_FIRST_TRANSMISSION.md`.

Deep handoff: `docs/handoffs/NFL_HANDOFF_2026-10-26_WEEK7_POSTMORTEM_PLAYER_FIRST_ENVIRONMENT_CONTINUITY.md`.
Do not recursively read older dated handoffs.

### ABSOLUTE ASSIGNMENT

PRISM is an **INDIVIDUAL-PLAYER PROJECTION MODEL**.

> **THE PLAYER SHOULD HAVE THE RULES. THE POSITION SHOULD NOT HAVE THE PLAYER.**

> **ANOMALIES ARE QUESTIONS, NOT TRENDS.**

Position MAE/RMSE is downstream QA only. Identity/state/opponent work is additive.

Always distinguish: `SOURCE READY` / `SOURCE BLOCKED` / `PREDICTIVE CLOSED` / `QUALIFIED`.

### CONTROL ROOM

- Claude = executor / research lead
- GPT-5.6 = independent auditor / project control
- shared lane = Issue #535

Before substantive work: read `AGENTS.md`; read only this TOP checkpoint; read the deep handoff; fetch newest #535; reconcile live `main` / PRs / Actions.

### LIVE STATE

`main` at checkpoint: `c35ed937bf4fdcec03c9476d5b6f7ed6dbcc4e90` (plus this handoff commit).

Research branch (user-authorized), **merge base equals main's head**:

`research-prism-player-first-identity-environment-2026-10-26` @ **`febb1acf`**

Green run `40402625625`. **82 tests** under pandas 1.5.3 / numpy 1.26.4 (the versions CI installs). 13 artifacts committed to git.

Newest #535 comment: **6427513335**

Audit trail: `6425282770` (substrate ACCEPTED, 5 priorities) → `6426050333` (priorities 1-4, target `aa70663c`) → `6426366490` (**BLOCKED** on A+B+materiality) → **`6427513335`** (A+B fixed, materiality decomposed, target `febb1acf`).

**No paid OddsAPI is authorized. Betting remains CLOSED.**

### THE THREE AUDIT ITEMS — ALL CLOSED

**BLOCKER A — `NO_BASELINE` bypassed known contamination. FIXED.**
Injury flags now resolve *before* state assignment; contamination no longer sits behind having a baseline:
`contaminated = abnormal | returned_from_absence | (~enough & has_limiting)`.
The asymmetry is deliberate: with a sufficient baseline and a normal snap count a designation is observably non-limiting and the week stays HEALTHY_FULL; with a short baseline that cannot be assessed, so it forces the week out.

Measured effect — **151 player-weeks** that were usable despite known contamination are now withheld:

| state | before | after |
|---|---|---|
| `NO_BASELINE` | 1,430 | **1,279** |
| `UNRESOLVED_CONTEXT` | 1,078 | 1,171 |
| `RETURN_FROM_INJURY` | 213 | 228 |
| `INJURY_LIMITED` | 67 | 110 |
| `HEALTHY_FULL` | 6,786 | 6,786 |

`no_baseline_with_limiting_designation = 0`, enforced in CI.

**BLOCKER B — trade-unsafe share. FIXED** as `TARGET_TEAM_TENURE_SHARE_V1`.
Numerator restricted to `posteam == team_now`; accepted weeks intersected with target-team tenure; denominator `team_now` over those identical weeks; no tenure → **NaN** plus `no_target_team_tenure_evidence`, never the previous team's role. 29,930 of 35,196 cards correctly publish no share. `opp_total_current_season` stays all-teams (a fact, not a role claim) beside `opp_total_current_season_target_team`.

**MATERIALITY — DECOMPOSED.** `scripts/research/decompose_position_prior_fallback_materiality_v1.py`, priors loaded from `scripts/modeling/bayesian_v2.py` by file path so they cannot drift from production.

| | cards | % eligible |
|---|---|---|
| pooled `position_prior_only` | 5,422 / 10,960 | 49.5% |
| — **zero-prior**, cannot move allocation | **1,730** | — |
| — **allocation-moving** | **3,692** | **33.7%** |
| Week 7 `position_prior_only` | 649 / 1,414 | 45.9% |
| **Week 7 allocation-moving** | **430** | **30.4%** |

The entire zero-prior bucket is two cells: `carry × TE` 1,058 (prior 0.00) and `target × QB` 672 (prior 0.00) — summing exactly to 1,730. **49.5% overstated severity; 30.4% is the Week-7 figure.**

Room-level mass is the number to carry forward: Week-7 carry rooms SEA **1.42**, BAL 1.36, MIA 1.24, PIT target 1.24 — against a room that normalises to ~1.0. Players with no opportunity evidence contribute more than a full room's worth of share mass before normalisation.

### ADDITIONAL FINDING — TENURE AUTHORITY (not audit-requested)

The tenure fallback derived from *opportunity* when the participation team column was missing. That inflates the share: it cannot see a week a player was on the team and took no carry or target, so those weeks leave the denominator while touch weeks keep the numerator complete — the same error as Blocker B through a graceful degradation. The audit's spec permits "opportunity/roster state", so this is **stricter than required**; a share known to be biased upward is worse than a missing one. Only the participation authority may establish tenure.

Target week 1 has no earlier weeks under strict-prior, so no tenure authority can exist: 5,924 week-1 cards report `no_prior_participation` and publish no share. The CI gate is semantic, not positional — whatever the source, a card that did not use the participation authority must publish no tenure and no share.

### CARD STATE

35,196 cards (`COMPLETE_ROSTER_UNIVERSE`), 10,960 model-facing (`MODEL_FACING_OPPORTUNITY_UNIVERSE`, its own committed artifact), 3,053 players, 32 teams, seasons 2025+2026, target weeks 1-7.

Gate: `PLAYER_STATE_ANOMALY_GATE_V2_INJURY_RESOLVED`. `PARTIAL_GAME` **SOURCE BLOCKED** for 2026 (`load_participation` raises `Season must be between 2016 and 2025`; PBP is no substitute, it names only ball-touchers).

Bridge: player-level, 0 ambiguous, partial coverage structurally impossible. Binding constraint is nflverse roster `pfr_id` completeness (74.6%). **99.80% coverage on opportunity-eligible cards with current-season opportunity** is the statistic that matters, not `bridge_match_rate`.

Threshold sweep: `MODERATELY_THRESHOLD_SENSITIVE`, spread **13.66%** (362 of 2,651 eligible cards). **No threshold selected; carried forward as uncertainty.**

Boundary: `DESCRIPTIVE_STATE_ONLY_NO_FIT_NO_PROMOTION`, `sportsbook_inputs_used = 0` on every row.

### KNOWN GAPS — STATED, NOT HIDDEN

- **1,171 player-weeks remain unexplained.** No injury report explains a benching, blowout rest, healthy inactive or committee rotation.
- `PARTIAL_GAME` source-blocked for 2026.
- Team shares semantically correct but **descriptive/provisional, not predictive role authority** per the audit.
- 13.66% threshold sensitivity unresolved by design.
- `PARTIAL_PARTICIPATION_RATIO = 0.5`, `MIN_BASELINE_WEEKS = 2` deliberately **unfitted**.
- Role elasticity and the full teammate-competition map deferred to V2.

### WEEK-7 FORENSIC / TRANSMISSION — UNCHANGED, STILL OPEN

Week-7 ticket failure was **construction**, not model (`6419138959`, accepted in `6420198257`). **Construction failure does NOT close opponent/environment transmission**, and the accepted card does not resolve the Walker / JSN / Kupp / Barner findings by itself.

Still open: `def_rush_epa` unconsumed by generic RB rules; explosive-play suppression unconsumed; WR/TE/RB and outside/slot YPT dropped; YBC/stuff dropped; generic game script hardcodes `pass_share = 0.57` (`scripts/modeling/rules_v2.py:98`); lead/trail does not alter the generic split; Stage-2 conditional allocation unwired.

**Environment is not game total / spread / implied points.** That is one input. It is the football situation: opportunity topology, game script, teammate competition, defensive scheme and personnel.

**Kenneth Walker III remains the preferred first generic-skill transmission case** (GPT version at `cca416bd`).

### STAGE-2 — FAIL CLOSED BY CHRONOLOGY

`PRISM_FUTURE_STAGE2_TRANSLATION_NOT_READY__FAIL_CLOSED` (`6421086265`). Blocker: `target/future stat rows already present; refuse audit: weeks=[7]`. Do not weaken or filter around the guard, do not call Week 7 prospectively ready, **do not reconstruct a Week-7 Stage-2 shadow postgame**, do not reuse Week 7 as a retrospective substitute. The next target must be a genuinely future window frozen before its stats exist.

### BETTING-LAYER CORRECTIONS — STILL IN FORCE

Alt direction conflicting with canonical main-line direction ⇒ **PASS**; raw yard cushion ≠ safety; use mean + SD + exact threshold + `P(clear)` + exact price; never label ~41% probability "Safe"; surface repeated-player and same-game concentration; no universal anchor across tiers; ATD uncertified and separate. Do not invent exposure caps from Week 7.

### EXISTING SCIENCE BOUNDARIES — PRESERVE

QB M89/M90 + C2 active; WR-R15 active; TE-R5P active; RB P3/R22 Week-1-only and **not** active Week 7; Stage-2 qualified in research but not production-wired; direct WR/CB unavailable/gated; ATD uncertified; RB rushing Stage-1 source reset `SOURCE BLOCKED`; RB/TE team-context reset candidates `PREDICTIVE CLOSED`.

### CARRIED-FORWARD REPO DEBT

- Repo CI on main still fails one known stale no-credit exact-string `FETCH_LIVE_ODDS` assertion. **Do not weaken the live-odds guard to make it pass.**
- `claude/nice-cori-4qkwqs` is an **orphan branch with no merge base**; its tooling (Week-6 graders, gate-salvage script, Week-6 result docs) must be ported file-by-file.
- `requirements.txt` leaves `pyarrow` unbounded; workflows pin `pyarrow==17.0.0` individually.
- `docs/handoffs/` contains near-duplicate V4 archive and next-chat-prompt pairs from a GPT chat timeout. Harmless (the root ledger is unambiguous) but pick by date, not by guess.

### EXACT CURRENT NEXT ACTIONS

1. **re-audit `febb1acf`** against Blocker A, Blocker B and the materiality decomposition — do NOT redo them;
2. **no predictive or scoring use until that re-audit clears it**;
3. if it clears, freeze any new predictive mechanism **before** scoring;
4. Walker-first read-only opponent/environment transmission decomposition;
5. extend to representative WR/TE identities;
6. betting-layer construction V2 spec (direction conflict → PASS, clearance probability, price/EV, exposure caps, probability-aware tier labels);
7. port the stranded `claude/nice-cori-4qkwqs` tooling file-by-file;
8. grade Week 7 once the graders are ported to main.

No paid OddsAPI. No sportsbook upstream. No aggregate-MAE optimization. No post-hoc subgroup rescue. No Week-7 fitting. No reopening closed science without new causal evidence.

The next chat MUST re-fetch #535 and live GitHub before asserting that this remains current.
