# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## TOP CHECKPOINT — 2026-10-27 — V6 — A+B CLOSED / DEFAULTS MATERIALITY RETRACTED / WEEK-7 TRACE SECURED / FIVE-PART REPORT REPRODUCES BOTH AUDIT ROUTES / NO PREDICTIVE CLEARANCE

**Model:** PRISM — Player Role & Individual Situation Model

### WHAT THE NEXT SESSION SHOULD DO FIRST

Read newest #535 (**6430580774**) and await / act on the audit of the five-part
materiality report. **Do not restate the retracted DEFAULTS figures** (30.4%
allocation-moving, 1,730 zero-prior, SEA 1.42) — see RETRACTIONS below.

Displaced checkpoints, archived verbatim, **not** for resuming:
`docs/handoffs/NFL_HANDOFF_2026-10-26_V6_ARCHIVED_TOP_MATERIALITY_DECOMPOSED.md` (V5, contains retracted claims),
`..._V5_ARCHIVED_TOP_IDENTITY_BLOCKERS_REQUIRED.md` (V4),
`..._V4_ARCHIVED_TOP_AUDIT_PRIORITIES_DONE_REAUDIT_PENDING.md` (V3),
`..._V3_ARCHIVED_TOP_IDENTITY_CARD_ACCEPTED_SUBSTRATE.md` (V2),
`..._V2_ARCHIVED_TOP_WEEK7_FORENSIC_PLAYER_FIRST_TRANSMISSION.md`.

Deep handoff: `docs/handoffs/NFL_HANDOFF_2026-10-26_WEEK7_POSTMORTEM_PLAYER_FIRST_ENVIRONMENT_CONTINUITY.md`. Do not recursively read older dated handoffs.

### ABSOLUTE ASSIGNMENT

PRISM is an **INDIVIDUAL-PLAYER PROJECTION MODEL**.

> **THE PLAYER SHOULD HAVE THE RULES. THE POSITION SHOULD NOT HAVE THE PLAYER.**

> **ANOMALIES ARE QUESTIONS, NOT TRENDS.**

Position MAE/RMSE is downstream QA only. Always distinguish `SOURCE READY` / `SOURCE BLOCKED` / `PREDICTIVE CLOSED` / `QUALIFIED`.

### CONTROL ROOM

Claude = executor / research lead. GPT-5.6 = independent auditor / project control. Shared lane = Issue #535.

Before substantive work: read `AGENTS.md`; read only this TOP checkpoint; read the deep handoff; fetch newest #535; reconcile live `main` / PRs / Actions.

### LIVE STATE

`main` at checkpoint: `53aed043` (plus this handoff commit).

Research branch (user-authorized): `research-prism-player-first-identity-environment-2026-10-26` @ **`ef3b13a2`**

Green runs: card `40402625625`, trace+report `40408929482`. **98 tests** (pandas 1.5.3 / numpy 1.26.4).

Newest #535: **6430580774**

Audit trail: `6425282770` substrate ACCEPTED → `6426366490` BLOCKED on A+B+materiality → `6427513335` A+B fixed → `6427742545` **A+B CLEARED**, materiality blocked → `6427971422` DEFAULTS claim withdrawn → `6428042413` provenance supplied + priced-board objection → **`6430580774`** five-part report.

**No paid OddsAPI is authorized. Betting remains CLOSED.**

### RETRACTIONS — DO NOT RESTATE THESE AS FINDINGS

Two of my own published claims are withdrawn. Both were caught by audit, not by me.

**1. The DEFAULTS-based materiality figures.** "30.4% allocation-moving", "1,730 zero-prior cards", "SEA carry room 1.42 prior mass" were computed from the static `DEFAULTS` table. Production reaches `DEFAULTS` only as `_weighted_group_mean()`'s THIRD fallback (weighted group mean → median → constant). Measured from the secured artifact: **every** fallback row has a nonzero posterior in both markets — exactly-zero rush posteriors 0, exactly-zero target posteriors 0. `DEFAULTS` says `rush_share[TE]=0.00` and `tgt_share[QB]=0.0`; production says 0.002982 and 0.000706. **Exactly-zero fallback does not occur in production.** The fields are renamed `nonzero_default_prior_candidate` / `default_prior_mass_proxy`.

**2. Using the priced board to bound a football population.** "At most 13 could have entered any allocator" is invalid — a player can move the finite room and never receive a prop. This is the §3 violation. Also retracted: the "36× larger" framing, which compared 464 **card** rows to 13 priced players. The correct football-universe comparison is **34 vs 13 = 2.62x**.

Coincidence to avoid: the audit's 13 rush-top-five admitted rows and the 13 priced fallback players are the same number and different populations.

### WEEK-7 TRACE SECURED (§5a window used)

Paid artifact `12456298985` (run `39849371821`, head `4dc178b5`, expires **2027-01-20**) is read and its pre-sportsbook explanatory state is **durable in git** at `data/research/week7_opportunity_trace/` — 213 artifact files catalogued, 50 trace files. Sportsbook/decision columns excluded by denylist; extractor raises if one survives.

Key preserved files: `week7_trace__data__football_simulation_universe.csv` (396 rows: `prior_games`, `current_games`, `bayes_evidence_state`, `bayes_position_group`, `bayes_rush_share`, `bayes_tgt_share`, `rules_rush_share`, `rules_tgt_share`), `..._target_entitlement_v1_trace.csv` (396 rows, **`entitlement_tgt_share`**), `..._model_bayesian_diagnostics.csv`.

### FIVE-PART REPORT — BOTH AUDIT ROUTES REPRODUCE EXACTLY

`scripts/research/report_week7_fallback_materiality_v1.py`, output `week7_fallback_materiality_report_v1.json`.

1. **Full football universe**: 396 rows / 28 teams; `position_prior_only` **34** = true zero-history **34** across **21** teams; `current_games=0` **69**.
2. **Nonzero posterior**: QB rush 0.111763 / tgt 0.000706; RB 0.359568 / 0.079174; TE 0.002982 / 0.129391; WR 0.008382 / 0.165842. **Zero exactly-zero posteriors in either market.**
3. **Rush allocator**: **13** rows admitted across **11** of 28 teams, final allocation probability **1.824446** — reproduces the audit to six decimals by applying production's own `_top_n_shares` + room normalisation from `simulation_v2.py`. Heaviest: LAR 0.320205, PIT 0.265311, MIN 0.252108, KC 0.228210, CLE 0.216676.
4. **Target room**: zero-history entitlement **1.264686** of **26.600000** = **4.7545%**, **17** players ≥ 0.05 — reproduces the audit exactly.
5. **Priced board = SPORTSBOOK AVAILABILITY ONLY**: 2,894 rows / 328 players; fallback 60 rows / 13 players. **Must never bound parts 1–4**; the warning is embedded in the JSON.

**Methodological rule learned the hard way:** the rush route REPRODUCES by re-deriving; the target route does NOT. Four derivations were tried (raw sharpened 2.745449, per-team normalised 1.775639, normalised unsharpened 1.813099, residual-style 1.686857) — none close. Production's `entitlement_tgt_share` is the end of `rules_tgt_share` → `m38_explicit_entitlement_tgt_share` → `post_m38_tgt_share` → `team_scale`. **Where production records its own answer, read it; do not re-derive it.** The field was missed on the first extraction because its name was guessed; route columns are now captured by PATTERN and a full column-name inventory is recorded per file.

### FORWARD ARCHIVE CONTRACT — WRITTEN, NOT YET ON MAIN

`archive-market-track-record-v1.yml` now preserves explanatory state on every slate, reusing the same extractor (one definition of a sportsbook column). Placed after the paid-raw-inputs step and deliberately **soft-failing** with a loud `::warning::` — the archive's first duty is the paid boundary, and an enrichment must never jeopardise it, which is exactly how Weeks 3–5 were lost. `audit_repo.py --strict` passes.

**This lives on the research branch and takes effect only once it reaches `main`. It should be there before the next live slate.** A production workflow change was deliberately not pushed to main unilaterally.

### RESIDUAL §5a GAP — NARROWED, NOT CLOSED

Explanatory state is durable for **Week 7 only**, recovered from a surviving artifact. Weeks 1–6 priced boards are archived but their explanatory state was never preserved and those artifacts are gone. The forward contract stops the bleeding; it does not recover history.

### IDENTITY CARD STATE

35,196 cards (`COMPLETE_ROSTER_UNIVERSE`), 10,960 model-facing (`MODEL_FACING_OPPORTUNITY_UNIVERSE`, own committed artifact), 3,053 players, 32 teams, 2025+2026, weeks 1-7.

Blockers A and B **CLOSED** by audit `6427742545`:
- **A**: contamination resolves before baseline sufficiency; `contaminated = abnormal | returned_from_absence | (~enough & has_limiting)`. 151 player-weeks moved from usable to withheld (`NO_BASELINE` 1,430→1,279). `no_baseline_with_limiting_designation = 0`, CI-enforced.
- **B**: `TARGET_TEAM_TENURE_SHARE_V1` — numerator, accepted weeks and denominator all restricted to `team_now`; no tenure → NaN. Tenure only from the **participation authority**; the opportunity-derived fallback was removed because it inflates the share (cannot see no-touch tenure weeks). The auditor **accepted this stricter reading**.

Gate `PLAYER_STATE_ANOMALY_GATE_V2_INJURY_RESOLVED`; `PARTIAL_GAME` **SOURCE BLOCKED** for 2026. Bridge player-level, 0 ambiguous; **99.80%** coverage on opportunity-eligible cards with current-season opportunity. Sweep `MODERATELY_THRESHOLD_SENSITIVE`, **13.66%**, no threshold selected.

### KNOWN GAPS

- 1,171 player-weeks unexplained (no injury report explains a benching, blowout rest, healthy inactive or committee rotation).
- `PARTIAL_GAME` source-blocked for 2026.
- Team shares semantically correct but **descriptive/provisional, not predictive role authority**.
- 13.66% threshold sensitivity unresolved by design; `PARTIAL_PARTICIPATION_RATIO = 0.5` and `MIN_BASELINE_WEEKS = 2` deliberately **unfitted**.
- Role elasticity and the full teammate-competition map deferred to V2.

### WEEK-7 FORENSIC / TRANSMISSION — STILL OPEN

Week-7 ticket failure was **construction**, not model (`6419138959`, accepted `6420198257`). **Construction failure does NOT close opponent/environment transmission.**

Still open: `def_rush_epa` unconsumed by generic RB rules; explosive-play suppression unconsumed; WR/TE/RB and outside/slot YPT dropped; YBC/stuff dropped; generic game script hardcodes `pass_share = 0.57` (`scripts/modeling/rules_v2.py:98`); lead/trail does not alter the generic split; Stage-2 unwired.

**Environment is not game total / spread / implied points.** It is the football situation: opportunity topology, game script, teammate competition, defensive scheme and personnel.

**Kenneth Walker III remains the preferred first generic-skill transmission case** (GPT version `cca416bd`). **Walker-first work waits until the materiality item is audited.**

### STAGE-2 — FAIL CLOSED BY CHRONOLOGY

`PRISM_FUTURE_STAGE2_TRANSLATION_NOT_READY__FAIL_CLOSED` (`6421086265`). Do not weaken or filter around the guard, do not call Week 7 prospectively ready, **do not reconstruct a Week-7 Stage-2 shadow postgame**, do not reuse Week 7 as a retrospective substitute.

### BETTING-LAYER CORRECTIONS — IN FORCE

Alt direction conflicting with canonical main-line direction ⇒ **PASS**; raw yard cushion ≠ safety; use mean + SD + exact threshold + `P(clear)` + exact price; never label ~41% probability "Safe"; surface repeated-player and same-game concentration; no universal anchor across tiers; ATD uncertified and separate. Do not invent exposure caps from Week 7.

### EXISTING SCIENCE BOUNDARIES — PRESERVE

QB M89/M90 + C2 active; WR-R15 active; TE-R5P active; RB P3/R22 Week-1-only and **not** active Week 7; Stage-2 qualified in research but not production-wired; direct WR/CB unavailable/gated; ATD uncertified; RB rushing Stage-1 source reset `SOURCE BLOCKED`; RB/TE team-context reset candidates `PREDICTIVE CLOSED`.

### CARRIED-FORWARD REPO DEBT

- Repo CI on main still fails one known stale no-credit `FETCH_LIVE_ODDS` assertion. **Do not weaken the live-odds guard to make it pass.**
- `claude/nice-cori-4qkwqs` is an **orphan branch with no merge base**; its tooling (Week-6 graders, gate-salvage script, Week-6 result docs) must be ported file-by-file.
- `requirements.txt` leaves `pyarrow` unbounded; workflows pin `pyarrow==17.0.0` individually.
- This cloud environment's network policy denies `productionresultssa18.blob.core.windows.net`, so a session cannot download Actions artifacts directly; artifact work must run on the runner.
- `docs/handoffs/` has near-duplicate V4 pairs from a GPT chat timeout. Pick by date, not by guess.

### EXACT CURRENT NEXT ACTIONS

1. await / act on the audit of the five-part report (`6430580774`);
2. **no predictive or scoring use of the Identity Card until cleared**;
3. get the forward archive contract onto `main` before the next live slate;
4. Walker-first read-only opponent/environment transmission decomposition (gated on item 1);
5. representative WR/TE identities;
6. betting-layer construction V2 spec (direction conflict → PASS, clearance probability, price/EV, exposure caps, probability-aware tier labels);
7. port the orphan `claude/nice-cori-4qkwqs` tooling file-by-file;
8. grade Week 7 once the graders are ported.

No paid OddsAPI. No sportsbook upstream. No aggregate-MAE optimization. No post-hoc subgroup rescue. No Week-7 fitting. No reopening A or B. No reopening closed science without new causal evidence.

The next chat MUST re-fetch #535 and live GitHub before asserting that this remains current.
