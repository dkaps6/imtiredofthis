# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## TOP CHECKPOINT — 2026-10-27 — V8 — WALKER W7 EXACT CASE CLEARED / GENERALIZATION CORRECTED ON SIX POINTS / ARCHIVE CONTRACT ON MAIN / NO PREDICTIVE CLEARANCE

**Model:** PRISM — Player Role & Individual Situation Model

### WHAT THE NEXT SESSION SHOULD DO FIRST

Read newest #535 (**6432895399**). Walker's W7 exact case is **CLEARED by both re-audits**; the source generalization was **corrected on six points** and the corrections are landed at `8ed1d177`. **Do not restate the retracted DEFAULTS figures** (30.4% allocation-moving, 1,730 zero-prior, SEA 1.42) **or the six withdrawn transmission overclaims** — see RETRACTIONS.

**Open question for the auditor, not for you to decide alone:** the two GPT chats disagree on named-player work (`6432370310` wants a compact JSN/Kupp/Barner trace; `6432388032` thinks it is reconfirmation). My reading, posted for their ruling: role-label assignment is itself an untraced transmission step, so it is a new mechanism question and the two positions reconcile.

Displaced checkpoints, archived verbatim, **not** for resuming:
`docs/handoffs/NFL_HANDOFF_2026-10-27_V8_ARCHIVED_TOP_WALKER_TRACED_PRE_CORRECTION.md` (V7, **contains the six withdrawn overclaims**),
`..._2026-10-27_V7_ARCHIVED_TOP_FIVE_PART_REPORT_REPRODUCED.md` (V6),
`..._2026-10-26_V6_ARCHIVED_TOP_MATERIALITY_DECOMPOSED.md` (V5, contains retracted claims),
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

`main` at checkpoint: **`00a9ae21`** (plus this handoff commit).

Research branch (user-authorized): `research-prism-player-first-identity-environment-2026-10-26` @ **`8ed1d177`**

Tests: **779 passed**, 1 known pre-existing failure, 1 skipped (pandas 1.5.3 / numpy 1.26.4). `audit_repo.py --strict` passes.

Newest #535: **6432895399**

Audit trail: `6425282770` substrate ACCEPTED → `6426366490` BLOCKED on A+B+materiality → `6427513335` A+B fixed → `6427742545` **A+B CLEARED** → `6427971422` DEFAULTS withdrawn → `6428042413` provenance + priced-board objection → `6430580774` five-part report → **`6431984732` WALKER UNBLOCKED** → `6432197295` trace + `6432265555` addendum → `6432370310` / `6432388032` **W7 CLEARED, generalization needs correction** → `6432895399` all six corrections landed.

**No paid OddsAPI is authorized. Betting remains CLOSED.**

### WALKER OPPONENT/ENVIRONMENT TRANSMISSION — W7 CLEARED, GENERALIZATION CORRECTED

Doc `docs/research/PRISM_OPPONENT_ENVIRONMENT_TRANSMISSION_TRACE_V1.md`; script `scripts/research/trace_opponent_environment_transmission_v1.py`; artifacts `data/research/opponent_transmission/`. Consumes, does not redo, the decomposition @ `cca416bd`. **Do not redo Walker.**

**CLEARED by both re-audits — the W7 exact case.** For the frozen W7 generic rushing path: Seattle's rush-defense context existed; `def_rush_epa` did not reach Walker's rushing mean; explosive suppression carried but unconsumed; RB YPT/YBC/stuff not wired into the generic player projection; **neither 0.60 box gate fired**; so opponent-caused opportunity, efficiency, mean and sigma movement were exactly **0**; receiving **did** move through the zone-coverage target-share rule.

**Also CLEARED:** the generic RB rushing path has a very thin opponent interface, and major run-defense families are stranded from it; `ml_v2`/`state_v2` are opponent-blind; the orphaned `pricing_v2` rush-EPA mechanism is not canonical and is not endorsed; `ypt_allowed_*` are not consumed by the production player path.

**THE CERTIFIED DATAFLOW FINDING** (exact, not inventory): `context_bridge_v3` populates **17** `TeamContext` fields; rules read **13**. `def_rush_epa` and `explosive_play_rate_allowed` are **TRANSPORTED-BUT-STRANDED** — carried onto the very object `matchup_multipliers()` receives, read by no rule. `def_pass_epa` is consumed by `qb_pass_synthesis_v1` (promoted); `proe` by `simulation_v2`.

**What is established about RB rushing, stated at correct strength:**

> The box gate IS a live opponent → RB rushing mean channel (`rules_ypc = base_ypc * rb_rush_eff_mult`, `simulation_rules.py:231`). **Outside** that gate, the generic RB rushing mean has no opponent run-defense transmission.

**`rush_stuff_rate`, `yards_before_contact_per_rb_rush`, `ypt_allowed_*` are referenced by NO production module at all** — built, stored, never consumed.

**Ensemble contract:** `COMPONENTS = ("mc_proj", "ml_proj", "state_proj")`. Bayesian v2 is **not** a fourth component (it is inside the MC baseline). So **2 of 3** independent generic components are opponent-blind; **MC can** express opponent effects where a rule exists and fires.

**Role generalization, corrected:** receiving **opportunity** has role-specific gates for every role *represented by the canonical multipliers* — but `_wr_role_labels()` labels slots plus only the top two perimeter WRs by `tgt_share`, so **WR3 and below get no target multiplier at all**. Receiving **efficiency** is NOT opponent-blind: pressure sets `pass_eff_mult` and `simulation_rules.py:230` applies `rules_ypt = base_ypt * pass_eff_mult`; what is absent is any **role-specific** conversion channel from `ypt_allowed_*`. **Sigma**: only the **direct explicit** volatility multiplier is pressure-driven; sigma also moves indirectly (`rec_sd` scales with `rules_ypt` and multinomial target counts), and **promoted QB C2 is a separate opponent-aware distribution route** (selector contract `[pass_opportunity_spot, pass_efficiency_spot, rush_opportunity_spot, rush_efficiency_spot, pred_qb_attempts, week]` from opponent `def_prior_*`). A generic-skill trace must not overwrite promoted QB architecture.

**Counts are INVENTORY, not causal:** 108 opponent columns on the player row, 88 football signal, **63 not referenced by any consumer**. These are literal-reference counts; a base-name hit does not prove the opponent-suffixed value is consumed (`proe` is the clearest counterexample). Use the `TeamContext` 17-vs-13 comparison for certified claims.

**NOT ESTABLISHED:** gate firing frequency for ANY gate; whether any threshold is "reachable" or "ordinary" in the league distribution; predictive value of consuming any stranded family. Production **does** persist the multiplier (`artifact_contracts.py:38`; `simulation_rules.py:236`) but **my own** Week-7 preservation filter kept column names and dropped every opponent value — a new §5a gap in my script, not in production. Fix the filter before the next live slate.

### RETRACTIONS — DO NOT RESTATE THESE AS FINDINGS

Two of my own published claims are withdrawn. Both were caught by audit, not by me.

**1. The DEFAULTS-based materiality figures.** "30.4% allocation-moving", "1,730 zero-prior cards", "SEA carry room 1.42 prior mass" were computed from the static `DEFAULTS` table. Production reaches `DEFAULTS` only as `_weighted_group_mean()`'s THIRD fallback (weighted group mean → median → constant). Measured from the secured artifact: **every** fallback row has a nonzero posterior in both markets. `DEFAULTS` says `rush_share[TE]=0.00` and `tgt_share[QB]=0.0`; production says 0.002982 and 0.000706. **Exactly-zero fallback does not occur in production.** Fields renamed `nonzero_default_prior_candidate` / `default_prior_mass_proxy`.

**2. Using the priced board to bound a football population.** "At most 13 could have entered any allocator" is invalid — a player can move the finite room and never receive a prop. This is the §3 violation. Also retracted: the "36× larger" framing, which compared 464 **card** rows to 13 priced players. The correct football-universe comparison is **34 vs 13 = 2.62x**.

Coincidence to avoid: the audit's 13 rush-top-five admitted rows and the 13 priced fallback players are the same number and different populations.

**3. Six transmission overclaims (V7), withdrawn after re-audit `6432370310`/`6432388032`.** All six were verified against source before acceptance; all six were correct.

| # | withdrawn | correct statement |
|---|---|---|
| 1 | `def_rush_epa` "consumed in the player path" | **transported-but-stranded**; transport ≠ consumption |
| 2 | "2 of 4 ensemble components" | **2 of 3**; Bayesian v2 is inside MC |
| 3 | "opponent run defense cannot reach an RB rushing mean, every RB every week"; "dead" gate | the box gate **is** live; claim holds only **outside** the gate; no frequency/severity wording |
| 4 | "opponent reaches per-role efficiency for essentially no one" | receiving YPT moves via shared `pass_eff_mult`; only the **role-specific** channel is absent |
| 5 | "opponent reaches sigma only through pressure" | only the **direct explicit** multiplier; sigma moves indirectly, and QB C2 is a separate opponent-aware route |
| 6 | "0.50–0.60 thresholds sit inside ordinary NFL profiles" | **removed**; frequency never measured |

Lesson to carry: I wrote a population-frequency claim two sections after writing NOT ESTABLISHED for the same quantity. **Source code cannot support a frequency statement.**

### WEEK-7 TRACE SECURED (§5a window used)

Paid artifact `12456298985` (run `39849371821`, head `4dc178b5`, expires **2027-01-20**) is read and its pre-sportsbook explanatory state is **durable in git** at `data/research/week7_opportunity_trace/` — 213 artifact files catalogued, 50 trace files. Sportsbook/decision columns excluded by denylist; extractor raises if one survives.

Key preserved files: `week7_trace__data__football_simulation_universe.csv` (396 rows), `..._target_entitlement_v1_trace.csv` (396 rows, **`entitlement_tgt_share`**), `..._model_bayesian_diagnostics.csv`.

**Known limitation (see V7 trace above):** the wanted-column filter preserved opponent/environment **names only**, not values.

### FIVE-PART REPORT — BOTH AUDIT ROUTES REPRODUCE EXACTLY

`scripts/research/report_week7_fallback_materiality_v1.py`, output `week7_fallback_materiality_report_v1.json`.

1. **Full football universe**: 396 rows / 28 teams; `position_prior_only` **34** = true zero-history **34** across **21** teams; `current_games=0` **69**.
2. **Nonzero posterior**: QB rush 0.111763 / tgt 0.000706; RB 0.359568 / 0.079174; TE 0.002982 / 0.129391; WR 0.008382 / 0.165842. **Zero exactly-zero posteriors.**
3. **Rush allocator**: **13** rows admitted across **11** of 28 teams, final allocation probability **1.824446** — reproduces the audit to six decimals via production's own `_top_n_shares` + room normalisation. Heaviest: LAR 0.320205, PIT 0.265311, MIN 0.252108, KC 0.228210, CLE 0.216676.
4. **Target room**: zero-history entitlement **1.264686** of **26.600000** = **4.7545%**, **17** players ≥ 0.05 — reproduces exactly.
5. **Priced board = SPORTSBOOK AVAILABILITY ONLY**: 2,894 rows / 328 players; fallback 60 rows / 13 players. **Must never bound parts 1–4.**

**Methodological rule learned the hard way:** the rush route REPRODUCES by re-deriving; the target route does NOT. Four derivations were tried (2.745449, 1.775639, 1.813099, 1.686857) — none close. Production's `entitlement_tgt_share` is the end of `rules_tgt_share` → `m38_explicit_entitlement_tgt_share` → `post_m38_tgt_share` → `team_scale`. **Where production records its own answer, read it; do not re-derive it.** Route columns are now captured by PATTERN with a full column-name inventory per file.

### FORWARD ARCHIVE CONTRACT — ON MAIN (DONE IN V7)

`archive-market-track-record-v1.yml` @ `00a9ae21` preserves explanatory state on every slate, reusing the same extractor (one definition of a sportsbook column). Placed after the paid-raw-inputs step and deliberately **soft-failing** with a loud `::warning::` — the archive's first duty is the paid boundary, and an enrichment must never jeopardise it, which is exactly how Weeks 3–5 were lost.

### RESIDUAL §5a GAP — NARROWED, NOT CLOSED

Explanatory state is durable for **Week 7 only**, recovered from a surviving artifact, and **without opponent/environment values**. Weeks 1–6 priced boards are archived but their explanatory state was never preserved and those artifacts are gone. The forward contract stops the bleeding; it does not recover history.

### IDENTITY CARD STATE

35,196 cards (`COMPLETE_ROSTER_UNIVERSE`), 10,960 model-facing (`MODEL_FACING_OPPORTUNITY_UNIVERSE`, own committed artifact), 3,053 players, 32 teams, 2025+2026, weeks 1-7.

Blockers A and B **CLOSED** by audit `6427742545`:
- **A**: contamination resolves before baseline sufficiency; `contaminated = abnormal | returned_from_absence | (~enough & has_limiting)`. 151 player-weeks moved from usable to withheld (`NO_BASELINE` 1,430→1,279). `no_baseline_with_limiting_designation = 0`, CI-enforced.
- **B**: `TARGET_TEAM_TENURE_SHARE_V1` — numerator, accepted weeks and denominator all restricted to `team_now`; no tenure → NaN. Tenure only from the **participation authority**; the opportunity-derived fallback was removed because it inflates the share. The auditor **accepted this stricter reading**.

Gate `PLAYER_STATE_ANOMALY_GATE_V2_INJURY_RESOLVED`; `PARTIAL_GAME` **SOURCE BLOCKED** for 2026. Bridge player-level, 0 ambiguous; **99.80%** coverage on opportunity-eligible cards with current-season opportunity. Sweep `MODERATELY_THRESHOLD_SENSITIVE`, **13.66%**, no threshold selected.

### KNOWN GAPS

- 1,171 player-weeks unexplained (no injury report explains a benching, blowout rest, healthy inactive or committee rotation).
- `PARTIAL_GAME` source-blocked for 2026.
- Team shares semantically correct but **descriptive/provisional, not predictive role authority**.
- 13.66% threshold sensitivity unresolved by design; `PARTIAL_PARTICIPATION_RATIO = 0.5` and `MIN_BASELINE_WEEKS = 2` deliberately **unfitted**.
- Role elasticity and the full teammate-competition map deferred to V2.
- Box-gate firing frequency unmeasured (above).

### WEEK-7 FORENSIC / TRANSMISSION — PARTIALLY CLOSED BY V7

Week-7 ticket failure was **construction**, not model (`6419138959`, accepted `6420198257`). **Construction failure does NOT close opponent/environment transmission.**

Mechanism now **traced and documented** (V7 section above): `def_rush_epa` carried-and-ignored; explosive suppression unconsumed; YPT/YBC/stuff never referenced; the single RB-rush gate at 0.60.

Still open and NOT traced: generic game script hardcodes `pass_share = 0.57` (`scripts/modeling/rules_v2.py:98`); lead/trail does not alter the generic split; Stage-2 unwired.

**Environment is not game total / spread / implied points.** It is the football situation: opportunity topology, game script, teammate competition, defensive scheme and personnel.

**The zero-history / position-prior entitlement defect is a SEPARATE cause** with its own trace and disposition. Do not merge it into transmission.

### STAGE-2 — FAIL CLOSED BY CHRONOLOGY

`PRISM_FUTURE_STAGE2_TRANSLATION_NOT_READY__FAIL_CLOSED` (`6421086265`). Do not weaken or filter around the guard, do not call Week 7 prospectively ready, **do not reconstruct a Week-7 Stage-2 shadow postgame**, do not reuse Week 7 as a retrospective substitute.

### INSIDE-5 TD OPPORTUNITY QUALITY — DESIGN ONLY

Pre-registration frozen at `docs/research/PRISM_INSIDE5_TD_OPPORTUNITY_QUALITY_PREREGISTRATION_V1.md` (`MIN_I5_SUPPORT = 10`). **Unscored and unpromoted** until audited and a first legal test window is frozen. Negative evidence stands: `inside_5`/`goal_to_go` exist nowhere in production; correlation with red-zone **0.890** (largely redundant); the clip is **not** the main compression problem; 5 well-supported divergent backs make it a narrow role-quality refinement at best.

### BETTING-LAYER CORRECTIONS — IN FORCE

Alt direction conflicting with canonical main-line direction ⇒ **PASS**; raw yard cushion ≠ safety; use mean + SD + exact threshold + `P(clear)` + exact price; never label ~41% probability "Safe"; surface repeated-player and same-game concentration; no universal anchor across tiers; ATD uncertified and separate. Do not invent exposure caps from Week 7.

### EXISTING SCIENCE BOUNDARIES — PRESERVE

QB M89/M90 + C2 active; WR-R15 active; TE-R5P active; RB P3/R22 Week-1-only and **not** active Week 7; Stage-2 qualified in research but not production-wired; direct WR/CB unavailable/gated; ATD uncertified; RB rushing Stage-1 source reset `SOURCE BLOCKED`; RB/TE team-context reset candidates `PREDICTIVE CLOSED`.

### CARRIED-FORWARD REPO DEBT

- Repo CI on main still fails one known stale no-credit `FETCH_LIVE_ODDS` assertion. **Do not weaken the live-odds guard to make it pass.**
- `claude/nice-cori-4qkwqs` is an **orphan branch with no merge base**; its tooling (Week-6 graders, gate-salvage script, Week-6 result docs) must be ported file-by-file.
- `requirements.txt` leaves `pyarrow` unbounded; workflows pin `pyarrow==17.0.0` individually.
- This cloud environment's network policy denies `productionresultssa18.blob.core.windows.net`, so a session cannot download Actions artifacts directly; artifact work must run on the runner.
- Some unit tests write status JSON into canonical `data/`; treat as residue, do not commit.
- `docs/handoffs/` has near-duplicate V4 pairs from a GPT chat timeout. Pick by date, not by guess.

### EXACT CURRENT NEXT ACTIONS

1. get the auditor's ruling on the named-player disagreement, then run the compact JSN/Kupp/Barner transmission table scoped to: role label assigned, opponent fields present, which gates fire, finite-room effect, teammate-vs-same-defense difference (read-only, no scoring window);
2. **no predictive or scoring use of the Identity Card or any transmission candidate until cleared**;
3. fix the preservation filter to retain opponent values + `rules_*_mult`, so box-gate reachability becomes measurable on the next live run;
4. trace the remaining untraced items: hardcoded `pass_share = 0.57`, lead/trail game script;
6. betting-layer construction V2 spec (direction conflict → PASS, clearance probability, price/EV, exposure caps, probability-aware tier labels);
7. port the orphan `claude/nice-cori-4qkwqs` tooling file-by-file;
8. grade Week 7 once the graders are ported.

No paid OddsAPI. No sportsbook upstream. No aggregate-MAE optimization. No post-hoc subgroup rescue. No Week-7 fitting. No reopening A or B. No reopening closed science without new causal evidence. **Week 7 can identify a missing mechanism; it cannot score the fix.**

The next chat MUST re-fetch #535 and live GitHub before asserting that this remains current.
