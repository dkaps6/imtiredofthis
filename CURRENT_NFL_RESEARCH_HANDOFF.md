# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## TOP CHECKPOINT — 2026-10-28 — V16 — **CORRECTION: PRODUCTION IS NOT CONSTANT-SIGMA** / LIVE 2026 BACKTEST DONE / CLAUDE ISSUE #535

**Model:** PRISM — Player Role & Individual Situation Model. **Repo:** dkaps6/imtiredofthis. **Current state:** one production fix merged to `main` (`9aa6058c`); next lane identified and not yet started.

**Read in this exact order**:
1. AGENTS.md (canonical 2026 operating contract, player-first first principle).
2. **ONLY THIS NEWEST ACTIVE TOP V14**. Do not recursively read prior V2–V13 checkpoints.
3. `docs/research/PRISM_PLAYER_DISPERSION_IS_A_CONSTANT_V1.md` — the active lane.
4. `docs/research/PRISM_RATE_SCALE_PATH_ATTRIBUTION_V1.md` — what shipped and why.
5. Prior deep continuity if needed: `docs/handoffs/NFL_HANDOFF_2026-10-27_V13_RATE_GUARD_PLAYER_ALLOCATION_CONTINUITY.md`; displaced V13 root archived at `docs/handoffs/NFL_HANDOFF_2026-10-28_V13_PRE_V14_ROOT_ARCHIVE.md`.
6. GitHub Issue https://github.com/dkaps6/imtiredofthis/issues/535 — fetch comments after **6457028486** FIRST. GPT ↔ Claude Code work channel.

**Most important immediate next task:** the dispersion lane's target has **changed**. It is no longer "replace a constant" — the constant was never deployed. It is now **widen and calibrate the Monte Carlo distribution in `run_pricing_v2.py`**, measured by `fair_prob` calibration and Brier against the market on live 2026 data, where production currently scores 0.2800 vs the market's 0.2491. The per-tail quantile work stays useful as a *description* of how floors and ceilings scale, but it is **not** a drop-in replacement and its `Q0` gains must never be quoted as gains over production. Second task: preserve production decision-gate evidence on every live run so weeks 6+ become bet-gradable.

### NON-NEGOTIABLE USER SCIENCE
**THE PLAYER SHOULD HAVE THE RULES, NOT THE POSITION HAVE THE PLAYER.** Named-individual projections from identity, injury/state, real opportunity, teammate competition, defense, role, script and market-specific football logic. Positional MAE/RMSE are QA only. **ANOMALIES ARE QUESTIONS, NOT TRENDS.** Maintain promoted Bayesian/ML/State/MC/QB/RB layers; do not scrap prior work.

**NEW, user-stated 2026-10-28 and now evidence-backed:** *"You cannot treat all RBs and all WRs and all QBs and all TEs the same. Positionally sure, as individual players absolutely not."* A generic weight on Drake London and Josh Downs is wrong because their **distributions** differ, not just their means. **Averaged MAE is structurally blind to this** — a model can have every mean right and every floor and ceiling wrong and score identically. Do not use averaged error as the primary metric for a player-level distribution claim; use per-player quantile calibration.

### ⚠️ READ FIRST — MAJOR CORRECTION, V16
**Production does NOT price from a constant sigma or a normal CDF.** Full doc:
`docs/research/PRISM_CORRECTION_PRODUCTION_IS_NOT_CONSTANT_SIGMA_V1.md`.

Real path: `full-slate.yml:573` → `run_pricing_with_full_roster_universe_v3.py` →
v4/v5/v6 → `...v1.py` → `import scripts.run_pricing_v2 as pricing`, and
`run_pricing_v2.py:400,410` does `p_over = np.mean(adjusted_outcomes > line)` and
`model_sd = np.std(adjusted_outcomes)`. **It prices from the empirical Monte Carlo
distribution.** `run_pricing_v2` imports only `_fair_market_prob` / `_fair_odds` from
`pricing_v2` and **never references `SIGMA`**. Live W7 board confirms: `model_sd` has
139 distinct values for `rec_yards` (5.64–33.79), 71 for `rush_yards`, 139 for
`receptions`. Dispersion **is** player-specific and the distribution is **not** forced
symmetric.

**This is the rate-scale error repeated one layer over** — a defect located in a module
the workflow does not call. `tests/test_production_pricing_path_contract.py` (6 tests)
now pins the real path so it cannot recur. **Before any future claim about "what
production does", grep the workflow for the entry point it actually runs.**

**What this invalidates:** every `Q0` comparison in all four quantile studies. "42.2% →
69.9%" is an improvement over a textbook baseline, **not over production**. The
McLaurin/Wilson "priced identically" claim is **false**.
`PRISM_PLAYER_DISPERSION_IS_A_CONSTANT_V1` is wrong in its title and premise. All nine
affected docs carry a correction banner.

**What survives:** the data findings — floor rises 1.7–2.4× slower than ceiling in all
three markets; centrality→floor `r=+0.620`; per-player volatility weakly persistent
(`r≈0.25`); the three opponent-side nulls. These were measured from outcomes and say
nothing about production code.

### LIVE 2026 BACKTEST — weeks 1, 2, 6, 7 (`backtest_2026_live_calibration_v1`)
**Production's probabilities are severely overconfident.** `fair_prob` calibration,
1,571 deduped player-market-weeks: predicted 0.061 → **actual 0.447**; predicted 0.844 →
**actual 0.538**. The realised over-rate is nearly flat (0.45–0.60) across a reported
0.78-wide range. Brier **model 0.2800 vs market 0.2491**. Overall predicted over-rate
0.406 vs actual 0.488, so also **biased toward unders**.

**Direction vindicated, mechanism refuted:** distributions too narrow — yes. Because of
a constant — no. Because **the Monte Carlo is too tight**.

Mean accuracy vs market: overall model MAE 17.53 vs vegas 16.85 (+0.68), closer on
47.5%. **Beats the market on `pass_yards`** (53.28 vs 55.18, closer 54.1%) — the one
market with a promoted synthesis. Worst on `rush_rec_yards` (+4.87, closer 40.8%).
**Improving weekly:** gap +1.24 → +0.66 → +0.57 → **+0.19**; closer 43.8% → **50.9%**.

### OPERATIONAL GAP — fix this on the next live run
**Bet-level grading is blocked for weeks 6 and 7.**
`data/market_track_record/PRODUCTION_DECISION_GATES_V1.json` preserves decision-gate
evidence for **weeks 1–2 only**, so the canonical grader fails closed later — correctly.
Weeks 3–5 are already permanently lost. **Every live run must preserve gate evidence at
the same time it archives the board**, or paid boards survive but stay ungradable as a
betting record.

### SUPERSEDED — was "the first real win" in V15
**Per-tail quantile fit beats production's constant sigma, on naive data.**
`PRISM_DIRECT_QUANTILE_{PREREGISTRATION,RESULTS}_V1`, frozen `abe11db5`, scored
`a2df5060`. Fit 2020–2022, scored **2017–2019 (never inspected)**, 3,992 rows, 184
players. Arm **Q2** (per-tail fit on projected mean + role centrality) passes all four
frozen rules.

- well-calibrated players **38.6% → 59.8%**; coverage median **0.520 → 0.597** (target 0.60);
- players improved: p20 **83.7%**, p80 **59.8%** — a majority at **both** tails;
- high-mean floor bias **+11.7 → −4.7** with the ceiling **not** paying for it (−3.1 → +0.2);
- zero crossed quantiles; robust at min-games 6/8/10; Q2 beats Q0 in all three
  contaminated reported seasons with the p80 majority holding in every one.

**The structural finding.** Production's constant sigma implies **both tails move 1:1
with the mean** and role is irrelevant. Fitted: **floor slope 0.335, ceiling slope
0.698** — a receiver's floor rises far more slowly than his average, which is exactly
why production overstates its best players' floors by ~12 yards. And **centrality loads
on the ceiling 2.5× more than the floor** (+107.3 vs +43.3): being central to an
offence is mostly an **upside** property.

**This explains the prior study's centrality null.** There centrality entered through a
**single sigma**, which cannot lift one tail more than the other. Once tails are
independent its effect is tail-specific. That null was an artefact of functional form,
**not** evidence against role — do not cite it as such.

**Recorded so nobody repeats the mistake:** Q1 (mean only) scores **higher** on the
headline (63.6% vs 59.8%) at every threshold and is **not supported**, because it
improves the ceiling for only 45.7% of players — the exact failure mode of the
preceding study. Reading the coverage column alone picks Q1 and repeats it. Rule 2
existed to prevent that and did.

**NO production change is authorised.** Integration needs its own change, its own
tests, explicit user authorisation, and the design answer above.

### SHIPPED EARLIER THIS SESSION
- **`PERCENTY_COLS` rate-scale fix MERGED to main `9aa6058c`.** `scripts/make_team_form.py:331` now includes `coverage_man_rate`, `coverage_zone_rate`, `middle_open_rate`, `middle_closed_rate`, so Sharp percents convert at ingest as the box rates already did. `middle_open` gate now discriminates (13/32) instead of firing for every team. 10 tests incl. idempotence. Suite 892 passed; the single `test_full_slate_main_push_defaults_to_no_credit_mode` failure is **pre-existing** (confirmed failing on main before the merge).
- **The previously proposed guard `ab29a096` was OFF-PATH and is NOT the gate fix.** `simulation_rules` imports `load_model_contexts` from `context_bridge` **(v1)**, which reads `team_form.csv` + `cb_coverage_team.csv` and **never** reads `team_context_v3.csv`; `apply_rules_to_metrics` also discards the teams dict (`_, players = load_model_contexts()`) and uses the TeamContext baked into each PlayerContext. The guard landed only as defense-in-depth at the artifact boundary; it divides only when `1.0 < v <= 100.0` so it cannot double-convert. **Do not describe it as the gate fix.**

### RESEARCH CURRENT STATUS — NEW FINDINGS
- **THE ACTIVE LANE. Dispersion is a per-market constant.** `scripts/pricing_v2.py:244` → `sigma = SIGMA[market]` (`rec_yards` **26.0**) with a **normal** CDF, so every player's floor and ceiling are equidistant from his mean and **no player property enters dispersion at all**. Cost: realized per-player SD spans **10.0–53.8**; 29 of 121 receivers >30% under-dispersed. McLaurin (mean 58.2, SD 19.0) and Michael Wilson (mean 59.2, SD 51.5) are **1 yard of mean apart, 2.7× apart in dispersion, priced identically**; actual floors 46.6 vs 15.2. Bias is **systematic**: production over-disperses role players (floor −4.3, ceiling +7.8) and **compresses the stars** (floor +6.1, ceiling −4.7). This is the worst place to be wrong because floor/ceiling *are* the over/under.
- **Centrality buys a floor (user hypothesis CONFIRMED).** Floor as × of own mean: 0.295 / 0.432 / 0.554 across centrality terciles. `corr(centrality, floor_ratio)=+0.620`; `corr(centrality, rel. volatility)=−0.656`. London 0.57× vs Downs 0.37× with near-identical ceiling ratios (1.52 vs 1.53).
- **Honest limit on the fix:** per-player volatility is real but **weak** — positive in 5/5 persistence tests (disjoint halves 2023/2024/2025 + two cross-season pairs) at `r(SD)=+0.15..+0.57`, far below the mean's `r=+0.59..+0.74` and unstable between years. **A player's own measured SD must NOT be used raw** (Wilson's 51.5 is mostly noise on 15 games). What is reliable: SD rises monotonically with the mean (21.7/28.3/35.9) and the mean is strongly persistent → use **projected mean + role centrality**. Relationship is **sublinear** (CV 0.79/0.70/0.58), so `sigma = k·mu` is wrong; needs an intercept.
- **Matchup multipliers are overstated ~5–8× and carry NO documented promotion.** 1,072 2025 TE games, player-relative to own mean: zone gate implied multiplier **1.035** (diff +0.034 ± 0.091, **not significant**); WR 0.991, RB 1.059, also not significant. Production applies **1.265** to TEs when both gates fire. Quartile trend 0.977/0.961/1.010/1.052 suggests ≤~5% at the extreme. Only `docs/` references are this research line. Shrinking them is better-supported than the shipped fix but needs its own preregistration. **Not bundled.**
- **RETRACTION — game script is NOT forecastable and the constant 0.57 is near-optimal.** Predicting team-week rush rate out of sample from football-only inputs: own prior rush rate **0.0814**, adding an opponent strength gap **0.0817 (worse)**, production's constant 0.43 **0.0831** — within **2%** of best. The realized 0.162–0.759 spread is **in-game randomness**, not pre-game signal. An earlier reading of that spread as a large unexpressed lever was **wrong**. The volume/game-script channel is dead for backs *and* receivers.
- **Defensive QUALITY is forecastable** (disjoint halves): def stuff rate `+0.411`, def rush YPC `+0.381`, def pass EPA `+0.368`, def pass YPA `+0.270`. Quality is a stable trait; script is not.
- **PREREGISTERED STUDY, H1 NOT SUPPORTED** (`PRISM_PLAYER_SPECIFIC_MATCHUP_{PREREGISTRATION,RESULTS}_V1`, frozen `da7e7c1c` before scoring). Player **profile** (aDOT) × defense **scheme** (man/zone) does not improve named-player receiving yards: holdout MAE M0 22.560, M1 (positional) **22.389**, M2 (player-specific) 22.398. Frozen rule 3 FAILED (gain sat with middle-aDOT players). Secondary: on targets the positional model is **worse than the player's own baseline**. Also: **M1 beats M0 by only 0.171 yards (0.76%)** — the positional matchup layer is worth under 1% even working as designed, so the inert gates are **cheap, not catastrophic**.
- **EXPLORATORY, NOT VALIDATED** (`PRISM_PLAYER_BY_OPPONENT_QUALITY_EXPLORATORY_V1`): quality × quality helps where profile × scheme did not. RB rushing yards 25.797 → positional 25.791 → `prior_ypc × def_ypc_allowed` **25.680**. WR/TE 24.596 → positional 24.612 (worse) → `prior_ypt × def_ypa_allowed` **24.377**. Positional adds nothing or hurts; player-specific helps. Magnitudes small (0.45%, 0.89%). **2025 is spent as a confirming sample** — confirmation requires held-back 2023/2024.
- **Alignment remains UNTESTED, not refuted.** `run_player_form_v2.py` collapses LWR/RWR/SWR to position family (`out["position"] = out["position_group"]`), so slot-vs-boundary — the strongest player-specific matchup axis — exists in no artifact. Keep separate from `PRISM_SLOT_ROLE_UNREACHABLE_V1` (SWR collapse + no downstream SLOT literal ⇒ `slot_target_mult` is **unreachable dead code**; must NOT be silently changed).

### STILL TRUE FROM PRIOR CHECKPOINTS — DO NOT REDO
Walker Week-7 exact case **CLEARED, never redo**. Opponent transmission trace cleared. JSN/Kupp/Barner role trace cleared (Kupp's −28% is M38 hierarchy, NOT opponent). 2025 gate prevalence proxy descriptive-only. 2026 PBP exists; 2026 nflverse participation (`defenders_in_box`, `defense_man_zone_type`) **unavailable** — 2025 is the last season inside the participation window. `middle_closed_rate` is consumed by nothing. Real W7 gate state: `wr1_target_mult` = **1.0 on all 396 rows**, `rules_pass_rate` = **0.57 on all 1,412 rows**. **REPLAY BLOCKER** stands: `replay-paid-full-slate-artifact-once.yml` is pinned to an expired Week-5 artifact and restores prebuilt context; do not blindly repoint it. The W7 fixture is a **column subset** (`usecols=keep`) missing `neutral_pace_last5` and `plays_est`, so it **cannot** establish absolute baseline parity — `rules_plays_est` differs from archive by a per-team constant (within-team std exactly 0.0 across 28 teams).

### NOW-SETTLED DISPERSION FACTS (do not re-derive)
Production `pricing_v2.py:244` is `sigma = SIGMA[market]` (`rec_yards` **26.0**) with a
normal CDF; origin predates available git history. Its intervals contain **50%** of
outcomes against a **60%** target. A single mean-scaled sigma fixes the floor for 77%
of players but breaks the ceiling for 69% — a symmetric family **cannot** calibrate
both tails. A lognormal matched on (mean, sigma) gives the right asymmetry direction
but narrows the central interval and scores worse. **Per-tail direct fitting is the
answer; sigma-and-a-family is not.** A player's own measured SD must **not** be used
raw (r≈0.25 persistence).

### MUST-PASS ACCEPTANCE FOR ANY DISPERSION MERGE
1. **Preregister and freeze before scoring**: the `sigma(projected_mean, centrality)` form, eligibility thresholds, the held-back data, and the decision rule. Commit the preregistration before any result exists.
2. **Per-player quantile calibration is the primary metric** — does each player's 20th/80th percentile contain his outcomes at the right rate — reported as a **distribution across players**. Averaged MAE is explicitly disallowed as the primary read.
3. **Do not use a player's own measured SD raw.** Only `r≈0.25` of that spread is signal; shrink toward the group or derive from the persistent parts (projected mean, centrality).
4. Form must have an **intercept** (relationship is sublinear); `sigma = k·mu` is falsified.
5. Anomaly gate per AGENTS.md: `DNP_INACTIVE` / `PARTIAL_GAME` / `UNRESOLVED_CONTEXT` state-labelled before any observation becomes evidence.
6. Sportsbook separation: no market variable at any stage. Betting remains **CLOSED**.
7. GPT independent audit, post ruling to #535. **The user must expressly authorize any merge.**

### REMAINING AUTHORIZED RESEARCH QUEUE
Dispersion lane is the single active project; do not open a parallel study. Queued, not approved: confirmation of player × opponent-quality on held-back 2023/2024; shrinking the overstated matchup multipliers toward supported values; alignment (LWR/RWR/SWR) restoration as a separate player-first identity proposal; whether the identity card's richer axes (`inside_5_opp`, `third_down_opp`, `two_minute_opp`, trajectory, `opp_week_to_week_sd`) improve the named-player baseline beyond `prior_ypt × prior_tgt_per_game`; Inside-five TD; TE-R5P; entitlement-stage teammate redistribution. **No paid OddsAPI, no sportsbook upstream, no Walker redo, no W7 outcome fitting, no weakening of a truth/integrity gate.**

### COMMUNICATION / DURABILITY
Claude communication: GitHub Issue #535. Do not confuse executor local pytest counts with independently CI-verified tests. Do not claim the user approved a merge. Update this ONE root TOP after material milestones and archive displaced checkpoints under `docs/handoffs/`. Per AGENTS.md §5a, results worth citing go in git, never only in an expiring Actions artifact.

**Session context the next chat should not have to rediscover:** the user corrected the executor three times on 2026-10-28 and was right each time — (1) the receiving null did not cover rushing/Walker, (2) the same test should run for WRs, (3) MAE is the wrong metric for an individual-player claim. The third correction is what surfaced the dispersion constant. Two days produced one shipped fix and a verified map of where leverage is **not**; the user is time-pressured with Week 8 imminent and does not want another lane change.
