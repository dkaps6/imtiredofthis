# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## TOP CHECKPOINT — 2026-11-10 — V25 — **D3 REJECTED ON A CLEAN HOLDOUT / ALT RUNGS GRADED AGAINST REAL PRICES FOR THE FIRST TIME / THE SELECTION RULES ARE NOW CODE, NOT JUDGEMENT / TWO LIVE DISTRIBUTION DEFECTS NAMED**

**Model:** PRISM — Player Role & Individual Situation Model. **Repo:** dkaps6/imtiredofthis.
**Branch:** `claude/nice-cori-4qkwqs`, synced to `main`.

**V24 is superseded, not retracted.** Archived verbatim at
`docs/handoffs/ARCHIVE_NFL_RESEARCH_HANDOFF_V24_PRE_V25_2026-11-10.md`: the certified Week 9
paid board, the five alternate-line provider keys, the four board-killing gates and the
durability fixes all stand. **Do not re-derive them.**

---

### 1. THE V22 ITEM IS CLOSED — D3 IS NOT PROMOTED

`PRISM_D3_W9_HOLDOUT_RESULTS_V1.md`, rows `data/research/d3_w9_holdout_graded_v1.csv`.

Holdout `data/market_track_record/w9_prospective_frozen_v1.csv` was captured
**2026-11-04T00:41:10Z**, before any Week 9 game, reading stats no later than Week 8,
478 players, arms `baseline` and `D3`. Graded on 314 frozen players with Week 9 offensive
snaps.

D3 wins MAE (receiving 13.604 vs 14.035, rushing 5.424 vs 6.197; closer per-player on 63.5%
and 82.0%). **It is not a real win.** RMSE is unchanged and marginally worse in both. Bias
worsens in both (receiving -3.308 to -3.984, rushing -1.362 to -2.346) on a slate whose
realized means sat above both arms. D3 wins MAE by shrinking toward zero.

On the mandated metric — side accuracy against the line — the two arms are **the same model**:

| market | baseline | D3 | n | disagreements |
|---|---|---|---|---|
| `rush_yards` | 52.7% | 52.7% | 74 | **0** |
| `rec_yards` | 55.3% | 55.9% | 170 | **5** (D3 3/5) |

**D1 overlay is UNTESTED, not rejected.** It depends on the `injuries.csv` schema repair that
has not landed, so it was inert at capture. **That same defect is why a player who was Out in
weeks 7 and 8 and Questionable in week 9 reached a shipped card.**

**Bar for any future D-arm:** beat baseline on side accuracy by more than the disagreement
count explains, do not worsen bias, and improve the **top usage quartile** specifically.

---

### 2. ALT RUNGS GRADED AGAINST REAL PRICES — THE FIRST TIME THIS REPO HAS EVER DONE IT

`PRISM_ALT_FIRST_PRICED_GRADING_2026_W9.md`, rows `data/research/alt_graded_2026_wk09_v1.csv`
(4,306 rungs, 13 final Week 9 games).

**`backtest_alt_clearance_table_v1.py` contains zero references to `offered_american`.** It
validated P(clear) at synthetic rungs and never compared one to a price. "The table is
calibrated" was true and **carried no profitability claim**. Week 9 is the first week alt
prices were archived.

Flat 1u on every available alt OVER rung: **-13.4% ROI.** By market:

| market | n | hit% | ROI |
|---|---|---|---|
| **`rec_yards`** | 1401 | 59.3% | **+2.9%** |
| `pass_yards` | 651 | 46.9% | -0.7% |
| `rush_yards` | 673 | 50.2% | -18.5% |
| `rush_rec_yards` | 321 | 42.7% | -33.9% |
| **`receptions`** | 1260 | 31.0% | **-30.0%** |

**No price band is profitable blind** — every band from <=-250 through >+1000 loses, and deep
OVER rungs are taxed hard (frac 1.5-2.0x returns -42.2%, >2.0x returns -59.4%). Price
filtering is not a strategy.

---

### 3. THE SELECTION RULES, IN THE ORDER FOOTBALL DEMANDS

Each was found by a specific failure. All are enforced in
`scripts/research/build_alt_cards_v1.py` with 24 regression tests in
`tests/test_build_alt_cards_gates_v1.py`. **Plain-English statement for the user, who cannot
read the code and should not have to: `docs/research/YOUR_RULES_PLAIN_ENGLISH.md`.**

1. **ON THE FIELD** — `snap_pct >= 0.30`, strictly prior. Below 40% snaps the sign inverts:
   **27.3% hit against a 69.8% breakeven, -66.2% ROI.** Found by shipping a 15.6%-snap RB3.
   The only screen that cost nothing: +1.9pp hit and +1.8pp ROI for 6bp of payout.
2. **REAL ROLE** — absolute `main_line` floor (`rec_yards` >= 18, `rush_yards` >= 15). The
   book's small line is a usage statement. Sub-10 receiving lines **recorded zero 29.4%** of
   the time.
3. **NOT AGAINST OURSELVES** — `cushion = model_proj / alt_threshold >= 1.5`. Cushion < 1.0 is
   a rung above our own projection and returns **-24.6%**. By band: 1.5-2.0x **+7.0%**,
   2.0-3.0x +5.7%; restricted to -150..-400 prices, 1.5-2.0x returns **+13.5%**.
4. **REAL ALTERNATE** — `frac_of_line` outside [0.85, 1.15] and <= 1.5.
5. **OWN EVIDENCE** — `own_games >= 5`, `own_rate >= 0.60`, denominator **games played**
   (`load_snap_counts`), never games with a recorded stat.
6. **MATCHUP** — exclude the toughest defensive quartile. Toughest **-1.8%** (no edge at all),
   softest **+16.9%**. Doubles ROI 4.4% -> 10.2%. Offensive pace is inert by comparison.
7. **AVAILABLE** — Out/Doubtful/Questionable and manual quarantine.
8. **ONE LEG PER CARD ACROSS TIERS.** Tiers sharing legs are one bet wearing five hats; a
   single marginal leg removed 80% of a shipped board.

**Dropped as a rule:** one-leg-per-game. It was a convenience for independence maths, not a
finding, and it cost real options.

---

### 4. TWO LIVE DISTRIBUTION DEFECTS — THESE ARE PRODUCTION BUGS

**(a) `model_sd` is far too narrow.** `PRISM_ALT_PRICING_VALIDATION_2026_W9.md`, rows
`data/research/prism_probability_calibration_2026_v1.csv`. A gamma matched to
(`model_proj`, `model_sd`) reproduces production `fair_prob` to MAE 0.009-0.048 across
line/mean ratios 0.03x-22x, so the summary is faithful to the draws — and the draws are
overconfident. Graded weeks 2/6/7/8 (n=1,447, snap denominator): the **0-10% band hits 39.5%**
and the **70-100% band hits 49.3%**. Everything collapses toward 46%. Measured
`actual/model_proj` for `rec_yards` runs 0.47 at p25 to 1.67 at p75 with an **11.4%
literal-zero rate** the simulation does not represent. **Every `edge_pct` the board emits
inherits this.**

**(b) High-end ceiling compression, measured prospectively.**
`PRISM_ALT_CEILING_AND_COVERAGE_2026_W9.md`. By main-line quartile, TOP tier bias is
**-12.5 yards** (`rec_yards`) and **-7.3** (`rush_yards`) — where the market line ran **+8.1**
on rushing TOP, so **PRISM is 15.4 yards worse than the line exactly where the big tickets
are.** Largest misses: Lamb -128.0 (proj 67.0, actual 195), Golden -110.8, Watson -105.8,
Boutte -101.3, Nacua -71.8 (proj 78.2, actual 150). This is the RB P3 limitation AGENTS.md §2
requires be tracked prospectively; **it is at least as severe for WR `rec_yards`.**

Consequence: because `p_prism` binds on 35.8% of `rec_yards` and 54.8% of `rush_yards` rungs,
a filter requiring PRISM to beat the price **deletes the high-usage OVER rungs that pay**.
Nacua produced zero eligible legs; all 24 of his rungs hit, to >139.5 at +429.

---

### 5. ANTI-FINDINGS — DO NOT RE-DERIVE

- **Long parlays are not reachable ex ante.** Ranked before kickoff, the best any scheme
  managed at 22 legs was 19/22. Hindsight-perfect tickets exist every week and are not
  buyable. The published +1,818,000 figure was winners sorted by price.
- **Price-ranking breaks tickets.** Within a qualifying pool, ranking by price goes 12/15 and
  re-selects exactly the low-main-line players the role floor exists to kill.
- **Cushion-ranking is correct but caps payout.** It always lands on the deepest, most
  expensive rung. Hit rate and ROI both *rise* with cushion, so price and reliability are
  aligned, not traded off. 15 legs cushion-ranked went **15/15 at +1,935**.
- **Alt coverage had a 16% hole** (37 of 231 priced players), clustered by kickoff window —
  BAL/JAX played Thursday, CIN/ATL kicked 09:30 ET against a 09:08 ET capture. Bijan Robinson
  was never rejected by a filter; **he had no ladder to reject.** Emit a coverage manifest.

---

### 6. IMMEDIATE NEXT ACTIONS, IN ORDER

1. **Repair `injuries.csv`.** `build_injuries_weekly` must emit its declared 11-column schema
   and `audit_2026_production_readiness.py` must assert it so a file without `status` fails
   closed. **This unblocks D1 and fixes the gate that let a returning-from-injury player onto
   a shipped card.** Production repair, no preregistration needed.
2. **Finish grading Week 9** — SNF (TB@CHI) and MNF (BUF@MIN) are now final and were excluded
   from the 13-game grading above.
3. **Run the per-cell alt table-support audit** (n, players, post-hoc flag) — still REQUIRED
   before treating any alt row as bettable, still not run.
4. **Week 10 head-to-head on identical rows:** gated cushion rule vs the triple-intersection
   vs blind, with the matchup gate actually applied.
5. **Resolve the MIN QB wk9 quarantine row** — Murray stayed Out, Wentz started.
6. Then: widen simulation dispersion to the measured residual spread per market and projection
   tercile, and model the bust rate explicitly. Re-audit every `edge_pct` consumer after.

---

### 7. PROMOTED AT V25

**Nothing.** No coefficient, weight, feature, artifact schema or pricing path changed. D3 was
tested and rejected. The selection gates are research tooling under `scripts/research/`, not a
production path. `AGENTS.md` §3 holds throughout: `cushion` uses `model_proj` **ordinally**
against a rung and no sportsbook variable constructs any projection.

---

### NON-NEGOTIABLE USER SCIENCE
**THE PLAYER SHOULD HAVE THE RULES, NOT THE POSITION HAVE THE PLAYER.** Named-individual
projections from identity, injury/state, real opportunity, teammate competition, defense,
role, script and market-specific football logic. Positional MAE/RMSE are QA only.
**ANOMALIES ARE QUESTIONS, NOT TRENDS.** Maintain promoted Bayesian/ML/State/MC/QB/RB
layers; do not scrap prior work.

**Averaged MAE is disallowed as the primary metric for any individual-player distribution
claim.** A model can have every mean right and every floor and ceiling wrong and score
identically. Use per-player quantile calibration, or — for a decision claim — side
accuracy against the line, which needs no distributional assumption.

### ⚠️ STILL BINDING — THE V16 CORRECTION
**Production does NOT price from a constant sigma or a normal CDF.** Path:
`full-slate.yml:573` → `run_pricing_with_full_roster_universe_v3.py` → v4/v5/v6 → `...v1.py`
→ `import scripts.run_pricing_v2 as pricing`; `run_pricing_v2.py:400,410` does
`p_over = np.mean(adjusted_outcomes > line)` and `model_sd = np.std(adjusted_outcomes)`.
`pricing_v2.SIGMA` is **never** referenced on the production path.
`tests/test_production_pricing_path_contract.py` pins this. **Before any claim about "what
production does", grep the workflow for the entry point it actually runs.**

### ⚠️ THE SAME ERROR HAPPENED AGAIN IN V18 — AND WAS CAUGHT
A measurement put production's top-1 within-RB share at **0.2226** (near-uniform) and
blamed duplicate `(player, market)` rows in `metrics_ready.csv` consuming the top-5 rushing
pool — Walker appears 10×, and `0.95/5 = 0.19`. **Retracted.** Production calls
`_full_roster_simulate` → `_build_full_universe`, which builds from
`player_form_consensus.csv`, *not* `metrics_ready`, and raises on duplicate
`(team, player_clean_key)`. Preserved universe: 396 rows, 28 teams, **0 duplicates**. The
bug does not exist in production.

**This is now three off-production-path claims in this project** (rate-scale guard,
constant-sigma, duplicate rows). The rule is not optional: **follow the import graph from
the workflow to what is actually called, before measuring anything.**

### RB RUSHING — THE FULL DIAGNOSIS (settled; do not re-derive)
Live 2026 rows, weeks 1/2/6/7, n=182 non-QB `rush_yards` with actuals:

| component | bias | MAE |
|---|---|---|
| `mc_proj` | **−16.15** | 25.38 |
| `ml_proj` | **+0.79** | 26.40 |
| `state_proj` | −3.68 | 26.66 |
| `model_proj` final | −8.46 | 24.39 |
| vegas | −2.43 | **22.78** |

Excess bias vs market on identical rows **−6.03 [−8.03, −4.03]** — significant and
selection-immune. Bias by within-team model rank is negative at **every** rank: rank 1
**−22.34 [−29.04, −15.64]**, rank 2 −5.21 [−10.62, +0.21], rank 3+ −5.13.

### ⚠️ ROOT CAUSE — and the FOURTH retraction, which overturned the lane's premise

**RETRACTED:** the parent diagnosis's *"team RB carry volume is right: 22.59 simulated vs
23.11 actual (bias −0.52)"*. Measured on the **exact production simulation universe**
(`week7_trace__data__football_simulation_universe.csv` — 396 rows, 28 teams, 0 duplicates,
the frame `_full_roster_simulate` passes to `canonical_simulate`) by **running the real
`simulate()`**: realized RB carries **19.77/team vs 23.15 actual, bias −3.38
[−6.36, −0.53]**. An independent analytic estimate agrees at 19.70. **Team rush mass is
LOW, not right.** The old 22.59 almost certainly came from the priced `metrics_ready`
population — the same off-path error as the other three retractions.

That false premise is why the lane chased *allocation* instead of *mass*, and why both
fixes were doomed from the start.

**The mechanism** (proposed by GPT audit 6477974816 AUDIT TARGET 2; confirmed here):
`rules_rush_share` is estimated per player **independently**, with no team-sum constraint.

| quantity | value |
|---|---|
| teams over the 0.95 cap | **28 / 28** |
| raw team rush-share sum | mean **1.2729**, max **1.6877** |
| cap multiplier applied to every modeled player | mean **0.7574**, min 0.5629 |
| RB carries pre-cap → post-cap | 26.60 → **19.77** (−6.83 = −28.03 yd/team) |

The two errors do not cancel: raw shares over-allocate by **+3.26** carries, the cap
over-corrects to **−3.38**. Yards lost to compression by rank — **rank 1 13.18, rank 2
7.88, rank 3+ 5.28** — against observed shortfalls of −22.34 / −5.21 / −5.13. A uniform
multiplicative haircut lowers **every** rank at once, which is exactly the pattern that
killed fix A (mass-preserving, so the missing mass was never inside the RB distribution)
and fix B (reweighting, so the defect is upstream of all three components).

**Do NOT just remove the cap** — pre-cap mass is over by +3.26, so that overshoots. The
defect is the missing reconciliation step. **This unifies with the Bayesian finding:**
`bayesian_v2._posterior_mean` shrinks each player toward a position prior independently,
which both **inflates total mass** and **flattens relative shape** (the ~9pp concentration
gap). One structural defect, two symptoms.

### FIX A — CLOSED NEGATIVE (K1 fired), BUT IT LOCATED THE CAUSE
γ fitted on raw trailing shares, nflverse 2013–2025, 6,296 team-weeks = **0.76**, not >1
(walk-forward sd 0.022 across twelve origins). Raw trailing share is *more* concentrated
than reality. **K1 → ship nothing.** γ=0.76 is mean reversion, which the promoted
empirical-Bayes baselines already apply.

**The real finding.** Production's RB rush-share vector, measured on the deduped universe
with `rules_rush_share`, controlled for back count:
**top-1 share 0.5173 vs football's 0.6277, gap −0.1103, 95% CI [−0.1453, −0.0741]**,
present in every back-count cell.

**Layer decomposition — the cause is the Bayesian layer:**

| stage | top-1 | target | gap |
|---|---|---|---|
| PlayerForm `rush_share_prior` | 0.6897 | 0.6658 | **+0.0239** |
| PlayerForm `rush_share` (blend) | 0.6288 | 0.6421 | **−0.0133** ← on target |
| post-Bayes | 0.5173 | 0.6277 | **−0.1103** |
| post-rules | 0.5173 | 0.6277 | −0.1103 ← untouched |

Paired on the same 25 teams: **−0.0895 [−0.1370, −0.0420]**, reduced in 21/25. Back-count
confound controlled (subset size changes for 9/25): restricted to the 16 teams where it did
**not** change → **−0.0933 [−0.1292, −0.0575]**, 15/16 teams. **`apply_bayesian_to_metrics`
over-shrinks RB rush shares by ~9pp; `apply_rules_to_metrics` passes it through
bit-identically on every team.** Week 8's archived PlayerForm is still near target
(0.6148 vs 0.6444), so this is a standing property of the layer.

γ=1.75 reproduces the target and is worth **+10.3 yd** to the top back of the −22.34 gap.
**But it is mass-preserving, so backs 2+ lose exactly that, while every rank is already
low — kill condition K4.** A transform is shipped **INERT**
(`RB_CARRY_CONCENTRATION_PROMOTED=False`, effective γ=1.0 identity, dual authority) as a
fallback only. 13 tests.

### FIX B — CLOSED NEGATIVE (K2 fired), AND IT INDICTS THE METHOD
`rush_yards`, fit wk1–2 (n=89), test wk6–7 (n=79), candidates frozen before scoring:

| candidate | weights | bias | MAE | side acc |
|---|---|---|---|---|
| C0 incumbent | 0.557/0.443/0 | −8.54 | **23.66** | 53.2% |
| C1 refit | **1.0/0/0** | −15.53 | 24.55 | 51.9% |
| C2 pure ML | 0/1/0 | **+0.26** | 24.34 | 53.2% |
| C3 equal (control) | ⅓ each | −6.42 | 23.91 | 53.2% |

**K2 fired:** the naive control C3 also passed G1–G3, so the finding is "the incumbent is
stale", not "ML is right", and nothing is promoted. C2 has the right level and still fails
G1 (CI **[−4.12, +10.31]** straddles zero at n=79); **K3** fires too (bias flips wk6 +4.1 →
wk7 −3.3). **C1 indicts the method:** the *promoted* fitter, given recent data, chose
**100% MC** and made bias worse, because squared-error NNLS prefers the biased-but-sharper
component.

**Structural, code-level finding:** raw NNLS coefficients are **[1.5165, 0, 0], sum
1.5165** — the fitter *detected the level defect and asked for a 52% scale-up*;
`_normalise_nonnegative` divides it away and returns `(1.0, 0, 0)`. The blend is a strict
convex combination with no intercept, so **the ensemble cannot correct level drift except
by reweighting.** `rush_rec_yards` the same (sum 1.3357). 6 characterization tests pin it.

**Decision effect, wording corrected per GPT audit 6477974816:** fixing the level **changed
26 of 79 decisions and produced zero NET gain** — the flipped sides split **13/13**, so hits
stay **42/79 for C0, C2 and C3 alike** (verified row by row; a real coin-flip coincidence,
not a bug). C2 picks OVER on 58/79 while actuals went over on 37/79 — an unbiased mean is
not a calibrated side. **"Changes no decision" was false and is withdrawn.**

**Also withdrawn:** the stronger inference that every RB level fix must therefore be judged
primarily on side accuracy. That inverts the project's objective. PRISM optimizes
named-player projection quality (AGENTS.md first principle); side accuracy is a *downstream*
decision metric, reported but not definitional. At n=79 with a 13/13 split this is also an
underpowered test — it shows no evidence of betting gain from that substitution, not that a
more correct player projection is valueless.

### MC DISPERSION CALIBRATION — STILL SHIPPED BUT **INERT**
`k` remains `CANDIDATE_PENDING_PARITY_REPLAY`, NOT promoted. W7 exact-path parity failed
closed because the W7 fixture lacks the ML/State prediction columns. **This is now
unblocked going forward:** the `eb109a6a` archival fix is working and wk08 replay inputs
are landing (`data/market_track_record/replay_inputs/2026_wk08/`, including
`model_ml_diagnostics`, `model_state_diagnostics`, `player_form_consensus`).

**The wk08 manifest reports `"complete": false`, and that is CORRECT — do not "fix" it.**
`required_missing` is `['data/metrics_ready.csv', 'data/model_rule_simulation_inputs.csv']`.
Both are downstream of sportsbook props, and the wk08 runs were **no-live-odds dry runs**
(no `boards/2026_wk08.csv`, no `raw_props/2026_wk08.csv`, no `game_odds/2026_wk08.csv`).
Full Slate uploads `data/**` and `outputs/**`, so those files would have been captured had
they existed; they do not exist because there were no props to price. The preservation
script is failing closed and recording the reason, which is the intended behaviour.
**A replay input set therefore cannot be complete on a no-odds run by construction — the
set completes on the next LIVE-ODDS run.** That is the event to wait for before attempting
the parity replay.

Promotion still requires baseline parity established **before** any treatment result is
read, and `k` must not be retuned on W7.

### SHIPPED TO PRODUCTION (cumulative, this lane)
`PERCENTY_COLS` rate-scale fix (`9aa6058c`) — the only behaviour-changing merge. MC
dispersion transform (`b34f85aa`) — wired, **inert**. RB carry concentration — wired,
**inert**. Replay input preservation (`eb109a6a`) — working.

### STILL TRUE FROM PRIOR CHECKPOINTS — DO NOT REDO
Three opponent-side nulls. Walk-forward per-tail quantiles in all three markets: floor
slope 1.7–2.4× slower than ceiling; centrality→floor `r=+0.620`; per-player volatility
weakly persistent (`r≈0.25`). Live 2026 backtest: production probabilities severely
overconfident (Brier 0.2800 vs market 0.2491). Weeks 3–5 2026 boards are **permanently
lost** to a 7-day retention setting (AGENTS.md §5a).

### MUST-PASS ACCEPTANCE FOR ANY MERGE IN THIS LANE
1. **Preregister and freeze before scoring** — form, eligibility, held-back data, decision
   rule, kill conditions. Commit the preregistration before any result exists.
2. Include a **naive control** in every candidate set. K2 exists because C3 caught a result
   that would otherwise have read as a win.
3. **Averaged MAE is never the primary read.** For a decision claim, show side accuracy.
4. Anomaly gate per AGENTS.md: `DNP_INACTIVE` / `PARTIAL_GAME` / `UNRESOLVED_CONTEXT`
   state-labelled before any observation becomes evidence.
5. Sportsbook separation: no market variable at any stage upstream. Betting remains
   **CLOSED**; no paid OddsAPI fetch without express user authorization.
6. GPT independent audit, post ruling to #535. **The user must expressly authorize any
   merge.**

### REMAINING AUTHORIZED RESEARCH QUEUE
**Active, in this order (agreed through #535 comments 6478960995 and 6479098849):**
**(1) roster/availability EXISTENCE integrity** — Ourlads is the sole existence authority
(458 → 458 → 458 → 396 → 396; the availability layer never adds a row), so build a union
existence universe plus a game-day refresh. **(1a) a deterministic identity defect inside (1):**
`canonicalize_player_name_safe` **does not strip name suffixes**, so nflverse
`Mike Washington Jr.` (`mikewashingtonjr`) never joins the universe's `Mike Washington`
(`mikewashington`). 29 of 489 players with 2026 wk1–7 carries/targets (5.9%) keep a suffix
in their clean key; 4 realised mismatches at W7. **This silently breaks grading joins too**,
so treat it as a data-integrity fix, not cosmetics.
**(2) team rush-budget reconciliation**, which must not run before (1).
**(3) WR/TE existence-ceiling diagnostic — DONE**, see
`docs/research/PRISM_EXISTENCE_CEILING_IS_SYSTEM_WIDE_V1.md`: the ceiling is **system-wide**
(absent share 4.16% of targets vs 7.41% of carries) but **materiality is RB-concentrated** —
the worst rushing absence was a 15-carry lead back, the worst receiving absence 5 targets.
14/17 absent receivers were `ACT` in an earlier week, so the dominant class is deterministic
provider omission, not roster uncertainty. **Do not retune target shares from it.**
**(4) Bayesian evidence-dependent shrinkage**, which the auditor rules should be evaluated
only once the roster/budget contract is coherent.

**Standing capability gap — NOT a parallel lane, do not open it in competition with the
above:** we use **no NextGen Stats at all**. One reference repo-wide, Migration 75,
explicitly `production_actionable = False`; zero NGS on the production path.
`load_nextgen_stats` **works for 2026 weeks 1–7** (rushing/passing 253 rows each, receiving
617) and carries `expected_rush_yards`, `rush_yards_over_expected`, `efficiency`,
**`percent_attempts_gte_eight_defenders`**, `avg_separation`, `avg_yac_above_expectation`.
`PRISM_STRANDED_SIGNAL_DISPERSION_PREREGISTRATION_V1.md` wrongly recorded the 2026 NGS feed
as unavailable — it conflated NGS with **participation**, which genuinely does fail for
2026; that doc now carries a correction banner. **Target-week NGS rows are OUTCOMES**, so
any use must be strictly-prior-weeks-only and walk-forward. Open questions put to the
auditor (#535 comment 6479259425): is NGS publication timing pregame-safe for a Thursday
run, and does M75's QB-efficiency null already close RB rushing efficiency and per-player
box counts? Separately, `nfl_data_py` is a **fallback only** (5 files, every reference an
`except` branch) behind `nflreadpy` (55 files), and is what pins `pandas<2.0`.

Queued, not approved:
whether the same over-shrink affects WR/TE target shares; the `_normalise_nonnegative`
scale-discard as a deliberate ensemble change; preserving decision-gate evidence so weeks
6+ become bet-gradable; completing the wk08 replay input set so parity becomes possible;
shrinking the overstated matchup multipliers (contraindicated for TE, excess bias vs vegas
−0.14); alignment (LWR/RWR/SWR) restoration; Inside-five TD; TE-R5P. **No paid OddsAPI, no
sportsbook upstream, no W7 outcome fitting, no weakening of a truth/integrity gate.**

### COMMUNICATION / DURABILITY
Claude communication: GitHub Issue #535. Do not confuse executor local pytest counts with
independently CI-verified tests. Do not claim the user approved a merge. Update this ONE
root TOP after material milestones and archive displaced checkpoints under `docs/handoffs/`.
Per AGENTS.md §5a, results worth citing go in git, never only in an expiring artifact.

**Session context the next chat should not have to rediscover:** the user has corrected the
executor repeatedly and been right each time; the most consequential correction was that
MAE is the wrong metric for an individual-player claim. Two and a half days have produced
**one** behaviour-changing shipped fix, three retracted off-path claims caught before they
shipped, and a verified map of where leverage is **not** — including the corrected finding that the tested RB level substitution **changed 26 of 79 decisions but produced zero net side-accuracy gain (13/13 on flipped decisions)**. The user is time-pressured
with Week 8 imminent and does not want another lane change. The active lead is now **roster/existence integrity first**, then **coverage-aware team rush-budget reconciliation**, then **Bayesian evidence-dependent shrinkage**. Do not skip that order.


### V20 NEXT-CHAT RESTART PROTOCOL — DO THIS, DO NOT RECONSTRUCT FROM MEMORY
1. Read `AGENTS.md`.
2. Read this V20 root handoff completely.
3. Fetch Issue #535 comments newer than **6479259425** before doing anything else.
4. Re-fetch current `main` SHA and audit any new Claude commit before accepting its claims.
5. Continue **AUDIT TARGET 6**: establish whether the candidate existence/roster source is truly pregame-safe, and separately measure WR/TE existence loss. Do not design/promote a reconciliation until roster authority is leakage-safe.
6. Preserve the order: **roster/existence integrity → coverage-aware rush-budget reconciliation → Bayesian evidence-dependent shrinkage**.
7. Do not reopen betting, fetch paid odds, retune W7, promote MC dispersion, remove the 0.95 cap, or normalize incomplete rosters to 1.0 without explicit user authorization and preregistered evidence.
8. Every material result goes into git and #535; update this ONE root TOP and archive the displaced version.

### USER COMMUNICATION CONTRACT
The user wants plain-English translation of Claude's technical updates and an independent answer to: **Did this actually improve PRISM's named-player projections, or did we only learn something?** Always distinguish:
- **research/discovery** — what we learned;
- **production change** — what code actually changed live behavior;
- **validated improvement** — what survived honest production-path evidence.
Do not call a discovery an improvement. Do not agree blindly. Do not optimize positional MAE. Keep the named player in his actual football situation as the scientific unit.
