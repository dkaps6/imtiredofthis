# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## TOP CHECKPOINT — 2026-11-04 — V23 — **THE BETTING LAYER WAS ANCHORED ON THE WRONG NUMBER AND IS NOW FIXED / THE LINE IS A NEAR-PERFECT MEDIAN / THE MODEL ADDS A LITTLE, IN THE TWO MARKETS WE CAN TEST / `rush_rec_yards` HAS NO CALIBRATED ENSEMBLE AT ALL / W3-W5 CLOSED FORENSICALLY** / CLAUDE ISSUE #535

**Model:** PRISM — Player Role & Individual Situation Model. **Repo:** dkaps6/imtiredofthis.
**Branch:** all V22/V23 work is on `claude/nice-cori-4qkwqs` (PR #697 open, unmerged).
**Production changes: ZERO.** Every file touched is under `scripts/research/`, `docs/research/`,
`docs/production/` or `data/research/`. The one exception remains V22's
`scripts/audit_2026_production_readiness.py` injury-contract gate.

**V22's findings all stand.** The ceiling (93% of players at the information limit, the alpha
under-rated −0.0392), universal compression with three sources, allocator scale invariance, the
opportunity-vs-efficiency shrinkage split, and the three code defects (D1/D3 validated, D2
withdrawn) are unchanged and archived in
`docs/handoffs/ARCHIVE_NFL_RESEARCH_HANDOFF_V22_PRE_V23_2026-11-04.md`. **Do not re-derive them.**

---

### WHAT V23 ESTABLISHED — AND IT IS THE FIRST RESULT THAT CHANGES A BET

`docs/research/PRISM_ALT_LINE_ANCHOR_RESULTS_V1.md` (preregistered first, in
`..._PREREGISTRATION_V1.md`). **MARKET-ASSISTED layer. No football projection touched.**

**1. The sportsbook main line is a near-perfect median, in every volume quartile.**
Realized `P(actual > line)` sits at **0.476–0.508** across all four line quartiles of every market,
2024+2025. This independently explains the V22 edge decomposition's −0.0211: **main-line
side-picking is unbeatable by construction**, and that lane is closed for good.

**2. The model's projection is compressed in volume, and it is live in production.**
Median `model_proj / vegas_line` by line quartile on the 2026 boards:

| market | Q0 low | Q1 | Q2 | Q3 high | spread |
|---|---|---|---|---|---|
| `rec_yards` | 1.366 | 1.035 | 0.961 | 0.827 | **53.9pp** |
| `rush_yards` | 1.221 | 1.070 | 0.817 | 0.790 | 43.1pp |
| `rush_rec_yards` | 1.078 | 0.821 | 0.807 | 0.730 | 34.8pp |
| `receptions` | 1.116 | 0.922 | 0.854 | 0.791 | 32.5pp |
| **`pass_yards`** | 1.077 | 1.026 | 1.012 | 0.933 | **14.4pp** |

`pass_yards`, the one market with a promoted synthesis, is again the clean control at a quarter the
spread. **This is V22's compression finding measured against the line instead of against outcomes.**

**3. The model's probabilities are catastrophically miscalibrated on volume.** Model `p_over`
against realized over-rate, 2024+2025: `rec_yards` gap +0.260 in Q3, **`rush_rec_yards` +0.448** —
the model asserts a **3.9%** chance of the over where reality is **48.7%**. **Never bet off
`p_over`, `fair_prob` or `edge_pct` on a high-volume player.**

**4. The committed alt-line table was anchored on the projection and is DEPRECATED.**
`data/research/alt_line_clearance_curve_2024_2025_v1.csv` expressed thresholds as a fraction of the
**projection**, so a fixed fraction meant a different real threshold for every player. Measured on
held-out 2025, `rush_yards` at `frac_of_proj = 0.6`:

| volume | line ~ | realized | promised 70.4% | true break-even vs the −238 quoted |
|---|---|---|---|---|
| Q0 | 8.5 | **60.9%** | −9.5pp | **−156 — we were overpaying** |
| Q3 | 69.0 | **90.4%** | +20.0pp | **−943 — we were refusing it** |

`rush_rec_yards` Q3 at the same fraction clears **96.0%**, a true break-even of **−2379** against
−473 quoted. **The safe tier the model kept failing to find was being filtered out by the table's
own anchor.** It also failed decile calibration at **8.92pp**.

**5. SHIPPED: the market-anchored, volume-stratified replacement.**
`data/research/alt_line_market_anchored_table_v1.csv` (150 cells, ≥100 rows and ≥40 players each),
bands in `data/research/alt_line_quartile_boundaries_v1.csv`, per-bet lookup in
`docs/research/PRISM_ALT_LINE_DECISION_CARD_V1.md`. **It needs no odds pull** — the main line comes
off the app. Internal check: at `frac = 1.00` clearance is **0.497–0.561** in every quartile, the
line behaving as a median exactly as (1) says.

**6. The model DOES add something to bet selection — in the two markets with power to test it.**
Δ Brier (B3 − B2) over players, 2025: `receptions` **−0.00418** [−0.00545, −0.00305] and
`rec_yards` **−0.00102** [−0.00134, −0.00070], the only two markets clearing the 150-player bar,
both PASS, with `rec_yards` holding prospectively on 2026. The three that fail
(`pass_yards`, `rush_yards`, `rush_rec_yards`) are all K5-underpowered, so their failure is weakly
evidenced. **I predicted this would fail. It did not — the first time in this programme the model
beat my expectation of it.** The effect is **0.5–2% relative** and is **not** shown to convert to
profit; do not build a tier on it.

### THE OPEN PRODUCTION DEFECT V23 FOUND — THIS IS THE NEXT FIX

**`rush_rec_yards` has no calibrated ensemble.** `data/model_ensemble_weights.csv` carries
`pass_yards`, `rush_att`, `rush_yards`, `rec_yards`, `receptions` — **and no row for
`rush_rec_yards`**, which therefore ships `ensemble_status = uncalibrated_mc_only`,
`method = mc_fallback_no_oos_weights`, i.e. **raw Monte Carlo with zero weight on ML and State**,
both of which are computed and present.

Consequence on the 2026 boards: `proj − line = **−14.74**`, the model says OVER on **19.2%** of
rows where truth is ~49%. Composing it from the two **calibrated** components recovers only
**2.3 of the 14.4 yards**, so missing calibration is necessary but not sufficient — the players on
that board are team alphas, and **−12.1 / 71.9 = −16.8% matches V22's alpha haircut of −16.2%
almost exactly.** Those two findings are very likely the same defect seen from two directions.

It is also the **worst-EV market in the programme (−0.1665)** and the second most *tractable* by
relative dispersion. **A market that is predictable and that we are bad at is the right target.**

### WHAT V23 DID NOT ESTABLISH — THE HALF THAT IS MISSING

**The table gives the price you need. It does not give the price the book offers.** The committed
boards carry **main-line odds only**; no alt-line price exists anywhere in this repository.
Converting the card into realized profit needs either the user reading offered alt prices off the
app, or an **explicitly authorized** alt-line odds acquisition. **This is a data question, not a
modelling one, and it is the single remaining blocker on the money path.**

Two disclosed defects in what shipped: the stratified arm **misses decile calibration by 0.38pp**
(3.38 vs a 3pp bar), so the **conservative CI-low** break-even is the operative number; and
**`receptions` is parameterised wrongly** for an integer market (fractional thresholds collapse;
Q1 does not exist) and is **omitted from the card** until rebuilt on absolute thresholds.

### CLOSED FORENSICALLY — W3/W4/W5 2026, DO NOT RE-INVESTIGATE

`docs/production/MISSING_WEEKS_3_5_FORENSIC_CLOSURE_V1.md`. **Nothing was deleted**; git shows only
ADDs. **W3's pricing failed mechanically** so no board was ever produced (run `36330757181`; its
one surviving artifact is the 90-day RB PD2 research capture). **W4 and W5 boards did exist** —
runs `36934563481`/`36935917903` and `37852811339`/`37860592615`, artifact ids and sha256 digests
recorded in the handoffs — and `list_workflow_run_artifacts` now returns **`total_count: 0` for all
four**. Root cause: `full-slate.yml` carried split retention, **90 days for the research capture and
7 for the bundle holding the board**, and that `7` survived until **2026-10-19**, when `a6068d6a`
raised it and `584782f7` added archive-on-pricing-failure. W6 was preserved by hand the same day.
**That one date explains the entire shape of the archive.** Five of eight 2026 slates are gradeable
at row level: W1, W2, W6, W7, W8. W4/W5 lines came from paid acquisitions that cannot be refetched.

### ALREADY RUN — DO NOT REPEAT

Everything on V22's failed list, plus:

- **`rec_yards` M89-style residual synthesis** — `PRISM_REC_YARDS_SYNTHESIS_RESULTS_V1.md`.
  **Failed K2/K4/K5** on 1,014 rows of 2026 W2+W6. Its §4 limitation ("a multi-season fit is not
  available at any effort") **was wrong** — `docs/research/overnight/clean_v1_full_stack_vegas_benchmark_detail.csv`
  has 17,715 rows over 2024+2025 with all four components, lines, both sides' odds and outcomes.
  **A refit on that data is a legitimately different study and is still open**, but note its §5
  argument (receiving is noise-dominated at rel. dispersion 0.915) and that V23 found the *level*
  is fine in production for `rec_yards` (−1.85, 48.2% over) — the compression is in the *spread*,
  not the mean, so a mean-correcting synthesis is aimed at the wrong moment.
- **Alt-line anchor choice** — settled above. The anchor is the **line**. Do not re-test it.
- **Main-line side-picking** — closed twice now, by EV decomposition and by (1) above.

### IMMEDIATE NEXT ACTIONS, IN ORDER

1. **Fit and promote a `rush_rec_yards` ensemble row** following the existing
   `ENSEMBLE_WEIGHT_2023_HOLDOUT_FIT` lineage pattern (nonnegative OOS linear blend, fit one
   season, blind holdout the next). Preregister first. This is a **real production fix with a
   measured cost** and it is the highest-value open item.
2. **W9 is unplayed** — `data/market_track_record/w9_prospective_frozen_v1.csv` (478 players) is
   frozen and carries **projections only, no lines**. The D1 injury overlay is still pending on it
   (`load_injuries` max week was 8 at capture). Grade per the frozen rule in
   `PRISM_W9_PROSPECTIVE_CAPTURE_V1.md` after Monday night.
3. **Rebuild `receptions`** on absolute integer thresholds so it can go back on the card.
4. **Resolve D1's root cause** — needs runtime instrumentation of a live run; the archived trace
   copy is lossy and the builder does emit all 11 columns with `status` populated. Cause is OPEN.

### PROMOTED AT V22

**Nothing.** No coefficient, weight, feature, artifact schema or pricing path changed. `D1--D3`
for receiving is *ready* and awaits the W9 holdout.

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
