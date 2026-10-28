# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## TOP CHECKPOINT — 2026-10-28 — V19 — **ROOT CAUSE FOUND: PLAYER RUSH SHARES ARE NEVER RECONCILED INTO A TEAM OPPORTUNITY BUDGET** / "TEAM VOLUME IS RIGHT" **RETRACTED** / CLAUDE ISSUE #535

**Model:** PRISM — Player Role & Individual Situation Model. **Repo:** dkaps6/imtiredofthis.

**Current state:** the RB `rush_yards` level defect is now fully diagnosed and **both**
candidate fixes from the parent diagnosis are closed negative under their own frozen kill
conditions. One real cause is located upstream, with a confidence interval. Nothing new is
promoted. Live pricing is unchanged by anything in this lane.

**Read in this exact order**:
1. AGENTS.md (canonical 2026 operating contract, player-first first principle).
2. **ONLY THIS NEWEST ACTIVE TOP V18.** Do not recursively read prior V2–V17 checkpoints.
3. **`docs/research/PRISM_RB_RUSH_SHARE_MASS_BUDGET_V1.md` — THE ROOT CAUSE. Read before
   anything else in the RB lane; it retracts the premise both earlier fixes were chosen under.**
4. `docs/research/PRISM_RB_CARRY_CONCENTRATION_RESULTS_V1.md` — fix A, and the shape half.
4. `docs/research/PRISM_RB_ENSEMBLE_REWEIGHT_RESULTS_V1.md` — fix B, and why reweighting
   cannot work.
5. `docs/research/PRISM_CORRECTION_PRODUCTION_IS_NOT_CONSTANT_SIGMA_V1.md` — still binding.
6. Displaced V17 root archived at
   `docs/handoffs/NFL_HANDOFF_2026-10-28_V17_PRE_V18_ROOT_ARCHIVE.md`.
7. GitHub Issue https://github.com/dkaps6/imtiredofthis/issues/535 — read
   **[comment 6477740922](https://github.com/dkaps6/imtiredofthis/issues/535#issuecomment-6477740922)**
   FIRST. It is a deliberately **standalone state-of-play catch-up**: a recycled context can
   start there without scrolling the other 967 comments. Then fetch anything after it.
   GPT ↔ Claude Code work channel.

**Most important immediate next task:** reproduce the **team rush-share mass budget**
measurement on a prospective live-odds universe, then design the reconciliation. Raw
`rules_rush_share` sums to **1.2729** per team (28/28 teams over the 0.95 cap), so
`_allocate_counts` applies a uniform **0.7574** haircut and RB carries land at **19.77 vs
23.15 actual (−3.38)**. Do NOT simply remove the cap — pre-cap mass is **over** by +3.26.
The defect is that per-player shares are never reconciled as a team allocation.
Preregister before scoring, as always.

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
**Active:** Bayesian RB rush-share over-shrink (the only live lead). Queued, not approved:
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
shipped, and a verified map of where leverage is **not** — including the now-measured fact
that fixing the RB rushing level does not change a bet decision. The user is time-pressured
with Week 8 imminent and does not want another lane change. The Bayesian over-shrink is the
lead precisely because it is upstream of both failed fixes and is a player-identity fix.
