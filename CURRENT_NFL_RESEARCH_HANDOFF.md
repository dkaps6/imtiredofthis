# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## TOP CHECKPOINT — 2026-11-04 — V22 — **THE CEILING IS MEASURED: 93% OF PLAYERS ARE AT THE INFORMATION LIMIT, THE ALPHA IS NOT / COMPRESSION HAS THREE SOURCES / THREE CODE DEFECTS FOUND, ONE COMBINATION READY / NOTHING PROMOTED** / CLAUDE ISSUE #535

**Model:** PRISM — Player Role & Individual Situation Model. **Repo:** dkaps6/imtiredofthis.
**Branch:** all V22 work is on `claude/nice-cori-4qkwqs` (107 commits ahead of `main`, PR #697 open).
**Production changes: ZERO.** Every file touched is under `scripts/research/` or `docs/research/`.

### WHAT V22 ESTABLISHED — THE CEILING, AND WHY NINE STUDIES CAME BACK NULL

`docs/research/PRISM_IRREDUCIBLE_FLOOR_AND_ALPHA_HAIRCUT_V1.md`

A model that knows every player's **true** target rate exactly still scores **M1 = 0.03245**,
because it is graded against one game and multinomial luck is irreducible. Production (E1) scores
**0.04203**. **Real headroom is +0.00958, not the +0.01690 previously claimed** — ORACLE_BOTH
scored *below* the physical floor, and re-run on simulated data where it can only fit noise it
returns 0.02505 against its 0.02513. **~43% of that headroom was realization-fitting.**

**Three of four tiers are at or below the luck floor** (5–12% has +0.0013 of room, 12–20%
+0.0037). **That explains the entire history of null results**: five estimator corrections and
four signal studies were aimed at a pooled metric where **600 of 769 rows were already at the
information limit.** They were not bad studies. They were studies of a solved problem. It also
explains why a naive opportunity count beat every sophisticated arm in three of four studies — at
the noise floor the naive estimator is correct and sophistication only adds variance.

**Scored against true rate, with no realization noise and no outcome selection:**

| player's TRUE rate | n | E1 − TRUE | \|E1 − TRUE\| |
|---|---|---|---|
| <5% | 452 | +0.0043 | **0.0068** |
| 5–12% | 149 | +0.0044 | **0.0172** |
| 12–20% | 107 | −0.0066 | **0.0212** |
| **20%+** | **61** | **−0.0392** | **0.0440** |

**The model is excellent for 93% of players and under-rates each team's true alpha by −0.0392 —
16% of his rate, ~1.1 targets per game.** Mechanism: phantom players occupy *relative* share, so
the alpha's allocation probability is reduced by exactly (1 − waste). Predicted −0.0390 against an
observed −0.0393.

**Also corrected at V22:** 63% of the previously-reported −0.1009 top-tier under-allocation was a
regression artifact from binning on outcome. Real gap +0.0274.

### COMPRESSION IS UNIVERSAL — AND HAS THREE INDEPENDENT SOURCES

`PRISM_ALPHA_COMPRESSION_BOARD_CHECK_V1.md`, `PRISM_ROLE_SHRINKAGE_FIX_RESULTS_V1.md`

Tiered **pregame-only**, `(model − vegas)/vegas`: **10 of 10 role × market cells compress,
monotone in every one.** WR was the mild case; **TE `rec_yards` +58.8pp and RB `rush_yards`
+73.9pp** are worse. `rush_rec_yards` is the worst projection cell (−13.7% to −23.6%).

**The natural experiment:** `pass_yards` is the only market that bypasses generic shrinkage
(M89/M90 synthesis) and the **only** market without the defect — **+9.4pp and ±1% bias.** It is
now a **positive control**: an arm that damages it is disqualified.

The three sources:
1. **`bayesian_v2._posterior_mean`** — `GROUP_STRENGTH` is a fixed weight that never decays, and
   `PRIOR_PLAYER_CAP` truncates the player's own history at 6–10 games. At Week 8 a receiver with
   a full prior season has 17 prior games of which **6 count**, so the role constant permanently
   holds **18.75%** of every estimate. The target is **not** `DEFAULTS` — it is
   `_weighted_group_mean`, the `prior_games`-weighted position mean (WR **0.144–0.148**, RB 0.064,
   TE 0.122).
2. **`rules_v2.matchup_multipliers`** — multiplies zero-sum shares with **no renormalisation**.
   Ceilings: **WR1 1.000 (floor 0.95), TE 1.265, RB 1.320.** WR1 is the only role that cannot
   exceed 1.0. Live values recovered from production: **WR 1.0000, QB 1.0000, TE 1.2560, RB 1.1956.**
3. **Opportunity-share estimation itself** (the first two feed it).

### THE KEY STRUCTURAL FACT — THE ALLOCATOR IS SCALE-INVARIANT

`PRISM_ALLOCATOR_CAP_SCALE_INVARIANCE_V1.md` — **read this before touching any share work.**

`_allocate_counts` gives probability **`share_i / team_sum`** once the cap binds. **Any uniform
per-team rescale is a mathematical no-op**, and team opportunity is pinned at 95% of attempts for
any sum ≥ 1.0. Confirmed in the live harness: team targets span **0.3 across eight arms**.

**This retracted three earlier V22 claims:**
- the "+0.5574 of excess mass competing for removal" framing — **there is no excess mass**; the cap
  discards it. Team sums (1.5110 vs realized 0.9536) are vector properties with **no production
  consequence**;
- D1's A2-vs-A3 result — **identical in 54/54 team-weeks at 0.000e+00.** The
  discard-vs-redistribute lesson was wrong;
- D2's five arms are **two** production models: `{C0,M1,M3}` and `{M2,M4}`. M1 was not a weaker
  fix, it was **no fix**.

Only **relative** shares reach the simulation. All compression findings survive, because they
measured relative shape.

### THE SPLIT THAT UNIFIED NINE NULLS

**Opportunity shrinkage is wrong. Efficiency shrinkage is CORRECT.** No arm passed G1 on `ypt`,
`ypc` or `receptions_per_target`, and every arm harmed a tier; shrinking a player's efficiency
toward his own prior-season efficiency was the single worst result in the study. **Efficiency
genuinely does not persist**, so the group prior is doing real work there. **Do not reopen
efficiency shrinkage.**

**Best estimator arm: A4 / λ=1.00 — the role-STATE anchor** (position × depth-rank, not bare
position). `tgt_share` bias **0.01079 → 0.00798, a 26% reduction**, 86.6% breadth, compression
−55.3%, no tier harmed, and it halves the team-sum excess. **Gain is linear in λ**, so there is no
cheap sweet spot.

**A5 — the literal "player has the rules" arm (shrink toward the player's own prior) — LOST to A4
on non-overlapping CIs and was worst on efficiency.** The winning unit is the player's **current
role-state**, neither the bare position nor his own thin history. Refinement, not refutation.

**But A4 costs side accuracy: −1.13pp (V1), −0.88pp (V2).** The P-versus-S frontier is real and
measured twice. **Nothing promotes on one objective alone.**

### THREE CODE DEFECTS — MEASURED, WITH ONE COMBINATION READY

| | defect | status |
|---|---|---|
| **D1** | **`injuries.csv` ships 5 of 11 columns** — no `status` field, so `_injury_limited` returns False for every player in every week. The injury rule is **dead code** and an OUT player keeps **100%** of his share, sometimes boosted (Tyler Warren 0.1957 → 0.2476 while Out). 124 OUT skill players, **0.0% played, retention exactly 0.0000.** QUESTIONABLE: 64.3% played, **0.8535 target / 0.5718 rush** retention. | **fix validated**, inert in production until the artifact is repaired |
| **D2** | matchup multipliers break the share constraint. Share fix gains +0.00441 but on **yards scores −0.128** — it moves opportunity to higher-`ypt` WRs and receiving was already 9.1% high. | **DO NOT SHIP** |
| **D3** | sd floor + clip at zero give **2.40 phantom receiving yards at zero targets**, 1.19 rushing. | **fix validated** (A1, the zero-opportunity gate) |

**READY: `D1--D3` for receiving.** Per-player gain **+1.015 [+0.569,+1.524]**, ratio
**1.091 → 1.045**, bottom tier over-projection **−55.8%**, no tier past the 2% bar. D1 and D3 are
**synergistic** (+0.069), not independent.

**BLOCKED: D3 for rushing.** Production's rushing aggregate is **0.984 — accidentally right**,
because phantom yards offset a genuine under-projection. D3 takes it to **0.860**. The 14% gap
decomposes into **three equal factors**: attempts **0.959** (26.0 model vs 27.1 realized), the cap
**0.941**, `ypc` **0.953**. Positional reallocation recovers **+0.9%** and is nearly a dead end —
because the model's `ypc` is ~4.2 for everyone against realized **RB 4.44 / QB 3.55**, so fixing
*who* carries does not change yards. `PRISM_RUSH_ALLOCATION_RESULTS_V1.md`.

### PROCESS — EIGHT SPECIFICATION ERRORS, ALL CAUGHT, AND THE FIX

Allocator V1's non-mass-preserving transform · the board check's lookahead tiering · ROLE_SHRINKAGE
V1's copied G7 · V1's unexercised G5 · V2's infeasible G7 threshold · D1's G2 contradicting its own
arm · the rush study's near-zero-denominator 2% bar · **the rush study omitting kill conditions
entirely.**

Two fixes adopted and both worked: **scope every gate to the arms it governs** (D3 was the first
clean run), and this **mandatory preregistration checklist** — research question · arms · gates ·
**kill conditions** · prediction with named risks · scope.

**Prediction record: 8 hits, ~17 misses. The named RISK landed in nearly every study.** The
preregistered risk statement is the productive element of this methodology; the predictions are
close to worthless. Keep writing both; weight only the risks.

### IMMEDIATE NEXT ACTION — THE PROSPECTIVE WEEK, AND IT EXPIRES AT KICKOFF

**Every V22 study failed to promote for the same reason: no holdout.** W7+W8 only; **W3–W5 2026
are permanently lost** (AGENTS.md §5a). **Week 9 is unplayed as of 2026-11-04.**

1. **Capture a W9 prospective board NOW, before kickoff** — production vs `D1--D3`, frozen and
   committed. This is the holdout that unblocks promotion and it is the only item with a deadline.
2. **Repair `injuries.csv`**: `build_injuries_weekly` must emit its declared 11-column SCHEMA, and
   `audit_2026_production_readiness.py` must assert it so a file without `status` **fails closed**.
   Production repair, no preregistration needed. **D1 does nothing until this lands.**
3. **`ypc` is position-blind** — a third of the rushing gap and the reason correct allocation does
   nothing. A player-identity problem, not a calibration one.
4. **The 0.95 cap** costs **5.9% on every market.** ⚠️ **It is already protected** — V20 §7 forbids
   removing it without explicit user authorization and preregistered evidence. Treat as
   authorization-gated, not unnoticed.

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
