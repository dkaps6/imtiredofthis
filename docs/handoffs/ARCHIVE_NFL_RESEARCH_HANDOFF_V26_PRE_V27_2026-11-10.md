# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## TOP CHECKPOINT — 2026-11-10 — V26 — **D1 TESTED AND SPLIT: OUT/DOUBTFUL ZEROING SHIPPED TO PRODUCTION, EDGE CLAIM REJECTED / THE INJURY STATE CONTRACT IS FIXED AND THE CHAIN IS VERIFIED END TO END**

**Model:** PRISM — Player Role & Individual Situation Model. **Repo:** dkaps6/imtiredofthis.
**Branch:** `claude/nice-cori-4qkwqs`, synced to `main`.

**V25 is superseded, not retracted.** Archived verbatim at
`docs/handoffs/ARCHIVE_NFL_RESEARCH_HANDOFF_V25_PRE_V26_2026-11-10.md`. Its alt-line
findings, the two distribution defects and the anti-findings all stand and are
summarised in §5–§7 below. **V25 said D1 was "untested, not rejected" and listed the
W10 capture as the step that would unblock it. Both are now wrong.**

---

### 1. PRODUCTION CHANGED — the first production change in several checkpoints

`scripts/modeling/simulation_rules.py`: `OUT_DOUBTFUL_OPPORTUNITY_RETENTION = 0.0`,
previously `0.50`. A player carrying OUT / DOUBTFUL / IR / PUP now receives **zero**
modeled target and rush share.

**The next Full Slate board is therefore not directly comparable to weeks 1–9 for any
aggregate diagnostic.** Expect lower team opportunity mass and lower positional means.

Evidence (`PRISM_D1_W9_HOLDOUT_RESULTS_V1`): OUT retention **0.0000** over 364 prior
cases with **0.0% played**; DOUBTFUL **0.0000** over 23 cases, 0.0% played. The earlier
study reached the same place independently — A2 (0.00) beat A1 (0.50) by +0.00257
[+0.00139, +0.00392], CI excluding zero.

At 0.50, production projected a mean **17.74 receiving and 8.55 rushing yards for 24
players the league had publicly ruled out**, every one of whom recorded 0.0. Those 27
OUT/DOUBTFUL players are 5.6% of the universe and contribute **7.4% of total receiving
absolute error**. Promoted as a **correctness repair**; no statistical test is required
to call that wrong.

Pinned by `tests/test_out_doubtful_zero_opportunity_v1.py` (12 tests).

---

### 2. THE INJURY STATE CONTRACT IS FIXED, AND THE CHAIN IS VERIFIED WITH REAL DATA

`build_injuries_weekly` used to label **any** non-empty pull `official_report` — it
branched on `if not df.empty` and never checked whether a row carried a designation.
Clubs publish practice participation Wed/Thu and the game designation **Friday**, so a
midweek pull said "official report", `status` was null, `_injury_limited` never fired,
and every unavailable player kept full opportunity. **Production's implied retention was
1.00 even though its code said 0.50.**

Now: `classify_report_state()` decides from the rows. New state **`practice_only_report`**.
`VALID_STATES` pins four values. The three validators that hard-coded the old two-value
set and would fail closed — `validate_2026_provider_artifacts`,
`validate_full_slate_data_quality_v1`, `repair_injuries_nflcom_v1` — all accept it, and
the first now **raises** when a status claims `official_report` with zero designations.
`audit_2026_production_readiness` gains a P0 for that mislabel and for any unknown state.
19 tests in `tests/test_injury_report_state_contract_v1.py`.

**Verified end to end on real week 9 data, not by reading code:** builder emits 362 rows
/ `official_report` / 154 designations → `context_bridge` populates `injury_status` and
`injury_designation` → `_injury_limited` fires for Out and Doubtful and correctly does
**not** fire for Questionable. **Week 9 would have zeroed 81 OUT/DOUBTFUL players.**

---

### 3. D1 — TESTED. THE HOLDOUT DID NOT NEED TO WAIT, AND THAT MISTAKE COST THREE WEEKS

A valid holdout needs a universe built from weeks ≤8, retention calibrated on weeks ≤8,
real week 9 designations and week 9 actuals. **All four already existed.** Calibrating on
≤8 and testing on 9 is out-of-sample; the original study's defect was fitting *and*
testing on W7+W8, not a missing frozen file. **"Prospective capture" and "valid holdout"
are not the same thing.** Do not repeat this.

**Grading population is the whole story.** Played-only (n=314, snap denominator) leaves
12 designated players, **all QUESTIONABLE**, because every OUT and DOUBTFUL player is
excluded for not playing — D1's main effect is structurally invisible and K4 fires. Full
universe (n=478, non-producers at actual 0) has 48 designated across all three tiers and
is the correct population for a projection claim.

Full universe: `d1d3` improves receiving MAE 13.391 → **11.929**, RMSE 20.957 →
**19.975**, rushing bias +1.111 → **+0.031**. **D1 improves RMSE, which D3 alone never
did** (its RMSE was flat at 22.846 vs 22.841). D3 wins MAE by shrinking the middle; D1
fixes the tails.

**The edge claim is rejected and K1 fires.** Receiving side accuracy reads 54.8% → 58.1%,
but `d1d3` vs base is **12 of 18 disagreements (binomial p = 0.238)** and **only 2 of
those 18 involve a designated player** — the other 16 are undesignated players flipping on
residual-mass perturbation of the multinomial draw. Only **8 of 186** priced receiving
rows are designated at all, because books pull lines for players who are OUT.

**Carry this forward: simulation noise moves more side calls than the arm does.** A
single-week side-accuracy comparison on re-simulated universes has a noise floor wide
enough to swallow a real effect.

**NOT promoted: the QUESTIONABLE multiplier** (0.8535 targets / 0.5718 rush). n=21,
2 of 18 disagreements, and the designation is applied inconsistently by club — the
preregistered named risk. Needs weeks, not one slate.

**D3 remains rejected** on its own holdout (`PRISM_D3_W9_HOLDOUT_RESULTS_V1`). Note
`d1d3` beats `d1` on both markets, so the joint arm stays under test.

---

### 4. KNOWN INCONSISTENCY, PINNED BY TEST RATHER THAN QUIETLY FIXED

An OUT **WR1 alpha** still routes through `_injury_target_overrides`, which keeps half
his share and redistributes the rest 60/30/10. **So an OUT WR1 retains opportunity while
an OUT WR2 now gets zero.** That legacy rule is separately promoted, has its own test
(`test_alpha_injury_redistribution_preserves_legacy_60_30_10`), and **redistributes** —
the D1 study's arm A3, which lost half of A2's gain with a CI crossing zero. Replacing it
needs its own preregistration. `test_out_alpha_still_uses_the_legacy_redistribution_rule`
keeps it visible.

**Also measured:** teammates of an OUT player are **not** under-projected (rec bias +0.06
vs +0.50 for teams with nobody out) because `_allocate_counts` uses a residual bucket
rather than redistributing zero-sum. But the **0.95 share cap** means zeroing an OUT
player relieves a squeeze, so teammate bias moves +0.06 → **+1.79**. A real cost,
outweighed in aggregate. The 0.95 cap stays untouched (V20 §7 protects it).

---

### 5. ALT LINES — SELECTION RULES, ENFORCED IN CODE (unchanged from V25)

`scripts/research/build_alt_cards_v1.py`, 24 tests, plain-English statement for the user
at `docs/research/YOUR_RULES_PLAIN_ENGLISH.md`. In football order: **on the field**
(`snap_pct >= 0.30`; below 40% snaps the sign inverts to −66.2% ROI) · **real role**
(absolute `main_line` floor; sub-10 receiving lines record zero 29.4% of the time) ·
**not against ourselves** (`cushion = model_proj / alt_threshold >= 1.5`; below 1.0
returns −24.6%) · **real alternate** (`frac_of_line` outside [0.85, 1.15], ≤ 1.5) ·
**own evidence** (games played, not games with a stat) · **matchup** (toughest quartile
−1.8%, softest +16.9%) · **availability** · **no leg shared across tiers**.

First grading against real prices (`PRISM_ALT_FIRST_PRICED_GRADING_2026_W9`, 4,306 rows,
13 games): blind −13.4%; `rec_yards` the only profitable market (+2.9%), `receptions`
worst (−30.0%); **no price band wins blind**. The old backtest never referenced
`offered_american`, so no profitability claim had ever existed.

---

### 6. TWO LIVE DISTRIBUTION DEFECTS — STILL OPEN

**(a) `model_sd` is far too narrow.** Graded weeks 2/6/7/8 (n=1,447): the 0–10% band hits
**39.5%**, the 70–100% band hits **49.3%**. Everything collapses toward 46%. Measured
`actual/model_proj` for `rec_yards` runs 0.47 at p25 to 1.67 at p75 with an **11.4%
literal-zero rate** the simulation does not represent. **Every `edge_pct` the board emits
inherits this.**

**(b) High-end ceiling compression.** TOP-quartile bias −12.5 (`rec_yards`) and −7.3
(`rush_yards`), where the market line ran **+8.1** on rushing TOP — PRISM is **15.4 yards
worse than the line** exactly where the big tickets are. Largest W9 misses: Lamb −128.0
(proj 67.0, actual 195), Nacua −71.8 (proj 78.2, actual 150).

---

### 7. ANTI-FINDINGS — DO NOT RE-DERIVE (unchanged from V25)

Long parlays are **not reachable ex ante** (best scheme went 19/22; the published
+1,818,000 was winners sorted by price) · **price-ranking breaks tickets** (12/15, and it
re-selects the low-main-line players the role floor exists to kill) · **cushion-ranking is
correct but caps payout** (hit rate and ROI both *rise* with cushion; 15 legs went 15/15
at +1,935) · **alt coverage had a 16% hole** clustered by kickoff window — Bijan Robinson
was never rejected by a filter, he had no ladder to reject.

---

### 8. IMMEDIATE NEXT ACTIONS, IN ORDER

1. **The OUT WR1 alpha rule** (§4). The evidence is already in hand; it needs a short
   preregistration because it replaces promoted behaviour. Highest value, smallest scope.
2. **Finish grading Week 9** — SNF (TB@CHI) and MNF (BUF@MIN) are final and were excluded
   from the 13-game alt grading.
3. **Per-cell alt table-support audit** (n, players, post-hoc flag). Still REQUIRED before
   treating any alt row as bettable, still not run. Everything in §5 ran ahead of it.
4. **Friday 2026-11-13: run `capture_prospective_board_v2.py --week 10`** after
   designations publish, before Sunday kickoff. **Now for accumulating QUESTIONABLE weeks,
   not for unblocking D1.** K4 (<20 designated) is the binding constraint, so accumulate
   weeks rather than expect one slate to settle it. TNF WAS@NYG is excluded automatically.
5. **Resolve the MIN QB wk9 quarantine row** — Murray stayed Out, Wentz started.
6. Then: widen simulation dispersion to the measured residual spread per market and
   projection tercile, and model the bust rate explicitly. Re-audit every `edge_pct`
   consumer after.

---

### 9. PROMOTED AT V26

**`OUT_DOUBTFUL_OPPORTUNITY_RETENTION = 0.0`** in `scripts/modeling/simulation_rules.py`,
as a correctness repair. Nothing else. The alt selection gates are research tooling under
`scripts/research/`, not a production path. `AGENTS.md` §3 holds throughout: `cushion`
uses `model_proj` **ordinally** against a rung, and no sportsbook variable constructs any
projection anywhere in this work.

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
