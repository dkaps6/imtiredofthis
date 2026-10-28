# PRISM — complete work ledger, 2026-10-26 → 10-28

Built from the commit record and the promotion flags, not from recall. Every row is
traceable to a commit or an artifact. Written because the lane turned several times and
the volume became hard to track.

**One-line summary:** three behaviour-changing merges shipped, two changes are built but
deliberately inert, two candidate fixes are closed negative, four significant claims were
retracted, and the one lane that matches the project's stated objective — player-specific
evidence weighting — is measured and unstarted.

---

## 1. SHIPPED — live in production

| change | commit | what it does | evidence |
|---|---|---|---|
| **`PERCENTY_COLS` rate-scale fix** | `9aa6058c` | `coverage_man_rate`, `coverage_zone_rate`, `middle_open_rate`, `middle_closed_rate` arrived as percents (24.0–64.6) and were compared against **fraction** thresholds, so the `middle_open` gate fired for **all 32 teams** instead of 13 — a constant, not a discriminator | supported multiplier **1.035** vs production's **1.265** |
| **Replay input preservation** | `eb109a6a` | archives the model-side pricing inputs each run, so a pricing replay stops being impossible | wk08 sets now landing; confirmed working |
| **5 identity overrides** | `6cedd3c3` | reconnects prior-game history for players who were being projected from a **generic position prior** because the Ourlads spelling never joined the nflverse name | each verified by `gsis_id`; validator 25 rows / 0 round-trip failures |
| Dual promotion authority | `40cb909e` | safety only: an artifact can no longer activate itself without the in-code flag agreeing | symmetric gate, tested |

The identity overrides: `Deandre Carter → DeAndre Carter` (11-yr vet), `Hollywood Brown →
Marquise Brown` (7-yr vet), `Drew Ogletree → Andrew Ogletree`, `Audric Estim → Audric
Estimé`, `Ben Vansumeren → Ben VanSumeren`. **5 of 396 players; only Estimé (140 prior
opportunities) and Brown (102) are materially affected.**

## 2. BUILT BUT INERT — promotable only if the named blocker clears

| change | flag | blocker |
|---|---|---|
| **MC dispersion calibration** (`b34f85aa`) | `CALIBRATION_REQUIRED_FOR_LIVE_PRICING = False`; artifact `CANDIDATE_PENDING_PARITY_REPLAY` | needs an **exact production-path baseline-parity replay**. The W7 fixture's ML/State diagnostics are stripped, so parity is **UNAVAILABLE BY SOURCE**. Needs a complete replay set from a **live-odds** run |
| **RB carry concentration** γ=1.75 (`c82825d3`) | `RB_CARRY_CONCENTRATION_PROMOTED = False`, effective γ=1.0 identity | **K4**: mass-preserving, so it moves ~10.3 yd from backs 2+ to the lead back while *every* rank is biased low. Also now **superseded** — the missing mass was never inside the RB distribution |

Neither changes a live projection. Both have dual-authority gates and tests asserting the
candidate constant alone cannot move a number.

## 3. CLOSED NEGATIVE — not in the running

| item | why it closed |
|---|---|
| **Fix A — RB carry-concentration sharpening** | **K1**: γ on raw trailing shares fitted at **0.76**, not >1 (2013–2025, 6,296 team-weeks, walk-forward sd 0.022). Raw trailing shares are *more* concentrated than reality |
| **Fix B — ensemble reweight for `rush_yards`** | **K2**: the naive equal-weight control also passed the gates, so the finding is "the incumbent is stale", not "ML is right". **K3**: the pure-ML candidate's bias flips sign across test weeks. Also `_normalise_nonnegative` divides away the scale correction NNLS actually asks for (raw coefs sum **1.5165**) |
| **Three opponent-side signal nulls** | banked (`76fbde37`) so no future session re-runs them |
| **Player dispersion as a player property** | H1 not supported; per-player volatility only weakly persistent (`r≈0.25`) |
| **"Dispersion is a constant"** | the *claim* is retracted as off-path (see §5); the measurement that motivated it was against a textbook baseline |

## 4. OPEN — measured, unstarted, or blocked

Ordered by how directly each serves the project's stated objective.

| # | item | status |
|---|---|---|
| **1** | **Bayesian shrinkage is count-based, capped, and generic** | **measured, not started.** `_posterior_mean` gives the position prior a **constant** weight (`GROUP_STRENGTH=3.0`) and weights a player's own history by **game count, capped at 6**. Measured per player on the live universe: the generic position prior is **33.2% mean / 25.0% median / 20.0% floor** of a `rush_share` projection and **35.0% / 25.0% / 20.0%** of a `tgt_share` projection. No player, however established, can get below **20%**. A 7-year starter's prior season counts the same as a 6-game sample |
| 2 | **Roster existence union** | designed, approved by the auditor, unbuilt. **Ourlads is the sole authority on player *existence***: W7 chain `roles_ourlads` 458 → status 458 → availability 458 → eligible **396** → universe **396**. The availability layer **never adds a row** |
| 3 | **Team rush-budget reconciliation** | designed, not built, and **must not precede (2)**. `team_modeled_mass_target = 1 − unresolved_roster_rush_mass`, residual state-derived rather than a hard-coded 0.05 |
| 4 | **Canonicalizer suffix gap** | **29 of 489** players with 2026 wk1–7 carries/targets (5.9%) retain a suffix in the clean key; 4 realised mismatches at W7. Breaks **grading** joins too. Open question with the auditor on name-collision risk before touching shared identity plumbing |
| 5 | `_normalise_nonnegative` scale-discard | code-level finding, pinned by 6 characterization tests, no change proposed |
| 6 | **NextGen Stats entirely unused** | queued/untested. Works for 2026 wk1–7. Carries `expected_rush_yards`, `rush_yards_over_expected`, `percent_attempts_gte_eight_defenders`, `avg_separation`. Target-week rows are **outcomes** — any use must be strictly-prior and walk-forward |
| 7 | W7/W8 parity replay | blocked on a complete **live-odds** replay set. wk08 sets read `"complete": false` **correctly** — those runs had no odds, so `metrics_ready` does not exist |
| 8 | Decision-gate evidence preservation | so weeks 6+ become bet-gradable |

## 5. RETRACTIONS — four significant, plus corrections

These matter more than most of the positive findings, because each one was load-bearing.

| # | retracted claim | truth |
|---|---|---|
| **1** | the rate-scale guard `ab29a096` fixes the TE multiplier | the guard sat in a module the workflow **never calls**. Real fix was `PERCENTY_COLS` |
| **2** | production prices from a **constant sigma** through a normal CDF | it prices from the **empirical Monte Carlo sample**; `pricing_v2.SIGMA` is never referenced on the live path. **9 docs bannered**, contract test added. Invalidates every `Q0` quantile comparison as "better than production" |
| **3** | production's top-1 RB share is **0.2226** (duplicate `(player, market)` rows eat the top-5 pool) | off the production path. `_build_full_universe` builds from `player_form_consensus`, **not** `metrics_ready`, and raises on duplicates |
| **4** | **"team RB carry volume is right"** (22.59 vs 23.11, −0.52) | **FALSE.** On the production universe, running the real `simulate()`: **19.77 vs 23.15, bias −3.38 [−6.36, −0.53]**. This was the most consequential retraction — it is *why* the lane chased allocation instead of mass, so **both** candidate fixes were chosen under a false premise |

Smaller corrections, all accepted: three selection artifacts in the RB diagnosis (filtering
on the outcome); "fixing the level changes no decision" → **changes 26 of 79, zero net
gain**; "empirical budget = 1.0" → **conditional on a complete roster**; "Bayes is why mass
inflates" → **pre-Bayes mass is already invalid, Bayes amplifies it**; stacked-box
terminology; and a doc that wrongly said the 2026 NGS feed was unavailable (it conflated
NGS with **participation**).

## 6. DURABLE MEASURED FACTS — cite these, don't re-derive them

**RB `rush_yards`, live 2026 wk 1/2/6/7, n=182:**

| component | bias | MAE |
|---|---|---|
| `mc_proj` | −16.15 | 25.38 |
| `ml_proj` | **+0.79** | 26.40 |
| `state_proj` | −3.68 | 26.66 |
| final `model_proj` | −8.46 | 24.39 |
| vegas | −2.43 | **22.78** |

- excess bias vs market on identical rows **−6.03 [−8.03, −4.03]** (selection-immune)
- bias by within-team rank is negative at **every** rank: **−22.34 / −5.21 / −5.13**
- team rush-share sum **1.2729** (28/28 teams over the 0.95 cap), uniform haircut **0.7574**
- existence ceiling: **7.41%** of carries, **4.16%** of targets outside the universe
- production probabilities are overconfident: **Brier 0.2800 vs market 0.2491**
- floors scale **1.7–2.4× slower** than ceilings in all three markets, era-stable
  (walk-forward, twelve origins) — but every "beats production" comparison is **void** per
  retraction #2
- generic position-prior share of a projection: **rush 33.2% / targets 35.0%**, floor 20%

## 7. Standing rules that produced these results

Preregister and freeze before scoring. **Include a naive control** — K2 exists only because
one was in the frozen set. Averaged MAE is never the primary read for a player-level claim.
Follow the import graph from the workflow to what is actually called before claiming what
production does — three of four retractions came from skipping that. Betting remains
**CLOSED**; no paid odds fetch without explicit authorization.
