# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## TOP CHECKPOINT — 2026-11-05 — V24 — **THE 2026 WEEK 9 PAID BOARD IS CERTIFIED AND IN GIT / ALTERNATE LINES WORK AND ARE PRICED / FOUR BOARD-KILLING GATES FIXED, TWO OF THEM OURS / MIN QB WITHHELD ON PURPOSE** / CLAUDE ISSUE #535

**Model:** PRISM — Player Role & Individual Situation Model. **Repo:** dkaps6/imtiredofthis.
**Branch:** `claude/nice-cori-4qkwqs`. **PRs #701-#706 are MERGED to `main`.** Full Slate is
dispatchable by anyone from `main`; nothing in this checkpoint lives only in a chat.

**V23 is superseded, not retracted.** Its findings all stand and are archived verbatim in
`docs/handoffs/ARCHIVE_NFL_RESEARCH_HANDOFF_V23_PRE_V24_2026-11-05.md`: the main line is a
near-perfect median (realized P(over) 0.476-0.508 in every quartile, so main-line side-picking
is closed for good), compression is a slope rather than a level, the measured betting
contribution is 0.56pp on `rec_yards` and 2.28pp on `receptions`, and W3-W5 2026 are closed
forensically. **Do not re-derive them.**

---

### WHAT ACTUALLY SHIPPED AND IS LIVE

**1. The Week 9 2026 paid board is certified and committed.** Run 42105885230.

| artifact in git | contents |
|---|---|
| `data/market_track_record/boards/2026_wk09.csv` | priced board, **1,538 rows after quarantine**, 30 teams |
| `data/market_track_record/alt_props/2026_wk09_full.csv` | **8,563 alternate offers** |
| `data/market_track_record/alt_props/2026_wk09_full_edges.csv` | **8,563 priced alt edges** (4,644 priceable) |
| `data/market_track_record/alt_props/2026_wk09_thu.csv` | 738 Thursday offers |
| `raw_props/` + `game_odds/` `2026_wk09*` | the paid boundaries, non-clobbering |

Board markets: `rec_yards` 516, `receptions` 512, `rush_yards` 286, `pass_yards` 116,
`rush_rec_yards` 114.

**2. ALTERNATE LINES ARE CONFIRMED SERVED BY THE PLAN.** All five keys HTTP 200 on first live
execution: `player_pass_yds_alternate`, `player_rush_yds_alternate`,
`player_reception_yds_alternate`, `player_receptions_alternate`,
`player_rush_reception_yds_alternate`. 738 rows for ONE game, 8,563 for a 15-game slate.
`player_anytime_td` is OUT of the pull by standing instruction and must stay out.

**3. `scripts/price_alt_line_offers_v1.py` prices them off realized clearance, never the
model.** Deliberate: `pricing_v2._prob_over` is a Normal approximation wrong by up to +0.448
in the tails, which is where every alt line lives.

---

### THE FOUR GATES THAT KILLED PAID RUNS, AND WHO CAUSED THEM

Two were real gates catching real problems. Two were ours. Stated that way on purpose.

**(a) THEIRS — provider name spelling. Cost TWO paid runs.** Ourlads says "Matt Hibner";
nflverse and the books say "Matthew Hibner" (gsis 00-0040879, BAL TE). Run 42094118645 died in
`repair_live_prop_identity`; run 42099277230 died five steps later in
`materialize_live_props_for_model` on the same name. Fixed three ways:
- three verified rows in `data/manual_name_overrides.csv` — Hibner, **Joshua->Josh Palmer**
  (BUF, 00-0036988), **Mitch->Mitchell Tinsley** (CIN, 00-0038839). All three are nflverse
  `player_display_name`, confirmed across 2025 AND 2026 weekly stats, which is the exact field
  `player_form_v2` builds `player_game_logs` from. NOT roster `full_name`, NOT `football_name`
  (which says "Mitch").
- `scripts/audit_provider_name_parity_v1.py` runs in Full Slate **BEFORE the paid fetch**,
  strict only when a paid fetch is imminent. It found Palmer and Tinsley, both Week 9 teams,
  before they cost anything.
- a namespaced `il:` initial+surname fallback in `_infer_team` only, which cannot widen the
  quarantine or override matchers and still fails closed on a cross-team surname collision.

**(b) THEIRS — the MIN starter conflict. The gate saved a bad bet.** Ourlads/C2 certify
**Kyler Murray** QB1 and he started weeks 3-8 (29/37/30/24/29 attempts). DraftKings AND FanDuel
post **only Carson Wentz**, at 207.5, and post no Murray offer at all. The nflverse week-9
injury feed has **no MIN rows**, because MIN plays Monday and designations publish Friday.
Nothing reconciles them, so per the AGENTS.md anomaly gate the state is `UNRESOLVED_CONTEXT`.
Our row projected **229.45 against 207.5** — a +21.9 yard "edge" on a QB whose team's
established starter has no prop posted. **Withheld via
`data/manual_final_board_quarantine.csv` (MIN, 2026, wk9); NO starter override was added.**
A Friday-or-later re-run resolves it. Same pattern as the Wentz MIN wk2 row and four other QBs.

**(c) OURS — the alt pricer was validated against a schema production does not write.**
It died on `KeyError: 'line'`. `outputs/props_raw.csv` is the GROUPED artifact, with per-book
lines inside `offers_json` and no `line` column; the W8 archive it was tested against is the
older FLAT shape. `flatten_main()` now reads either and fails loudly on neither. Also fixed
there: `cleared_conservative` is a one-sided DOWNWARD haircut, so `1 - cleared` gave UNDERs no
haircut at all and made every UNDER outrank every OVER on `ev_conservative`, the column the
board sorts by.

**(d) OURS — removing `player_anytime_td` orphaned a gate that required it.**
`validate_anytime_td_model_status_v1` demanded >=1 priced ATD row. Now: not in `MARKETS` + zero
rows -> labelled `ATD_MARKET_NOT_OFFERED_BY_CONFIGURATION`; **in `MARKETS` + zero rows -> still
fails closed**; market list unreadable -> fails closed. `validate_certified_full_slate_stack_v1`
GAINED an assertion: a not-offered claim is only valid with zero priced rows.

---

### DURABILITY DEFECTS FOUND AND FIXED — AGENTS.md 5a

1. **The archiver's two-file allowlist** meant 738 paid alt rows existed only in a 90-day
   artifact. It now carries `alt_props/` and `alt_edges/` and Full Slate's own label-bearing
   filenames.
2. **The archiver clobbered `${ARCH_STEM}.csv`**, so a Sunday capture would have silently
   destroyed Thursday's `game_odds`/`raw_props` boundary. First capture keeps the canonical
   stem (replay/parity tooling reads that exact name); later ones get `_partN`.
3. **`workflow_dispatch` added to the archiver** so a past run's stranded artifact can be
   re-archived. Used to rescue run 42094118645.
4. **`boards/` is a union-by-key ledger, so a quarantine could not propagate.** The certified
   1,538-row capture merged back to 1,544 and **republished the withheld Wentz row,
   indistinguishable from a certified one**. The season/week-scoped final-board quarantine is
   now enforced on the ledger after the merge, using the existing matcher.

---

### OPERATIONAL TRAPS — READ BEFORE DISPATCHING ANYTHING

- **Merging to `main` cancels an in-flight paid Full Slate run.** `concurrency` is
  `full-slate-${{ github.ref }}` with `cancel-in-progress`, and it fires when the competing run
  is **created**, not when it starts. Cancelling the push run afterwards is too late.
  **Do not merge while a paid run is in flight.**
- **The zero-credit replay path works and should be the default for a re-run.**
  `reuse_paid_odds_run_id` + `_artifact_id` + `_artifact_digest`, `fetch_live_odds=false`. It
  verifies run id, artifact id, digest and expiry before use. It requires
  `outputs/props_raw_compact.csv`, which only exists if the compact step succeeded, so a run
  that died before it cannot be replayed.
- **Credits: ~193 remaining** after Week 9 (~109 used on the full-slate main-market fetch).
  A Sunday main+alt run for the remaining games is roughly 160.
- **`[LIVE_ODDS_AUTHORIZED]` in a push commit message to `main` spends paid credits**, since
  `e0e6c832`. `test_full_slate_workflow_contract::test_full_slate_main_push_defaults_to_no_credit_mode`
  is RED because of it and was left red deliberately — it is a user decision, not a bug to
  paper over. **Never put that marker in a commit message.**
- Four tests fail on a clean checkout and are NOT ours: that contract test, two `test_ml_v2`
  cases, `test_slot_role_unreachable_v1`. They reproduce with all V24 changes stashed.

---

### NOT DONE — DO NOT PRESENT THESE AS FINISHED

- **The 25 positive-EV-conservative alt rows are NOT a bet list.** They concentrate in
  `receptions` at 0.5/1.5 main lines on long prices; the cell inspected had `n=122, players=77`
  with `post_hoc_line_restriction=True`; and one row is `Kirk Cousins rush_yards OVER 0.5`,
  the table applied to a population it was never fit on. **A per-cell table-support audit
  (n, players, the post-hoc flag) is REQUIRED before any money.** Not run.
- **Week 9 is not graded.** Grade after Monday 2026-11-09 from `boards/2026_wk09.csv`.
- **MIN pass yards is absent from the Week 9 board by design**, not by failure.
- **`injuries_source_status.json` still mislabels practice-only pulls as `official_report`.**
  Root cause known since V22: NFL publishes game designations Friday, so `status` is null
  midweek and `_injury_limited` can never fire. Tracked as P2
  `injury_questionable_untreated`; it is also why (b) above could not be resolved.
- **`anytime_td` must stay out of any ticket construction.**

### IMMEDIATE NEXT ACTIONS, IN ORDER

1. Grade Week 9 after Monday from `boards/2026_wk09.csv` + `alt_props/2026_wk09_full_edges.csv`.
   This is the first slate where alt edges can be graded at row level.
2. Run the **per-cell alt table-support audit** before treating any alt row as bettable.
3. If a Sunday/Friday re-run is wanted: dispatch Full Slate from `main` with
   `fetch_live_odds=true`, `fetch_alt_lines=true`, an `event_window` covering only the
   remaining games, and a fresh `odds_capture_label`. Check `~193` credits first.
4. Re-examine MIN QB once Friday designations publish; remove the wk9 quarantine row only if
   the football authority actually supersedes Murray.
5. Then resume the V23 queue (archived): `rush_rec_yards` ensemble is fitted and live;
   the volume-conditional and shrunk-volume-conditional lanes are CLOSED NEGATIVE.


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
