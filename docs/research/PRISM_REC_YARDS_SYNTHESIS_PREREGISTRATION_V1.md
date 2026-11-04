# PRISM — REC_YARDS FOOTBALL-ONLY RESIDUAL SYNTHESIS, PREREGISTRATION V1

**Frozen:** 2026-11-04, **before any arm is computed.**
**Status:** PREREGISTRATION. No result here. Nothing promoted.
**Motivation:** `PRISM_EDGE_DECOMPOSITION_EV_V1.md` — `pass_yards` is the **only** market with
positive EV (**+0.0824**, CI [+0.0042,+0.1605]) and the **only** one routed through a dedicated
player-level synthesis. It is also the only market without compression. Three independent
corroborations of one architecture.

---

## 0. Prior-work check — this is adjacent to a study that already failed

**`ensemble reweighting` is on the failed list** (V22 handoff: five estimator corrections, all
null). A residual model whose only inputs are the component projections would be close to a
reweighting and **risks repeating a known-null test.**

**The distinction that makes this a different study**, stated before running:

| | ensemble reweighting | this synthesis |
|---|---|---|
| form | convex combination of components | **additive capped correction** to the ensemble |
| inputs | components only | components **+ dispersion + football context** |
| can express | a fixed blend | *"when components disagree, correct differently"*; defense- and role-conditional shifts |

**If the fitted model's gain is explained by the component features alone (arm A1 below), this
IS reweighting repeated, and the correct conclusion is that it was already tested and is null.**
A1 exists to detect exactly that.

## 1. Research question

**Does an M89/M90-architecture residual correction improve `rec_yards`, measured on EV as well as
projection error — and does it need football context, or only the components?**

## 2. Architecture — a faithful translation of the promoted QB fit

`model/qb_pass_synthesis_v1.json`: ridge (α=20.0) on the **residual**
`actual − ensemble_proj`, with a **residual cap of 45.0 yards**, standardized features, explicit
imputer statistics and missingness indicators, trained on pre-season rows and validated
prospectively.

Translated for receiving: **`rec_synthesis_proj = ensemble_proj + clip(ridge(features), ±cap)`**,
cap preregistered at **30.0 yards** — scaled from M89's 45.0 by the ratio of market scale
(`rec_yards` mean ≈ 32 against `pass_yards` ≈ 215 would imply far less; 30 is deliberately
**generous** so the cap does not silently do the work, and it is **not** tunable).

## 3. Features — football-only, three nested arms

**AGENTS.md §3 is binding: no sportsbook variable may enter any feature.** `vegas_line`,
`vegas_over_odds`, `fair_prob`, `market_prob`, `edge_pct`, `edge_abs` are all present on the
board and all **FORBIDDEN**. M89's own 21-feature contract contains zero market features.

- **A1 — components only:** `mc_proj`, `ml_proj`, `state_proj`, `ensemble_proj`, `model_sd`,
  `ensemble_weight_mc/ml/state`. **This is the reweighting-equivalent control (§0).**
- **A2 — A1 + the decomposition:** reconstructed `pred_targets`, `pred_ypt`, `catch_rate`,
  `tgt_share`, team `plays`, `pass_rate` — the receiving analogue of M89's
  `pred_attempts`/`pred_ypa`/`off_*` block.
- **A3 — A2 + player priors and opponent context:** prior-season `tgt_share`/`ypt`, prior games,
  `rules_role` (WR1/WR1_5/other), and defense receiving-yards-allowed and pass-rate-faced — the
  analogue of M89's `qb_prior_*` and `def_*` blocks.

Reconstructed features come from the same nflverse machinery validated at **r = +0.967**
(`PRISM_D3_PHANTOM_YARDS_RESULTS_V1` M4), computed from **prior weeks only**.

## 4. Sample and split — and the limitation that bounds this study

| | |
|---|---|
| fit | **W1 + W2 + W6** 2026 (~1,600 `rec_yards` rows) |
| validate | **W7 + W8** 2026 (~1,070 rows), untouched by the fit |
| prospective | **W9**, added to the frozen board **before kickoff** if §5 passes |

**M89 trained on three full seasons (1,331 rows across 2023–25). This has three weeks of one
season.** Historical component projections do not exist for receivers —
`data/backtests/full_stack_vegas_benchmark_v1/non_qb_detail.csv` carries no `mc_proj`/`ml_proj`/
`state_proj` — so a multi-season fit is **not available at any effort short of re-running history**.

**This is therefore a thinner study than M89 by construction, and overfitting is the primary
risk, not the secondary one.** Ridge, the residual cap, three nested arms and a genuine holdout
are the whole defence.

## 5. Gates — each scoped to the arms it governs

- **M1 — all arms.** Zero sportsbook variables in the feature matrix. Asserted by name against a
  forbidden list; **failure voids the run.**
- **M2 — all arms.** No W7/W8/W9 row may enter the fit. Asserted on the fitting frame's week
  values; **failure voids the run.**
- **M3 — all arms.** At the zero-coefficient limit the synthesis must reproduce `ensemble_proj`
  exactly (≤1e-12), confirming it is a pure additive correction.
- **S1 — all arms.** On **W7+W8 held out**: per-player bias versus `ensemble_proj` must improve,
  95% CI excluding 0, bootstrapped **over players**.
- **S2 — all arms.** **EV per $1** on W7+W8, priced at the board's posted odds, must improve
  versus the shipped `model_proj`. *(EV is why this study exists; S1 alone is the error this
  programme has already made.)*
- **S3 — all arms.** No pregame tier degrades by more than 2% relative, with an **absolute floor
  of 1.0 yard** (the fix adopted after the rush study's mis-scaled bar).
- **S4 — A2, A3 only.** The arm must beat **A1** on S1, CI excluding 0. **This is the §0
  reweighting test: if it fails, the football context adds nothing and this is a repeat.**

## 6. Kill conditions

**K1** M1, M2 or M3 breached → **run void** · **K2** no arm passes S1 → the residual is not
learnable from these features · **K3** S2 fails for every arm → **projection error improves and EV
does not, which is the central tension of this programme and must be published, not worked
around** · **K4** S3 breached — name the tier and arm · **K5** **S4 fails → this is ensemble
reweighting repeated; record it as a duplicate of a known-null study and promote nothing** ·
**K6** the residual cap binds on >10% of rows → the cap is doing the work rather than the model,
and the fit is not trustworthy · **K7** fewer than 250 players in the holdout → underpowered.

## 7. Prediction

**A1 passes S1 weakly and A3 beats it** — M89's power came substantially from its `def_*` and
`qb_prior_*` blocks, and the receiving analogue should behave likewise. **S2 is where I expect
trouble:** EV improvement requires moving rows *across* the line, and `rec_yards` sits at
**−0.0328** EV with a −114 median price, so the synthesis must find **0.8pp** of win rate to
break even. **I do not expect that from three weeks of fitting data.**

**Named risk, and the one I would bet on:** with ~1,600 rows and 20+ features, **A3 will fit the
noise** and look best in-sample while failing the W7/W8 holdout. If A1 holds up and A3 collapses,
that is the signature of overfitting, not of the components being sufficient — and the honest
read is *"needs more data"*, not *"context does not matter."*

Second risk: **W1 and W2 are early-season weeks** where current-season evidence is thinnest, so
the fitted correction may encode an early-season regime that does not apply in W7+W8.

My record: **8 prediction hits, ~18 misses**, with the named risk landing in nearly every study.
**Weight the risks.**

## 8. Scope

`rec_yards` only. No production code modified. No odds fetched — prices come from the committed
boards and are used **only** to score S2, never as features. The QB synthesis, the RB P3 Week-1
route and the 0.95 cap are all untouched. A pass licenses **one** follow-up: adding a third arm to
the frozen W9 board before kickoff, plus a production validation run — **not** a pricing-path
change.
