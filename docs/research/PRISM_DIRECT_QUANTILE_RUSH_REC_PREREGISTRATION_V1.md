# PRISM — Per-tail quantiles for rushing yards and receptions: PREREGISTRATION (V1)

> ⚠️ **CORRECTED — read `PRISM_CORRECTION_PRODUCTION_IS_NOT_CONSTANT_SIGMA_V1.md` first.**
> This document's `Q0` baseline assumes production prices from a constant σ through a
> normal CDF. **It does not.** The live path runs `run_pricing_v2.py`, which prices from
> the empirical Monte Carlo distribution and reports a per-player `model_sd`;
> `pricing_v2.SIGMA` is never referenced on the production path. Every comparison below
> is therefore against a textbook baseline, **not** against the deployed system, and its
> gains must not be quoted as gains over production. The measurements of how floors and
> ceilings scale in the data remain valid.

**Frozen before any scoring.** This is a **replication**, not a search: the model form,
estimator, evaluation and decision rule are carried over **unchanged** from
`PRISM_DIRECT_QUANTILE_PREREGISTRATION_V1` (frozen `abe11db5`, supported at `a2df5060`
for WR/TE receiving yards). Nothing is tuned per market. If a market fails, that is the
finding.

## 0. Why these two markets

Each has its own undocumented constant in `scripts/pricing_v2.py:36-43`:

| market | production σ | normal? |
|---|---|---|
| `rush_yards` | **24.0** | yes |
| `receptions` | **1.8** | yes |

The receiving-yards study showed a constant σ with a normal forces both tails to move
1:1 with the mean and makes role irrelevant, and that fitting the tails independently
fixes it (well-calibrated players 38.6% → 59.8% on naive data). Both constants above
carry the same structural defect and neither has been tested.

Preliminary structure check on **2022** (already inspected, so not a holdout, used only
to confirm the markets are shaped like the one that worked):

| market | mean down-distance | mean up-distance | low-mean down/up | high-mean down/up |
|---|---|---|---|---|
| RB `rush_yards` | 27.88 | 21.97 | 21.73 / 14.68 | 34.67 / 29.33 |
| WR/TE `receptions` | 1.43 | 1.31 | 0.92 / 0.84 | 1.92 / 1.82 |

Both show the floor further from the mean than the ceiling, and both scale with the
mean while production uses a constant. Receptions is the milder case.

## 1. Hypothesis

**H1 (per market, independently).** Fitting p20 and p80 directly, each as its own
function of the player's projected mean and role share, calibrates individual floors
and ceilings better than production's constant-σ normal — at **both** tails.

**H0.** The constant is adequate for that market, or the tails cannot be separated from
the projected mean alone. Reported as such; the markets are judged **separately** and a
win in one does not carry the other.

## 2. Data, frozen

- **Fit:** 2020, 2021, 2022.
- **Primary holdout: 2014, 2015, 2016 — never inspected for any market.** Chosen over
  2017–2019 precisely because those were spent as the receiving-yards holdout. The only
  decisive evaluation.
- **Reported, NOT decisive:** 2017–2019 and 2023–2025.
- Season length: **week 17** before 2021, **week 18** from 2021.
- Snap counts are available from 2012, so the anomaly gate is intact across all windows.
- **No sportsbook variable at any stage.**

## 3. Markets, player axis and eligibility — frozen

**RB `rush_yards`**
- `mu_hat` = prior `rushing_yards / carries` × prior `carries / game`
- `centrality` = prior share of his team's carries
- eligibility: ≥ **60** prior carries, ≥ **3** prior games with ≥5 carries, ≥ **5**
  carries in week *W*

**WR/TE `receptions`**
- `mu_hat` = prior `receptions / targets` × prior `targets / game`
- `centrality` = prior share of his team's targets
- eligibility: ≥ **20** prior targets, ≥ **3** prior games with a target, ≥ **1** target
  in week *W* (identical to the receiving-yards study)

Common: week ≥ **4**; `MIN_HOLDOUT_GAMES = 8` for per-player coverage, with sensitivity
at 6 and 10 reported.

## 4. Anomaly gate (AGENTS.md, non-negotiable)

Unchanged: `DNP_INACTIVE` excluded from prior and outcome, `PARTIAL_GAME` excluded from
outcomes, `UNRESOLVED_CONTEXT` fails closed. Counts reported; nothing silently dropped.

## 5. Models, frozen — carried over verbatim

`mu_hat` is **identical in every arm**, so only the tails vary.

- **Q0 — production:** `p20 = mu_hat − 0.8416·σ_market`, `p80 = mu_hat + 0.8416·σ_market`
  using that market's own constant (24.0 or 1.8)
- **Q1 — per-tail:** `p20 = a₂₀ + b₂₀·mu_hat`, `p80 = a₈₀ + b₈₀·mu_hat`, each minimising
  its **own** pinball loss
- **Q2 — plus role:** adds `c·centrality` to each tail

Estimator: pinball minimisation, least-squares warm start then Nelder-Mead. No
regularisation tuning, no feature search, no per-market variation. These three arms are
the entire candidate space.

## 6. Receptions is a discrete count — disclosed in advance

Receptions are small integers (2022: mean 2.83, sd 2.40; modal values 1, 2, 3).
Production's σ = 1.8 implies an interval of ±1.52 **receptions**. Per-player coverage of
an integer outcome is therefore intrinsically lumpy and **exactly 0.60 may be
unattainable**, which can suppress rule 1 for reasons unrelated to model quality.

This is stated **before** scoring so it cannot become an excuse afterwards. The rules
are **not** relaxed for it. Pinball loss is continuous in the prediction and is
unaffected by the discreteness, so rules 2–4 remain fully informative; if receptions
fails only rule 1 while passing 2–4, that is reported exactly that way and **counted as
H0 not supported for that market**, with the discreteness noted as a candidate
explanation rather than a rescue.

## 7. Evaluation, frozen — per-player, never averaged

Identical to the receiving-yards study. Averaged MAE remains **disallowed** as a primary
metric. Headline is the fraction of players whose predicted p20–p80 contains within
**±0.10** of **0.60** of their holdout games; the full distribution of per-player
coverage is reported; plus per-player pinball at p20 and p80 and the floor/ceiling bias
by mean tercile.

## 8. Decision rule, frozen — identical four, applied per market

**H1 is supported for a market only if ALL FOUR hold on the 2014–2016 naive holdout:**

1. the fraction of well-calibrated players **increases** versus Q0;
2. per-player pinball improves for a **majority** of players at **p20 AND p80**;
3. in the **high-mean tercile**, `|floor_bias|` **decreases** AND `|ceiling_bias|` does
   **not increase**;
4. `p20_hat < p80_hat` on **every** row.

If any fails, H1 is not supported for that market. No arm re-specified, no subgroup
searched, no threshold moved, no market dropped after seeing its result. Q1 vs Q2 is
decided by the same rules; if Q1 suffices, the simpler form wins.

## 9. What this study does NOT do

- does not change production; integration remains blocked on the open design question
  (how two fitted quantiles become a probability for an arbitrary line without
  reintroducing a symmetric family);
- does not alter `mu` in any arm;
- does not test `pass_yards` or `rush_rec_yards`, which also carry constants;
- does not revisit the opponent side, settled as weak in `PRISM_OPPONENT_SIGNAL_NULLS_V1`.
