# PRISM — IS THE `pass_yards` EDGE REAL? PREREGISTRATION V1

**Frozen:** 2026-11-04, **before any statistic is computed.**
**Status:** PREREGISTRATION. No result here. Nothing promoted. **No bet is authorized by this.**

---

## 0. Prior work — two selection approaches are already CLOSED, and are not re-tested

`PRISM_LIVE_SIDE_GRADING_CORRECTION_V1` §3 established, and this study does **not** revisit:

- **Model confidence carries no information.** `N(model_proj, model_sd)` probability is flat at
  49–57% realized across every band and badly miscalibrated — a row claimed at 88% lands at
  **53.7%**.
- **`edge_pct` carries no information.** Deciles run 51.6, 51.6, 51.6, 52.2, 58.6, 54.1, 58.0,
  43.9, 52.9, 55.1; the top decile CI is **[47.3%, 62.6%]**. Not monotone.

Its surviving conclusion was *"select on market cell."* `PRISM_EDGE_DECOMPOSITION_EV_V1` then
refined that on EV: **`pass_yards` +0.0824, everything else ≤ 0.** **This study asks only whether
that one cell is durable.**

## 1. Research question

**Is the `pass_yards` EV edge a real, repeatable edge — or a small-sample artifact?**

If real, it is a betting operation on its own: ~110 rows per week at +8% per unit. If not, the
honest answer is that **the model has no currently exploitable edge anywhere**, and that must be
said plainly.

## 2. The methodological flaw this study exists to correct

`PRISM_EDGE_DECOMPOSITION_EV_V1` bootstrapped **over rows**. `pass_yards` is **554 rows from
roughly 32 quarterbacks across 5 weeks** — rows for the same quarterback are **not independent**
(same player, same team tendencies, correlated weekly form). **Bootstrapping over rows overstates
precision.**

**Primary analysis here resamples over QUARTERBACKS (clusters), not rows.** The CI will widen,
possibly through zero. **That is the point: if the edge does not survive clustering, it was never
measured properly.** The row-level CI `[+0.0042, +0.1605]` barely excluded zero to begin with.

## 3. Tests — all pregame-computable, none re-treading §0

- **T1 — cluster bootstrap.** EV per $1 with 95% CI resampling **over QBs**. The headline number.
- **T2 — week stability.** EV per week, 5 weeks. A real edge should not be one week.
- **T3 — concentration.** EV per QB. **If two or three quarterbacks carry the whole result, it is
  not a market edge, it is a few lucky players.**
- **T4 — side.** Over versus under.
- **T5 — price bands.** EV by posted price. **Never tested.** At these margins juice dominates:
  `receptions` had a −129 median and lost while others at −114 did not.
- **T6 — the naive control.** EV of betting `pass_yards` **against** the model (always take the
  other side), and of a coin flip. If the model's side is no better than its inverse, the "edge"
  is the market's structure, not the model.

## 4. Decision rules — set now

- **G1** T1's cluster-bootstrap 95% CI lower bound is **above 0**.
- **G2** T2 shows **≥4 of 5 weeks positive**.
- **G3** T3 shows the edge is **not** concentrated: removing the single best QB leaves the CI
  lower bound above 0.
- **G4** T6's inverse-side EV is **clearly worse** than the model's side.

**All four must hold** for `pass_yards` to be called a real edge. **G1 alone is not enough** —
one favourable sample can pass a single gate.

## 5. Kill conditions

**K1** G1 fails → **the edge is not established; say so plainly and authorize no betting on it** ·
**K2** G2 fails → it is a one-week result · **K3** G3 fails → **it is a few quarterbacks, not an
edge**, and must be reported that way · **K4** G4 fails → the apparent edge is market structure,
not model skill · **K5** any T5 price band is used to define a betting rule on this sample →
**that is in-sample selection and is forbidden**; price bands are **descriptive only** here and a
rule derived from them requires its own holdout.

## 6. Prediction

**G1 fails, or passes marginally.** 554 rows across ~32 clusters is thin, the row-level CI already
nearly touched zero, and clustering will roughly halve the effective sample size. **I expect the
CI to span zero.**

**I expect G3 to be the informative failure** — that a handful of quarterbacks carry it. 554 rows
is ~17 per QB, and one QB whose yardage ran hot for five weeks could move the whole figure.

**Named risk:** I may want this to be true. It is the only positive result on the betting side and
I have just spent a session finding negatives. **If G1 passes and G2/G3 fail, the answer is NO,
not "promising."**

My record this session: **8 prediction hits, ~19 misses**, with the named risk landing nearly
every time.

## 7. Scope

`pass_yards`, 2026 W1/W2/W6/W7/W8 committed boards. **No odds fetched** — prices come from the
committed boards. **No bet is authorized by this study** regardless of outcome; a positive result
licenses one prospective test, not a wager. No production code modified.
