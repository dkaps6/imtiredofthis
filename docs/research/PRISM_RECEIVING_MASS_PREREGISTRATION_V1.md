# PRISM — RECEIVING MASS BIAS, PREREGISTRATION V1

**Frozen:** 2026-11-03, **before any arm is scored.**
**Status:** PREREGISTRATION. No result in this document. Nothing promoted.

---

## 0. Objective — stated first, because it determines every design choice below

This repository builds **individual-player projections**. The named player in his current
football situation is the primary scientific unit. Position-level or pooled MAE is a
**downstream diagnostic that improves as a consequence** of better named-player projections —
it is never the thing being optimised.

That is not boilerplate here. It is the direct cause of the defect this study investigates:

> `PRISM_RECEIVING_MASS_DECOMPOSITION_V1.md` found the receptions ensemble sitting at the
> **pooled-MAE optimum** (w_ml ≈ 0.5, MAE 1.658) and that this optimum **is a biased
> estimator** — every individual priced receiver projects ~0.5 receptions low. The nearly
> unbiased component (`ml_proj`, bias −0.105) holds only 34% of the weight.

The aggregate metric did not merely fail to catch the defect. **Fitting to it created the
defect.** So this study inverts the objective:

**Bias on the individual player is the primary metric. Pooled MAE is a reported guardrail and
is not permitted to decide a gate.**

---

## 1. Research question

**Does correcting the receptions/`rec_yards` bias improve projections for individual named
players — broadly across players, not just in aggregate — and is the ensemble weighting the
mechanism, or is it a simple additive offset?**

---

## 2. Sample

Graded live rows, 2026 weeks 1, 2, 6, 7, 8 (`backtest_2026_live_calibration_v1_rows.csv`).
Component projections (`mc_proj`, `ml_proj`, `state_proj`) exist only on archived boards, so
this sample cannot be extended backwards — weeks 3–5 are permanently lost (AGENTS.md §5a).

| market | rows | players | players ≥3 weeks | players ≥4 weeks |
|---|---|---|---|---|
| receptions | 636 | 224 | 124 | 83 |
| rec_yards | 641 | 226 | 127 | 83 |

**Minimum to proceed: 100 players with ≥3 graded weeks.** If not met, report underpowered and
read no gate. This is stated now because the Week 8 grading failed by reading an undersized cell.

`receptions` is primary; `rec_yards` is secondary and reported separately. **A direction
disagreement between them is a finding, not something to average away** (K5).

---

## 3. Arms

- **C1 — production**, `ensemble_proj` exactly as shipped (w_mc 0.658 / w_ml 0.341 / w_state 0.001).
- **D0 — naive offset control.** `C1 + δ`, where **δ is the pooled mean bias estimated on 2026
  weeks 1–2 only** and applied to weeks 6–8. Out-of-sample by construction.
  *This is the control that matters.* If D0 matches the reweighting arms, the finding is "add a
  constant", not "the ensemble is mis-weighted", and nothing complex should be promoted.
- **D1 — pure ML.** `ml_proj`, the least-biased available component.
- **D2 — bias-targeted blend.** `w·ml_proj + (1−w)·mc_proj` with **w chosen on weeks 1–2 only**
  to minimise |pooled bias|, then applied unchanged to weeks 6–8. No tuning on the evaluation
  weeks.

All hyperparameters (δ, w) are fit on weeks 1–2 and frozen before weeks 6–8 are scored.

---

## 4. Primary metric — per player, never pooled

For each player *p* with ≥3 graded weeks, compute his **own mean signed error**
`bias_p = mean(prediction − actual)`.

- **M1 — player-level bias:** mean over players of `|bias_p|`. Each player counts once,
  regardless of how many weeks he was priced, so high-volume players cannot carry the result.
- **M2 — breadth:** the fraction of players whose `|bias_p|` moves toward zero. A gain
  concentrated in a handful of players is not a player-level improvement.

Confidence intervals by bootstrap **resampled over players**, 10,000 iterations.

**Reported as guardrails only, never deciding a gate:** pooled MAE, pooled bias, and
per-position aggregates.

---

## 5. Gates — all must pass

- **G1 — individual bias:** arm beats C1 on **M1**, 95% CI excluding 0.
- **G2 — breadth:** arm improves **M2** for **≥60%** of players, CI lower bound above 50%.
  *This is the player-first gate. An arm that fixes the pooled number while helping only a
  minority of named players fails.*
- **G3 — MAE guardrail:** pooled MAE does not worsen by more than **0.06** (≈3.5% of 1.663) at
  the CI's unfavourable bound. A guardrail, not an objective — it can veto, never approve.
- **G4 — beats the naive offset:** arm beats **D0** on M1, CI excluding 0.
- **G5 — no heterogeneity harm:** the arm does not make `|bias_p|` worse for more than **25%**
  of players by more than 0.25 receptions. Protects the tail AGENTS.md warns against erasing.

## 6. Kill conditions

- **K1** — no arm passes G1.
- **K2** — an arm passes G1 but fails G2: aggregate gain without player-level breadth. Report;
  promote nothing.
- **K3** — G3 breached: the bias fix costs too much accuracy to be a projection improvement.
- **K4** — D0 matches the best arm within CI: the mechanism is an additive offset, **not**
  ensemble weighting. Report that plainly and do not promote a reweighting.
- **K5** — `receptions` and `rec_yards` disagree in direction on G1: report both, promote
  neither, treat the markets separately thereafter.

## 7. Side accuracy — reported, explicitly NOT a gate

`PRISM_RECEIVING_MASS_DECOMPOSITION_V1.md` §4 showed that removing this bias **degrades**
betting side accuracy (mc 54.73% → ensemble 54.11% → ml 52.36%), because the model's downward
bias happens to align with the market's own downward bias (model 2.853 < line 3.204 < actual
3.376).

**That degradation is expected and must not be read as failure.** Per AGENTS.md §3 and the
first principle, the projection is built independently of the sportsbook; tuning it to improve
side accuracy is market contamination entering through a calibration objective — which is how
this defect arrived in the first place. Side accuracy is recorded here so that a future session
sees the drop and understands it was anticipated and accepted, not overlooked.

## 8. Diagnostic — reported, no gate

Is `mc_proj` the source? The RB lane found the same simulator over-allocating rush mass and
then haircutting it (`rules_rush_share` summing to 1.2729 against a 0.95 cap). Report whether
per-team summed `mc_proj` receptions shows an analogous allocation shortfall. Hypothesis-
generating only; any follow-up gets its own preregistration.

## 9. Scope

A pass licenses exactly one follow-up: implementing the winning correction **behind a version
flag, default off**, re-running the full production validation suite, and **grading it
prospectively** before any live use. `data/model_ensemble_weights.csv` is promoted/frozen
calibration (AGENTS.md §5) and is not edited by this study.

`rush_rec_yards` — which runs uncalibrated on `mc_fallback_no_oos_weights` at −17.49 bias — is
**out of scope** and needs its own treatment. It must not be bundled in.

No sportsbook data enters any arm.
