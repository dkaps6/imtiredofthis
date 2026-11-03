# PRISM — RECEIVING MASS BIAS, PREREGISTRATION V2

**Frozen:** 2026-11-03, **before any arm is computed.**
**Supersedes:** `PRISM_RECEIVING_MASS_PREREGISTRATION_V1.md`, which stopped at its own
sample-size check (`PRISM_RECEIVING_MASS_RESULTS_V1_UNDERPOWERED.md` — 44 players against a
minimum of 100). **No arm was scored under V1**, so no outcome informs this revision.

---

## 0. Objective — unchanged, and still the reason for every choice below

This repository builds **individual-player projections**. Pooled or position-level MAE is a
downstream diagnostic that improves *as a consequence* of better named-player projections; it is
never the optimisation target.

That is causal here. `PRISM_RECEIVING_MASS_DECOMPOSITION_V1.md` found the receptions ensemble
sitting at the **pooled-MAE optimum**, and that optimum **being a biased estimator** — every
individual priced receiver projects ~0.5 receptions low, while the nearly unbiased component
(`ml_proj`, bias −0.105) holds 34% of the weight. **Fitting to the aggregate metric created the
defect.**

So: **bias on the individual player is primary. Pooled MAE is a guardrail that may veto, never
approve.**

---

## 1. The only change from V1: leave-one-week-out cross-fitting

V1 split fit (weeks 1–2) from eval (weeks 6–8). That left a three-week evaluation window, so a
≥3-week per-player bar required a player to be priced in every eval week — only 44 qualified.

**V2 instead cross-fits.** For each graded week W ∈ {1, 2, 6, 7, 8}:

> fit the arm's hyperparameters on **all graded weeks except W**, then apply them unchanged to
> week W.

Every row is therefore evaluated **out-of-sample**, no row contributes to its own fit, and each
player accumulates up to five evaluated weeks.

**Sample under this protocol (receptions): 636 rows, 224 players, 124 with ≥3 graded weeks.**
Minimum to proceed remains **100 players with ≥3 weeks** — met. If a run shows otherwise, report
underpowered and read no gate.

Everything else below is carried over from V1 unchanged.

---

## 2. Arms

- **C1 — production.** `ensemble_proj` as shipped (w_mc 0.658 / w_ml 0.341 / w_state 0.001).
- **D0 — naive offset control.** `C1 + δ`, δ = mean bias on the held-in weeks.
  *The control that matters.* If D0 matches the reweighting arms, the mechanism is an additive
  constant, not mis-weighted components (**K4**).
- **D1 — pure ML.** `ml_proj`. No fitting, so cross-fitting is a no-op for this arm.
- **D2 — bias-targeted blend.** `w·ml_proj + (1−w)·mc_proj`, w chosen on the held-in weeks to
  minimise |pooled bias| over a grid w ∈ {0, 0.05, …, 1.0}.

---

## 3. Primary metric — per player, never pooled

For each player *p* with ≥3 evaluated weeks: `bias_p = mean(prediction − actual)`.

- **M1 — individual bias:** mean over players of `|bias_p|`. **Each player counts once**,
  so high-volume players cannot carry the result.
- **M2 — breadth:** fraction of players whose `|bias_p|` moves toward zero versus C1.

Bootstrap **resampled over players**, 10,000 iterations.

Guardrails, reported but never deciding a gate: pooled MAE, pooled bias, per-position aggregates.

---

## 4. Gates — all must pass

- **G1 — individual bias:** arm beats C1 on **M1**, 95% CI excluding 0.
- **G2 — breadth:** arm improves **M2** for **≥60%** of players, CI lower bound above 50%.
  *The player-first gate: fixing the pooled number while helping a minority fails.*
- **G3 — MAE guardrail:** pooled MAE does not worsen by more than **0.06** at the CI's
  unfavourable bound. May veto, never approve.
- **G4 — beats the naive offset:** arm beats **D0** on M1, CI excluding 0.
- **G5 — no heterogeneity harm:** the arm does not worsen `|bias_p|` for more than **25%** of
  players by more than **0.25** receptions.

## 5. Kill conditions

- **K1** no arm passes G1 · **K2** G1 passes but G2 fails (aggregate gain, no breadth) ·
  **K3** G3 breached · **K4** D0 matches the best arm within CI (mechanism is an offset;
  promote no reweighting) · **K5** `receptions` and `rec_yards` disagree in direction on G1
  (report both, promote neither).

`receptions` is primary; `rec_yards` secondary and reported separately.

## 6. Side accuracy — reported, explicitly NOT a gate

Removing this bias is **expected to degrade** betting side accuracy (decomposition §4: mc
54.73% → ensemble 54.11% → ml 52.36%), because the model's downward bias shadows the market's
own (model 2.853 < line 3.204 < actual 3.376). **That drop is anticipated and accepted.**
Tuning the projection to protect it would be market contamination entering through a
calibration objective — which is how this defect arrived. Recorded so a future session sees the
drop and knows it was foreseen.

## 7. Diagnostic — reported, no gate

Whether per-team summed `mc_proj` receptions shows an allocation shortfall analogous to the RB
lane's rush-mass defect. Hypothesis-generating only.

## 8. Scope

A pass licenses one follow-up: implement behind a version flag, default off; full production
validation suite; **prospective grading before any live use**.
`data/model_ensemble_weights.csv` is promoted/frozen calibration (AGENTS.md §5) and is not
edited by this study. `rush_rec_yards` (uncalibrated, −17.49 bias) is **out of scope**.

No sportsbook data enters any arm.
