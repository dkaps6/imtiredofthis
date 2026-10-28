# PRISM — RB `rush_yards` ensemble reweight (fix B): PREREGISTRATION (V1)

**Frozen before any candidate was scored.**

Fix B from `PRISM_RB_RUSHING_LEVEL_DIAGNOSIS_V1.md`. Fix A is banked as
`PRISM_RB_CARRY_CONCENTRATION_RESULTS_V1.md`: it fired K1 as frozen, located the real
cause (the Bayesian layer over-shrinks RB rush shares by ~9pp), and established that a
mass-preserving allocation fix **cannot** move a level because every within-team rank is
biased low. B is the level fix.

## 1. The defect, and why the incumbent weights cannot address it

`data/model_ensemble_weights.csv` gives `rush_yards` **mc 0.5570 / ml 0.4430 / state
0.0**, fit on `all_2024_oos_frozen_for_2025` (3,207 rows, calibration MAE 12.59). On live
2026 rows, the component it leans on hardest is the one that has drifted:

| component | bias | MAE |
|---|---|---|
| `mc_proj` | **−16.15** | 25.38 |
| `ml_proj` | **+0.79** | 26.40 |
| `state_proj` | −3.68 | 26.66 |
| `ensemble_proj` | −9.06 | 24.47 |
| vegas line | −2.43 | 22.78 |

The weights were fit before that drift. `fit_market_weights` minimizes squared error via
`LinearRegression(fit_intercept=False, positive=True)`, which favours `mc_proj` because
MC has the **best MAE of the three despite the worst bias** — good discrimination, wrong
level. Squared-error fitting will keep choosing it.

**Structural note, frozen as a stated expectation and not as a result:**
`_normalise_nonnegative` divides the fitted coefficients by their sum, so the blend is a
strict convex combination with no intercept. Any *scale* correction NNLS discovers is
therefore divided away, and the attainable bias is bounded by the least-biased component.
That bound is not binding here — `ml_proj` at +0.79 is reachable at `ml_weight = 1.0` —
but it means the ensemble has no mechanism to correct a component's level drift beyond
reweighting. Whether NNLS actually wants a scale correction here will be **measured and
reported as a diagnostic**, and it will not alter any candidate.

## 2. Candidate set — frozen, no additions after scoring

| id | weights | free parameters |
|---|---|---|
| **C0** | incumbent: mc 0.5570 / ml 0.4430 / state 0.0 | 0 (baseline) |
| **C1** | NNLS refit via the promoted `fit_market_weights`, fit on **weeks 1–2 only** | 3 |
| **C2** | pure ML: mc 0 / ml 1 / state 0 | 0 |
| **C3** | equal weight: 1/3 each | 0 |

C3 is a **naive control**. If C3 also beats C0 then "any change helps" and the result is
about the incumbent being stale, not about ML being right. Including it is the guard
against claiming a mechanism that the data does not single out.

## 3. Train/test separation — frozen

- **Fit:** 2026 weeks **1–2** only (`rush_yards`, n=89 rows with all three components and
  an actual). C1 is the only candidate that consumes it; C0/C2/C3 have no fit.
- **Test:** 2026 weeks **6–7** only (n=79). Never used for fitting.
- Weeks 3–5 do not exist: those boards were permanently lost to a 7-day retention setting
  (AGENTS.md §5a).

**Disclosed, and it limits the strength of the conclusion:** `ml_proj`'s +0.79 bias was
measured on all four weeks, weeks 6–7 included. So the *motivation* for C2 has seen the
test rows even though C2 itself fits nothing. This is therefore a **strong consistency
test, not a blind one**. The genuinely blind confirmation is prospective — week 8 onward.
No amount of care here converts weeks 6–7 into clean holdout, and it will not be
described as such.

## 4. Metrics — frozen

**Primary, both reported, neither alone decisive:**

1. **signed bias** on the test rows — does the candidate put the level in the right place;
2. **side accuracy versus the sportsbook line** — fraction of rows where
   `sign(model_proj − vegas_line) == sign(actual − vegas_line)`. This is the decision that
   actually gets made, and it needs **no distributional assumption**.

**Guardrail:** test MAE. Reported, and a candidate that fixes bias while materially
degrading MAE does not pass.

**Averaged MAE is explicitly NOT a primary metric** and no claim will rest on it, per the
standing instruction that a position-average cannot settle an individual-player question.

**Brier score is deliberately NOT used.** Production prices `p_over` from the empirical
Monte Carlo sample (`np.mean(adjusted_outcomes > line)`), and the archived rows carry only
`model_sd`, not the sample. Scoring a probability here would require assuming a normal CDF
— the exact textbook-baseline error corrected in
`PRISM_CORRECTION_PRODUCTION_IS_NOT_CONSTANT_SIGMA_V1.md`. Side accuracy is the
assumption-free substitute.

**Uncertainty:** paired bootstrap (4,000 draws, seed 42) on test-row differences versus
C0, since all candidates are evaluated on identical rows.

## 5. Markets — frozen

- **`rush_yards`** is the primary and the only promotion candidate.
- **`rush_rec_yards`** is secondary/exploratory (n=63 fit / 53 test). It inherits the
  rushing defect and is reported, but it is **not** a promotion candidate at this n.
- `rec_yards`, `receptions`, `pass_yards` are reported **for context only**. Their weights
  are separately promoted lineages and this study does not touch them.

## 6. Promotion gates — frozen

A candidate may be promoted for `rush_yards` only if **all** hold on weeks 6–7:

- **G1** test signed bias strictly closer to zero than C0's, with a paired bootstrap CI
  on the improvement that excludes zero;
- **G2** test side accuracy not worse than C0's;
- **G3** test MAE not materially worse than C0's (tolerance: within 1.0 yard);
- **G4** C3 (equal weight) does **not** also satisfy G1–G3; if it does, the finding is
  "the incumbent is stale", no specific candidate is promoted, and the matter goes back to
  a historical refit;
- **G5** no sportsbook variable enters any weight or fit. Lines are used only as the
  benchmark in the side-accuracy metric and in reporting (AGENTS.md §3).

## 7. Kill conditions — frozen

- **K1** no candidate satisfies G1 → bank as a null; the incumbent weights stand.
- **K2** C3 passes the gates → "any change helps"; promote nothing, escalate to a
  historical refit on generated component predictions.
- **K3** the winning candidate's advantage over C0 reverses between week 6 and week 7 →
  unstable at this n; bank as suggestive and wait for week 8.
- **K4** required component columns unavailable → fail closed, do not substitute.

## 8. Write target — frozen

The fit writes a **research** weights file under `data/research/`, following the precedent
of `scripts/research/fit_2023_ensemble_weights_v1.py`, which states it "does not touch
`data/model_ensemble_weights.csv`". Production's frozen artifact is modified **only** if
the gates pass, and then as an explicit versioned migration with its lineage recorded, per
AGENTS.md §5.

## 9. Scope

No sportsbook variable enters any weight or fit. Betting remains **CLOSED**. No paid odds
fetch is authorized or performed by this work.
