# PRISM — REC_YARDS SYNTHESIS: FAILS EVERY GATE, AND IT WITHDRAWS "PASS_YARDS IS THE TEMPLATE"

**Date:** 2026-11-04
**Preregistration:** `PRISM_REC_YARDS_SYNTHESIS_PREREGISTRATION_V1.md`, frozen before any arm ran.
**Verdict:** **K2, K4 and K5 all fire. NOTHING PROMOTED.** The synthesis makes `rec_yards` worse
on projection error, worse on EV, and harms the top tier.
**And the motivating premise was backwards** — see §5.
**Artifacts:** `data/research/rec_yards_synthesis_v1_holdout.csv`,
`..._fit.json`, `scripts/research/evaluate_rec_yards_synthesis_v1.py`.

---

## 1. Mechanical gates pass

| gate | result |
|---|---|
| **M1** no sportsbook variable in any feature | **PASS** — forbidden list checked by name, NONE present |
| **M2** no W7/W8/W9 row in the fit | **PASS** — fit weeks `[2, 6]` |
| **M3** additive form reproduces `ensemble_proj` at the zero-coefficient limit | PASS by construction |
| **K6** residual cap binding | **0.0% on all arms** — the cap is not doing the work |

**W1 dropped out of the fit legitimately**: reconstructed features require prior weeks, and there
are none before W1. **Fit is 1,014 rows across W2+W6**, holdout 1,052 rows / 186 players — thinner
than the ~1,600 the preregistration anticipated.

## 2. S1 — the synthesis does not improve projections

Per-player bias on the held-out W7+W8, versus `ensemble_proj`:

| arm | mean \|bias\| | gain | 95% CI | S1 |
|---|---|---|---|---|
| `ensemble_proj` | **19.483** | — | — | — |
| `model_proj` (shipped) | 19.483 | +0.000 | — | — |
| `syn_A1` components | 19.742 | **−0.259** | [−1.203, +0.698] | **fail** |
| `syn_A2` + decomposition | 19.788 | **−0.304** | [−1.348, +0.727] | **fail** |
| `syn_A3` + priors/defense | 19.691 | **−0.207** | [−1.276, +0.866] | **fail** |

**K2 fires.** Every arm is point-negative. **Worth recording separately:** `model_proj` equals
`ensemble_proj` exactly on these rows, so **`rec_yards` currently receives no downstream
correction at all** — the ensemble *is* the shipped projection.

## 3. S4 — K5 fires: this is ensemble reweighting repeated

| contrast | gain | 95% CI | |
|---|---|---|---|
| A2 vs A1 | −0.0455 | [−0.3780, +0.2966] | CI spans 0 |
| A3 vs A1 | +0.0517 | [−0.4879, +0.5984] | CI spans 0 |

**The football context adds nothing over the component projections.** Per §0 of the
preregistration, that is the signature of **ensemble reweighting, which is already on the failed
list** — so this is a **duplicate of a known-null study** and is recorded as such.

**The §0 prior-work check did its job.** It was written specifically to catch this case, and
without it the honest conclusion ("we already tested this") would have been unavailable.

## 4. S2 — EV gets substantially worse

Held-out W7+W8, priced at the board's posted odds:

| arm | n | win% | **EV/$1** | 95% CI | Δ vs shipped |
|---|---|---|---|---|---|
| **`model_proj`** (shipped) | 1052 | **55.32%** | **+0.0429** | [−0.0150, +0.1006] | — |
| `syn_A3` | 1052 | 52.66% | −0.0073 | [−0.0627, +0.0500] | **−0.0502** |
| `syn_A2` | 1052 | 51.52% | −0.0290 | [−0.0858, +0.0271] | **−0.0720** |
| `syn_A1` | 1052 | 51.14% | −0.0359 | [−0.0934, +0.0214] | **−0.0788** |

**The synthesis destroys 5–8 points of EV.** S2 fails for every arm.

**A caveat that cuts against my own framing:** `rec_yards` on W7+W8 returns **+0.0429** for the
shipped model, while the full W1–W8 sample returns **−0.0328**. The holdout happens to be a
*favourable* sample for the baseline, and its CI spans zero, so "the baseline is profitable here"
is not established either. What **is** established is that the arms are clearly worse than it.

**S3:** the top tier degrades **−4.8% / −4.5% / −4.1%** against a 2% bar. **K4 fires.**

## 5. The motivating premise was backwards — `pass_yards` is not a template

`PRISM_EDGE_DECOMPOSITION_EV_V1` §7 concluded: *"`pass_yards` is the template … the next build is
a `rec_yards` synthesis in the M89/M90 mould."* **I had the causation the wrong way round.**

Relative dispersion of the realized outcome, `sd(actual)/mean(actual)`:

| market | rel. dispersion | \|err\|/mean |
|---|---|---|
| **`pass_yards`** | **0.341** | **0.251** |
| `rush_rec_yards` | 0.548 | 0.408 |
| `receptions` | 0.717 | 0.491 |
| `rec_yards` | **0.915** | 0.579 |
| `rush_yards` | 0.978 | 0.530 |

**`pass_yards` is 2.7× less noisy than `rec_yards`.** A quarterback throws ~35 passes and his
volume is close to deterministic (team pass rate × plays), so the ensemble's residual is mostly
**systematic** and a ridge correction can learn it. A receiver draws 2–8 targets from a
multinomial allocation, so his residual is mostly **luck** — and **no correction can learn luck.**

**`pass_yards` is not profitable because it has a synthesis. It has a workable synthesis because
it is a low-noise market.** This is the same conclusion as
`PRISM_IRREDUCIBLE_FLOOR_AND_ALPHA_HAIRCUT_V1` arriving from a different direction: receiving is
dominated by allocation noise, and 93% of receivers already sit at the information limit.

**Falsifiable consequence: no synthesis architecture will rescue `rec_yards`.** The ceiling is not
a modelling deficiency, it is the market. That claim should be tested by *not* spending further
effort there, and revisited only if someone finds a lower-noise decomposition of receiving.

## 6. Prediction scored

- **"A3 beats A1"** — **WRONG.** CI spans zero; context added nothing (K5).
- **"S2 is where I expect trouble … I do not expect 0.8pp of win rate from three weeks"** —
  **RIGHT**, and understated: EV fell by 5–8 points rather than merely failing to rise.
- **Named risk, "A3 will fit noise and fail the holdout"** — **RIGHT in substance**, though all
  three arms failed rather than A1 surviving. A3 does carry the largest correction sd (6.76 vs
  A1's 5.41) and is the least-bad arm, consistent with overfitting plus a weak signal.
- **Second risk, early-season regime** — untested; W1 dropped from the fit for an unrelated reason.

## 7. Where this leaves the betting side

**The redirect is more valuable than the failed arm.** Ranking markets by tractability rather than
by current performance:

| market | rel. disp | current EV | read |
|---|---|---|---|
| `pass_yards` | **0.341** | **+0.0824** | low noise, good model — **working** |
| **`rush_rec_yards`** | **0.548** | **−0.1665** | **low noise, bad model — the real opportunity** |
| `receptions` | 0.717 | −0.0167 | middling |
| `rec_yards` | 0.915 | −0.0328 | **noise-dominated — stop building here** |
| `rush_yards` | 0.978 | −0.0008 | noise-dominated |

**`rush_rec_yards` is the standout.** It is the **second most tractable market** by dispersion and
the **worst performing** — which means the loss is the *model's*, not the market's. It is also
independently the worst projection cell in the programme (−13.7% to −23.6% bias). **A market that
is predictable and that we are bad at is a far better target than one we are already near the
floor of.**

No production code modified. No odds fetched; prices came from the committed boards and scored S2
only, never as features.
