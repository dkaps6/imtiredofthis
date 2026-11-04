# PRISM — ENSEMBLE METHOD: NOTHING PROMOTED, AND IT REDIAGNOSES THE DEFECT

**Date:** 2026-11-04
**Preregistration:** `PRISM_ENSEMBLE_METHOD_PREREGISTRATION_V1.md`, frozen before any arm ran.
**Verdict:** **S4 fails in every market and S1N fails in two. NOTHING PROMOTED. No code changed,
no weight changed.**
**And it converts "the model is 25–40% low" into something sharper: the error is a SLOPE, not a
LEVEL.** See §3 — that is the result.
**Artifacts:** `data/research/ensemble_method_v1_results.json`,
`scripts/research/evaluate_ensemble_method_v1.py`.

---

## 1. Mechanical gates and the control

| gate | result |
|---|---|
| **M1** no sportsbook variable as feature or target | **PASS** — target `actual`, features `mc_proj`/`ml_proj`/`state_proj`; 19 forbidden columns absent |
| **M2** no 2025 row in any fit | **PASS** — every fit frame is 2024 only |
| **M3** forcing C2 to sum to 1 reproduces C1 | **PASS** — max \|Δ\| **0.00e+00** across 5 markets |
| **M4 — POSITIVE CONTROL** | **PASS** — `pass_yards` C2 vs C0 per-player MAE **+0.1585 [−1.2910, +1.5535]**, CI spans 0, **not damaged** |

**K2 did not fire.** The arm was not disqualified globally — it was refused on its own merits,
which is a stronger result than being killed by the control.

**`rush_att` was not evaluated and was not touched**, as named risk 4 required: the benchmark
carries five markets and `rush_att` is not among them.

## 2. The constraint is doing real work

**S2 and S3 pass decisively in all four non-QB markets. The level diagnosis was right.**

| market | C2 weight sum | level C0 → C2 | S2 gain | S3 gain (vs refit) |
|---|---|---|---|---|
| `rush_yards` | **1.4197** | 0.7499 → **1.0301** | **+22.0pp** | +24.4pp |
| `rush_rec_yards` | 1.2315 | 0.7773 → **0.9572** | **+18.0pp** | +18.0pp |
| `rec_yards` | 1.1833 | 0.8181 → **1.0232** | +15.9pp | +11.2pp |
| `receptions` | 1.1873 | 0.8503 → **1.0466** | +10.3pp | +7.2pp |
| `pass_yards` | 1.0599 | 0.9539 → 1.0150 | +4.0pp — fails the 5pp bar | fails |

The regression really does want to scale the components up 18–42%, and removing the normalisation
really does fix the mean. **And it is still refused**, because of what it costs:

| market | per-player MAE C0 → C2 | Δ 95% CI | S1N |
|---|---|---|---|
| `rec_yards` | 18.6380 → 19.5799 | **+0.9419 [+0.6121, +1.2758]** | **FAIL — significantly worse** |
| `receptions` | 1.3542 → 1.4184 | **+0.0642 [+0.0322, +0.0964]** | **FAIL — significantly worse** |
| `rush_rec_yards` | 26.2614 → 26.9956 | +0.7343 [−0.4596, +1.9115] | pass (ns) |
| `rush_yards` | 18.3545 → 18.9683 | +0.6138 [−0.1451, +1.3470] | pass (ns) |

**And S4 fails in every single market.**

## 3. THE RESULT: the error is a slope, not a level

Per line quartile, C0 → C2 relative MAE change. **Every market has the same shape.**

| market | Q0 (low volume) | Q1 | Q2 | Q3 (high volume) |
|---|---|---|---|---|
| `rec_yards` | **+12.1%** | +4.0% | +2.5% | **−0.8%** |
| `receptions` | **+8.7%** | — | +1.6% | **−1.2%** |
| `rush_rec_yards` | **+11.6%** | +8.5% | −2.4% | **−8.4%** |
| `rush_yards` | +9.0% | **+12.9%** | −4.7% | **−6.1%** |

**A uniform scale-up helps the high-volume players and hurts the low-volume ones, monotonically.**
That is not a level error being corrected — it is a **slope** error being slid along.

**This is V23's compression finding arriving from a third direction and it closes the question.**
V23 measured `model_proj / vegas_line` at **1.366** in `rec_yards` Q0 and **0.827** in Q3. The
low-volume players were never low — **they were already 37% HIGH.** Scaling everyone up by 18%
necessarily makes them worse. A single multiplicative constant cannot fix a population that is
over-projected at one end and under-projected at the other; it can only choose which end to be
wrong about.

**C3 settles the remaining possibility.** An additive intercept is no better: `rec_yards` 19.7590
per-player MAE against C2's 19.5799, `rush_yards` 19.0085 against 18.9683, and its intercepts are
small (+3.485, +2.953, +0.348, −2.653). **Neither a multiplicative scalar nor an additive scalar
works, because neither is a function of volume.**

**Falsifiable consequence: any correction to these markets must be monotone in opportunity.**
A scalar — in the weights, in an intercept, or in a post-hoc multiplier — cannot pass S4, and I
expect any future scalar arm to fail it in the same shape.

## 4. The preregistered rule did its job, and I am holding to it

§4 of the preregistration said, before any number was seen: *promote if and only if M1–M4 pass,
AND S1N, AND S2 (≥5pp level), AND S3, AND S4* — with S1 superiority reported and not required, and
with the sentence *"if this rule feels convenient after the fact, it is on record before the fact."*

| market | S1N | S2 | S3 | S4 | K6 | verdict |
|---|---|---|---|---|---|---|
| `pass_yards` | pass | **fail** | **fail** | pass | pass | **no** |
| `rec_yards` | **fail** | pass | pass | **fail** | pass | **no** |
| `receptions` | **fail** | pass | pass | **fail** | pass | **no** |
| `rush_rec_yards` | pass | pass | pass | **fail** | pass | **no** |
| `rush_yards` | pass | pass | pass | **fail** | pass | **no** |

**K4** fires for `rec_yards` and `receptions`; **S4** fires for all four; `pass_yards` has no level
problem to fix and correctly fails S2/S3. **Nothing promotes.**

Named risk 3 said: *"If C2 improves level and worsens per-player MAE significantly, §4's rule
correctly refuses it — and I should not then argue the metric was wrong."* **That is exactly what
happened, and I am not arguing the metric was wrong.** A 22pp level gain is genuinely attractive
and it is not worth a 12.9% regression on the low-volume quartile.

## 5. Predictions scored

- **"C2 passes S2 and S3 decisively in the four non-QB markets, levels to 0.92–0.97"** — **RIGHT**
  on direction and on three of four magnitudes; `rush_yards` and `rec_yards` overshot past 1.02.
- **"lands non-significant on S1 per-player MAE in most of them"** — **PARTLY WRONG.** ns in
  `rush_yards` and `rush_rec_yards`, but **significantly worse** in `rec_yards` and `receptions`.
- **"`pass_yards` is where I expect trouble and the likeliest way this ships nothing"** —
  **WRONG.** M4 passed comfortably; `pass_yards` was never the problem. **The low-volume quartile
  of every market was**, and I did not name it.
- **Named risk 2, "the constraint may be doing real regularising work I am about to remove"** —
  **RIGHT, and it is the whole result.**
- **Named risk 3, "level and MAE may disagree and the rule must hold"** — **RIGHT**, landed exactly.
- **Named risk 4, "`rush_att` cannot be evaluated and must be left alone"** — **RIGHT**, honoured.

## 6. Where this leaves the lane

**Closed:** scalar corrections to the non-QB ensemble. Unnormalised weights, an intercept, or any
post-hoc constant multiplier. **Do not re-test a scalar.** The mechanism is understood and S4 will
refuse it.

**Open, and now well-posed:** a **volume-conditional** correction — weights or a recalibration that
is a monotone function of projected opportunity, so the low end is left alone and the high end is
lifted. The evidence for it is strong and consistent across three independent measurements (V22's
compression in 10/10 role×market cells, V23's `proj/line` slope, and §3 here). **That is the next
study and it needs its own preregistration.**

**Worth noting for whoever writes it:** `pass_yards` reaches 0.954 level under the *incumbent*
weights and is the only market with a volume-conditional correction already in production — the
M89/M90 residual synthesis, whose features include `pred_attempts`. **The thing that works is
already volume-conditional.** That is the fourth independent pointer in the same direction.

No production code or weight was modified. No odds fetched.
