# PRISM — ROLE-CONSTANT SHRINKAGE FIX, RESULTS V1

**Date:** 2026-11-03
**Preregistration:** `PRISM_ROLE_SHRINKAGE_FIX_PREREGISTRATION_V1.md` + amendment V1a, both frozen
before any arm ran.
**Verdict: NOTHING PROMOTED.** K4 fired on the best arm; G5 was never evaluated; G7's
mass-preservation half is mis-specified by me. **But the scientific result is clear and it
splits the problem in two.**
**Artifacts:** `data/research/role_shrinkage_arms_v1.csv` (112,800 rows),
`data/research/role_shrinkage_group_means_v1.csv`. Walk-forward W2–W8 2026, prior = 2025.

---

## 1. The amended quantity, now measured

V1a withdrew the claim that the shrinkage target is `DEFAULTS` (WR 0.16). The **actual**
empirical anchor, `prior_games`-weighted within position group:

| metric | RB | WR | TE |
|---|---|---|---|
| `tgt_share` | **0.064–0.070** | **0.1440–0.1480** | **0.117–0.126** |

`DEFAULTS` would have been 0.08 / 0.16 / 0.12. The empirical anchor is **lower** for RB and WR,
confirming V1a's prediction that weighting by `prior_games` across backups pulls the target
below the starter population it is applied to.

## 2. Implementation validated

| nesting check | expected | observed |
|---|---|---|
| A2 at κ→∞ reproduces C0 | ~0 | **8.1e-13** |
| A1 reproduces C0 where `prior_games ≤ cap` | 0 | **0.0e+00** |
| A4 with rank anchors collapsed to position = C0 | 0 | **0.0e+00** |

## 3. Objective P — the result splits cleanly by metric type

Per-player bias, each player counted once, bootstrapped over players.

### Opportunity shares — the defect is real and fixable

| arm | `tgt_share` gain | 95% CI | breadth | `rush_share` gain | G1 |
|---|---|---|---|---|---|
| A1 no cap | −0.00005 | [−0.00023,+0.00013] | 50.6% | +0.00003 | fail |
| A2 decaying group weight | +0.00150 | [+0.00117,+0.00183] | 86.4% | +0.00147 | **PASS** |
| A3 both | +0.00095 | [+0.00056,+0.00132] | 84.5% | +0.00089 | **PASS** |
| **A4 role-rank anchor** | **+0.00280** | **[+0.00232,+0.00330]** | **86.6%** | **+0.00297** | **PASS** |
| A5 player-own anchor | +0.00102 | [+0.00057,+0.00149] | 70.7% | +0.00089 | **PASS** |

### Efficiency — the shrinkage is CORRECT; every arm is worse

| arm | `ypt` gain | `ypc` gain | `receptions_per_target` gain |
|---|---|---|---|
| A2 | +0.00614 (CI crosses) | −0.01968 | −0.00047 |
| A4 | −0.00256 | +0.01316 (CI crosses) | −0.00024 |
| **A5** | **−0.14074** | **−0.16774** | **−0.00839** |

**No arm passes G1 on any efficiency metric, and every arm harms a tier (G3).** A5 — shrinking a
player's efficiency toward his own prior-season efficiency — is the single worst result in the
study.

**This unifies nine prior null results.** Every signal study that tried to forecast efficiency
failed because **efficiency genuinely does not persist**, so the group prior is doing real work
and loosening it only adds variance. Opportunity is learnable; efficiency is not. The shrinkage
is wrong in exactly one of the two places it is applied.

## 4. G2 — compression, and A4's margin

Gradient = mean(est − true) in Q1 minus Q4, tiers from the player's C0 estimate (pregame).

| arm | `tgt_share` | vs C0 | `rush_share` | vs C0 |
|---|---|---|---|---|
| C0 | −0.0182 | — | −0.0155 | — |
| A2 | −0.0147 | −19.2% | −0.0115 | −26.1% |
| A5 | −0.0131 | −27.7% | −0.0094 | −39.3% |
| **A4** | **−0.0081** | **−55.3%** | **−0.0036** | **−76.6%** |

A4 removes **more than half** the share compression, and the gain lands exactly where it was
needed: the Q4 tier improves **+0.0105** (`tgt_share`) and **+0.0119** (`rush_share`) while no
tier is harmed.

## 5. A4 also fixes the allocator leak at source — unplanned corroboration

Mean team sums of the share estimates:

| | C0 | **A4** | truth |
|---|---|---|---|
| `tgt_share` | **1.0886** | **1.0340** | ≤ 1.0 |
| `rush_share` | 1.0440 | **0.9905** | ≤ 1.0 |

C0 sums to **8.9% more than a team's entire opportunity**, which is impossible. A4 roughly halves
the `tgt_share` excess and lands `rush_share` essentially on 1.0. This is the
`PRISM_MC_OPPORTUNITY_LEAK_V1` finding being reduced **in the estimator** rather than patched in
the allocator — and it was not something the arm was designed to do.

## 6. K4 — the best arm costs side accuracy

Objective S, reduced-form projector, `rec_yards`, W2–W8, n=1950, level-neutralised so only shape
differs:

| arm | side accuracy | Δ vs C0 | G4 |
|---|---|---|---|
| C0 | 51.18% | — | — |
| A1 | 52.10% | +0.92pp | PASS |
| A2 | 51.38% | +0.21pp | PASS |
| A3 | 51.59% | +0.41pp | PASS |
| A5 | 52.00% | +0.82pp | PASS |
| **A4** | **50.05%** | **−1.13pp** `[−2.67,+0.46]` | **FAIL** |

**K4 fires on A4.** The arm with by far the best projections and the largest compression
reduction is the only arm that loses side accuracy. Its CI crosses zero, so the loss is not
certain — but G4's bar was a point estimate ≥ C0 − 0.5pp, and **a bar is not renegotiated after
seeing the number.**

This is the **second independent measurement** of the two objectives diverging, after
`ALPHA_COMPRESSION_BOARD_CHECK_V1` §3. The preregistration named this outcome in advance and
required it be published rather than worked around: **some of this market's side edge depends on
a biased projection.** The arms that keep the side edge (A2, A3, A5) deliver roughly **a third**
of A4's projection gain. That is a real frontier, not a free lunch.

### Reduced-form validation (so §6 can be read at all)

r = **+0.921** against production `model_proj`; compression gradient **+61.6pp** vs production's
**+57.5pp**. Faithful on shape, with a **+14% level offset** — which is why §6 is
level-neutralised and why its **absolute** accuracies are not production's (production scores
51.38% on the same rows). **Arm-to-arm deltas are the only quantity §6 supports.**

## 7. Why nothing is promoted

- **K4** on A4, the only arm worth promoting on projections.
- **G5 never evaluated.** `ypa` was excluded from the run, so the `QB pass_yards` positive
  control was not exercised. An unmeasured required gate is a failed one (V1a).
- **G7 mis-specified by me.** Its mass-preservation half demands team sums unchanged to 1e-9 —
  copied from the allocator study, where mass was fixed by construction. Here the arms are
  *estimators*, so of course the sums move, and §5 shows moving them is the **point**. The gate
  as written fails every arm including the good ones. **I am recording it as mis-specified rather
  than reinterpreting it to pass**, and that alone blocks promotion under this preregistration.
  Its nesting half is sound and passed (§2).
- Efficiency metrics fail G1 and G3 across the board.

**This is the third specification error of the same family in one session** — allocator V1's
non-mass-preserving transform, the board check's lookahead tiering, and now G7. All three were
caught, but the pattern is now the dominant failure mode in this program and is worth stating
plainly: **the arms keep being right while the scaffolding around them is wrong.**

## 8. Prediction scored

§7 predicted *"A5 and A3 reduce compression most; A1 alone under-delivers; A4 partially works."*

- **A1 under-delivers — right.**
- **A5/A3 reduce compression most — wrong.** A4 did, by roughly double.
- **"A4 partially works" — badly understated.** A4 was the best arm on every projection measure.

The **named risk was right again**: *"If G1 passes while G4 fails, the honest conclusion is that
this market's side edge partly depends on a biased projection."* That is precisely the finding.

**Four consecutive studies in which my stated prediction missed and my named risk landed.** The
preregistration discipline is carrying this program; the predictions are not.

## 9. What A4 vs A5 says about the governing principle

A5 — the literal *"the player should have the rules, not the position have the player"* arm —
**lost to A4 on every share metric** (+0.00102 vs +0.00280, non-overlapping CIs) and was the
**worst arm on efficiency**.

The honest reading is a **refinement, not a refutation**. A5 anchors on the player's own
prior-season rate, which is thin or stale exactly when role has changed. A4 anchors on the
population of players **in the same current role-state** (depth rank), which is both
player-specific *in what it conditions on* and stable *in how it is estimated*.

So the unit that wins is **the player's current role-state** — neither the bare position nor the
player's own history. That is AGENTS.md's own ordering (*player identity → current role/state*)
coming out of the data rather than being asserted, and it says the next anchor should be built
from **role-state features** (depth, snap share, teammate competition), not from either extreme.

## 10. Authorized next step

A corrected preregistration V2 that: fixes G7 to test **sum-toward-1.0** instead of
sum-unchanged; includes **`ypa`** so G5 is exercised; restricts arms to **share metrics only**,
since §3 settles that efficiency shrinkage is correct; and treats the **P-versus-S frontier as
the object of study** rather than a gate to pass, because §6 makes it the central fact.

No sportsbook variable entered any arm. Market lines were used for grading only (AGENTS.md §3).
No gate was weakened to make an arm pass.
