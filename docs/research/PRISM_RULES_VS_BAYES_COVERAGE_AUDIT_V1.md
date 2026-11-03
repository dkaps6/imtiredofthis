# PRISM — RULES-vs-BAYES COVERAGE AUDIT

**Date:** 2026-11-03
**Prompted by:** V2 §7's unquantified scope limitation, and the suspicion that the rules layer
is itself flawed. **Both halves are now measured. The suspicion was right, and my warning about
coverage was wrong.**
**Sample:** committed production simulation universes, W7 + W8 2026, 799 rows, 64 team-weeks.
Small — every figure below should be re-measured on a wider trace before it drives code.

---

## 1. The coverage question, answered — the fix is live where it matters

`simulation_v2.py` resolves `_num(row,"rules_tgt_share","bayes_tgt_share",…)`, so a populated
`rules_` value wins. `rules_applied = 1` on 97.4% of board rows, which is what prompted the
worry. But *applied* is not *changed*:

| metric | rows where `rules_` ≠ `bayes_` | Bayesian value actually used |
|---|---|---|
| **`rush_share`** | **0.0%** | **100%** |
| `tgt_share` | 42.6% | 57.4% |

And the `tgt_share` override is **entirely position-specific**:

| position | n | % overridden | mean shift | team mass added |
|---|---|---|---|---|
| **WR** | 301 | **0.0%** | — | — |
| **QB** | 148 | **0.0%** | — | — |
| **TE** | 165 | **100.0%** | **+0.02941** | +4.85 |
| **RB** | 162 | 94.4% | +0.01484 | +2.27 |
| FB | 23 | 95.7% | +0.01165 | +0.26 |

**My V2 §7 warning — "a +26% estimator gain could reach production as nothing at all" — was
wrong, and wrong in the pessimistic direction.** For **WRs, where the entire compression finding
lives, the Bayesian value is used 100% of the time**, and `rush_share` is a pure passthrough
everywhere. The V1/V2 fix is **fully live on exactly the population it was built for.**

## 2. The rules layer inflates TE and RB target share, and only theirs

The override is **always positive** and lands on **every TE** (+0.0294 each) and 94% of RBs
(+0.0148), never on a WR or QB. Across 64 team-weeks it injects **+7.38 of phantom target
share — about +0.115 per team-week.**

A team's receiving shares must sum to ≤1.0. They do not:

| | mean team sum | excess |
|---|---|---|
| `bayes_tgt_share` (estimator) | **1.3756** | +0.3756 |
| `rules_tgt_share` (**what the simulation uses**) | **1.5074** | +0.5074 |

**Attribution: 74% of the leak is the Bayesian estimator; 26% is the rules layer.** So the rules
layer is a real, separately-fixable contributor — but it is **not** the main cause, and I should
not have implied otherwise while the number was still unmeasured.

## 3. The depth chart **is** in the universe, and the WR hierarchy does not use it

**Correction:** I reported earlier that no depth-chart column reaches the simulation universe.
**That was wrong** — I grepped for `ourlads`/`depth` and the column is simply called `role`:

```
QB1 56 · QB2 56 · QB3 36 · RB1 56 · RB2 56 · RB3 50
TE1 56 · TE2 56 · TE3 53 · WR1 56 · WR2 56 · WR3 56 · WR4 55 · WR5 49 … WR7
```

Full depth lineage, WR1 through WR7, 301 WR rows.

**And the WR hierarchy sharpening does not consult it.** `rules_tgt_share` is **unchanged for
every WR**, so Migration 37's sharpening is not applied in the rules layer at all — it happens
downstream in `simulation_v2._sharpen_wr_target_shares`, which **re-derives its own ranking by
sorting the very shares it is about to modify** (`order = np.argsort(-wr)`). `rules_role` is
likewise built by sorting `tgt_share` (`simulation_rules.py:54-59`).

So the mechanism is circular: **the estimate picks the label, then the label multiplies the
estimate.**

**How much does the circularity cost today? Measurably little:**

| | agreement |
|---|---|
| `role` (depth lineage) WR1 == top `bayes_tgt_share` WR | **94.6%** |
| `rules_role` WR1 == top `bayes_tgt_share` WR | **94.6%** |
| `role` and `rules_role` agree on WR1 | **56/56** |

The self-derived label coincides with the external depth chart on **every observed case**. So the
circularity is **structurally unsound but currently near-harmless** — which is a different and
more useful finding than "the hierarchy is broken."

**The exposure is the 5.4%.** When the model is wrong about who the WR1 is, the sharpening
**reinforces its own error** instead of deferring to the depth chart. And the depth chart is
**sitting in the universe, unused for this purpose.** That connects directly to
`PRISM_IRREDUCIBLE_FLOOR_AND_ALPHA_HAIRCUT_V1` §4, where **perfect top-end ranking was worth
+0.01163 — the larger half of the allocator headroom.** Replacing a self-referential sort with
an external one is the cheapest available purchase on that.

## 4. A correction that limits V1 and V2

Production `bayes_tgt_share` sums to **1.3756** per team. My reconstruction's C0 summed to
**1.0886.**

**My reconstruction is not production-faithful on this quantity**, so V1 §5's and V2 §5's
team-sum figures — including "A4 moves sums 1.0886 → 1.0340" — describe **the reconstruction,
not production.** The direction of the finding stands; the magnitudes do not transfer.

The likely cause is consistent with earlier work: I built evidence as a **mean of per-week
shares** over a universe restricted to players with week-*k* stats, while production uses
PlayerForm's **ratio-of-totals** over a universe averaging 14.2 players per team, most of whom
never draw a target. **The gap between 1.0886 and 1.3756 is itself largely the universe-composition
fault** that `PRISM_MC_OPPORTUNITY_LEAK_V1` identified. V3 must either read the production
universe directly or state this gap as a bound.

## 5. What this changes

**Good news, and it is the answer to the question asked:** the V1/V2 share fix is **not** inert.
It is fully live for WRs and for all `rush_share`, which is the whole population the compression
finding concerns. **V2 §7's blocker is removed.**

**Three new, separable items, in value order:**

1. **The TE/RB `tgt_share` inflation** (+0.0294 / +0.0148, 100% / 94% of them, +0.115 team mass
   per team-week, 26% of the leak). This needs its own preregistration — it is a rules-layer
   defect, untouched by anything V1/V2 tested, and it is the first finding in this program that
   is *localized to specific positions by construction rather than by measurement*.
2. **Replace the self-derived WR ranking with the `role` depth lineage** in
   `_sharpen_wr_target_shares` and `rules_role`. Low risk (94.6% agreement means a small blast
   radius), and it removes a circular dependency while buying part of the +0.01163 ranking
   headroom.
3. **V3's reconstruction fidelity** (§4) must be fixed before any team-sum gate is set, which is
   also the right moment to repair V2's infeasible G7 threshold.

**Sample caveat, stated plainly:** 799 rows over two weeks. The positional pattern in §1 is stark
enough (0% vs 100%) that it is unlikely to be noise, but the magnitudes in §2 and the 94.6% in §3
are two-week estimates and must be re-measured on a wider trace before driving code.

No sportsbook variable was read. No production code was modified.
