# PRISM — WR RANKING SWAP, PREREGISTRATION V1

**Frozen:** 2026-11-03, **before any arm is computed.**
**Status:** PREREGISTRATION. No result here. Nothing promoted.
**Source finding:** `PRISM_RULES_VS_BAYES_COVERAGE_AUDIT_V1.md` §3.

---

## 0. The defect

Two places rank a team's WRs, and **both sort the quantity they then modify:**

```python
# simulation_rules.py:54-59  -> sets rules_role
perim.sort(key=lambda p: _num(p.features.get("tgt_share"), 0.0), reverse=True)
if perim: out[(team,_key(perim[0].player))] = "WR1"

# simulation_v2._sharpen_wr_target_shares -> applies the Migration 37 multipliers
order = np.argsort(-wr, kind="stable")
```

**The estimate picks the label; the label multiplies the estimate.** Meanwhile the universe
already carries an independent depth lineage in its `role` column (`WR1`…`WR7`, 301 rows), which
**neither site consults.**

## 1. Research question

**Does replacing the self-derived WR ranking with the external `role` depth lineage improve
projections — and, where it cannot be shown to, is removing the circular dependency still
justified?**

## 2. The power problem, faced up front

The two rankings **agree on 56 of 56 observed WR1s** (94.6% against the top `bayes_tgt_share`
WR). The effect can only exist on the **disagreement subset**, which in the two committed
universes (W7, W8) is roughly **3 WR1 cases.**

**This study is almost certainly underpowered on the population where the fix acts, and that is
stated now rather than discovered later.** Two consequences, both binding:

1. **The primary deliverable is a "do no harm" result, not a measured gain.** A mechanism that
   sorts the variable it modifies is structurally unsound regardless of its current hit rate; a
   fix for it is justified by **removing a circular dependency**, provided it demonstrably costs
   nothing. Claiming a measured improvement from ~3 cases would be exactly the error this program
   has made four times already.
2. **Power remedy, attempted first:** reconstruct depth rank for **all** of W1–W8 from
   `nflreadpy.load_depth_charts()`, giving an independent external ranking on the full sample
   rather than two weeks. **If that source is unavailable or does not resolve to the production
   universe, the study is reported as structurally-justified-only and the disagreement-subset
   numbers are published as descriptive, with n stated beside every one.**

## 3. Arms

- **C0** production: both sites self-derive from `tgt_share`.
- **R1** `rules_role` labels taken from the `role` depth lineage; sharpening order unchanged.
- **R2** `_sharpen_wr_target_shares` order taken from the depth lineage; `rules_role` unchanged.
- **R3** both.

## 4. Gates

- **G1 (do no harm — primary)** on the **full** WR population, R3's per-player bias must not be
  worse than C0's: 95% CI lower bound on the gain **above −0.0005** (a bound set now, at roughly
  a fifth of V2's smallest detected effect).
- **G2 (do no harm — side)** side accuracy delta CI lower bound **above −1.0pp**, matching V2 §3.
- **G3 (disagreement subset — descriptive only)** report bias and side accuracy on the rows where
  the two rankings differ, **with n beside every figure. This gate cannot be passed or failed**;
  it is published as a direction, and **no promotion may rest on it.**
- **G4 (mass)** team `tgt_share` sums must not increase. The sharpening is mass-preserving by
  contract (`sharpened *= total/sharpened.sum()`); a change here must keep that.
- **G5 (identity)** when the two rankings agree, every arm must reproduce C0 **to 1e-12**. This
  is the mechanical check that the swap changes *only* the disagreement rows.
- **G6** walk-forward; no lookahead. The `role` lineage must be the pregame depth chart, never a
  postgame reconstruction — **if `load_depth_charts` returns post-game-updated rows, it is
  disqualified as a source.**

## 5. Kill conditions

**K1** G1 breached → the depth lineage is a worse ranking than the model's own and the
circularity is **load-bearing**, which would be a genuinely surprising and publishable result ·
**K2** G2 breached · **K3** G4 breached · **K4** G5 breached → **run void** ·
**K5** G6 breached (lookahead in the depth source) → **source disqualified, run void** ·
**K6** the disagreement subset is empty on the available sample → **report as
structurally-justified-only; no statistical claim of any kind.**

## 6. Prediction

**No measurable effect on the full population** (94.6% agreement dilutes any gain ~18×), a
**directionally favourable but statistically meaningless** disagreement subset, and **G1/G2 pass
as do-no-harm.** The expected outcome is therefore *"swap it because the mechanism is wrong, not
because we measured it getting better."*

**Named risk:** the depth lineage may be **staler** than the model's own ranking — a depth chart
can lag an in-season role change that current-season target share has already captured. If so,
K1 fires and the circularity turns out to be doing useful work. **That would be the interesting
outcome and must not be argued away.**

My last five predictions: one hit, four misses, with the named risk landing every time. **This
prediction carries no evidentiary weight.**

## 7. Scope

`rush_share` and `_sharpen_rb_rush_shares` are out of scope. No sportsbook variable enters any
arm; market lines grade only. The RB P3 Week-1 route stays frozen. A gate found mis-specified is
**recorded as mis-specified** — which blocks promotion — not reinterpreted (V1/V2 precedent, now
invoked three times).

**Explicit non-goal:** this study does **not** attempt to fix the matchup-multiplier asymmetry or
the injury-status handling found in the same audit. Those are separate defects with their own
preregistrations and must not be bundled in to make this one look larger.
