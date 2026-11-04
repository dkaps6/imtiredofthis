# PRISM — ROLE-STATE ANCHOR, PREREGISTRATION **V2** (FRONTIER STUDY)

**Frozen:** 2026-11-03, **before any V2 arm is computed.**
**Status:** PREREGISTRATION. No result here. Nothing promoted.
**Builds on:** `PRISM_ROLE_SHRINKAGE_FIX_RESULTS_V1.md` (V1 promoted nothing: K4 on A4, G5
unevaluated, G7 mis-specified). V1's §10 authorized exactly this study.

---

## 0. What V1 settled, and what it did not

**Settled — not re-tested here:**
- **Efficiency shrinkage is correct.** No arm passed G1 on `ypt`, `ypc` or
  `receptions_per_target`, and every arm harmed a tier. `ypa` is an efficiency metric and is
  therefore **out of scope as a fix target.**
- **Share shrinkage is wrong and fixable.** A4 (role-rank anchor) gained +0.00280 on `tgt_share`
  and +0.00297 on `rush_share`, removed 55.3%/76.6% of the compression gradient, harmed no tier,
  and moved team sums from an impossible 1.0886 toward 1.0340.
- **A4 beat A5 decisively** (non-overlapping CIs). The winning unit is the player's **current
  role-state**, not the bare position and not his own history.

**Not settled — the object of this study:**
A4 is the only arm that **lost** side accuracy (−1.13pp, CI [−2.67,+0.46]) while the weaker arms
all gained. V1 §6 therefore established a **frontier** between the two objectives and could not
locate a point on it. **That frontier is now the object of study, not a gate to pass.**

## 1. Research question

**Is there a point on the position→role-state anchor path that captures most of the projection
gain without costing side accuracy — or is the trade-off strict?**

A strict trade-off is a legitimate and publishable answer.

## 2. Arms — a continuous frontier

For each share metric, the anchor becomes a blend of V1's two endpoints:

```
anchor(λ) = (1 − λ) · gm_position  +  λ · gm_position×rank_band
```

- **λ = 0.00** ≡ C0 production. **Identity check (G9).**
- **λ = 0.25, 0.50, 0.75**
- **λ = 1.00** ≡ V1's A4.
- **B** = λ = 1.00 **plus** V1's A2 decaying group weight (`w_group = GS·κ/(κ+n_player)`, κ
  fitted walk-forward). Tests whether the two mechanisms are **additive or redundant** — V1
  measured them only separately.

**Scope:** applied to **`tgt_share` and `rush_share` only.** `ypa`,  `ypt`, `ypc` and
`receptions_per_target` are **not touched by any arm.**

## 3. Gates

- **G1 (P)** some λ beats C0 on per-player bias for **both** share metrics, 95% CI excluding 0,
  bootstrapped **over players**.
- **G2 (compression)** that λ reduces the Q1→Q4 gradient on both share metrics.
- **G3 (no tier harmed)** no pregame tier degrades by more than 0.002.
- **G4 (S) — reframed as a frontier gate.** A λ is **viable** when its P gain CI excludes 0
  **and** its S delta CI lower bound is **above −1.0pp**. **G4 passes if at least one λ is
  viable.** If none is, the trade-off is strict and that is the finding. *The −1.0pp bound is set
  now, before any S number is seen.*
- **G5 (positive control) — MECHANICAL, not statistical.** `ypa` estimates must be
  **bit-identical to C0 (≤1e-12)** under every arm, confirming the `pass_yards` path is
  untouched. `bayes_ypa` is live (`simulation_v2.py:202` → `mc_proj` → ensemble base → M89/M90
  synthesis), so this is a real control. **V1's G5 failed by never being exercised; this version
  cannot.**
- **G6** walk-forward only; every figure out of sample.
- **G7 — CORRECTED.** V1's G7 demanded team sums stay *unchanged*, which was copied from the
  allocator study and is wrong for an estimator. The correct test: **`|team_sum − 1.0|` must not
  increase versus C0 in ≥90% of team-weeks.** A team's shares cannot exceed its opportunity, and
  C0 averages 1.0886, so moving toward 1.0 is the goal, not a side effect.
- **G8 (no lookahead)** every tier and threshold computed from pre-kickoff information only.
- **G9 (identity)** λ=0 reproduces C0 per row to ≤1e-12.

## 4. Sample, and the power problem — stated before running

**Objective P:** walk-forward W2–W8 2026, prior = 2025, evidence reconstructed from nflverse.
Per-player bias, each player once, bootstrapped over players.

**Objective S — widened deliberately.** V1 scored S on `rec_yards` alone (n=1950) and the A4−C0
CI was **3.1pp wide**, which cannot resolve a 1pp effect. V2 widens S to **`rec_yards` +
`receptions`, across WR/TE/RB** (expected ≈4,000–5,000 rows) to roughly halve that width.

**This may still be underpowered, and that is a declared risk, not an excuse to be invoked
later.** If every λ's S interval spans −1.0pp, **G4 is reported as INDETERMINATE and nothing is
promoted** — the same standing rule that an unmeasured gate is a failed one. The minimum
detectable effect will be reported **with** the result.

**Objective S is scored on the V1 reduced-form projector** (`share × team volume × efficiency`),
validated at r = **+0.921** against production `model_proj` with a compression gradient of
+61.6pp against production's +57.5pp. It carries a **+14% level offset**, so S is
**level-neutralised** and **only arm-to-arm deltas are interpretable** — never absolute accuracy.

## 5. Kill conditions

**K1** no λ passes G1 · **K2** G2 fails at that λ · **K3** G3 breached — name the tier ·
**K4** no λ is viable under G4 → **the trade-off is strict; publish the frontier, promote
nothing** · **K5** G5 breached (any arm perturbs `ypa`) → **run void** · **K6** G7 fails ·
**K7** G8 or G9 breached → **run void** · **K8** S intervals all span −1.0pp → **G4
INDETERMINATE, promote nothing** · **K9** B matches λ=1.00 within CI → the mechanisms are
redundant and the simpler arm carries forward.

## 6. Prediction — recorded, and weighted at zero

Expected: the frontier is **monotone in λ** for both objectives, with λ ≈ 0.50 capturing most of
the projection gain at a materially smaller S cost; **B is redundant** with λ=1.00 rather than
additive.

**Named risk:** the S frontier may be **flat and noise-dominated**, so G4 cannot discriminate and
K8 fires. A second risk: λ and S may be **non-monotone**, in which case any "best λ" is a fit to
noise and must not be promoted on this sample regardless of its numbers.

**My last four stated predictions all missed while the named risk landed each time**
(`ALPHA_TIER` B3-vs-B1, allocator γ, shrinkage arm ranking, and V1's own K4). **This prediction
carries no evidentiary weight and must not be cited as support for any outcome.**

## 7. Constraints

Share metrics only. `ypa` untouched and mechanically verified (G5). QB mean-projection research
stays frozen per AGENTS.md; `pass_yards` appears only as the control. The RB P3 Week-1
`WEEK1_STACK_OVERRIDE` route is frozen. No sportsbook variable enters any arm; market lines are
used for **grading only**. No gate may be weakened to make an arm pass — and per V1 §7, a gate
discovered to be mis-specified is **recorded as mis-specified**, which blocks promotion, rather
than reinterpreted.

**Known scope limitation:** the arms act on `bayes_*` columns, and `simulation_v2.py` resolves
`rules_*` **ahead of** `bayes_*` (`_num(row,"rules_tgt_share","bayes_tgt_share",…)`). Wherever
`simulation_rules.py` sets a `rules_` share, the Bayesian value is overridden and the fix is
inert. **This bounds the production effect of any promoted arm and must be quantified before
implementation**, not after.

## 8. Scope of a pass

One follow-up only: implementation behind a version flag, default off; full production
validation; prospective grading before live use; and a committed artifact, not an Actions run id.
