# PRISM — JOINT D1 × D2 × D3, PREREGISTRATION V1

**Frozen:** 2026-11-03, **before any arm is computed.**
**Status:** PREREGISTRATION. No result here. Nothing promoted.
**Prerequisite:** `PRISM_ALLOCATOR_CAP_SCALE_INVARIANCE_V1.md`, which collapsed six candidate
arms to three genuine interventions and withdrew the mass-competition risk this study was
originally meant to test.

---

## 0. What changed, and what the study is now for

The original motivation — *"four mechanisms competing to remove the same +0.5574 of excess mass,
so fixing them together risks over-correcting into under-allocation"* — **is withdrawn.** The
allocator is scale-invariant; there is no shared mass budget.

**The remaining question is narrower and real:** D1 and D2 both raise the *relative* share of
active, non-TE/RB players, and D3 cuts yards per unit of opportunity. **Do they compound, cancel,
or act independently?** That cannot be inferred from three isolated runs and is what this
measures.

## 1. Interventions — three, each already measured alone

| | change | acts on | measured alone |
|---|---|---|---|
| **D1** | inactive players' share → 0 | relative share | OUT 2.558 → 0.000 targets; teammates +5.7% |
| **D2** | delete the matchup target multipliers | relative share | TE −0.347, RB −0.134, WR +0.272 targets |
| **D3** | zero yards at zero opportunity | yards given opportunity | `rec_yards` bias +1.428 → +0.743 |

**All level-normalisation variants are excluded as proven no-ops** (D1's A3, D2's M1/M3/M4).

## 2. Arms — full 2×2×2 factorial

Eight arms, named by which interventions are ON: **`---`** (production), `D1--`, `-D2-`, `--D3`,
`D1D2-`, `D1-D3`, `-D2D3`, **`D1D2D3`**.

The factorial is the point: **a main-effects-only result would be indistinguishable from the three
isolated studies.** Interaction terms are the deliverable.

## 3. Metric — yards, because that is what is priced

Scored on the **faithful MC reproduction** validated in D3 at **r = +0.967** against shipped
`model_proj`, which already drives the real `_allocate_counts`, `_sharpen_wr_target_shares` and
`_sharpen_rb_rush_shares`.

- **Primary:** per-player bias on `rec_yards` and `rush_yards` against realized yards, each player
  counted once, bootstrapped **over players**.
- **Secondary:** tier table, **both ends reported** — D1/D2 act at the top (alpha), D3 at the
  bottom (bench), so a single-ended table would hide half of each effect.
- **Interaction:** for each pair and the triple, the observed combined gain **minus** the sum of
  the individual gains, with a bootstrap CI. **This is the quantity the study exists to produce.**
- Share-level diagnostics are reported but **may not decide anything** — per the prerequisite,
  share *values* do not transfer to production.

## 4. Gates — each scoped to the arms it governs

- **M1 — applies to all eight arms.** The reproduction must reuse the production allocator
  unmodified. Verified by `---` reproducing the D3 run's C0 to **1e-9** on mean yards.
  **Failure voids the run.**
- **M2 — applies to `D1--`, `-D2-`, `--D3`.** Each single-intervention arm must reproduce its
  own isolated study's effect direction and rough magnitude. **A contradiction voids that arm**,
  because it would mean the harness, not the intervention, changed.
- **M3 — applies to `--D3` only.** Team total allocated targets must be **unchanged** from `---`,
  since D3 acts after allocation. **Failure voids the run** — it would mean D3 is leaking into
  the share layer.
- **M4 — applies to `D1--` and `-D2-`.** Team total allocated targets must be unchanged from
  `---` to within Monte Carlo noise, confirming the scale-invariance result holds inside the full
  harness rather than only in isolation.
- **S1 — applies to `D1D2D3`.** The triple must beat `---` on per-player bias for **both**
  markets, 95% CI excluding 0.
- **S2 — applies to all arms.** No tier may degrade by more than **2% relative** at either end.
- **S3 — applies to all arms.** Projected/realized total yards must move **toward 1.000** and
  **must not undershoot below 0.95** — the over-correction check, restated in the only currency
  where over-correction can actually happen.
- **S4 — DESCRIPTIVE ONLY, all arms.** Side accuracy on W7+W8 board rows with its MDE printed.
  **Cannot be passed or failed.** Unlike D3, it **will** be computed here, because the triple's
  effect is larger and the direction is worth recording even if unresolvable.

## 5. Kill conditions

**K1** M1, M3 or M4 breached → **run void** · **K2** M2 breached for an arm → **that arm void**,
others readable · **K3** S1 fails → the combination does not work even though the parts do, and
that is the finding · **K4** S2 breached — name the tier and the arm · **K5** S3 undershoots →
**the combination over-corrects**, and the admissible recommendation becomes a *subset* of the
three, not the triple · **K6** all interaction terms' CIs span zero → the three are
**independent**, which licenses shipping them separately and is a clean, useful result.

## 6. Prediction

**The three are close to independent, and K6 fires.** D3 acts downstream of allocation, so its
interaction with D1 and D2 should be near-zero by construction. D1 and D2 both raise active-WR
relative share and so should interact **slightly negatively** — each has less room once the other
has acted — but by only a few percent of either main effect.

**The triple passes S1 on both markets**, with `rec_yards` bias moving from +1.428 toward roughly
+0.3 to +0.5, and **S3 lands near 1.00–1.02 without undershooting.**

**Named risk, and the one I would bet on being the interesting result:** `rush_yards` already
flipped **negative** under D3 alone (mean bias **+0.080 → −0.821**, because removing phantom yards
exposed genuine under-projection on the backs who carry). **Adding D1 and D2 — which move rushing
share toward active RBs — could push `rush_yards` further negative and breach S3's floor or S2 on
the top tier.** If so, **the right recommendation is D1+D2 for receiving and D3 for rushing
handled separately**, not the triple, and K5 governs.

A second risk: the eight arms share one two-week sample, so S2 at the extremes rests on ~50 rows
per tier per arm and may be too thin to detect a 2% degradation.

My record: **seven hits, seven misses**, with the named risk landing nearly every time. This
prediction carries no evidentiary weight.

## 7. Scope and limitation

W7+W8 2026 production universes, 799 player-weeks, 54 team-weeks. **No holdout** — W3–W5 are
permanently lost (AGENTS.md §5a). The interventions carry **no fitted parameters**; D1's
QUESTIONABLE retention (0.8535 target / 0.5718 rush) is the one calibrated quantity and was fitted
on W1–8 injury data, **not** on the W7/W8 yards being scored. **Promotion still requires a
prospective week or a prior-season reconstruction**, and this study cannot supply it.

`tgt_share`, `rush_share`, `rec_yards`, `rush_yards` only. No sportsbook variable enters any arm;
market lines are used in S4 for grading only. No production code is modified. The RB P3 Week-1
route stays frozen.
