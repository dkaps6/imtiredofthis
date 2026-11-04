# PRISM — D3: PHANTOM YARDS AT ZERO OPPORTUNITY, PREREGISTRATION V1

**Frozen:** 2026-11-03, **before any arm is computed.**
**Status:** PREREGISTRATION. No result here. Nothing promoted.
**Source:** `PRISM_RULES_SIM_BAYES_CODE_AUDIT_V1.md` D3.

---

## 0. Process change — every gate is now scoped to the arms it governs

Six gates in this session have been mis-specified, and **five of the six failed the same way: a
gate stated for "every arm" that one arm could not satisfy by construction** (the allocator's
mass transform, V1's copied G7, V2's infeasible threshold, D1's G2). That is a writing habit, not
bad luck.

**From here, each gate names the arms it applies to, and any gate that cannot apply to an arm says
so explicitly.** This preregistration is written that way.

## 1. The defect

```python
rec_sd   = np.maximum(6.0, np.sqrt(np.maximum(targets,1))*ypt*0.55)*vol_mult
rec_yards = np.clip(rng.normal(rec_mu, rec_sd), 0.0, None)
rush_sd  = np.maximum(3.0, np.sqrt(np.maximum(carries,1))*ypc*0.65)*vol_mult
```

Two compounding faults. The **sd floor applies regardless of opportunity**
(`np.maximum(targets,1)` treats zero targets as one), and the **clip at zero is asymmetric**, so
a mean-zero distribution clipped below returns a positive mean. Analytically:

| opportunity | μ | σ | **E[yards]** | bias |
|---|---|---|---|---|
| **0 targets** | 0.0 | 6.0 | **2.40** | **+2.40** |
| 1 target | 7.5 | 6.0 | 7.81 | +0.31 |
| 2 targets | 15.0 | 6.0 | 15.01 | +0.01 |
| **0 carries** | 0.0 | 3.0 | **1.19** | **+1.19** |

**A player who is never thrown to is projected 2.40 receiving yards.** The bias is concentrated
entirely on low-opportunity players — the population measured as over-projected by **+8% to +30%**
in `ALPHA_COMPRESSION_BOARD_CHECK_V1` §2, which this mechanism is large enough to explain.

**Note the floor is not simply wrong.** At one target it raises σ from 4.13 to 6.0, which is
defensible — a single target can be a 60-yard touchdown, so low-volume players genuinely have
high variance relative to their mean. **At zero opportunity there is no such tail: he cannot gain
yards.** The defect is the floor's application at zero, not its existence.

## 2. Arms — a nested ladder

| arm | opportunity = 0 | opportunity ≥ 1 |
|---|---|---|
| **C0** production | `clip(normal(0, floor), 0, ∞)` → **2.40** | `clip(normal(μ,σ), 0, ∞)` |
| **A1 — primary** | **exactly 0** | unchanged from C0 |
| **A2** | exactly 0 | **Gamma(μ, σ)** — mean exactly μ, sd exactly σ, support ≥ 0, **no clip bias** |
| **A3** | exactly 0 | clip-normal with the **floor removed** (`σ = √t·ypt·0.55·vol`) |

**A1 is primary** because zero opportunity yielding zero yards is a **football identity, not a
modelling choice.** A2 additionally removes the residual clip bias at low opportunity (+0.31 at
one target). A3 tests whether the floor itself is harmful.

**A1 ⊂ A2 and A1 ⊂ A3**, so the ladder separates the three effects cleanly.

## 3. Mechanical gates — arms named

- **M1 — applies to A1, A2, A3.** Zero opportunity must yield **exactly 0.0** yards, to 1e-12.
  **Failure voids the run.**
- **M2 — applies to A1 only.** A1 must be **bit-identical to C0 on every row with opportunity
  ≥ 1**, to 1e-12, confirming it changes only the zero cell. **Does not apply to A2 or A3, which
  deliberately change those rows.**
- **M3 — applies to A2 only.** The Gamma draw's sample mean and sd must match μ and σ within Monte
  Carlo tolerance (±1% at 200k draws). **Failure voids A2**, not the run.
- **M4 — faithfulness, applies to C0.** My reproduction of the yards block must track the shipped
  `model_proj` for `rec_yards`: Pearson r reported, and the level ratio reported. **If r < 0.85,
  the study is reported as ANALYTIC-ONLY and no arm may be promoted** — the §1 table stands on its
  own arithmetic regardless.

## 4. Statistical gates — arms named

- **S1 — applies to A1, A2, A3.** Per-player bias on `rec_yards` and `rush_yards` against realized
  yards must improve versus C0, 95% CI excluding 0, bootstrapped **over players**.
- **S2 — applies to A1, A2, A3.** No pregame tier degrades by more than 2% relative, **bottom
  tier reported first** (this defect lives at the bottom, unlike D1/D2).
- **S3 — applies to A1, A2, A3.** Total projected team receiving yards must move **toward** the
  realized total.
- **S4 — DESCRIPTIVE ONLY, applies to all arms.** Side accuracy on W7+W8 `rec_yards` board rows,
  reported with its minimum detectable effect. **Cannot be passed or failed; no promotion may rest
  on it.**

## 5. Kill conditions

**K1** M1 or M4 breached → **run void** · **K2** M2 breached → A1 is mis-implemented, **run
void** · **K3** M3 breached → **A2 disqualified**, A1 and A3 still readable · **K4** S1 fails for
A1 → removing phantom yards does not improve projections, which would be extraordinary and must be
published · **K5** S2 breached — name the tier · **K6** A1 ≈ A2 within CI → the residual clip bias
is immaterial and the **simpler arm carries forward.**

## 6. The limitation that caps A3 specifically

**This study scores the MEAN only.** The sd floor exists to widen the distribution for
low-volume players, and `model_sd` feeds pricing. **A3 removes that widening, so it could improve
the mean while understating tail risk — and this design cannot detect that harm.**

**Therefore A3 may not be promoted on this study's evidence even if it wins S1.** A promotion
case for A3 requires a separate distributional study (coverage of realized outcomes by predicted
quantile). A1 and A2 do not have this problem: A1 touches only the zero cell where there is no
tail, and A2 preserves σ exactly by construction.

## 7. Prediction

**A1 passes S1** and reduces low-tier `rec_yards` projections by roughly **1.5–2.5 yards** — a
player with a 2% share facing ~30 attempts draws zero targets in about 55% of iterations, so he
collects ≈0.55 × 2.40 phantom yards per iteration before the one-target term. That is the right
order of magnitude to be a **material part** of the measured +8% to +30% bench over-projection,
though not all of it. **Negligible effect on high-volume players.** **A2 adds a small further gain
and K6 is a live possibility.** **A3 wins S1 but cannot be promoted** (§6).

**Named risk:** my reproduction may not be faithful enough (M4). The real path applies a shared
per-game `pass_eff_shock`, a binomial pass-attempt draw and the multinomial allocation, and I am
rebuilding all of it from the universe's columns. **If r < 0.85 this becomes an analytic-only
result**, which would still establish the §1 bias but could not size its effect on projections.

My record: **six hits, seven misses**, with the named risk landing nearly every time. This
prediction carries no evidentiary weight.

## 8. Scope

`rec_yards` and `rush_yards` only. **D1 and D2 are OFF** — C0 is the shipped `rules_*` values as
archived, and the arms change only the yards draw, so the share layer is held constant. Per the
audit's interaction warning this keeps D3 measurable in isolation; **the mandatory joint D1+D2+D3
study follows separately.** No sportsbook variable enters any arm; market lines are used in S4
for grading only. No production code is modified.
