# PRISM — THE ALLOCATOR IS SCALE-INVARIANT, AND IT INVALIDATES PART OF TODAY'S FRAMING

**Date:** 2026-11-03
**Status:** STRUCTURAL CORRECTION. Found while designing the joint D1+D2+D3 study.
**It retracts claims in three documents committed earlier today and simplifies two others.**

---

## 1. The fact

`scripts/simulation_v2._allocate_counts`:

```python
clean = np.clip(np.nan_to_num(shares), 0.0, 0.95)
total_share = clean.sum()
if total_share > 0.95: clean *= 0.95 / total_share
residual = max(0.0, 1.0 - clean.sum())
probs = np.append(clean, residual); probs = probs / probs.sum()
allocations[i] = rng.multinomial(int(total), probs)[:n_players]
```

A player's allocation probability is **`share_i / team_sum`** once the cap binds. **Therefore any
uniform per-team rescale of the share vector is a mathematical no-op**, and the team's total
allocated opportunity is fixed:

| team sum of shares | allocated targets (of 30 attempts) | |
|---|---|---|
| 0.50 | 15.02 | 50% |
| 0.80 | 24.04 | 80% |
| **1.00** | **28.50** | **95%** |
| **1.51** | **28.50** | **95%** |
| 3.00 | 28.51 | **95%** |

And the top player's allocated targets are invariant to the sum: **7.100 / 7.146 / 7.134 / 7.132**
at sums 1.51 / 1.38 / 1.23 / 1.00.

**Only the *relative* shares reach the simulation. The level never does.**

## 2. What this retracts

### 2a. The audit's "competing for the same mass" framing — **withdrawn**

`PRISM_RULES_SIM_BAYES_CODE_AUDIT_V1` §remediation, and the closing section of
`PRISM_D1_INJURY_STATUS_RESULTS_V1`, both warned that **"D1, D2, D3 and the ROLE_SHRINKAGE arms
are competing to remove the same finite +0.5574 of excess mass"** and that fixing them together
risked over-correcting into under-allocation.

**There is no excess mass in the simulation.** The sums 1.5110 against a realized 0.9536 are real
properties of the share *vector* and have **no production consequence** — the cap discards the
difference before allocation. **The over-correction risk as stated does not exist.** Fixes can
still interact through *relative* shares, which the joint study must measure, but not through a
shared mass budget.

### 2b. D1's §3 lesson — **withdrawn**

`PRISM_D1_INJURY_STATUS_RESULTS_V1` §3 reported A3 (redistribute freed mass) losing to A2
(discard it), and drew the lesson: *"mass preservation is only a virtue when the mass is correct …
discarding freed opportunity is strictly better than redistributing it."*

**A3 is `A2 × (C0_sum / A2_sum)` per team — a uniform rescale. In allocation the two are
identical: 54/54 team-weeks, max|diff| 0.000e+00.** They are the same production model.

The measured difference came from scoring share *values* against realized shares, where the level
does matter, rather than scoring **allocations**, where it cannot. **The discard-versus-redistribute
choice is a no-op and the lesson drawn from it was wrong.** D1's §3 claim that it "retroactively
explains D2" is withdrawn with it.

### 2c. D2's five arms are **two** production models

| pair | max \|diff\| in allocated targets | |
|---|---|---|
| C0 vs M1 | **0.000e+00** | identical |
| C0 vs M3 | **0.000e+00** | identical |
| M2 vs M4 | **0.000e+00** | identical |
| **C0 vs M2** | **7.937e-01** | genuinely different |

`{C0, M1, M3}` are one model; `{M2, M4}` are another. **M1 is not a weaker version of the fix —
it is no fix at all.** D2's §3 said M1 "conflates the D2 fix with an across-the-board level
correction"; the direction was right but the statement understates it. The S1 gains reported for
M1, M3 and M4 are properties of the share vector and **do not transfer to production.**

**D2's recommendation survives unchanged — delete the multipliers (M2) — and is now the only real
D2 arm.**

## 3. What survives intact

- **D2's conclusion (M2).** The genuine effect, in allocated targets: **TE −0.347, RB −0.134,
  FB −0.099, WR +0.272**, team totals equal (1538.7 vs 1539.0). Deleting the multipliers moves
  roughly a third of a target per TE to the WRs. Real, and the only D2 effect there is.
- **D1's conclusion.** **OUT players 2.558 → 0.000 allocated targets**, teammates +5.7%, ~80
  targets redistributed across 54 team-weeks. Real. A2 and A3 are interchangeable — pick the
  simpler.
- **D3 entirely.** It operates on yards *given* allocation, downstream of the cap, so nothing here
  touches it. Its r = +0.967 reproduction already included the real allocator.
- **Every compression finding.** `ROLE_SHRINKAGE` V1/V2 and the allocator-shape studies all
  measured **relative** shape, which is exactly what reaches the simulation. Unaffected.
- **The alpha haircut, with its mechanism restated.** `IRREDUCIBLE_FLOOR_AND_ALPHA_HAIRCUT_V1` §6
  derived −0.0392 as `waste × true_rate` under a capped normaliser. The arithmetic is **right and
  the magnitude stands**, but the mechanism is better stated as: **phantom players occupy
  *relative* share, so the alpha's allocation probability is reduced by exactly (1 − waste).**
  `share_α/(sum with phantoms)` versus `share_α/(sum without)` is the same (1 − waste) factor, so
  the 16% figure is unchanged — it was never a level effect.

## 4. Why I did not catch this earlier

I read `_allocate_counts` on day one and quoted the `0.95` cap repeatedly — in
`PRISM_MC_OPPORTUNITY_LEAK_V1`, in the allocator-shape work, in both audits. **I treated the cap
as a clipping nuisance that discarded mass, and never asked what it implies about scale.** Every
team sum I reported (1.5110, 1.3772, 1.4270, 0.9536) was computed on a quantity the simulation
normalises away.

The tell was available and I walked past it twice: D1's A3 and D2's M1 are both *uniform
rescales by construction*, and I wrote both as substantive arms. A one-line check — does a
uniform rescale change the allocation? — would have collapsed six arms to two before any of them
ran.

**This is the seventh specification-class error of the session and the most consequential**,
because it is not a mis-set gate but a wrong model of the code the whole programme is measuring.

## 5. Consequence for the joint study

The design simplifies rather than complicates. There are **three genuine interventions**, not
four-plus-interactions:

1. **D1-zero** — set inactive players' share to 0 (changes relative shares);
2. **D2-delete** — remove the matchup target multipliers (changes relative shares);
3. **D3-gate** — zero yards at zero opportunity (downstream of allocation, independent).

All level-normalisation arms are discarded as no-ops. The joint study is a clean 2×2×2 on these
three, **scored on yards**, which is the common currency and what gets priced.

No sportsbook variable was read. No production code modified.
