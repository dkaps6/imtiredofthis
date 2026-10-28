# PRISM — player-evidence shrinkage: RESULTS (V1)

Preregistration: `PRISM_PLAYER_EVIDENCE_SHRINKAGE_PREREGISTRATION_V1.md`, frozen `5069b9bc`.
**18,809 player-weeks, 12 seasons (2014–2025), walk-forward rolling origin.** C0 calls the
real `bayesian_v2._posterior_mean`.

## Headline — one leg of the hypothesis won big, the other is dead

**H1 had two legs. Leg (a) — an allocative prior — is strongly supported. Leg (b) —
evidence-precision weighting — FAILED, and was actively harmful on its own.**

And the gate that guards the *player-specificity* framing, **G2, FAILS.** What works is a
**structurally correct prior**, not player-specific evidence weighting. **Nothing is
promoted on this preregistration**, because its gates as frozen are not all met.

## 1. Primary 1 — within-team relative share error (lower is better)

| candidate | mean err | median | vs C0 | 95% CI |
|---|---|---|---|---|
| **C0** incumbent | 0.14727 | 0.12319 | — | — |
| **C1** allocative prior only | **0.12142** | 0.08961 | **+0.02585** | [+0.02484, +0.02684] |
| **C2** precision weight only | 0.15712 | 0.13411 | **−0.00985** | [−0.01029, −0.00941] |
| **C3** both | **0.12091** | 0.08766 | **+0.02636** | [+0.02517, +0.02751] |
| **C4** uniform less shrinkage *(control)* | 0.12992 | 0.10062 | +0.01735 | [+0.01682, +0.01787] |

**The allocative prior is the entire win.** C1 alone delivers **+0.02585** of C3's
**+0.02636** — adding the precision weight buys **+0.00051**, i.e. nothing. And the
precision weight **alone makes things worse** (−0.00985).

**G3 passes:** C4 does not match C3 (+0.01735 vs +0.02636), so this is not merely "less
shrinkage". But C4 captures **66%** of the gain, so a substantial part of the defect *is*
simply over-shrinkage; the allocative structure adds the remaining third.

## 2. The convergent validation that makes this credible

Top-1 within-RB share, measured independently on this historical panel:

| | top-1 share | gap vs actual |
|---|---|---|
| **actual** | **0.6584** | — |
| C0 incumbent | **0.5535** | **−0.1049** |
| C1 allocative | 0.6417 | −0.0167 |
| C2 precision | 0.5270 | −0.1314 |
| **C3 both** | **0.6599** | **+0.0015** |
| C4 control | 0.5989 | −0.0595 |

**C0's −0.1049 gap independently reproduces the −0.1103 [−0.1453, −0.0741] measured on the
live W7 production universe.** Two different datasets, two different methods, the same
~10.5pp flattening. That is the strongest cross-validation in this lane.

And the allocative prior closes it almost exactly: **+0.0015**.

## 3. Primary 2 — G2 FAILS. The gain is not player-specific.

Improvement vs C0 by player evidence quartile:

| stratum | n | C0 err | C1 | C2 | C3 | C4 |
|---|---|---|---|---|---|---|
| Q1 thin | 4,703 | 0.18026 | +0.03137 | −0.01728 | **+0.03042** | +0.01779 |
| Q2 | 4,702 | 0.15835 | +0.02446 | −0.01119 | +0.02575 | +0.01628 |
| Q3 | 4,702 | 0.14314 | +0.02154 | −0.00917 | +0.02212 | +0.01527 |
| Q4 strong | 4,702 | 0.10731 | +0.02603 | −0.00176 | **+0.02714** | +0.02007 |

The gain is **roughly flat, and slightly LARGER for the thinnest-evidence players**
(Q1 +0.0304 vs Q4 +0.0271). G2 required the gain to be **concentrated in high-evidence
players**; it is not.

**This matters for how the result is described.** The project's hypothesis was that
*the same generic weight should not sit on Drake London and Josh Downs*. That framing
predicted the gain would land on established players. **It did not.** What actually fixes
the projection is that the prior being shrunk toward was **structurally wrong** — asserting
every back is a 32% back — and a depth-and-rank-aware prior repairs it for everyone.

The underlying complaint was still right: the position *was* projecting the player. But the
mechanism is the **prior's structure**, not the **evidence weighting**.

## 4. Guardrails

**Committee backs are not harmed — they improve.** K≥3, rank≥2, n=10,849:

| candidate | err | vs C0 |
|---|---|---|
| C0 | 0.12391 | — |
| C1 | 0.09846 | **+0.02545** |
| C3 | 0.09741 | **+0.02650** |
| C2 | 0.13080 | −0.00689 |
| C4 | 0.10831 | +0.01560 |

No lead-back-versus-committee trade-off. **K5 does not trigger.**

**G4 (team mass coherence) is NOT TESTABLE on this panel, and that is a real limitation.**
The historical panel is built from players with **realized carries**, so it carries ~3.0
backs per team and C0's un-normalized team sum is already **0.9824** — the panel never
exhibits production's **1.2729** inflation. That inflation comes substantially from modeling
backs who **do not play**, which this panel structurally cannot contain. **So nothing here
shows the allocative prior fixes the mass defect.** The shape result is unaffected, because
Primary 1 is measured on normalized shares.

## 5. Disposition — nothing promoted

| gate | result |
|---|---|
| G1 C3 beats C0, CI excludes 0 | **PASS** |
| G2 gain concentrated in high-evidence players | **FAIL** |
| G3 C4 does not match C3 | **PASS** |
| G4 team mass coherence | **NOT TESTABLE** on this panel |
| G5 committee backs not harmed | **PASS** (they improve) |

The preregistration says a candidate may be promoted **only if all gates hold.** G2 does
not. **Nothing is promoted here, and the gates are not being reinterpreted after the fact
to make the result shippable.**

What this earns is a clean, well-powered finding and a much narrower next study:

- **leg (b), evidence-precision weighting, is CLOSED NEGATIVE.** Harmful alone, negligible
  additive. Do not revisit without a new mechanism.
- **leg (a), the allocative prior, is a strong candidate** needing its **own**
  preregistration with gates written for *it* — one free structure (the depth×rank table),
  a prospective 2026 check, and an explicit test of whether it also moves the team mass on
  a universe that contains non-playing backs.

## 6. Scope

No coefficient, prior, cap or weight changed. `bayesian_v2` is untouched; C0 ran through its
real function and the variants are separate. Ensemble weights, the simulator, the 0.95 cap
and the roster path are untouched. The historical consensus is a **proxy** for production's
PlayerForm output, so absolute levels do not transfer — only the candidate comparison, which
holds because every candidate saw identical inputs. 2026 supplied no fitting input. No
sportsbook variable at any stage. Betting remains **CLOSED**. No odds fetch.
