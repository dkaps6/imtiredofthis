# PRISM — ALPHA-TIER USAGE SHARE, PREREGISTRATION V1

**Frozen:** 2026-11-03, **before any arm is scored.**
**Status:** PREREGISTRATION. No result in this document. Nothing promoted.

---

## 1. The finding this follows

`PRISM_SNAP_SIGNAL_AND_NOISE_FLOOR_V1.md` measured production's `tgt_share` posterior against
an oracle that knows each player's true full-season mean, on identical rows:

| actual week-W share | n | oracle | production C1 | headroom |
|---|---|---|---|---|
| <5% | 264 | 0.03369 | 0.04851 | +0.01482 |
| 5–12% | 476 | 0.03340 | 0.03644 | +0.00304 |
| 12–20% | 328 | 0.04265 | 0.04093 | **−0.00172** |
| **20%+** | **324** | **0.06643** | **0.08464** | **+0.01821** |

Production is **at or past the oracle for mid-share players** and total headroom is only
**0.00768**. Five candidate improvements to the general estimator were tested on 2026-11-03 and
all failed (time decay, disagreement weighting, dropping the prior for team-changers, recency,
snap-share decomposition). **The general estimator is not the problem.**

The headroom is concentrated in the **20%+ tier**. Independently,
`PRISM_LIVE_SIDE_GRADING_CORRECTION_V1.md` §6 found production projecting Ja'Marr Chase at
4.49 and Justin Jefferson at 4.12 receptions against markets implying 6+. Same tier, reached by
market pricing rather than oracle comparison.

---

## 2. Research question

**Is production's error on high-share receivers a systematic downward bias (compression toward
the position prior), or is it dispersion that no point estimate can remove?**

And if it is bias: **which part of the estimator causes it** — the pull toward the position
default, or the arithmetic (rather than conjugate) form of the blend?

---

## 3. Sample — sized so the gates can actually fire

The Week 8 grading failed because its cells came in below their own n threshold. This design
fixes that by pooling seasons.

Walk-forward, REG season only, predicting week W from the prior season plus weeks `1..W-1`:

| current season | prior season | weeks |
|---|---|---|
| 2024 | 2023 | 2–18 |
| 2025 | 2024 | 2–18 |
| 2026 | 2025 | 2–8 |

**No week-W information of any kind enters any arm**, including the tier definition (§4).

Eligibility, fixed in advance: positions WR/TE/RB; ≥1 target in week W; team ≥10 targets that
week; ≥3 prior current-season games.

**Minimum n to proceed: 400 ALPHA player-weeks.** If the pooled sample does not reach 400, the
study is reported as underpowered and no gate is read — the Week 8 failure mode is not repeated
by quietly reading an undersized cell.

---

## 4. Primary population — defined WITHOUT week-W data

**ALPHA = production's own C1 posterior ≥ 0.20.**

That is leakage-free (the posterior uses only prior season and weeks `1..W-1`) and
operationally the right question: *when production says a player is an alpha, how wrong is it?*

The §1 table used *actual* week-W share to define tiers, which is fine for a descriptive
measurement but unusable for selection. This study uses the predicted tier throughout.

**NON-ALPHA = C1 posterior < 0.20**, used only for the do-no-harm gate.

---

## 5. Phase 1 — diagnostic, with a kill condition

Measure **signed** error `(prediction − actual)` on ALPHA, with a player-level bootstrap CI.

**Preregistered prediction:** ALPHA signed error is **significantly positive** — production
over-predicts relative to... no. Production *under*-projects alphas, so prediction minus actual
is **significantly negative**. Stated explicitly to avoid post-hoc reinterpretation:
**we predict mean(prediction − actual) < 0 on ALPHA, CI excluding 0.**

**K0 — kill:** if ALPHA signed-error CI **includes 0**, the alpha gap is dispersion, not
compression. Arms B1–B3 all target compression and are unlikely to help. **Stop and report the
negative**; do not run Phase 2 looking for a different story.

Also reported, no gate: signed error for NON-ALPHA, and the ALPHA/NON-ALPHA bias difference.

---

## 6. Phase 2 arms — each isolates one mechanism

Notation as in `bayesian_v2._posterior_mean`: `g` position default, `P` prior-season share,
`C` current-season-to-date share, `n` current games, `m` prior games, `G`=3.0, `CAP`=6.0.

- **C1 — production**, exactly as shipped.
- **B1 — tier-aware group weight.** `G_eff = G * min(1, g / C)` when `C > g`, else `G`.
  Reduces the pull toward the position default in proportion to how far above it the player
  already is. *Isolates: the position default as the compressor.*
- **B2 — logit blend.** Identical weights to C1, but average `logit(x)` and invert.
  *Isolates: linear averaging as the compressor, holding the information content fixed.*
- **B3 — Beta-Binomial conjugate update.** The textbook estimator for a proportion from counts.
  Prior mean `μ₀` = C1's group+prior component (that is, C1 computed with the current term
  omitted); prior strength **κ = 100 team-target pseudo-observations, fixed a priori**.
  Posterior mean = `(μ₀κ + player_targets) / (κ + team_targets)`, counts summed over weeks
  `1..W-1`. *Isolates: the arithmetic form of the blend, holding the prior information fixed.*

κ and the B1 functional form are **fixed in advance and will not be tuned on the evaluation
set.** Sensitivity grids (κ ∈ {50, 100, 200}) are reported as robustness only.

---

## 7. Gates — all must pass

Primary metric: **paired per-player-week absolute error**, bootstrap resampled **by player**,
10,000 iterations.

- **G1 — ALPHA:** arm beats C1 on ALPHA, 95% CI excluding 0. *Primary gate.*
- **G2 — bias reduction:** arm's ALPHA signed error is closer to zero than C1's, CI excluding 0.
  *A win on |error| without reducing bias is not this hypothesis.*
- **G3 — do no harm:** arm does not lose to C1 on NON-ALPHA by more than 0.002 at the CI's
  unfavourable bound.
- **G4 — beats the naive control:** arm beats C0 (season-to-date share) on ALPHA, CI excluding 0.
- **G5 — season stability:** the ALPHA gain has the same sign in **all three** current seasons
  (2024, 2025, 2026) evaluated separately. *A gain that exists in only one season is not a
  mechanism.*

## 8. Kill conditions

- **K0** — Phase 1 bias CI includes 0 (see §5).
- **K1** — no arm passes G1.
- **K2** — an arm passes G1 but fails G2: the gain is not bias reduction. Report, promote nothing.
- **K3** — G3 fails: the fix buys alphas at the expense of everyone else.
- **K4** — G5 fails: single-season effect, treated as noise.

## 9. Scope

A pass licenses exactly one follow-up: wiring the winning form into `_posterior_mean` behind a
version flag, re-running the full production validation suite, and **grading it prospectively
before any live use**. It does not license changing live pricing, and it does not establish
that downstream yardage projections improve — a separate question with a separate test.

No sportsbook data enters any arm (AGENTS.md §3).
