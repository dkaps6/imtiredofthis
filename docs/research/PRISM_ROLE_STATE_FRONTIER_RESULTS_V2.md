# PRISM — ROLE-STATE ANCHOR, RESULTS V2 — **8 OF 9 GATES PASS; G7 BLOCKS ON MY OWN THRESHOLD**

**Date:** 2026-11-03
**Preregistration:** `PRISM_ROLE_SHRINKAGE_FIX_PREREGISTRATION_V2.md`, frozen before any arm ran.
**Verdict: NOTHING PROMOTED — K6.** The substance clears every scientific gate; the single
failure is a threshold I set without checking whether any estimator could meet it.
**Artifacts:** `data/research/role_state_frontier_v2.csv` (87,648 rows),
`..._teamsums.csv`, `scripts/research/evaluate_role_state_frontier_v2.py`.

---

## 1. Gate tally

| gate | result | |
|---|---|---|
| **G1** projection gain | **PASS** at every λ, both metrics | monotone, tight CIs |
| **G2** compression | **PASS** at every λ, both metrics | monotone |
| **G3** no tier harmed | **PASS** | gains land in Q4, where needed |
| **G4** side-accuracy frontier | **PASS** (λ=0.25, 0.50 viable) | but see §4 — weakly |
| **G5** `ypa` positive control | **PASS — 0.000e+00** | the gate V1 never exercised |
| **G6** walk-forward | PASS | |
| **G7** team sums toward 1.0 | **FAIL — 71–75% vs my ≥90% bar** | **K6. §5.** |
| **G8** no lookahead | PASS | |
| **G9** identity at λ=0 | PASS | |

## 2. Objective P — a clean, monotone frontier

Per-player bias, each player once, bootstrapped over players (n=1704).

| arm | `tgt_share` gain | 95% CI | breadth | `rush_share` gain | G1 |
|---|---|---|---|---|---|
| λ=0.00 (C0) | — | — | — | — | — |
| λ=0.25 | +0.00078 | [+0.00066,+0.00090] | 87.2% | +0.00081 | PASS |
| λ=0.50 | +0.00153 | [+0.00129,+0.00179] | 86.9% | +0.00160 | PASS |
| λ=0.75 | +0.00222 | [+0.00186,+0.00260] | 86.9% | +0.00239 | PASS |
| **λ=1.00** | **+0.00280** | **[+0.00233,+0.00329]** | 86.6% | **+0.00297** | PASS |
| B (λ=1 + decay) | +0.00296 | [+0.00244,+0.00349] | 86.6% | +0.00301 | PASS |

λ=1.00 cuts per-player `tgt_share` bias from 0.01079 to **0.00798 — a 26% reduction** — and the
gain is **linear in λ**, so λ buys exactly the fraction of the fix it names.

**G2 compression**, equally linear:

| | λ=0.25 | λ=0.50 | λ=0.75 | λ=1.00 |
|---|---|---|---|---|
| `tgt_share` | −13.8% | −27.7% | −41.5% | **−55.3%** |
| `rush_share` | −19.1% | −38.3% | −57.4% | **−76.6%** |

**G3**: no tier degrades at any λ, and every gain is concentrated in **Q4** (`tgt_share` +0.0105,
`rush_share` +0.0119 at λ=1.00) — the tier the whole program has been chasing.

## 3. K9 — the second mechanism is redundant

B (role-rank anchor **+** decaying group weight) versus λ=1.00 alone:

| metric | B − λ=1.00 | 95% CI | |
|---|---|---|---|
| `rush_share` | +0.00004 | [−0.00010,+0.00019] | **redundant** |
| `tgt_share` | +0.00015 | [+0.00000,+0.00031] | boundary |

**K9 substantially fires.** The decaying group weight adds nothing once the anchor is
role-state-conditioned. **Carry the simpler arm.** The two mechanisms were addressing the same
defect from different directions, not stacking.

## 4. Objective S — widened, and still cannot discriminate

n=**3888** (V1 had 1950), `rec_yards` + `receptions`, WR/TE/RB, level-neutralised.

| arm | side accuracy | Δ vs C0 | 95% CI | viable? |
|---|---|---|---|---|
| λ=0.00 | 52.11% | — | — | — |
| λ=0.25 | 51.75% | −0.36pp | [−0.82,+0.08] | **VIABLE** |
| λ=0.50 | 51.90% | −0.21pp | [−0.90,+0.49] | **VIABLE** |
| λ=0.75 | 51.95% | −0.15pp | [−1.00,+0.69] | no (boundary) |
| λ=1.00 | 51.23% | −0.88pp | [−1.88,+0.10] | no |

**Minimum detectable effect: 0.99pp.** The entire spread among λ=0.25–0.75 is **0.21pp** — a
fifth of what this sample can resolve. **The S frontier is flat and noise-dominated**, and it is
**non-monotone** (−0.36 → −0.21 → −0.15 → −0.88).

**That is the second named risk from §6 of the preregistration, and it fired:** *"λ and S may be
non-monotone, in which case any 'best λ' is a fit to noise and must not be promoted on this
sample regardless of its numbers."* **No λ may be selected on its S number.**

**G4 passes, and the reason matters.** λ=0.25 and λ=0.50 are viable because their intervals
**fail to exclude** harm, not because harm was excluded. **"Not shown to be harmful" is not
"shown to be harmless."** The one thing S now says with more confidence than V1 is that
**λ=1.00 probably does cost ≈0.5–1pp** — −0.88pp here against V1's −1.13pp, two independent
estimates agreeing in sign and size.

## 5. Why nothing is promoted — and what kind of failure this is

**G7 fails.** Mean `|team_sum − 1.0|` improves monotonically — **0.1377 → 0.1283 → 0.1197 →
0.1115 → 0.1044**, a 24% reduction — but the per-team-week non-worsening rate is **71–75%**
against my **≥90%** bar.

**The bar was mis-set.** A ≥90% per-unit non-worsening rate is near-uniform dominance, which
essentially no mean-improving estimator achieves; 70–80% is the normal range. I wrote the
threshold without computing what was attainable, and the preregistration's own rule is explicit:
*a gate discovered to be mis-specified is recorded as mis-specified, which blocks promotion,
rather than reinterpreted.* **So K6 fires and nothing is promoted.**

**This is the fifth specification error of the same family in one session** — the allocator's
non-mass-preserving transform, the board check's lookahead tiering, V1's copied G7, V1's
unexercised G5, and now V2's infeasible G7 threshold. Every one was caught, four of them by me
and before publication, but the pattern is unambiguous and worth naming without softening:
**the arms keep being right while the scaffolding around them is wrong.** The distinguishing
feature is that this failure is **arithmetic about my own bar**, not about football — V3 is a
threshold correction with a feasibility calculation, **not new science.**

## 6. Prediction scored

§6 predicted: *the frontier is monotone in λ for both objectives, λ ≈ 0.50 captures most of the
projection gain at a materially smaller S cost, and B is redundant.*

- **"B is redundant" — RIGHT** (§3). First correct prediction in five studies.
- **"λ≈0.50 captures most of the gain" — WRONG.** P is *linear* in λ, so λ=0.50 captures 55%, not
  "most". Linearity itself was unanticipated and is the more useful finding.
- **"monotone for both objectives" — WRONG for S**, which is flat and non-monotone.
- **Named risk #2 — RIGHT**, and it is what governs the interpretation (§4).

## 7. What V3 must do — and the gap that must close first

1. **G7 feasibility-corrected:** test the **mean** `|team_sum − 1.0|` reduction with a
   bootstrapped CI, plus a per-unit non-worsening floor set from a **measured** achievable rate
   (≈70%), computed and justified *before* freezing.
2. **Drop arm B** (K9) and drop λ=0.75/1.00 from contention on S grounds, or power S properly.
3. **Powering S is the real blocker.** At MDE 0.99pp this sample cannot resolve the effect that
   decides the question. W3–W5 2026 are **permanently lost** (AGENTS.md §5a), so more 2026 rows
   do not exist. V3 must either reconstruct a historical board from prior seasons or **accept
   that S cannot be resolved on available data and say so.**

**And before any implementation:** the arms act on `bayes_*`, while `simulation_v2.py` resolves
`rules_*` **first**. Wherever `simulation_rules.py` sets a `rules_` share the fix is **inert**.
**This remains unquantified and bounds the production value of everything above.** It must be
measured before a line of production code changes — otherwise a +26% estimator gain could reach
production as nothing at all.

No sportsbook variable entered any arm; market lines graded only. No gate was weakened.
