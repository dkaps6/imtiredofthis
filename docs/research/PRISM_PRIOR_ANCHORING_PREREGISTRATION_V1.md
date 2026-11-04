# PRISM — PRIOR-ANCHORING FIX, PREREGISTRATION V1

**Frozen:** 2026-11-03, **before any arm is scored.**
**Status:** PREREGISTRATION. No result in this document. Nothing promoted.

---

## 1. The defect this targets

`scripts/modeling/bayesian_v2._posterior_mean` blends three sources with weights that are
**not a function of the player**:

```python
weights = [GROUP_STRENGTH[metric]]                          # 3.0, constant
weights.append(min(float(prior_games), PRIOR_PLAYER_CAP[metric]))  # 6.0 cap, never decays
weights.append(float(current_games))                        # grows with the season
```

At 2026 Week 8 that is **9.0 of backward-looking weight (position default + prior season)
against 7.0 of current-season weight** — 1.29:1 against this year.

Nothing in that function responds to a role change, a team change, a QB change, a teammate's
departure, or a snap/route-rate shift. Confirmed: there is **no role-change or
state-transition detection anywhere** in `scripts/modeling/`, `scripts/simulation_v2.py`, or
`scripts/run_pricing_v2.py`, and the AGENTS.md anomaly-gate labels appear only in
`scripts/research/` and in docs — never in the projection path.

Live evidence that this bites: `PRISM_WEEK8_PREREGISTERED_GRADING_V1.md` §5–6 (ordering
conflicts with the market hit 50.7% vs 53.8% when orderings agree), and the 2026-11-02
Odunze/Burden case where the model ranked Chicago's receivers in the opposite order from the
market and the market was right.

---

## 2. Research question

**Does making the prior-season weight responsive to evidence improve next-week player usage
share, relative to the shipped fixed-weight posterior?**

This is deliberately tested at the level of the **quantity the defect lives in** (usage share),
not at downstream yardage MAE, so that a result cannot be confounded by the rest of the stack.

---

## 3. Data and walk-forward protocol

- **Prior season:** 2025, full (`nflreadpy.load_player_stats`, weeks 1–22).
- **Current season:** 2026, weeks 1–8.
- **Metrics:** `tgt_share` (primary), `rush_share` (secondary).
  - `tgt_share` = player targets / team targets, per team-game.
  - `rush_share` = player carries / team carries, per team-game.
- **Walk-forward:** to predict player *p* in 2026 week *W*, use only
  - his full 2025 season share (prior), and
  - his 2026 weeks `1..W-1` share (current).
  **No week-W information of any kind enters any arm.** W ranges over 2…8.
- **Eligibility, fixed in advance:** the player must have ≥1 target (resp. carry) opportunity
  in week W on a team with ≥10 team targets (resp. ≥10 team carries) that week, and must have
  at least one 2026 game before week W. Players with no 2025 data are retained and their prior
  term is simply absent, exactly as production handles it.

---

## 4. Arms

Let `g` = position default from `DEFAULTS[metric]`, `P` = prior-season share,
`C` = current-season-to-date share, `n` = current games played, `m` = prior games played,
`CAP` = `PRIOR_PLAYER_CAP[metric]` = 6.0, `G` = `GROUP_STRENGTH[metric]` = 3.0.

- **C0 — naive control.** Predict `C` (current-season-to-date mean only). No prior, no group.
- **C1 — production baseline.** Exactly as shipped:
  `w = [G, min(m, CAP), n]` over `[g, P, C]`.
- **A1 — time decay.** `w = [G, min(m, CAP) * max(0, 1 - n/D), n]`, with **D = 8** fixed in
  advance (prior weight reaches zero once the current season has as many games as the cap
  allows plus a margin). Tests "just trust this year more as the year goes on."
- **A2 — evidence-precision.** `w = [G, min(m, CAP) / (1 + (|C - P| / s)**2), n]`, with
  **s = 0.05** fixed in advance (a 5-percentage-point disagreement halves the prior's weight).
  Tests "the prior loses authority when this season's evidence contradicts it."

**A1 is in the design specifically to discriminate.** If A2 does not beat A1, then nothing is
gained by responding to *disagreement* and the finding is merely "decay the prior with time."

`s` and `D` are **fixed a priori and will not be tuned on the evaluation set.** A sensitivity
grid (`s ∈ {0.03, 0.05, 0.08}`, `D ∈ {6, 8, 12}`) will be reported as a robustness check only;
the preregistered values are the ones that decide the gates.

---

## 5. Primary metric and subgroups

**Primary:** per-player-week absolute error `|prediction − actual week-W share|`, compared
**paired** (same player-week, different arm), with a **paired bootstrap 95% CI on the mean
difference**, 10,000 resamples, resampled **by player** (not by row) so that players with many
weeks cannot dominate the interval.

Averaged MAE is reported but is **not** the decision object; the paired per-player-week
difference is. Per AGENTS.md, the result must also be readable at the player level:

**Preregistered subgroups, defined using only pre-week-W information:**
- **HIGH-DIVERGENCE:** `|C − P| ≥ 0.05`. These are the role-change cases the fix targets.
- **LOW-DIVERGENCE:** `|C − P| < 0.05`. Here the prior is still good and the fix must not harm.
- **NO-PRIOR:** no 2025 data. Reported for completeness; no gate.

---

## 6. Gates — all must pass to call the fix a win

- **G1 — overall:** A-arm beats C1 on mean paired absolute error, **95% CI excluding 0**.
- **G2 — mechanism:** A-arm beats C1 on **HIGH-DIVERGENCE**, 95% CI excluding 0.
  *This is the gate that matters. A win driven by anything other than the divergent cases is
  not this hypothesis.*
- **G3 — do no harm:** A-arm does not lose to C1 on **LOW-DIVERGENCE** by more than
  0.002 share at the CI's unfavourable bound.
- **G4 — beats naive:** A-arm beats **C0** on mean paired absolute error, CI excluding 0.
- **G5 — discrimination:** A2 beats A1 on **HIGH-DIVERGENCE**, CI excluding 0.
  *If G5 fails, the honest finding is time decay, not evidence precision, and A1 is what
  gets carried forward.*

## 7. Kill conditions

- **K1:** G1 fails → the change does not improve prediction. Stop; record the negative.
- **K2:** G2 fails → the gain is not coming from the cases the hypothesis is about. Stop,
  regardless of G1. A gain that is flat across divergence strata is a different effect wearing
  this hypothesis's name.
- **K3:** G3 fails → the fix buys divergent players at the expense of stable ones. Stop.
- **K4:** `tgt_share` and `rush_share` disagree in direction on G2 → report both, promote
  neither, and treat the metrics separately in any follow-up.

## 8. What a pass would and would not license

A pass licenses **one** thing: a follow-up that wires the winning weight function into
`_posterior_mean` behind a version flag, re-runs the full production validation suite, and is
graded prospectively before any live use.

It does **not** license changing live pricing, and it does **not** by itself establish that
downstream yardage projections improve — that is a separate question with a separate test.

Nothing in this study touches sportsbook data in any arm (AGENTS.md §3).
