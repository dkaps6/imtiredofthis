# PRISM — PRIOR-ANCHORING FIX, RESULTS V1 — **KILLED**

**Date:** 2026-11-03
**Preregistration:** `PRISM_PRIOR_ANCHORING_PREREGISTRATION_V1.md`, frozen in commit `08db512b`
**before** any arm was scored.
**Status:** NEGATIVE RESULT. Kill conditions K1 and K2 fired on both metrics.
**Nothing promoted. No production behaviour changed.**

---

## 1. Verdict

| gate | `tgt_share` A1 | `tgt_share` A2 | `rush_share` A1 | `rush_share` A2 |
|---|---|---|---|---|
| G1 overall | **FAIL** | **FAIL** | **FAIL** | **FAIL** |
| G2 mechanism (high-divergence) | **FAIL** | **FAIL** | **FAIL** | **FAIL** |
| G5 A2 beats A1 | **FAIL** | — | **FAIL** | — |

**K1 and K2 both fired on both metrics.** The hypothesis is dead.

Walk-forward, 2026 weeks 2–8, predicting week-W usage share from 2025 prior + 2026 weeks
1..W−1 only. `tgt_share`: 1,584 player-weeks, 399 players. `rush_share`: 781 player-weeks,
223 players. Paired bootstrap resampled by player, 10,000 iterations.

### `tgt_share` — paired gain vs C1 (positive = arm better)

| comparison | subset | mean gain | 95% CI | excludes 0 |
|---|---|---|---|---|
| A1 vs C1 | all | −0.00019 | [−0.00054, +0.00018] | no |
| A2 vs C1 | all | −0.00018 | [−0.00081, +0.00045] | no |
| A1 vs C1 | **high-divergence** | +0.00028 | [−0.00080, +0.00130] | no |
| A2 vs C1 | **high-divergence** | **−0.00059** | [−0.00287, +0.00167] | no |
| A1 vs C1 | low-divergence | −0.00046 | [−0.00082, −0.00009] | **yes (A1 HARMS)** |
| A2 vs A1 | high-divergence | −0.00086 | [−0.00242, +0.00060] | no |

`rush_share` fails identically; A2 additionally harms low-divergence
(−0.00073, CI [−0.00137, −0.00013]).

**A2's point estimate is negative on exactly the subgroup it was designed for.** The
evidence-precision idea does not merely fail to clear a bar — it points the wrong way where
the mechanism says it should be strongest.

The CIs on G2 are roughly ±0.0013–0.003, while the predicted perturbation from reweighting the
prior is on the order of 0.01. **The test had adequate power to detect the hypothesised
effect.** This is not an underpowered null.

---

## 2. The finding that refutes the 2026-11-02 diagnosis

On 2026-11-02, after the Odunze/Burden ordering error, the working diagnosis was that the
posterior is "a position-average machine with a player's name attached" and that prior-season
evidence outvoting the current season is the disease.

**The data says the opposite.**

### G4: the prior EARNS its weight

| comparison | metric | mean gain | 95% CI |
|---|---|---|---|
| C1 vs **C0** (drop prior and group entirely) | `tgt_share` | **+0.00493** | [+0.00311, +0.00676] |

Removing the prior and position default and using current-season-to-date alone is
**significantly worse**. (For `rush_share` the difference is not significant: −0.00310,
CI [−0.00993, +0.00317] — the prior neither helps nor hurts there.)

### EXPLORATORY — team changers, the cleanest "role changed" signal

Not preregistered; hypothesis-generating only. Players whose modal 2025 team differs from
their 2026 team — 281 player-weeks, 85 players.

| subgroup | C0 vs C1 (positive = dropping the prior is better) | 95% CI |
|---|---|---|
| **changed teams** | **−0.00592** | [−0.01056, −0.00124] **excludes 0** |
| same team | −0.00565 | [−0.00788, −0.00346] excludes 0 |

**Dropping the prior is significantly worse for team-changers, by the same magnitude as for
players who stayed put.** Last season's target share carries real, transferable signal about a
player even when his team, scheme and quarterback all changed. The prior is not stale in the
way the diagnosis claimed.

### EXPLORATORY — within-team ordering

The observed defect was an *ordering* error, not an accuracy error, so ordering was tested
separately. 209 team-weeks, WR/TE groups of 2+:

| arm | top-1 correct | rank correlation |
|---|---|---|
| C0 | 51.2% | 0.616 |
| **C1 (production)** | **56.5%** | 0.611 |
| A1 | 56.5% | 0.608 |
| A2 | 55.5% | 0.633 |

No arm beats production on top-1; no comparison excludes zero. **Reweighting the prior does
not fix ordering either.**

---

## 3. What is now established, and what is not

**Established:**
1. Reweighting the prior — by time decay or by disagreement — does not improve next-week usage
   share prediction, overall or on divergent players, for either metric.
2. It does not improve within-team ordering.
3. The prior is **load-bearing** for `tgt_share`, including for players who changed teams.

**Still true as a fact about the code, but no longer supported as a diagnosis:**
- The 9:7 backward-looking weight budget at Week 8.
- The absence of any role-change or state-transition detection in the projection path.
- The anomaly-gate labels existing only in `scripts/research/` and docs.

These remain accurate descriptions. What is refuted is the claim that **this weighting is why
projections fail.** Changing it does nothing, and removing it makes things worse.

**Not established / limitations:**
- `tgt_share` here is reconstructed from nflverse targets. Production builds its own from
  `player_form_consensus.csv`. **If those differ, this study tests a different quantity than
  production uses** — that reconciliation is unfinished and is the first thing a follow-up
  should check.
- Seven current-season weeks and 85 team-changers is a modest sample for the subgroup claims.
- The defect may live **downstream of the share estimate** — in the WR1 labelling path
  (`simulation_rules._wr_role_labels`) or in the allocation/entitlement machinery that consumes
  the label — rather than in the estimate itself. Untested.

---

## 4. Process note

The 2026-11-02 diagnosis was built from **one observed ordering error** (Odunze/Burden) plus a
reading of the code. That is exactly the failure mode AGENTS.md names: *anomalies are
questions, not trends.* A single case generated a confident structural narrative, which this
preregistered test then refuted.

The preregistration is what made that visible. Had the gates been written after seeing these
numbers, A1's +0.00028 on high-divergence `tgt_share` and A2's better rank correlation (0.633
vs 0.611) were both available to be told as a success story. Neither survives a bar set in
advance.

---

## 5. Next

Do **not** iterate on prior weighting. The honest next question is narrower and comes before
any new modelling:

**Does production's `tgt_share` match the share this study reconstructed?** If it does, the
ordering defect is downstream of the estimate and the WR1 labelling / entitlement path is where
to look. If it does not, then production is estimating something other than target share and
that discrepancy is the finding.

That is a measurement task, not a modelling one, and it should be answered before anything else
in this lane.
