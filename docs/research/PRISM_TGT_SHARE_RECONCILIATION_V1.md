# PRISM — `tgt_share` RECONCILIATION + WHERE THE ORDERING DEFECT ACTUALLY LIVES V1

**Date:** 2026-11-03
**Status:** MEASUREMENT + NEGATIVE RESULTS. Nothing promoted.
**Follows:** `PRISM_PRIOR_ANCHORING_RESULTS_V1.md`, which killed the prior-anchoring fix and
left one open limitation: *does production's `tgt_share` match the share that study used?*

---

## 1. Reconciliation: yes, they match

| | production | the study |
|---|---|---|
| source | `nflreadpy.load_player_stats(summary_level="week")` | same |
| formula | `player targets / team targets` | same |
| denominator scope | team targets summed **over the player's own games** | same |
| **aggregation** | **ratio of totals** (`player_form_v2.py:222`) | mean of per-game shares |

The one difference is aggregation. Re-running the entire study under production's exact
ratio-of-totals aggregation:

| arm | mean abs error (mean-of-games) | mean abs error (**ratio of totals**) |
|---|---|---|
| C0 | 0.05578 | 0.05582 |
| C1 | 0.05085 | **0.05078** |
| A1 | 0.05103 | 0.05097 |
| A2 | 0.05103 | 0.05101 |

**Every gate verdict is unchanged.** G1 and G2 still fail for both arms; C1 still beats C0
(+0.00504, CI [+0.00326, +0.00679]). The two definitions agree to roughly 1e-4.

**The prior-anchoring kill stands.** The study tested the quantity production actually uses.

---

## 2. The Odunze/Burden case: the prior DID flip it

Chicago, Week 8, using only data through Week 7 — exactly what production had:

| player | prior (2025) | current (2026 W1–7) | **C1 posterior** | **actual W8** |
|---|---|---|---|---|
| **Rome Odunze** | **0.2368** | 0.1696 | **0.1930** | 0.1765 |
| **Luther Burden III** | 0.1279 | **0.2098** | **0.1698** | **0.2647** |
| Colston Loveland | 0.1653 | 0.1875 | 0.1665 | 0.2647 |
| Kalif Raymond | 0.0610 | 0.1384 | 0.1134 | 0.0882 |

- posterior ordering: **Odunze > Burden > Loveland**
- actual W8 ordering: **Burden > Loveland > Odunze**
- market's `rec_yards` lines: **Burden 52.5 > Loveland 42.5 > Odunze 38.5**

**Current-season data alone had the ordering right** (Burden 0.2098 > Odunze 0.1696). The 2025
prior reversed it. So the 2026-11-02 reading of *this case* was correct: the prior caused this
ordering error, and the market agreed with current-season data over the posterior.

---

## 3. Reconciling §1 and §2 — it is a bias/variance trade, not a bug

Both are true:

- the prior **causes** ordering errors where a player's role genuinely changed;
- the prior **prevents** more error than it causes, on average, including for team-changers.

That is the ordinary behaviour of a shrinkage prior. The win is not available by loosening the
prior globally — that is what A1 and A2 tested, and both failed. The win requires **identifying
which players are the genuine-change cases**, and every cheap proxy tried so far fails to do it:

| candidate discriminator | result |
|---|---|
| `\|current − prior\|` divergence ≥ 0.05 | A1/A2 both fail G2 on this subgroup |
| changed NFL team 2025→2026 | dropping the prior is **worse** here too (−0.00592) |
| within-season trend / recency | see §4 — **points the wrong way** |

---

## 4. EXPLORATORY — recency is worse than season-to-date

Not preregistered; hypothesis-generating only. 766 players with weeks 1–3, 5–7 and 8 data:

| predictor of week-8 share | mean absolute error |
|---|---|
| season-to-date mean (weeks 1–7) | **0.01367** |
| last-3-game mean (weeks 5–7) | 0.01464 |

Recency gain **−0.00097**, CI [−0.00199, +0.00008]. The point estimate favours the **full
season-to-date mean**; weighting recent games more would make predictions worse.

And the narrative this was meant to support does not survive contact. Chicago's within-season
trends, early (W1–3) → late (W5–7):

| player | early | late | trend |
|---|---|---|---|
| Luther Burden III | 0.250 | 0.175 | **−0.074** |
| Rome Odunze | 0.142 | 0.171 | +0.029 |
| Colston Loveland | 0.098 | 0.246 | +0.148 |
| Kalif Raymond | 0.240 | 0.076 | −0.164 |

**Burden was not emerging — he was declining within 2026 and spiked in Week 8.** He led Odunze
on season-to-date share from the start. The "Burden has taken over as WR1" story was built from
one drive on 2026-11-02; the season data does not support it, and **Loveland** is the actual
riser nobody named.

That is the anomaly gate again: one drive produced a confident role narrative that the full
season contradicts.

---

## 5. The finding worth acting on: the role signal is missing, not mis-weighted

`player_form_v2.py:129-135` only populates `routes` if the source supplies it:

```python
if "routes" in x.columns: ...
elif "routes_run" in x.columns: ...
else: x["routes"] = np.nan
```

**`nflreadpy.load_player_stats(summary_level="week")` has no route or snap column at all.**
Confirmed for 2026. Therefore in production:

- **`route_rate` is NaN for every player**
- **`yprr` is NaN for every player**

Two of the eight modelled metrics are dead on arrival, and they are precisely the two that
measure *role* rather than *production*.

Meanwhile:

- `nflreadpy.load_participation(2026)` → **unavailable** (already known).
- **`nflreadpy.load_snap_counts(2026)` → AVAILABLE**, 11,292 rows, carrying `offense_snaps` and
  **`offense_pct`**. Production does not use it anywhere.

**Snap share is the role-state signal the model lacks.** It is available, it is weekly, it is
not an outcome statistic, and it distinguishes "this player's role changed" from "this player's
targets were noisy" — which is exactly the discriminator §3 shows is missing.

---

## 6. Next

Do not iterate further on weighting the existing metrics. The next question is:

**Does weekly `offense_pct` from `load_snap_counts` identify genuine role change, and does
conditioning the prior weight on a snap-share shift beat the shipped posterior?**

Before any modelling, two prerequisites:
1. **Publication timing.** Snap counts must be proven available *before* kickoff for a Thursday
   or game-day run. A post-hoc snapshot cannot become a pregame feature (same bar that
   AUDIT TARGET 6 sets for the roster feed).
2. **Coverage.** Confirm the 2026 snap table resolves to the same identity space as
   `player_form_consensus`.

Then preregister with gates sized to the n a multi-week sample can actually deliver — the
Week 8 grading failed on exactly that point.
