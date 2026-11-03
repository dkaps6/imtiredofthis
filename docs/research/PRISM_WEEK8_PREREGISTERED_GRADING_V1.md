# PRISM — WEEK 8 PREREGISTERED GRADING RESULT V1

**Date:** 2026-11-03
**Status:** PREREGISTERED TEST RESULT. Nothing promoted. No production behaviour changed.
**Preregistration:** `CURRENT_NFL_RESEARCH_HANDOFF.md` V21 / Issue #535 comment 6566359145,
frozen 2026-11-02, **before** Week 8 games completed.

---

## 1. What was preregistered

> **Hypothesis:** `pass_yards OVER > 52.4%` and `rush_yards UNDER > 52.4%` on W8.
> **Naive control:** all model picks pooled.
> **Kill condition:** either cell below breakeven on **n ≥ 40**.

Pick direction derived as `'over' if model_proj > vegas_line else 'under'` — never from the
`side` column (see `PRISM_LIVE_SIDE_GRADING_CORRECTION_V1.md`).

---

## 2. Primary result — **INCONCLUSIVE, the test was underpowered**

| cell | W8 record | hit | 95% CI | n ≥ 40? |
|---|---|---|---|---|
| `pass_yards` OVER | 9/17 | 52.9% | [31.0%, 73.8%] | **NO** |
| `rush_yards` UNDER | 22/36 | 61.1% | [44.9%, 75.2%] | **NO** |

**Neither cell reached the n ≥ 40 threshold the preregistration itself specified.** The kill
condition cannot fire and neither can a pass. Reporting either cell as "confirmed" would be
reading a result the design cannot support.

This is a **design failure, not a data failure**: one NFL week yields ~17 graded QB pass rows
and ~36 RB rush rows after dedup. A cell-level weekly test was never going to reach n=40. Any
future preregistration of this kind must either pool multiple weeks or set a threshold the
week can actually deliver.

**Naive control:** 190/362 = **52.49%**, CI [47.3%, 57.6%], breakeven 52.4%. Dead on the line.

---

## 3. The uptrend did NOT survive — this is the headline

| week | record | hit | 95% CI |
|---|---|---|---|
| 1 | 222/432 | 51.39% | [46.7%, 56.1%] |
| 2 | 210/414 | 50.72% | [45.9%, 55.5%] |
| 6 | 181/336 | 53.87% | [48.5%, 59.1%] |
| 7 | 219/389 | 56.30% | [51.3%, 61.1%] |
| **8** | **190/362** | **52.49%** | **[47.3%, 57.6%]** |
| **pooled** | **1022/1933** | **52.87%** | **[50.6%, 55.1%]** |

V21 recorded the W1→W7 climb as "a live hypothesis, not an established result." **Week 8
settles it: the climb was noise.** W8 regressed to the pooled mean and the pooled estimate
fell slightly (52.96% → 52.87%) once the new week was added.

The pooled CI still straddles breakeven. **There is no demonstrated overall directional edge** —
the point estimate sits ~0.5pp above a 52.4% breakeven, which is inside the noise and inside
the vig.

---

## 4. What DID hold: cell structure

| cell | in-sample W1–7 | out-of-sample W8 | delta | |
|---|---|---|---|---|
| `pass_yards` OVER | 38/62 = 61.3% | 9/17 = 52.9% | **−8.3** | regressed hard |
| `pass_yards` UNDER | 27/49 = 55.1% | 6/11 = 54.5% | −0.6 | held |
| **`rush_yards` UNDER** | 101/171 = 59.1% | 22/36 = **61.1%** | **+2.0** | **held** |
| **`rush_yards` OVER** | 55/113 = 48.7% | 11/29 = **37.9%** | **−10.7** | **worst cell, both periods** |
| **`receptions` OVER** | 94/170 = 55.3% | 23/41 = **56.1%** | **+0.8** | **held, n≥40** |
| `receptions` UNDER | 192/356 = 53.9% | 40/78 = 51.3% | −2.7 | broke |
| `rec_yards` OVER | 117/234 = 50.0% | 39/71 = 54.9% | +4.9 | — |
| `rec_yards` UNDER | 153/296 = 51.7% | 26/49 = 53.1% | +1.4 | — |

**The cell *ordering* is the stable object; the cell *levels* are not.**

- `rush_yards` UNDER and `receptions` OVER both held, and `receptions` OVER is the **only cell
  that cleared breakeven at n ≥ 40 out of sample**.
- `rush_yards` OVER was the worst cell in both periods and got worse (48.7% → 37.9%). This is
  the direct market-side signature of the RB mass-deflation defect in
  `PRISM_RB_RUSH_SHARE_MASS_BUDGET_V1.md`: the model under-projects RB rushing volume, so its
  rush OVERs lose and its rush UNDERs ride the same bias.
- **`pass_yards` OVER at 61.3% was overfit on n=62.** It ran 52.9% out of sample. Any decision
  priced off that 61.3% was priced off noise.

---

## 5. New result — the ordering-conflict rule has support

Derived 2026-11-02 after the model's within-team receiver ordering (Odunze > Burden)
contradicted the market's (Burden > Odunze) and the market proved right. Tested here for the
first time.

Within each `(week, team, market)` group of 2+ players, rank by `model_proj` and by
`vegas_line`:

| | record | hit | 95% CI |
|---|---|---|---|
| model rank **==** market rank | 583/1084 | **53.8%** | [50.8%, 56.7%] |
| model rank **≠** market rank (conflict) | 301/594 | **50.7%** | [46.7%, 54.7%] |

Week 8 alone (out of sample): agree **55.2%** (116/210) vs conflict **46.5%** (46/99).

Direction by conflict type: model ranks player **higher** than market 51.5% (157/305); model
ranks player **lower** 49.8% (144/289). **Both directions of disagreement underperform
agreement**, so the signal is disagreement with the market's *ordering*, not the direction of it.

**Status: SUGGESTIVE, not established.** The pooled gap is 3.1pp with overlapping CIs. But the
direction held out of sample and the mechanism is understood (stale role/target-share inputs,
§6), which is more than the pass-OVER level ever had.

---

## 6. Mechanism — why ordering conflicts arise

`scripts/modeling/simulation_rules.py:54-59` assigns WR1 / WR1_5 by sorting on `tgt_share`.
`tgt_share` comes from `bayesian_v2._posterior_mean`, whose weights are:

| input | weight at Week 8 |
|---|---|
| position default (`DEFAULTS["tgt_share"]["WR"]` = 0.16) | 3.0 |
| prior season | `min(prior_games, 6.0)` = **6.0**, capped and **never decayed** |
| current season (7 games) | 7.0 |

**Backward-looking weight 9.0 vs current-season 7.0 — a 1.29:1 margin against this season.**

No term in that function is a function of the player, the team, a role change, a teammate's
departure, or a snap/route-rate shift. A breakout receiver needs roughly a **27% current-season
target share** to out-rank an incumbent running 17% now but 24% last year.

Confirmed absences in the production projection path:
- **no role-change / state-transition detection** anywhere in `scripts/modeling/`,
  `simulation_v2.py`, or `run_pricing_v2.py`;
- the AGENTS.md anomaly-gate state labels (`HEALTHY_FULL`, `INJURY_LIMITED`, `PARTIAL_GAME`,
  `DNP_INACTIVE`, `RETURN_FROM_INJURY`, `UNRESOLVED_CONTEXT`) appear **only** in
  `scripts/research/` measurement files and in docs — **zero occurrences in the projection path**.

This is the same disease as the RB lane: **a prior that does not yield to current evidence**.
There it is `DEFAULTS["rush_share"]["RB"] = 0.32` applied per-player with no team constraint;
here it is prior-season target share, capped and undecayed.

---

## 7. What this means for production

**Nothing is promoted.** Specifically **not** promoted:
- the W1→W7 uptrend (refuted);
- `pass_yards OVER` 61.3% (did not replicate);
- the ordering-conflict rule (suggestive only, needs a proper preregistration).

What the evidence supports stating plainly:
1. **There is no demonstrated overall directional edge.** Pooled 52.87%, CI straddling breakeven.
2. **`rush_yards` OVER is a reliable negative** and is explained by a known, documented defect.
3. **Cell ordering is more stable than cell level.** Any future selection rule should be built
   on which cells are better than which, not on a level estimate from ~60 rows.

---

## 8. Next action

The highest-value open item is **not** another market study. It is the mechanism in §6:
make the weight on prior-season evidence a function of whether the player's situation is
still the same, rather than of games played. That is the "allocative prior vs
evidence-precision shrinkage" work authorised on 2026-10-28 and never completed.

Preregister it properly this time: pool enough weeks to reach the n the design needs, and
state the control before scoring.
