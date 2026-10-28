# PRISM — RB rushing: the root cause is an unreconciled team opportunity budget

**This supersedes the mechanism in `PRISM_RB_RUSHING_LEVEL_DIAGNOSIS_V1.md` and retracts
one of its load-bearing claims.** Prompted by GPT audit
[#535 comment 6477974816](https://github.com/dkaps6/imtiredofthis/issues/535#issuecomment-6477974816),
AUDIT TARGET 2, which proposed the mechanism and specified the decomposition.

## ⚠️ RETRACTION — "team RB carry volume is right" is FALSE

The parent diagnosis stated:

> "**team RB carry volume is right**: 22.59 simulated vs 23.11 actual RB carries per team
> (bias −0.52)"

Measured on the **exact production simulation universe** — the 396-row, 28-team,
zero-duplicate frame `_full_roster_simulate` hands to `canonical_simulate`, preserved at
`data/research/week7_opportunity_trace/week7_trace__data__football_simulation_universe.csv`
— by **running the real `simulate()`**:

| quantity | value |
|---|---|
| realized simulated RB carries / team | **19.77** |
| analytic post-cap estimate | 19.70 (agrees to 0.07) |
| actual week-7 RB carries / team | **23.15** |
| **bias** | **−3.38**, 95% CI [−6.36, −0.53] |

**Team RB carry volume is low by ~3.4 carries per team, significantly.** At ~4.1–4.25
YPC that is **−14 yards/team** of RB rushing mass that was previously believed to be
correct.

This is the **fourth** retraction in this lane and it was consequential: "team volume is
right" is precisely what pointed the investigation at *allocation* (who gets the carries)
instead of at *mass* (how many carries exist to give). Both fix A and fix B were chosen
under a false premise. The earlier 22.59 almost certainly came from the priced
`metrics_ready` population rather than the production universe — the same
off-production-path error as the other three.

## The mechanism

`rules_rush_share` is estimated **per player, independently**, with no constraint that a
team's shares form a valid opportunity budget. On the production universe they do not:

| quantity | value |
|---|---|
| teams whose raw rush-share sum exceeds the 0.95 cap | **28 / 28** |
| raw team rush-share sum | mean **1.2729**, min 0.9783, max **1.6877** |

Shares that should sum to ~1.0 sum to **1.27 on average**. `_allocate_counts` then applies
its backstop:

```python
total_share = clean.sum()
if total_share > 0.95:
    clean *= 0.95 / total_share
```

So **every modeled player on every team** is scaled by a mean multiplier of **0.7574**
(min 0.5629) — a ~24% haircut applied uniformly.

| | RB carries / team |
|---|---|
| pre-cap | 26.60 |
| post-cap | 19.77 |
| **lost to compression** | **6.83 carries = 28.03 yards** |

The two errors do not cancel: raw shares over-allocate by **+3.26** carries, and the cap
over-corrects to **−3.38**.

## Why this explains what concentration and reweighting could not

Yards lost to compression, by within-team rank, against the observed shortfalls:

| rank | yards lost to compression | observed shortfall | share explained |
|---|---|---|---|
| 1 | **13.18** | −22.34 | ~59% |
| 2 | **7.88** | −5.21 | >100% |
| 3+ | **5.28** | −5.13 | ~103% |

Compression is a **uniform multiplicative** haircut, so it lowers *every* rank at once —
which is exactly the pattern that killed both candidate fixes:

- **fix A** was mass-preserving, so it could not raise a level. Correct, and now explained:
  the missing mass was never inside the RB distribution to redistribute.
- **fix B** could only reweight between components. Also correct: the defect is upstream of
  all three components' blending.

The decomposition is **not additive** — compression exceeds 100% of the rank-2 and rank-3+
gaps, consistent with pre-cap totals being over by +3.26. Removing the cap naively would
overshoot. The fix is not "remove the cap" or "raise RB projections"; it is to **reconcile
independently estimated player shares into a valid team opportunity budget before Monte
Carlo allocation**, so the cap stops acting as a de facto scaler.

## Relationship to the Bayesian over-shrink finding

These are one defect seen from two angles. `bayesian_v2._posterior_mean` estimates each
player's share against a position-group prior **independently**, with no team-sum
constraint. That is *why* the shares sum to 1.27, and it is also why the within-RB
distribution flattens by ~9pp (`PRISM_RB_CARRY_CONCENTRATION_RESULTS_V1.md`). Independent
per-player shrinkage toward a population mean both **inflates total mass** and **flattens
relative shape**.

So the single structural statement is: **player rush shares are estimated independently
and never reconciled as a team allocation.** Mass and shape are both consequences.

## What is NOT claimed

- No coefficient, cap, or weight was changed. This is diagnostic only.
- The per-rank decomposition is not additive and is not a promotion case.
- 28 teams, one week. The same measurement must be repeated on a prospective live-odds
  universe before any change is designed, let alone promoted.
- Whether 0.95 is the right residual allowance is a separate question not addressed here;
  the finding is that the cap is being reached by every team and is therefore functioning
  as a uniform scaler rather than a backstop.

No sportsbook variable entered any measurement. Betting remains **CLOSED**. No odds fetch.

---

# Addendum — AUDIT TARGET 4: the empirical budget is ~1.0, and a separate roster-coverage defect

Requested by GPT audit
[#535 comment 6478678129](https://github.com/dkaps6/imtiredofthis/issues/535#issuecomment-6478678129),
which (a) corrected my attribution of the mass inflation and (b) specified this measurement.

## 1. Correction accepted: the budget violation exists BEFORE Bayes

GPT's stage-by-stage team sums reproduce **exactly** on my independent run — every figure
to the digit. My statement *"Bayes is why mass inflates to 1.27"* was **too strong**.

| stage | all-player team sum | post-top-5 sum | teams > 0.95 (post-top-5) |
|---|---|---|---|
| `rush_share_prior` | 1.0044 | **0.98104** | 18/28 |
| `rush_share_current` | 1.0664 | 1.05023 | 24/28 |
| PlayerForm `rush_share` blend | 1.1492 | **1.10856** | 25/28 |
| `bayes_rush_share` | 1.4098 | **1.27285** | 28/28 |
| `rules_rush_share` | 1.4098 | 1.27285 | 28/28 |

The accurate decomposition is GPT's: **the team-budget violation already exists in
PlayerForm** (1.1086, 25/28 teams over), **Bayes materially amplifies it**
(1.1086 → 1.2729), the rules layer leaves it untouched, and the simulator crushes it to
0.95. Only the ~9pp *relative-shape* flattening is localized at Bayes. **Fixing Bayes
alone would still leave an invalid team budget.**

## 2. What should the budget be? Measured, not chosen

Definition used, per GPT: *modeled team rush budget = 1 − expected share of real rush
attempts belonging to players legitimately outside the modeled pregame universe.*

2026 week 7, the 28-team pregame universe, all nflverse carriers (not RB-only):

| quantity | value |
|---|---|
| actual team rush attempts | 27.44 |
| attempts by **modeled** players | 25.52 |
| attempts outside the universe | 1.93 |
| **carries by genuinely unmodeled POSITIONS** (K/P/DL/OL/DB/LB) | **0.00 — on 27/27 teams** |
| **empirical budget** | **1.0000** |

**Not one carry in the slate went to a player outside the skill-position universe.**
Production's hard-coded `0.95` reserve has **no empirical basis** at this n. The budget
target is ~1.0, so **the fix is a renormalization, not a re-estimation on account of
unmodeled rushers** — which is the distinction GPT asked for.

## 3. A SEPARATE defect this surfaced: pregame roster coverage

All out-of-universe carries belonged to **skill-position players**, so under GPT's
discipline they are reported separately and are **not** evidence of unmodeled rushing.

**One was my own bug, caught and fixed.** `canonicalize_player_name_safe` returns a
**tuple** `(display_name, clean_key)`; an earlier draft did `str()` on the whole tuple, so
the join matched nothing and real universe members — Dak Prescott, C.J. Stroud,
J.K. Dobbins, J.J. McCarthy — were misfiled as out-of-universe. Fixed to take element
`[1]`; the figures above are post-fix.

**The remainder is real.** After the fix, 9 players / 52 carries sat outside. Tracing them
through every roster artifact in the W7 trace:

| player | carries | in raw `roles_ourlads` | verdict |
|---|---|---|---|
| **Jordan Mason (MIN)** | **15** | **no** | absent at source |
| Jonah Coleman (DEN) | 9 | no | absent at source |
| Dylan Sampson (CLE) | 6 | no | absent at source |
| Mike Washington Jr. (LV) | 6 | **yes** (all artifacts) | **alias/join gap, not absence** |
| Velus Jones Jr. (SEA) | 5 | no | absent at source |
| Elijah Mitchell (PHI) | 3 | no | absent at source |
| Anthony Tyus III (CAR) | 3 | no | absent at source |
| Michael Carter (TEN) | 3 | no | absent at source |
| Khalil Herbert (ATL) | 2 | no | absent at source |

**8 of 9 are missing from the Ourlads depth chart itself**, not filtered out downstream —
so this is **provider coverage** (AGENTS.md §4: Ourlads supplies current roster/depth
roles), upstream of Bayes, the rules layer and the simulator alike. MIN is the clearest
case: the universe modeled Aaron Jones, Deejay Dallas, Demond Claiborne and Max Bredeson,
while **Jordan Mason took 15 carries and was never modeled at all.** For those teams the
RB allocation is computed on an incomplete depth chart, and no amount of budget
reconciliation fixes that.

**Anomaly gate caveat (AGENTS.md).** This measures *pregame roster completeness*; it does
not establish that each case was knowable pregame. Practice-squad elevations, late
activations and in-game injury replacements are legitimate reasons a pregame depth chart
would not list a player. Each of the 8 needs state-labelling before any of it is called a
provider bug. What is established is the **magnitude**: 1.93 carries/team, mean 7.41% of
team rush attempts, median 0.0, p90 26.3%, max 51.7%, with 18/27 teams at zero.

## 4. Denominator semantics — checked, and it runs the other way

GPT flagged that sacks/kneels/aborted plays could make production's rush pool *larger*
than official carries, which would belong in the budget contract rather than a 5% fudge.

| quantity | value |
|---|---|
| model `rush_att` = `plays × (1 − pass_rate)` | 25.01 |
| actual rush attempts (sum of `carries`) | 27.44 |
| **difference** | **−2.43**, 95% CI [−5.64, +0.78] |

The difference is **negative and not significant**: production's pool is, if anything,
modestly *smaller* than official rush attempts — the opposite of the inflation hypothesis.
So there is no sack/kneel padding to account for, and a third possible contributor to the
RB shortfall (a slightly short rush pool) is **suggestive but unproven** at n=27.

## 5. Where this leaves the architecture

Consistent with GPT's proposal, and nothing here is promoted:

1. estimate each player's rush entitlement from player-specific evidence;
2. apply **evidence-dependent** shrinkage (established ≠ uncertain/new);
3. **reconcile the team vector to a validated modeled-mass target** — measured here at
   ~1.0, not 0.95;
4. send a valid vector into Monte Carlo;
5. keep a hard cap only as a corruption guard that should almost never fire.

The present `0.95` should not serve as both the model's target and its emergency guard.

**Limitations:** one week, 27–28 teams, a single preserved pregame universe. The
measurement must be repeated on a prospective live-odds universe before a reconciliation
is preregistered, and the 8 absent backs must be state-labelled first.

No coefficient, cap or weight changed. No sportsbook variable in any measurement. Betting
remains **CLOSED**. No odds fetch.
