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
