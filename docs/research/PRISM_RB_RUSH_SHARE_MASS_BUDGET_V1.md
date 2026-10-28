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

---

# Addendum 2 — the roster-coverage defect is structural: Ourlads is the SOLE existence authority

Per GPT audit
[#535 comment 6478960995](https://github.com/dkaps6/imtiredofthis/issues/535#issuecomment-6478960995),
which state-labelled the nine outside carriers from official transaction timing and ruled
on sequencing. Their classification is **verified below against nflverse weekly roster
status**, and it holds.

## 1. Correction accepted: the budget claim was unconditional and should not be

My *"empirical budget = 1.0000"* was too strong. GPT's wording is the correct one:

> **Conditional on a complete skill-position roster**, there is no evidence for a blanket
> 5% non-skill rusher reserve; the football budget should be near 1.0.

**1.0 was not the realized modeled-roster budget of that snapshot.** The source universe
came from Full Slate run **39849371821** (run id confirmed in the trace provenance), a
*Thursday* run — so the snapshot was incomplete relative to the eventual active roster.
**"Skill-position class coverage = 100%" is not "the pregame modeled roster was 100%
complete."** The realized share belonging to players actually present in that Thursday
universe is ~**92.59%** on the mean-of-team-fractions definition.

## 2. The nine carriers are three defect classes, verified against nflverse

nflverse `load_rosters_weekly(2026)` status by week (`ACT` active, `RES` reserve/IR,
`DEV` practice squad, `INA` inactive):

| player | carries | weekly status trail | class |
|---|---|---|---|
| Mike Washington Jr. (LV) | 6 | ACT w1–w8, present in **all** repo artifacts | **alias/join defect** |
| Velus Jones Jr. (SEA) | 5 | ACT **w3, w4, w5, w6**, w7 | **provider omission** |
| Anthony Tyus III (CAR) | 3 | ACT **w3, w4, w6**, w7 | **provider omission** |
| Michael Carter (TEN) | 3 | ACT from **w2** onward | **provider omission** |
| Jordan Mason (MIN) | **15** | ACT w1, **RES w2–w5**, ACT w7 | return from reserve |
| Jonah Coleman (DEN) | 9 | ACT w1–w2, **RES w3–w6**, ACT w7 | return from reserve |
| Dylan Sampson (CLE) | 6 | ACT w1, **RES w2–w6**, ACT w7 | return from reserve |
| Khalil Herbert (ATL) | 2 | **DEV w2–w6**, ACT w7, DEV w8 | practice-squad elevation |
| Elijah Mitchell (PHI) | 3 | first appears **ACT w7** | late addition |

**Totals: 17 carries deterministic defects, 30 return-from-reserve, 5 elevations.**

**One refinement to GPT's framing:** the provider omissions are not marginal
stale-by-a-day cases. Velus Jones was `ACT` for **four consecutive weeks** before week 7,
Tyus for three, Carter since week 2. These are **sustained** omissions from the depth
chart, which strengthens rather than softens that bucket.

## 3. The structural finding: the availability layer can only NARROW, never ADD

This is why the 17 deterministic carries were unrecoverable. Row counts through the
production chain on the W7 trace:

| artifact | rows |
|---|---|
| `roles_ourlads` | **458** |
| `roles_ourlads_status_v1` | 458 |
| `current_player_availability` | 458 |
| `roles_current_production_eligible_v1` | **396** |
| `football_simulation_universe` | **396** |

The availability/injury certification layer **never adds a row** — it takes Ourlads' 458
and narrows to 396. So **Ourlads is the sole authority on player *existence***. A player
Ourlads does not list cannot enter the universe no matter what the injury, inactives or
availability feeds say about him. Velus Jones was on the active roster for a month and was
structurally unreachable.

That confirms GPT's recommendation 2 and makes it concrete: **Ourlads may remain the
depth-role source, but it cannot remain the sole existence source.**

## 4. Sequencing ruling — accepted

Roster/availability integrity **first**, budget reconciliation **second**. The reason is
the one I raised and GPT confirmed: forcing an incomplete Thursday roster to a 1.0 target
assigns Jordan Mason's 15 carries to Aaron Jones and Max Bredeson. That makes the team
aggregate right while making named-player projections worse — the exact aggregate-vs-player
trap this project exists to avoid.

Order of work, per GPT, adopted:

1. fix deterministic identity/coverage defects (Washington alias; audit stale Ourlads
   omissions against the official active roster);
2. build a **union availability universe** — depth chart ∪ official active/transaction feed
   ∪ IR/designated-for-return ∪ relevant practice-squad elevation candidates ∪ inactives —
   rather than Ourlads-only existence;
3. require a **final game-day roster refresh** before authoritative player projections;
4. only then reconcile the player rush vector to the validated team mass target;
5. hold a **dynamic unresolved-roster residual** before roster lock instead of handing that
   mass to known players;
6. after lock, with a complete eligible universe, the residual collapses toward zero and the
   budget approaches 1.0;
7. keep the hard cap as corruption protection, never as the normal allocator.

## 5. The budget contract, as designed (not promoted)

```
team_modeled_mass_target = 1 - unresolved_roster_rush_mass
```

with the residual **state-derived**, not a universal 0.05:

- final official active/elevation roster complete → target ≈ 1.0;
- Thursday with a credible return/elevation candidate unresolved → **preserve** a residual
  rather than redistribute it;
- a known active player missing from Ourlads → **fail/repair roster integrity**, do not
  silently inflate the residual.

This separates two laws cleanly: **team conservation** (a valid budget) and **player
entitlement** (who gets it). The player-first objective is preserved — the budget is a
conservation law; its division stays player-specific.

## 6. What this W7 audit does and does not establish

- **does not** support a fixed 5% non-skill reserve — no carry went to a non-skill position;
- **does** show final eligible skill players can account for essentially all official carries;
- **does** show pregame roster uncertainty is real and **time-dependent**;
- therefore points to a **dynamic roster-coverage residual, not another hard-coded constant**.

One week, 27–28 teams, one preserved pregame universe. No coefficient, cap or weight
changed. Betting remains **CLOSED**. No odds fetch.

---

# Addendum 3 — the priors are MARGINAL, not ALLOCATIVE. That is the root of both symptoms.

Measured on the exact production universe. `bayesian_v2.DEFAULTS["rush_share"]` is
`{QB: 0.14, RB: 0.32, WR: 0.02, TE: 0.00}` and `GROUP_STRENGTH["rush_share"] = 3.0`, applied
**per player with no team-sum constraint**.

| quantity | value |
|---|---|
| modeled RBs per team | 3.32 |
| modeled QBs per team | 2.68 |
| **team rush-share sum implied by the PRIORS ALONE** | **mean 1.5414**, min 1.1600, max 1.8200 |
| **teams whose priors alone exceed 1.0** | **28 / 28** |
| RB block alone (0.32 × 3.32) | **1.063** |
| actual post-Bayes post-top-5 sum | 1.2729 |
| correlation(prior-implied, post-Bayes) | r = **0.526** |
| RB prior needed for a coherent budget at this depth | **≈0.226**, not 0.32 |

**`RB: 0.32` is a correct *marginal* statement** — a typical featured back takes about 32%
of his team's carries. **It is incoherent as an *allocation*** once it is applied to every
one of a team's 3.32 modeled backs. The prior table describes *a player*; it was never
reconciled as *a team*.

## This unifies every symptom in the lane

1. **Mass inflation.** Shrinking each player toward an incoherent prior with constant
   weight 3.0 lifts the team sum from PlayerForm's **1.1086** to **1.2729**. The priors
   themselves imply **1.5414**; truncation to the top five is what lands it at 1.27.
2. **Shape flattening.** The same constant pull drags the lead back's within-RB share from
   football's **0.6277** to **0.5173** — the ~9pp gap — because every back is pulled toward
   the same 0.32 regardless of how much his own evidence says otherwise.
3. **Every rank biased low.** `_allocate_counts` then crushes the inflated mass back with a
   **uniform** 0.7574 haircut, which lowers rank 1, rank 2 and rank 3+ simultaneously — the
   pattern that killed fix A (mass-preserving, so the missing mass was never inside the RB
   distribution) and fix B (reweighting, so the defect is upstream of all three components).

One cause, three symptoms. **And it is a player-identity defect, not an infrastructure
one:** the position is projecting the player, exactly as the project's first principle
forbids.

## Why this can be attacked before the roster work

The auditor sequenced the Bayesian layer third, on the grounds that the budget violation
pre-exists it. That is true but secondary: pre-Bayes mass is **1.1086**, the priors imply
**1.5414**, and Bayes is where 1.11 becomes 1.27. The dominant inflater is the prior table.

Crucially, **the 0.95 cap is a uniform multiplicative haircut, so it preserves relative
shares.** A shape improvement is therefore measurable *through* the broken budget: validate
on within-team relative share accuracy, which the haircut cannot mask. Mass and shape can
be fixed in the same change without waiting on roster existence — which remains a real,
separate defect worth fixing on its own merits.

## The two coupled changes (designed, NOT promoted, NOT yet preregistered)

**(a) Make the shrinkage target allocative instead of marginal.** The prior a player is
pulled toward should be depth-aware — what share the *k*-th back on a team with *K* modeled
backs gets — so that the priors for one team sum to a valid budget by construction. The
football target is already measured: top-1 within-RB share **0.6277** at 3 backs,
**0.7058** at 2, **0.5625** at 4 (2013–2025, 6,296 team-weeks, ranked by pregame proxy).

**(b) Make the shrinkage weight evidence-precision, not game count.** Replace
`min(prior_games, CAP)` with a weight reflecting how *informative* a player's own history
is — inverse variance of his own observations — and drop the cap that credits a 17-game
veteran with 6 games. This is textbook empirical Bayes (shrinkage ∝ noise/(noise+signal),
not ∝ 1/n) and it is player-specific by construction: a back with a stable role shrinks
less than an erratic one at equal game count.

Measured today, the generic position prior is **33.2% mean / 25.0% median / 20.0% floor**
of a `rush_share` projection and **35.0% / 25.0% / 20.0%** of a `tgt_share` projection. **No
player, however established, can get it below 20%.**

Nothing changed. No coefficient, prior, cap or weight touched. Betting remains **CLOSED**.
