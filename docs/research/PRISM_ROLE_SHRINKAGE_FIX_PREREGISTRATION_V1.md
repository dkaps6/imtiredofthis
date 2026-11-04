# PRISM — ROLE-CONSTANT SHRINKAGE FIX, PREREGISTRATION V1 (ALL ROLE GROUPS)

**Frozen:** 2026-11-03, **before any arm is computed.**
**Status:** PREREGISTRATION. No result here. Nothing promoted.
**Scope note:** "role" throughout means *players currently occupying that role*, not a position
as a modeling unit. The defect under test is precisely that the code treats the role as the unit.

---

## 0. The defect, in code

`scripts/modeling/bayesian_v2.py::_posterior_mean`:

```python
weights = [GROUP_STRENGTH[metric]]                                  # constant, never decays
values  = [group_mean]                                              # = DEFAULTS[metric][pos]
if prior ok:  weights.append(min(prior_games, PRIOR_PLAYER_CAP[metric]))   # CAPPED
              values.append(prior_value)
if current ok: weights.append(current_games)                        # uncapped
               values.append(current_value)
return np.average(values, weights=weights)
```

Two properties combine into a permanent pull toward a **role constant**:

1. `GROUP_STRENGTH` is a **fixed** weight (3.0–6.0) that **never decays** with player evidence.
2. `PRIOR_PLAYER_CAP` **truncates the player's own history** at 6–10 games.

At Week 8 a receiver with a full prior season has 17 prior games, of which **6 count**. Budget:
3.0 role + 6.0 capped prior + 7.0 current = 16.0, so the **role constant permanently holds
18.75%** of every estimate. For a true-0.28 WR against `DEFAULTS["tgt_share"]["WR"] = 0.16`
that is a −0.0225 pull before the allocator's separate haircut.

**This is the literal mechanism behind "the position has the player."**

## 1. The defect is universal across role groups — diagnostic, pregame-tiered

`(model_proj − vegas_line)/vegas_line`, quartiles from each player's **prior-weeks-only** level
(no outcome in the binning; W1,W2,W6,W7,W8 boards):

| role | market | n | Q1 low | Q2 | Q3 | Q4 top | compression |
|---|---|---|---|---|---|---|---|
| WR | receptions | 1006 | +7.3% | −4.5% | −11.0% | −14.6% | +21.9pp |
| WR | rec_yards | 1008 | +27.5% | +13.7% | −6.2% | −15.6% | +43.2pp |
| TE | receptions | 502 | +30.1% | +7.5% | −2.4% | −9.2% | +39.3pp |
| **TE** | **rec_yards** | 502 | **+47.9%** | +23.6% | +11.3% | **−10.8%** | **+58.8pp** |
| RB | receptions | 420 | +6.8% | +2.5% | −4.2% | −14.0% | +20.8pp |
| RB | rec_yards | 430 | +43.5% | +24.3% | +5.0% | −1.0% | +44.5pp |
| **RB** | **rush_yards** | 636 | **+54.8%** | −6.3% | −14.3% | **−19.1%** | **+73.9pp** |
| RB | rush_rec_yards | 386 | −1.1% | −16.4% | −23.6% | −23.4% | +22.3pp |
| QB | rush_yards | 380 | +283.4%¹ | +6.1% | −6.1% | −25.7% | +309.1pp¹ |
| **QB** | **pass_yards** | 442 | **+4.1%** | +3.9% | +2.8% | **−5.3%** | **+9.4pp** |

**10 of 10 cells, monotone in every one.** ¹ QB `rush_yards` Q1 is a small-denominator artifact
(+1.5 yards on a 2.8-yard base); its Q4 (−25.7%) is the real figure.

Projection-level, same pregame tiers, `(model − actual)/actual`:

| role · market | Q1 | Q4 | |
|---|---|---|---|
| WR receptions | −14.6% | −14.5% | uniform **level** bias |
| WR rec_yards | −16.9% | −15.1% | uniform **level** bias |
| TE rec_yards | −6.7% | −16.7% | compression |
| RB receptions | −8.7% | −17.9% | compression |
| RB rush_yards | +14.2% | −17.7% | strong compression |
| **RB rush_rec_yards** | **−13.7%** | **−23.6%** | worst cell on the board |
| **QB pass_yards** | **+0.9%** | **+0.5%** | **essentially unbiased** |

## 2. The natural experiment — and the control it gives us

**`pass_yards` is the only market that bypasses generic shrinkage**, routing through the promoted
M89/M90 football-only synthesis. It is also the **only** market without the defect: +9.4pp
compression against +20.8 to +73.9pp elsewhere, and ±1% projection bias against −4.7% to −23.6%.

This is strong corroboration that the shrinkage is causal rather than incidental, and it supplies
a **positive control**: an arm that damages `pass_yards` is damaging a market that is already
right, and is disqualified regardless of its other numbers.

## 3. Arms

All arms change only `_posterior_mean`'s weighting. No new features, no sportsbook inputs.

- **C0** production as-is. **Control.**
- **A1** remove `PRIOR_PLAYER_CAP` — the player's full prior history counts at face value.
- **A2** decay the group weight with player evidence:
  `w_group = GROUP_STRENGTH · κ/(κ + n_player)`, κ fitted walk-forward per metric.
- **A3** A1 **+** A2.
- **A4** replace the single role constant with a **within-role-rank anchor** (separate defaults
  by depth rank, e.g. WR1/WR2/WR3+). Finer, but still role-based — included to test whether the
  defect is the *constant* or the *granularity*.
- **A5** **player-specific anchor:** the shrinkage target becomes the player's **own prior-season
  rate** where it exists, falling back to the role constant only for rookies and players with no
  history. **This is the arm that directly implements "the player should have the rules, not the
  position have the player," and A4 vs A5 is the test that separates the two readings.**

## 4. Sample, metrics, and the two objectives

**Sample:** 2026 W1–W8 for the side objective (5,746 gradeable rows, pushes removed) and
nflverse weekly player stats for the projection objective. Walk-forward only: fit on weeks
`1..k`, apply to `k+1`. **Historical seasons may be used for the projection objective; 2023 and
2024 remain held back** for the already-designed `prior_eff × def_eff_allowed` confirmation and
must not be spent here.

**Objective P (projection):** per-player bias against actual, each player counted once; tier
tables mandatory; bootstrap resampled **over players**. Pooled MAE is a guardrail that **may
veto, never approve.**

**Objective S (side):** accuracy of `sign(model_proj − vegas_line)`, **never** the `side` column.

**These are scored separately and reported together.** `PRISM_ALPHA_COMPRESSION_BOARD_CHECK_V1`
§3 measured them moving in **opposite** directions — a real projection fix lost 4.30pp of side
accuracy — so **no arm may be promoted on one objective alone.**

## 5. Gates — all must pass

- **G1 (P)** beats C0 on per-player bias, 95% CI excluding 0, bootstrapped over players.
- **G2 (compression)** the Q1→Q4 gradient shrinks toward zero in **≥8 of 10** role × market cells.
- **G3 (no tier harmed)** no pregame tier degrades by more than 0.002 share-equivalent / 2% relative.
- **G4 (S)** side accuracy **does not degrade**: point estimate ≥ C0 − 0.5pp, CI lower bound
  above C0 − 2.0pp.
- **G5 (positive control)** `QB pass_yards` projection bias stays within ±2% and its compression
  stays ≤ +12pp. **Breaking the one healthy market disqualifies an arm outright.**
- **G6 (walk-forward)** every figure out of sample. No in-sample fit reported as a result.
- **G7 (mass preservation)** for share metrics (`tgt_share`, `rush_share`), team sums must be
  unchanged pre-normaliser to 1e-9, and at the identity parameter each arm must reproduce C0 per
  row to 1e-12. **Mechanical preconditions — failure voids the run rather than producing a
  finding.** (Lesson of `PRISM_ALLOCATOR_SHAPE_RESULTS_V1_MISSPECIFIED`.)
- **G8 (no lookahead)** every tier, threshold and filter computed from pre-kickoff information
  only, stated per figure. (Lesson of this morning's withdrawn §5.)

## 6. Kill conditions

**K1** no arm passes G1 · **K2** G2 under 8/10 · **K3** G3 breached — name the tier ·
**K4** G4 breached: the fix improves projections and costs side accuracy, which is the §3 result
repeating and **must be reported as such, not worked around** · **K5** G5 breached ·
**K6** in-sample only · **K7** G7 or G8 mechanical failure → **run void** ·
**K8** A4 matches A5 within CI → the defect is *granularity*, not the role constant, and the
simpler arm carries forward.

## 7. Prediction, recorded to be scored

**A5 and A3 reduce compression most; A1 alone under-delivers**, because for a player without a
prior season the cap is not the binding constraint — the undecaying group weight is.
**A4 partially works**, which would mean the defect is the constant's coarseness rather than its
existence.

**Named risk, stated in advance:** weakening shrinkage **raises variance for low-evidence
players**, so G3 may fail on Q1 and G4 may fail outright — the Q1 cells are exactly where the
over-projection lives, and some of that over-projection is the prior doing its job. **If G1
passes while G4 fails, the honest conclusion is that this market's side edge partly depends on a
biased projection**, which would be a genuinely uncomfortable finding and must be published
rather than buried.

**My last two stated predictions were both wrong** (`ALPHA_TIER` picked B3, B1 won; allocator
shape predicted γ≫1, got 1.05). The useful half of both was the named risk. This prediction
carries no evidentiary weight.

## 8. Constraints

Per AGENTS.md: **broad QB mean-projection research stays frozen** — `pass_yards` appears here
**only as a positive control**, not as a feature hunt; QB `rush_yards` is a separate market and
is in scope. The **RB P3 Week-1 `WEEK1_STACK_OVERRIDE` route is frozen** and must be bit-identical
after any arm. No sportsbook variable may enter any model computation; market lines are used for
**grading only** (§3). No truth or integrity gate may be weakened to make an arm pass.

## 9. Scope of a pass

A pass licenses **one** follow-up: implementation in `scripts/modeling/bayesian_v2.py` behind a
version flag, default off; the full production validation suite; and **prospective grading before
any live use.** Promotion requires both objectives, all eight gates, and a committed artifact —
not an Actions run id (AGENTS.md §5a).

---

# AMENDMENT V1a — §0 MISDESCRIBED THE SHRINKAGE TARGET

**Added:** 2026-11-03, **before any arm was computed.** Correcting the record, not the design.

§0 wrote the shrinkage target as `values = [DEFAULTS[metric][pos]]`. **That is wrong.**
`_posterior_mean` receives `group_mean` from `_weighted_group_mean`, which is the
**prior-season-games-weighted mean of that position group's own players on the current slate**:

```python
mask = df["bayes_position_group"].eq(pos) & pv.notna() & pg.gt(0)
if mask.any(): return float(np.average(pv.loc[mask], weights=pg.loc[mask]))
if pv.notna().any(): return float(pv.loc[mask].median())
return float(DEFAULTS[metric][pos])        # last-resort fallback ONLY
```

So `DEFAULTS` (the 0.16 WR constant) is a **third-choice fallback**, not the normal target. The
arithmetic example in §0 — "a −0.0225 pull against 0.16" — is therefore **withdrawn**; the true
pull is toward the empirical position-group mean, whose value is measured and reported with the
results rather than assumed here.

**The defect under test is unchanged and so is every arm.** Shrinking a named player toward a
*position-group* number with a weight that never decays, while capping his own history, is the
same mechanism whether that number is hardcoded or empirical — it is still the position holding
the player. If anything the empirical target is **more** compressive than the constant, because
it is weighted by `prior_games` across every player in the group, including backups, which pulls
it below the starter population it is applied to. That is now an explicit quantity to report.

**No gate, arm, metric or kill condition changes.** G5's positive control and the A4/A5 contrast
are unaffected.

## Known limitation of this run, stated in advance

`data/player_form_consensus.csv` is pipeline-generated and **not committed**, so
`build_bayesian_baseline` cannot be driven from a clean checkout. Player prior/current evidence
is therefore **reconstructed from nflverse weekly stats**, the same underlying source PlayerForm
consumes.

More importantly: **Objective S cannot be evaluated on production MC.** Regenerating
`model_proj` for five arms across eight weeks requires the full simulation and pricing stack.
Objective S will instead be scored on a **reduced-form projector** (`share × team volume ×
efficiency`), which **must first be validated** against the committed boards' own `model_proj`,
and that validation reported. **If the reduced form does not reproduce production's level and
compression gradient, Objective S is reported as NOT EVALUATED and no arm may be promoted** —
G4 is a required gate and an unmeasured gate is a failed one, per §4's rule that no arm promotes
on one objective alone.

`route_rate` and `yprr` are excluded: production sets `routes = NaN` for every player
(`player_form_v2.py:135`), so both metrics are inert in the live stack and testing them would
measure a path that does not run.
