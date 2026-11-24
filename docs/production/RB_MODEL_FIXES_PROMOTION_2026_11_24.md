# Two RB model fixes promoted 2026-11-24

**Operator instruction (2026-11-24):** "go ahead and ship both fixes into the model."

Both fixes:
- passed preregistered out-of-sample gates before this date;
- are football-only (no sportsbook variable reaches either one);
- are versioned and write audit columns;
- fail closed where an input is missing.

## 1. `RB_CARRY_SHARE_RECENCY_V1` (PlayerForm)

**Change.** For an RB/FB with ≥ 3 current-season games, the current-season carry share that
feeds the Bayesian baseline becomes `0.15 x season-to-date + 0.85 x last 3 games with today's
team`. Everything else is unchanged: prior share, game counts, other metrics, other positions.

**Code:**
- `scripts/player_form_v2.py::apply_rb_carry_share_recency`;
- called from `build()` after positions are resolved.

**Audit columns** in `player_form_consensus.csv`:
- `rush_share_current_season` (the old value);
- `rush_share_last3`;
- `rb_carry_recency_applied`;
- `rb_carry_recency_version`.

**Kill switch:** `RB_CARRY_SHARE_RECENCY_V1=0`. Emergency use only; production default is ON.

**Evidence:**
- **Input level** (`RB_CARRY_SHARE_RECENCY_RESULTS_V1.md`): `a` was fitted on 2023-2024 only;
  the CI excludes 0 on the 2025 and 2026 holdouts.
- **End to end** through Bayes, top-5 and the 0.95 cap
  (`RB_CARRY_SHARE_RECENCY_END_TO_END_RESULTS_V1.md`): expected-carry MAE gain
  - +0.073 [+0.035, +0.112] in 2025;
  - +0.046 [+0.013, +0.078] in 2026 weeks 4-10;
  - lead backs improve in both.

**Forward check, 2026 week 11, on production's archived week-11 PlayerForm**
(`data/research/rb_recency_week11_production_replay_v1.csv`):
- 51 RBs who played;
- expected-carry MAE 4.531 → 4.514 (+0.017, CI [-0.157, +0.173]);
- flat, as expected for a ~1-2% effect on one week.

**Limitation, stated plainly:**
- This does **not** fix the week-11 Emmett Johnson miss (production 5.1 expected carries,
  6.7 with recency, actual 23).
- That miss comes from the Bayesian layer giving backs with no history the position-group mean
  share: Nathan Carter (0 games) 0.32 and Jaydn Ott (1 game) 0.26, above Johnson's 0.23.
- That is the allocative-prior / budget defect (`ALLOCATIVE_PRIOR_*`, D1). It is under a
  frozen prospective test with an interim after week 14.

## 2. `RB_RECEPTIONS_OPPONENT_ADJ_V1` (pricing)

**Change.** For non-Week-1 RB receptions rows, the final football mean becomes
`mean x (1 + 0.1932 x D)`:
- `D = n/(n+4) x (opponent's prior per-game RB receptions allowed / league - 1)`;
- strictly prior, same season.

It is applied in `run_pricing_v2.price()` where the other promoted syntheses set the target
mean, before mean alignment, dispersion and integer alignment, so the distribution stays
consistent.

**Code:**
- `scripts/modeling/rb_receptions_opponent_adj_v1.py`;
- context builder `scripts/build_rb_receptions_opponent_context_v1.py`, wired in
  `full-slate.yml` after the QB C2 context, before any paid odds;
- `data/rb_receptions_opponent_context.csv` is added to the replay input set.

**Audit columns** in `props_priced_clean.csv`:
- `rb_receptions_opp_adj_applied`;
- `rb_receptions_opp_adj_version`;
- `rb_receptions_opp_adj_D`;
- `rb_receptions_opp_adj_factor`.

**Fails closed when:**
- the context is missing, empty or for another slate;
- the version does not match;
- `D` is non-finite;
- an opponent is not in the context;
- zero RB receptions rows were adjusted on a non-Week-1 slate.

**Evidence** (`OPPONENT_POSITION_DEFENSE_ADJUSTMENT_RESULTS_V1.md`, preregistered at c8cdb98):
- the only market x position to pass all three gates;
- MAE improved in both folds;
- `b > 0` in both fits;
- P(improve) 0.988 on the 2026 fold.
- `b` and `k` match `data/research/opponent_position_defense/promoted_coefficients_v1.json`
  (pinned by a test).

**Forward check, 2026 week 11 FanDuel RB receptions** (n=35): MAE 1.404 → 1.405. Flat. The
adjustment is ±7% at most, a small, validated refinement.

## Validation before merge

- `compileall` and `audit_repo.py --strict`: passed.
- `tests/test_rb_model_fixes_2026_11_24.py`: 9 tests.
- The full `pytest` suite.
- A Full Slate replay of the paid week-11 Sunday boundary (run 45024058278, no new credits).
