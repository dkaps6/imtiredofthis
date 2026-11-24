# Two RB model fixes promoted 2026-11-24

**Operator instruction (2026-11-24):** "go ahead and ship both fixes into the model."

Both fixes:
- passed preregistered out-of-sample gates before this date;
- are football-only (no sportsbook variable reaches either one);
- are versioned and write audit columns;
- fail closed where an input is missing.

## 1. `RB_CARRY_SHARE_RECENCY_V2` (PlayerForm)

**V1 was replaced the same day, before any paid run used it.** The PR #716 review found three
gaps between V1 and its validation, and all three were confirmed in the code:
- V1's test changed every position's share;
- production's last-3 window skipped zero-touch games;
- there was no anomaly gate.

V2 fixes all three. It passed a preregistered retest:
`RB_CARRY_SHARE_RECENCY_END_TO_END_{PREREGISTRATION,RESULTS}_V2.md`.

**Change.** For an RB/FB with >= 3 current games:
- His last 3 offensive-snap games with today's team, carries filled as 0, are each
  state-labelled from nflverse snap counts and the official injury report.
- If all three are `HEALTHY_FULL`, the current carry share that feeds Bayes becomes
  `0.15 x season-to-date + 0.85 x last 3`.
- Otherwise the season-to-date share is kept, as the AGENTS.md anomaly gate requires.

Everything else is unchanged: prior share, game counts, other metrics, other positions.

**Code:**
- `scripts/modeling/rb_carry_recency_v2.py`, one module used by both production and the
  validation;
- wired in `scripts/player_form_v2.py::rb_recency_windows` / `build_rb_recency_windows`.

**Audit columns** in `player_form_consensus.csv`:
- `rush_share_current_season`;
- `rush_share_last3`;
- `rb_carry_recency_window_weeks`;
- `rb_carry_recency_window_states`;
- `rb_carry_recency_applied`;
- `rb_carry_recency_status`, one of `APPLIED`, `WITHHELD_ANOMALY`,
  `WITHHELD_SOURCE_INCOMPLETE`, `FEWER_THAN_3_GAMES`, `NO_SEASON_SHARE`,
  `SOURCE_UNAVAILABLE:<why>` or `NOT_RB`;
- `rb_carry_recency_version`.

**Sources and failure:**
- Snap counts, players and injuries come from nflreadpy. They are an optional enrichment.
- If a source fails, every RB keeps season-to-date with `SOURCE_UNAVAILABLE:<why>`, and the
  failure is printed in the PlayerForm step.
- If a team's completed game is missing from the snap source, that team's RBs keep
  season-to-date with `WITHHELD_SOURCE_INCOMPLETE`.
- Kill switch: `RB_CARRY_SHARE_RECENCY=0`.

**Evidence (V2 test, the deployed intervention, carries MAE):**

| holdout | gain | 95% CI | lead backs | lead 95% CI |
|---|---|---|---|---|
| 2025 | +0.046 | [+0.022, +0.070] | +0.059 | [+0.014, +0.104] |
| 2026 wk 4-10 | +0.040 | [+0.020, +0.060] | +0.066 | [+0.018, +0.113] |

The gate holds recency back on 17% (2025) and 12% (2026) of RB rows.

**Limitation, stated plainly:**
- Recency does **not** fix the week-11 Emmett Johnson miss.
- With V2 the gate withholds his weeks 10-11 surge as `UNRESOLVED_CONTEXT`.
- The root cause is the Bayesian layer giving backs with no history the position-group mean
  share. That is the allocative-prior / budget defect (D1), under a frozen prospective test
  with an interim after week 14.

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

| check | result |
|---|---|
| `compileall` and `audit_repo.py --strict` | passed |
| `audit_2026_production_readiness.py` (local) | READY, no findings |
| `tests/test_rb_model_fixes_2026_11_24.py` | 17 passed (V2 recency: game states, zero-touch window, team codes, RB-only apply, source failure, incomplete snaps, kill switch; opponent adjustment; wiring) |
| full `pytest` | 1,317 passed, 3 failed. **All 3 fail identically without this change**: two need git history this sandbox lacks, one is a pre-existing workflow-contract assertion on the `FETCH_LIVE_ODDS` line. |
| **pricing loop, week-11 inputs**: `run_pricing_v2.price()` on the archived week-11 rule-stage inputs (pass_yards excluded: QB synthesis needs player logs outside the replay set) | 1,610 side-rows priced. **35 RB receptions players adjusted** (140 side-rows). **0 rows touched outside RB receptions.** Final `model_proj` = ensemble x factor, to 1.2e-5. |
| Full Slate no-odds dry run, week 12 (run 45461996815, **V1**) | PlayerForm: `RB_CARRY_SHARE_RECENCY_V1 applied to 99 RB/FB rows`. Context step: passed. The run stopped at the strict readiness audit on `injury_status_all_null`: the week-12 injury report is not published until Wednesday. Environmental, not this change. |
| Full Slate replay of the paid week-11 boundary (run 45461066034) | **Not possible for a completed week.** It stops at `Ourlads role universe is empty` before PlayerForm, because the current-roles availability artifact is empty once every game is final. This replay limitation is noted for the infrastructure backlog. |

**The paid-run exercise of the pricing seam** is the week-12 Thursday capture. The local pricing
run above exercised the same loop.
