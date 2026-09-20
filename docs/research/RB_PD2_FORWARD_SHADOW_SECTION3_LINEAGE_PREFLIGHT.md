# RB-PD2 Forward / Shadow Confirmation V1 — §3 Historical Difficulty Lineage Preflight

**STATUS: PRE-IMPLEMENTATION LINEAGE AUDIT ONLY. NO CANDIDATE SCIENCE CHANGE. NO PRODUCTION CHANGE.**

This note records the collision-free §3 preparation completed while Claude's §7 capture repair is still pending. It does **not** implement §3, §8, or §9 and does not alter the frozen forward plan.

Authority:
- frozen plan: `docs/research/RB_PD2_FORWARD_SHADOW_CONFIRMATION_V1_PLAN.md`
- freeze commit: `0de96f69194f1fca242515b613b814eab0c22d35`
- active branch: `research-rb-pd2-forward-shadow-confirmation-v1`

## 1. Exact authorized 2025 projection lineage

The repository already has a leakage-safe 2025 component-rebuild path suitable for the frozen §3.1 history contract.

Canonical reconstruction pattern:

1. `scripts/backtest/build_historical_inputs.py --season 2025 --prior-season 2024 --weeks 1-18`
2. combine authoritative schedule / team-week history for the required historical window;
3. build canonical historical player logs with `scripts/backtest/historical_player_logs.py`;
4. run:
   `scripts/backtest/walk_forward.py --season 2025 --prior-season 2024 --weeks 1-18 --iterations 2000`
5. use the exact frozen `rush_yards` ensemble weights from `data/model_ensemble_weights.csv`.

The same pattern is already exercised by:
- `.github/workflows/backtest-historical-benchmark-clean-rebuild-v1.yml`
- `.github/workflows/research-distribution-widening-2024-2025-holdout-v1.yml`

The frozen 2025 RB rushing-yard weights are:

- MC = `0.5569542426070742`
- ML = `0.4430457573929258`
- State = `0.0`
- fit scope = `all_2024_oos_frozen_for_2025`
- promotion lineage = `RB_STACK1_RUN_33535308110_FOR_P3`

No 2025 outcome enters those weights.

## 2. Leakage boundary in the historical component builder

`scripts/backtest/walk_forward.py` constructs each target week from an explicit pregame universe and uses the deterministic weekly simulation seed `42 + week`.

`scripts/backtest/component_predictions.py::predict_week()` builds MC / ML / State component predictions first. Only after those projections exist does it call `build_actual_rows()` and merge the realized target-week result into the component frame for grading.

Therefore, for §3 history:
- the 2025 target mean must be formed from the pregame component projections plus the already-frozen 2025 weights;
- realized 2025 rushing yards may be joined only after that projection exists;
- the resulting projection error is grading/history data, never a predictor input for the same game.

## 3. §3 history row contract

The forward difficulty-state builder should consume one football row per eligible historical RB/HB/FB `rush_yards` player-game with at least:

- season
- week
- team
- opponent
- event/game identity when available
- `player_clean_key`
- position
- football projection mean used for that game
- realized rushing yards, only after completion
- explicit projection-lineage label / code SHA or artifact manifest where available

For 2025 the football projection mean is the frozen generic-ensemble value:

`ensemble_proj = 0.5569542426070742 * mc_proj + 0.4430457573929258 * ml_proj`

with State weight exactly zero, subject to the repository's existing ensemble availability contract.

## 4. Stable player identity — do not silently re-key forward history

Forward §3 should use the canonical `player_clean_key` carried by the leakage-safe component output and by the accepted §7 capture contract.

Reason: the older historical PD2 helper in `evaluate_rb_pd2_multiseason_current_route_v1.py` can recompute a simplified alphanumeric key from `player` when building its 2021-2024 panel. That was valid inside that frozen historical experiment, but a forward state spanning 2025 historical components and live 2026 captures must not create two identity systems.

Required fail-closed behavior:
- nonblank `player_clean_key`;
- no fallback that silently changes a known canonical clean key;
- duplicate football identity under one clean key is an integrity error;
- any explicit alias migration must be separately auditable rather than inferred on the fly.

## 5. Exact frozen difficulty mechanic to preserve

For each target row, using completed prior eligible rows for the same canonical player only:

- history window = last 8;
- minimum prior games = 4;
- target historical error = `projection_mean - actual_rush_yards`;
- `prior8_yard_mae = mean(abs(prior errors))`.

The percentile mapping must reuse the exact frozen `strict_prior_difficulty_scores()` semantics from `scripts/research/evaluate_rb_pd2_yard_difficulty_mc_width_v1.py`:

- maintain a sorted reference of finite prior `prior8_yard_mae` values;
- reference minimum = 100;
- score all rows in a target `(season, week)` before inserting any row from that week;
- `difficulty_score = bisect_right(reference, prior8_yard_mae) / len(reference)`;
- persist `difficulty_reference_n`;
- persist `difficulty_reference_max_ord`;
- require `difficulty_reference_max_ord < target_ord`.

No coefficient, onset, history window, minimum-history rule, reference floor, or percentile convention may change.

## 6. 2025 -> 2026 chronology

The forward state should seed from authorized completed 2025 history and then advance through 2026 only with legitimate completed pregame-lineage rows.

Already-played 2026 rows:
- may become later predictor history only if their pregame football projection lineage can be reconstructed leakage-safely;
- may enter the rolling percentile reference only after completion;
- may **never** be backfilled as prospective confirmation observations.

If 2026 Week 1 is used as prior predictor history, mechanically reassert the frozen Week-1 P3 = STACK1 parent parity before admitting it.

For the first prospective target row, the difficulty state must therefore be derivable entirely from rows whose outcomes were known before the target lock.

## 7. Separation from §7 / §8 / §9

This preflight intentionally does not bind to Claude's unaccepted first §7 artifact shape.

After §7 is accepted, §3 should provide the historical state needed by §9:
- canonical player identity;
- prior eligible game count;
- prior8 yard MAE;
- difficulty score;
- reference N;
- reference max ordinal.

§8 then applies the frozen `widen_mean_neutral()` transform to the exact persisted §7 empirical baseline array. §9 combines the accepted §7 capture, §3 state, and §8 candidate into the immutable pregame lock.

No exact empirical draw array is required to build §3 history itself.

## 8. Tests to require when §3 implementation begins

At minimum:

1. exact 2025 `rush_yards` weight constants / fit-scope / lineage are asserted;
2. target-game outcome cannot enter projection construction;
3. same-player last-8 / min-4 semantics match the frozen historical helper;
4. same-week reference rows are deferred until the entire week is scored;
5. `difficulty_reference_max_ord < target_ord` for every scoreable target;
6. canonical `player_clean_key` is preserved across 2025 history and synthetic 2026 captures;
7. a cross-season 2025 -> 2026 last-8 case is correct;
8. a completed 2026 row cannot enter a prior target's history/reference;
9. an already-played 2026 row can be admitted as later predictor history only with explicit pregame-lineage certification;
10. Week-1 P3 / STACK1 parity is required if Week 1 contributes 2026 history.

## 9. Current implementation boundary

Do not implement §3 against the rejected §7 contract.

Wait for Claude's repaired §7 SHA, inspect it mechanically, and accept/integrate only if the six required §7 repair tests pass. Then implement §3 / §8 / §9 against the accepted artifact contract.

This note changes no frozen science and exposes no prospective candidate outcome.
