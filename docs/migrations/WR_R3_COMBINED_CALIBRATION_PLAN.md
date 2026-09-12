# WR-R3 Combined Calibration — Frozen Plan

**STATUS: RESEARCH ONLY — NOT PROMOTED.** This plan is frozen before any candidate result is computed. Production M38/WR-R15, weights, sportsbook gates, and Full Slate remain unchanged.

## Authority and question

This finishes the exact next step authorized by `WR_R3_PLAYER_ERROR_PERSISTENCE_RESULT.md` and reconciled in `docs/research/overnight/WR_R3_COMBINED_CANDIDATE_STATUS.md`.

R3 detected three strictly-prior same-player signals across 2020-2025: signed error persistence, absolute-error/difficulty persistence, and 30+ yard extreme-miss persistence. The test here is whether a **predeclared combination** of those signals improves the exact M38 full-stack receiving-yard architecture without violating aggregate, player-level, phase, role, or tail guards.

No sportsbook line, odds, price, result-derived threshold, target-game outcome, or target-game PBP may enter the candidate.

## Frozen parents / evidence

- Branch cut from `main` SHA `55650f2d0d7d8c4c30f68ed33833c76becd22ff9`.
- Exact M38 parent: `b98518d97b3038f471aee9ae3201009b2c70bb29`.
- R3 feature source: WR-R1 run `34058453941`, artifact `wr-r1-multiseason-2020-2025`, paired M38 casebook.
- Historical target seasons: 2020-2025; target week uses only strictly earlier games.
- Canonical 2025 M38 receiving-yard parent must reproduce before candidate grading:
  - all-position `rec_yards` rows = `4,647`;
  - MC MAE = `17.099904733366`;
  - WR evaluation rows = `2,130`;
  - WR RMSE = `25.196099510686`;
  - WR bias = `-5.238640833495`;
  - WR correlation = `0.567945850835`.

Any parent/cohort drift is a hard failure and candidate results are non-authoritative.

## Strict-prior R3 features

For each player-game, sorted by `(season, week)`, use at most the immediately previous 8 same-player M38 games, minimum 4. The current game is never included.

- `prior8_m38_bias = mean(mc_proj_m38 - actual_m38)`.
- `prior8_m38_mae = mean(abs(mc_proj_m38 - actual_m38))`.
- `prior8_m38_miss30_rate = mean(abs(mc_proj_m38 - actual_m38) >= 30)`.
- `prior_games = 4..8`.

For Lane B percentiles, each target row may compare its strictly-prior feature value only with feature rows from chronologically earlier player-games across the historical casebook. No current/future feature row enters its reference distribution. Require at least 100 strictly-prior reference rows; otherwise Lane B multiplier = 1.0.

## Candidate: one frozen combined run

Both lanes are frozen now and evaluated together. Component arms may be emitted for diagnosis, but **no component result may be used to retune the combined candidate**.

### Lane A — conservative pre-MC mean calibration

Interpret positive prior bias as historical overprojection and negative prior bias as historical underprojection.

Frozen receiving-yard correction:

`confidence = prior_games / 8`

`yard_delta = clip(-0.20 * confidence * prior8_m38_bias, -8.0, +8.0)`

M38 target entitlement is unchanged. Convert `yard_delta` only into the WR's pre-MC receiving-efficiency input:

`delta_ypt = yard_delta / max(expected_m38_targets, 1.0)`

`candidate_rules_ypt = clip(base_rules_ypt + delta_ypt, 2.0, 20.0)`

`expected_m38_targets` must come from the same M38 sharpened target-share allocator and team plays/pass-rate inputs used by `simulation_v2`; target shares, team target mass, catch rate, plays, pass rate, and non-WR inputs are unchanged.

This lane is inserted after canonical Bayesian/rules context has produced the legitimate pre-MC inputs and **before** `simulation_v2.simulate()`. It is not a post-simulation projection patch.

### Lane B — player-specific mean-neutral uncertainty widening

Compute strictly-prior empirical percentiles for `prior8_m38_mae` and `prior8_m38_miss30_rate` as described above:

`difficulty_score = percentile(prior8_m38_mae)`

`extreme_score = percentile(prior8_m38_miss30_rate)`

`uncertainty_score = 0.5 * difficulty_score + 0.5 * extreme_score`

`width_mult = 1.0 + 0.30 * clip((uncertainty_score - 0.50) / 0.50, 0.0, 1.0)`

Thus lower-half uncertainty rows are unchanged and the most uncertain rows widen by at most 30%.

Apply only as:

`candidate_rules_volatility_mult = clip(base_rules_volatility_mult * width_mult, 0.75, 1.50)`

No target share, mean YPT, catch rate, team volume, or role input is changed by Lane B. The evaluator must separately emit a Lane-B-only arm using the same seeds and enforce the mean-neutrality gate below, because the simulator's non-negative yard floor can otherwise create a small mechanical mean drift.

## Monte Carlo / pairing

- Use exact M38 `simulation_v2` from parent `b98518d...`.
- 2,000 iterations per target week.
- seed = `42 + week`, matching canonical walk-forward.
- Baseline, Lane-B-only, and combined arms use identical week seeds.
- Candidate modifications occur only in pre-MC metric inputs described above.
- No ensemble, ML, State, WR-R15, TE-R5P, sportsbook, or downstream betting threshold is part of this R3 experiment.

## Frozen evaluation populations

### Primary full-stack integration population

Exact canonical 2025 M38 WR receiving-yard rows: 2,130. Aggregate/phase/role/tail/distribution gates are evaluated here.

### Six-season individual-error population

2020-2025 M38 WR receiving-yard rows from the same historical reconstruction lineage. Player-history features remain strict-prior. This population is used only for season consistency and player-level gates; the 2025 canonical full-stack population remains the primary architecture gate.

## Frozen gates

The candidate can receive `WR_R3_COMBINED_CALIBRATION_INTEGRATION_WIN` only if **every hard family** below passes. Otherwise disposition is `NO_ACTIONABLE_WR_R3_COMBINED_CALIBRATION`. A pass still does not promote production.

### A. Parent / leakage hard gates

1. Exact 2025 parent row/metric reproduction listed above within `1e-9` for row count/MAE and `1e-6` for RMSE/bias/correlation.
2. Exactly 2,130 canonical 2025 WR evaluation rows.
3. Zero same/future-game history violations.
4. Zero sportsbook inputs.
5. M38 target entitlement/team mass unchanged by candidate to `1e-12`.

### B. Aggregate 2025 point-accuracy gates

1. Combined MAE improves by at least 1.0% vs baseline.
2. Combined RMSE is non-worse vs baseline.
3. Absolute bias is non-worse vs baseline.
4. Correlation is non-worse within tolerance `0.005`.
5. W2-18 MAE non-worse.
6. W13-18 MAE non-worse.
7. At least 2 of WR1/WR2/WR3 role slices have non-worse MAE; no role slice may worsen by more than 1.0%.

### C. Large-miss / tail guards

On canonical 2025 WR rows, combined candidate must not worsen pooled rates of absolute errors >=20, >=30, or >=40 yards. It also must not worsen:

- 50+ yard underprojection rate (`actual - prediction >= 50`), or
- error rate among actual 100+ receiving-yard games.

### D. Distribution / Lane-B gates

Using raw 2,000-draw distributions on canonical 2025 WR rows:

1. Lane-B-only pooled signed mean shift vs baseline must have absolute value <=0.25 yards.
2. Lane-B-only mean absolute row-level mean shift vs baseline <=0.50 yards.
3. Absolute error of empirical 80% interval coverage versus nominal 0.80 is non-worse for all eligible rows.
4. The same 80% coverage-gap metric is non-worse in the top quartile of strictly-prior `uncertainty_score`.
5. Combined 80% coverage-gap is non-worse than baseline.

### E. Individual / season gates

Across 2020-2025 strict-prior eligible WR rows:

1. At least 6,000 eligible player-games (same order-of-magnitude floor used by prior WR player-calibration work).
2. Median qualifying-player MAE delta (combined - baseline) < 0 among players with at least 8 eligible target games.
3. Combined MAE improves in at least 4 of 6 seasons.
4. 2024 and 2025 must both improve.
5. Pooled >=20, >=30, and >=40-yard miss rates are all non-worse.

## Output / disposition contract

Persist at minimum:

- result JSON with every frozen gate and disposition;
- 2025 paired row-level baseline/Lane-B/combined point metrics and interval diagnostics;
- 2020-2025 paired row-level baseline/combined predictions with R3 features;
- aggregate, season, phase, role, tail, player, and distribution summaries;
- leakage/source audit proving strictly-prior construction and no sportsbook use.

No threshold, coefficient, cap, percentile rule, cohort, or gate may be changed after candidate outputs are visible. Any follow-up requires a separately frozen experiment.