# Joint Pass/Receiving Conservation V1 — Frozen Implementation Semantics

This file is frozen **before the first V1 scientific run** and is part of the V1 plan. It resolves implementation details that the parent plan intentionally left to the candidate code. No value here may be changed after V1 results are visible.

## Target-share evaluation denominator

Because C1/C3 preserve the total modeled WR+TE+RB+FB receiver mass and only redistribute that mass among groups, the primary group-allocation scoreboard uses a like-for-like composition denominator on both sides:

- actual group share = group targets / (WR + TE + RB + FB actual targets)
- projected group share = group target probability / (WR + TE + RB + FB projected target probability)

Groups scored in the macro gate are `WR`, `TE`, and `RB_FB`.

The complete-team/all-receiver target shares from the ecosystem audit are still exported as a secondary diagnostic, but they are not substituted into the frozen C1/C3 macro gate after results.

## Current/B0 target probabilities

For each team-game:

1. read target shares in exact current priority order: `rules_tgt_share`, `bayes_tgt_share`, `target_share`, `tgt_share`, fallback 0;
2. apply exact promoted M38 WR sharpening `1.40 / 1.14 / 0.91 / 0.78`, preserving WR mass;
3. clip player shares to `[0, 0.95]`;
4. if player-share sum exceeds `0.95`, scale all player shares proportionally to total `0.95`;
5. residual target probability is `1 - player-share sum`.

C1/C3 start from these final B0 player probabilities, preserve non-pass-catcher probability, and redistribute only the WR/TE/RB/FB mass as specified in the parent plan.

## Empirical-Bayes group history

- Team history window: most recent **8** completed regular-season team-games strictly before cutoff.
- League-prior rows: immediately prior regular season plus completed current regular-season games strictly before cutoff.
- Grouping uses the historical row's already-observed past-game position; target-week position comes only from the pregame universe.
- Pseudo-count: **105 targets** exactly.
- If league prior has zero valid target opportunities, no candidate group calibration is produced for that team-game and C1/C3 retain B0 probabilities; this is counted and reported.
- If one calibrated group has no eligible target-week player with positive B0 probability, its unassignable probability goes to residual rather than to equal-share backups.

## Conserved receiving-yard draw

For C2/C3, all values below are fixed before the run.

### Player inputs

- Catch rate uses current priority order: `rules_catch_rate`, `bayes_receptions_per_target`, `receptions_per_target`, `catch_rate`, fallback `0.64`.
- Catch rate is clipped using current simulation semantics to `[0.001, 0.999]`.
- YPT uses: `rules_ypt`, `bayes_ypt`, `ypt`, fallback `7.5` when non-finite or <=0.
- Implied YPR = `YPT / catch_rate`, clipped to **[3.0, 35.0] yards/reception**.
- Player volatility multiplier uses `rules_volatility_mult`, clipped to `[0.75, 1.50]`, identical to B0.

### Residual receiver

- catch rate = `0.64`
- YPT = `7.5`
- implied YPR = `7.5 / 0.64 = 11.71875`
- volatility multiplier = `1.0`

### Raw yardage draw

For each player in each iteration:

- receptions ~ Binomial(targets, catch_rate)
- if receptions == 0: raw receiving yards = 0 exactly
- otherwise:
  - `mu = receptions * implied_YPR * pass_efficiency_shock`
  - `sd = max(3.0, sqrt(receptions) * implied_YPR * 0.55) * volatility_multiplier`
  - raw receiving yards = `max(0, Normal(mu, sd))`

The residual receiver uses the same formula with its frozen defaults.

Pass-efficiency shock remains the current simulation draw: `Normal(1.0, 0.09)` clipped to `[0.65, 1.35]`.

## Mean anchoring and exact conservation

For each team-game:

1. Sum all modeled + residual raw receiver-yard arrays into `raw_team_receiver_yards`.
2. Determine the frozen pregame mean anchor:
   - 2024-2025: exact M89/M90 `football_synthesis` for that team-week;
   - 2020-2023: exact B0 primary-QB MC mean from the same team-game run.
3. Constant scale = `anchor / mean(raw_team_receiver_yards)`.
4. Multiply **every modeled receiver and residual receiver iteration** by that same constant.
5. Candidate QB passing yards are defined as the iteration-wise sum of those scaled receiver arrays.

If raw team receiver mean is non-finite or <=0, that team-game is an integrity failure; no fallback scale is allowed.

The 2024-2025 B0 comparison QB distribution is the exact current primary-QB MC array multiplied by one constant so its mean equals the same M89/M90 `football_synthesis` anchor. This isolates distribution shape while holding the QB mean fixed.

## CRPS

Empirical CRPS for samples `x` and realized value `y` is:

`mean(|x-y|) - 0.5 * mean(|x-x'|)`

The implementation may use the algebraically equivalent sorted-sample O(n log n) form. No normal approximation is permitted.

Paired CRPS bootstrap:
- resampling unit = aligned team-game;
- resamples = **10,000**;
- seed = **5601**;
- reported probability = fraction of resamples where mean `(candidate CRPS - B0 CRPS) < 0`.

## Interval coverage

Central interval quantiles:
- 50%: p25-p75
- 80%: p10-p90
- 90%: p05-p95

Coverage is inclusive at both bounds. Absolute calibration error is `abs(empirical_coverage - nominal_coverage)`.

## MC execution

- iterations = **2,000** per team-game/variant for the V1 research run;
- target-week seed = existing convention **42 + week**;
- C2 and C3 use separate deterministic RNG streams derived from that week seed (`+200000` and `+300000`, respectively) so neither candidate's random-number consumption can alter the other candidate;
- B0 uses exact current `simulation_v2.simulate(..., seed=42+week)`;
- C1 uses exact current simulation with candidate target probabilities and `seed=42+week`, with M38 sharpening bypassed only because C1 probabilities already contain the exact M38 within-WR proportions.

## Scientific stopping rule

Mechanical repairs may restore the code to this frozen specification. They may not alter the specification. Any scientific change to window, pseudo-count, group definition, YPR bounds, residual defaults, volatility, MC seeds, or gates requires a new named migration after V1 is dispositioned.