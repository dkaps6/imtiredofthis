# Discrete Count Mean Alignment V1 — Frozen Plan

Date frozen: 2026-09-26

Status: **FROZEN BEFORE RESULT — RESEARCH ONLY**

Branch: `research-discrete-count-mean-alignment-v1`

## Structural problem

Production `run_pricing_v2.py` aligns a raw Monte Carlo distribution to the final football mean by multiplying every draw by:

`scale = target_mean / mc_mean`

That operation is appropriate for continuous yardage distributions, but it is also applied to discrete count markets.

For `receptions` and `rush_att`, the simulator produces integer counts. Multiplicative mean alignment can therefore turn physically discrete draws such as 3 receptions or 12 carries into values such as 3.7 receptions or 11.4 carries before fair probabilities are calculated.

This study asks whether that support violation materially harms distribution quality.

## Scope

Primary count markets:
- `receptions`
- `rush_att`

Historical seasons:
- 2024
- 2025

No Week-3 outcomes are used.

No production code is changed.

No sportsbook input enters football simulation.

## Historical authority

Rebuild the same deterministic 2,000-draw historical Monte Carlo distributions used by Historical Fair Probability Reconstruction V1:
- 2024: prior season 2023
- 2025: prior season 2024
- same `walk_forward.py`
- same `persist_historical_simulated_outcomes_v1.py`
- same frozen production ensemble weights
- same free Action Network-derived market archive where available

The old raw-array artifact from run `34712931786` has expired, so this study must rebuild the arrays from the existing deterministic workflow rather than approximate them.

## Arms

### A0 — current production-style continuous alignment

For raw integer MC draw `x_i`:

`z_i = x_i * target_mean / mc_mean`

This is the current pricing semantics.

### A1 — integer-preserving largest-remainder alignment

Start from the exact same continuous aligned value `z_i`.

1. `floor_i = floor(z_i)`
2. Compute `K = round(sum(z)) - sum(floor)`
3. Add one count to the `K` draws with the largest fractional remainders.
4. Stable draw index is the deterministic tie-breaker.

Properties:
- every candidate draw is a non-negative integer;
- each draw differs from its continuously aligned counterpart by less than 1 count;
- the candidate sample total equals `round(sum(z))`;
- candidate mean differs from the frozen target mean by at most `0.5 / N`;
- no parameter, threshold, coefficient, or random seed is fit.

No alternate rounding scheme may be tried after results are seen.

## Primary football-distribution metrics

For both `receptions` and `rush_att`:
- empirical CRPS vs actual count;
- 80% central interval coverage;
- 90% central interval coverage;
- mean absolute target-alignment error;
- fraction of current aligned draws that are non-integer;
- candidate integer-support audit.

Report:
- 2024 separately;
- 2025 separately;
- pooled 2024-2025.

## Receptions sportsbook-probability secondary metric

The free historical market archive contains receptions but not rush-attempt props.

For matched receptions rows only:
- same sportsbook line and odds;
- same target football mean;
- compare A0 vs A1 empirical `P(over)`;
- Brier score;
- log loss;
- count of rows whose over/under side probability changes;
- frozen STRONG/LEAN decision metrics may be reported descriptively, but no betting threshold is tuned.

No synthetic rush-attempt lines are allowed.

## Frozen gates

Mechanical gates:
1. raw `receptions` and `rush_att` MC arrays are integer-valued;
2. A1 arrays are integer-valued and non-negative;
3. A1 mean error to the frozen target is <= `0.5 / 2000 + 1e-12`;
4. A0 reproduces current multiplicative alignment exactly;
5. outcome/line information is not used by either alignment transform;
6. sportsbook inputs to football simulation = 0.

Science gates for a production-integration follow-up:
1. pooled receptions CRPS improves;
2. pooled rush-att CRPS improves;
3. neither market's CRPS worsens in either individual season;
4. pooled receptions Brier score improves on matched archived prop rows;
5. pooled receptions log loss is non-worse;
6. receptions Brier is non-worse in both 2024 and 2025.

All six science gates must pass to authorize a **separate** production-integration test.

Passing this study does not itself change production.

## Dispositions

If the current arm creates fractional count support but science gates fail:

`COUNT_SUPPORT_CONTRADICTION_CONFIRMED_DISCRETE_REPAIR_FAILED_CLOSED`

If mechanical and all science gates pass:

`DISCRETE_COUNT_MEAN_ALIGNMENT_V1_QUALIFIED_FOR_INTEGRATION_TEST`

If raw arrays are not actually discrete or lineage cannot be rebuilt:

`DISCRETE_COUNT_MEAN_ALIGNMENT_V1_NOT_INTERPRETABLE`

## No-rescue rules

Do not:
- search alternate rounding algorithms;
- tune by player/position/volume;
- use different methods for receptions and carries;
- use sportsbook lines inside the transform;
- optimize STRONG/LEAN thresholds;
- fit on 2026 outcomes;
- modify Week-3 RB Vacancy V1 or public-intent labels;
- change production from this diagnostic alone.
