# WR Full-Stack Integration Protocol

## Status

Standing WR research protocol, frozen before WR-ND6 results. This document does not alter the WR-ND6 diagnostic gates and does not authorize any production change.

## Purpose

A diagnostic signal is not a model win. Any WR signal or interaction that survives an isolated diagnostic must subsequently prove value inside the actual projection architecture before it can be considered for production.

## Canonical integration path

Any qualifying WR research signal must be tested through the existing historical Full Slate stack rather than through a detached correction layer:

1. historical PlayerForm / TeamForm and other timestamp-safe pregame context;
2. Bayesian baseline / shrinkage;
3. canonical rules and context engine;
4. `simulation_v2` Monte Carlo as the final distribution engine;
5. receiving-yard / reception distributions and summary projections;
6. historical walk-forward evaluation against the frozen current-model baseline.

Sportsbook or player-prop information remains downstream comparison only and may not enter football projections or integration features.

## Combination rule

Combinations remain allowed and are important, but they must be earned and frozen before their combined results are observed.

A later integration branch may test:

- a qualifying signal alone;
- a second independently qualifying signal alone;
- a small predeclared combination of qualifying complementary signals;
- a predeclared interaction only when the source diagnostic explicitly authorized that interaction.

Do not generate large post-result combinations of failed or near-miss signals. Do not retune failed diagnostic thresholds in order to create a combination candidate.

For WR-ND6 specifically, a player x defense explosive interaction is authorized only if at least one player-side and at least one defense-side candidate independently pass the frozen ND6 gate.

## Mean versus distribution

The integration target must match the mechanism discovered by the diagnostic.

- Mean/opportunity signals may alter legitimate pre-Monte-Carlo expectation inputs only through a separately frozen integration test.
- Ceiling / explosive signals should first be considered as distribution-shape, variance, mixture, or upper-tail information rather than automatically shifting the mean.
- Any mean shift must be separately justified by full-stack evidence.

Rules modify simulation inputs; they do not act as a second voting model after Monte Carlo.

## Required baseline

Each integration experiment must reproduce the exact current WR baseline before evaluating the candidate. While M38 remains the WR production research baseline, exact parity is:

- receiving-yard rows = 4,647;
- receiving-yard MC MAE = 17.099904733366;
- canonical WR evaluation rows = 2,130 where the entitlement casebook is relevant.

If the production baseline changes later, the integration plan must explicitly identify and reproduce that new frozen parent before testing.

## Required evaluation

The next integration plan must freeze its decision metrics and thresholds before candidate results are visible. At minimum, evaluate the metrics appropriate to the candidate mechanism, including:

- receiving-yard MAE, RMSE, bias, and correlation;
- phase stability, including W2-18 and W13-18;
- WR1 / WR2 / WR3 behavior where relevant;
- large overprojection and underprojection tails;
- for ceiling/distribution work, calibration of upper-tail events such as 100+ yards and large residual thresholds;
- Monte Carlo distribution behavior / fair probabilities where the candidate changes distribution shape.

A candidate that improves an isolated diagnostic but fails the full-stack integration test is rejected for production.

## Production rule

Production promotion requires a separate frozen full-stack integration win. Passing a diagnostic, passing a source audit, or looking favorable versus sportsbook lines is insufficient by itself.

No post-result threshold lowering, waivers, reinterpretation, or hidden tuning is permitted.
