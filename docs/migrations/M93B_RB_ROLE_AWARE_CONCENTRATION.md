# Migration 93B — RB Role-Aware Backfield Concentration

## Why this test exists

M93A established that the current RB allocation is too compressed in true high-workload games. A fixed `gamma=1.2` share sharpening improved 15+, 20+, 25+ and bell-cow outcomes in both 2024 and 2025, while preserving the legacy all-player rushing scoreboard.

However, the fixed rule also made legitimate 6-10 and 11-14 carry committee outcomes worse. The signal is real, but applying it to every backfield is too blunt.

M93B asks a narrower question:

> Using only information available before kickoff, can we identify which backfields are likely to be highly concentrated and activate share sharpening only there?

## Frozen parts of the model

M93B does **not** change:

- total projected RB carry pool
- team rushing opportunity
- M91 rushing efficiency / implied YPC
- receiving projection
- defensive matchup inputs
- Monte Carlo production coefficients
- sportsbook behavior
- production ensemble weights

This is still an allocation-only experiment.

## Pregame signals

The role/concentration classifier uses only baseline projections and historical games completed before the target week. Candidate inputs include:

- baseline projected RB1 share
- projected RB1/RB2 share gap
- baseline backfield HHI / number of active projected RBs
- current depth-chart `RB1` role flag
- lead-back trailing 1/3/5-game carries
- lead-back trailing 1/3/5-game RB-pool carry share
- recent targets and receptions as a secondary three-down-role clue
- recent 15+ / 20+ carry frequency
- team trailing 1/3/5-game top-RB share
- team trailing 1/3/5-game backfield HHI
- number of RBs recently receiving carries
- current lead-back workload trend
- strongest current competitor's recent carry/share/target usage

No target-week result enters these features.

## Development / validation design

To avoid fitting the role gate directly to the 2025 season:

1. Train on 2024 Weeks 1-12.
2. Use 2024 Weeks 13-18 as an internal development holdout to select from a small predeclared grid of:
   - concentration definition (`>=65%`, `>=70%`, `>=75%` actual RB-pool top share)
   - classifier probability gate (`0.50`, `0.60`, `0.70`)
   - activated gamma (`1.10`, `1.20`, `1.30`)
3. Refit the selected classifier on all of 2024.
4. Freeze the configuration and score 2025.

The classifier is intentionally a regularized logistic model rather than a large nonlinear learner. The purpose is to test whether pregame role information contains a stable concentration signal, not to win one season through hyperparameter search.

## Development constraints

A candidate cannot win the 2024 development selection merely by improving bell-cow games while destroying committees. It must:

- improve all-RB rushing-attempt MAE
- improve all-RB rushing-yard MAE
- not worsen 0-5 carry rushing-yard MAE
- improve 15+ carry rushing-yard MAE
- limit degradation in both 6-10 and 11-14 carry rushing-yard slices to no more than 0.5 yards each

The development objective then rewards overall rushing-yard gain, 20+ carry gain, and attempt gain.

## Validation gate

The frozen 2025 candidate advances only if it:

- improves all-RB attempts
- improves all-RB rushing yards
- does not worsen all-RB rush+receiving yards
- preserves/improves the 0-5 carry slice
- improves 20+ carry rushing yards
- keeps both middle-workload slices within the development tolerance
- does not regress the legacy all-player rushing-yard guard

M93B remains research-only regardless of outcome. A passing result would only justify carrying the role-aware concentration signal forward into the later opportunity architecture; it would not be promoted directly to production.
