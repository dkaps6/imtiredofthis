# PRISM Team Scoring-Territory Drive Creation V1 — Result

Date: 2026-10-16

Status: **TERMINALLY CLOSED / ALL THREE CELLS / NO PRODUCTION CHANGE**

Model:
**PRISM — Player Role & Individual Situation Model**

Frozen contract:
`docs/research/PRISM_TEAM_SCORING_TERRITORY_DRIVE_CREATION_V1_FREEZE.md`

Freeze commit:
`82f74487e68729efdbcc0868968fd6ddccd4eef1`

Canonical scored execution:
- run `39000352640` — **SUCCESS**;
- exact head `2baba745aa0fe50a9bcb88600051f28a783a197d`;
- artifact `12088459232`;
- digest `sha256:4816dbb88dd85911999171bad60700ab4eccbd62266396af302d689f661be881`.

Overall disposition:

`PRISM_TEAM_DRIVE_CREATION_CLOSED`

All three frozen cells closed:
- `RB_RUSH_INSIDE10_PRISM_TEAM_DRIVE_CREATION_CLOSED`;
- `RB_RECEIVING_REDZONE_PRISM_TEAM_DRIVE_CREATION_CLOSED`;
- `TE_RECEIVING_REDZONE_PRISM_TEAM_DRIVE_CREATION_CLOSED`.

## Frozen question

Conditional on the exact persisted canonical PRISM parent opportunity quantity, does the team's strictly-prior ability to create drives reaching the relevant scoring territory improve the next-game finite high-value opportunity pool?

Baseline:
- strictly-prior same-season league high-value / parent-opportunity rate;
- multiplied by the canonical `authority_denominator_mean`.

Candidate:
- the same baseline high-value rate;
- multiplied by the frozen same-team / league scoring-territory drive-entry-rate ratio;
- RB rushing territory = drives reaching opponent 10;
- RB / TE receiving territory = drives reaching opponent 20;
- candidate rate clipped to [0,1] exactly as frozen.

Player allocation:
- identical baseline/candidate qualified Stage-1 participation + Stage-2 exact active-set expected named share.

The candidate did not use:
- the team's recent direct high-value event rate;
- player recent scoring-area share;
- opponent/environment state;
- touchdown outcomes;
- sportsbook inputs;
- 2026 outcomes.

## Results

### RB rushing — inside-10

2024 development:
- team-games 416;
- positive-HV team-games 338;
- player-games 1,386;
- players 156;
- mean team drive-entry rate 0.222065;
- mean league drive-entry rate 0.216175;
- mean territory multiplier 1.027052;
- clipping share 0.0000;
- team CRPS gain **-0.043558**;
- team AE gain **-0.055967**;
- player CRPS gain **-0.004848**;
- player AE gain **-0.005305**;
- supported-player mean CRPS gain **-0.008379**;
- median supported-player gain **-0.004353**;
- positive supported-player share 0.3730.

2025 independent confirmation:
- team-games 416;
- positive-HV team-games 337;
- player-games 1,419;
- players 154;
- mean team drive-entry rate 0.214632;
- mean league drive-entry rate 0.212754;
- mean territory multiplier 1.008795;
- clipping share 0.0000;
- team CRPS gain **-0.022685**;
- team AE gain **+0.007145**;
- player CRPS gain **-0.000314**;
- player AE gain **+0.002544**;
- supported-player mean CRPS gain **-0.000160**;
- median supported-player gain **-0.001481**;
- positive supported-player share 0.4504;
- team-cluster CRPS-gain 95% CI **[-0.066222, 0.023085]**;
- game-cluster team CI **[-0.075755, 0.030928]**;
- player-cluster CI **[-0.008108, 0.007791]**;
- player game-cluster CI **[-0.006847, 0.006275]**;
- equal-player CI **[-0.007640, 0.007316]**.

Disposition:
`RB_RUSH_INSIDE10_PRISM_TEAM_DRIVE_CREATION_CLOSED`.

The 2025 expected-count AE improved slightly, but the primary proper score remained negative and all dependence-aware confirmation intervals failed the frozen positive-lower-bound gate. AE cannot rescue CRPS.

### RB receiving — red-zone targets

2024 development:
- team-games 416;
- positive-HV team-games 391;
- player-games 1,386;
- players 156;
- mean team drive-entry rate 0.367962;
- mean league drive-entry rate 0.358218;
- mean territory multiplier 1.026972;
- clipping share 0.0000;
- team CRPS gain **-0.098349**;
- team AE gain **-0.138330**;
- player CRPS gain **-0.000448**;
- player AE gain **-0.001903**;
- supported-player mean CRPS gain **-0.000863**;
- median supported-player gain **-0.000339**;
- positive supported-player share 0.4841.

2025 independent confirmation:
- team-games 416;
- positive-HV team-games 393;
- player-games 1,419;
- players 154;
- mean team drive-entry rate 0.372512;
- mean league drive-entry rate 0.368587;
- mean territory multiplier 1.010610;
- clipping share 0.0000;
- team CRPS gain **-0.076567**;
- team AE gain **-0.098317**;
- player CRPS gain **-0.001266**;
- player AE gain **-0.001164**;
- supported-player mean CRPS gain **-0.000780**;
- median supported-player gain **-0.000989**;
- positive supported-player share 0.4046;
- team-cluster CI **[-0.157742, 0.006453]**;
- game-cluster team CI **[-0.151482, 0.000568]**;
- player-cluster CI **[-0.003379, 0.000882]**;
- player game-cluster CI **[-0.003065, 0.000538]**;
- equal-player CI **[-0.002859, 0.001311]**.

Disposition:
`RB_RECEIVING_REDZONE_PRISM_TEAM_DRIVE_CREATION_CLOSED`.

### TE receiving — red-zone targets

2024 development:
- team-games 416;
- positive-HV team-games 391;
- player-games 1,334;
- players 138;
- mean team drive-entry rate 0.367962;
- mean league drive-entry rate 0.358218;
- mean territory multiplier 1.026972;
- clipping share 0.0000;
- team CRPS gain **-0.098349**;
- team AE gain **-0.138330**;
- player CRPS gain **+0.001583**;
- player AE gain **+0.003456**;
- supported-player mean CRPS gain **+0.001826**;
- median supported-player gain **-0.001299**;
- positive supported-player share 0.4628.

2025 independent confirmation:
- team-games 416;
- positive-HV team-games 393;
- player-games 1,305;
- players 141;
- mean team drive-entry rate 0.372512;
- mean league drive-entry rate 0.368587;
- mean territory multiplier 1.010610;
- clipping share 0.0000;
- team CRPS gain **-0.076567**;
- team AE gain **-0.098317**;
- player CRPS gain **+0.000770**;
- player AE gain **+0.003882**;
- supported-player mean CRPS gain **+0.000360**;
- median supported-player gain **+0.000022**;
- positive supported-player share 0.5000;
- team-cluster CI **[-0.157090, 0.006812]**;
- game-cluster team CI **[-0.155746, -0.000712]**;
- player-cluster CI **[-0.004734, 0.006371]**;
- player game-cluster CI **[-0.003045, 0.004487]**;
- equal-player CI **[-0.005903, 0.006272]**.

Disposition:
`TE_RECEIVING_REDZONE_PRISM_TEAM_DRIVE_CREATION_CLOSED`.

The small positive TE player-level point estimates in both seasons do **not** qualify:
- the upstream team red-zone pool is worse in both years;
- the 2025 team proper-score confirmation fails;
- player dependence-aware intervals cross zero;
- the frozen contract requires the whole finite-pool-to-player chain to pass.

Do not cherry-pick the TE row-level improvement.

## Comparison with the closed direct team high-value persistence V1

The drive-territory mechanism was generally **less harmful** than direct recent high-value event-rate persistence:
- RB rushing team CRPS gain moved from -0.082065 / -0.091510 to -0.043558 / -0.022685;
- receiving team CRPS gain moved from -0.160973 / -0.155009 to -0.098349 / -0.076567.

That is useful mechanism evidence:
- scoring-territory creation is a more football-grounded state than direct event-rate persistence;
- but this frozen transmission law still does not beat the strict-prior league baseline.

“Less bad” is not qualification.

## Scientific interpretation

Two distinct team-side opportunity-quality candidates have now failed:

1. direct same-team recent high-value event-rate persistence;
2. same-team scoring-territory drive-entry propensity transmitted multiplicatively into the high-value rate.

The second result narrows the problem:
- the canonical parent opportunity quantity remains intact;
- the player allocation bridge remains intact;
- the failure already appears in constructing the **team finite high-value pool**;
- simple recent team persistence, even when moved upstream to scoring-territory entry, is insufficient.

This exact drive V1 is terminally closed.

## Anti-rescue

Do not:
- change last-4/min-3;
- smooth/shrink the drive rate;
- tune the multiplier;
- switch to an additive/logit transform;
- change reach-10 / reach-20 thresholds;
- swap to plays-inside-territory as a post-score rescue of this same mechanism;
- reinsert direct team high-value event persistence;
- use player recent high-value share;
- subset lead backs / TE1;
- use TD outcomes;
- use sportsbook variables;
- use 2026 outcomes;
- weaken support or inference gates.

## Integrity

Canonical run:
- strict repository audit: PASS;
- freeze ancestry: PASS;
- immutable parent/digest checks: PASS;
- focused mechanics tests: PASS;
- parent denominator drift: 0 within frozen tolerance;
- Stage-2 allocation drift baseline vs candidate: 0;
- target-game outcomes used as predictors: 0;
- fitted coefficients: 0;
- threshold searches: 0;
- direct recent-team HV-rate candidate used: false;
- player recent HV-share candidate used: false;
- opponent/environment used: false;
- TD outcomes used: false;
- sportsbook inputs: 0;
- OddsAPI calls: 0;
- 2026 outcomes: 0;
- production changed: false.

## Next boundary

Do not reopen either closed team-side persistence mechanism.

The next architecture layer is opponent/environment transmission, but the project instruction for this opportunity-quality work explicitly prohibited adding opponent/environment **yet**. Therefore no opponent/environment candidate is scored in this result.

A future continuation may freeze one genuinely distinct opponent/environment mechanism only after treating both team-side opportunity-quality closures above as immutable scientific parents.
