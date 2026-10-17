# PRISM Team High-Value Creation Conditional Parent V1 — Result

Date: 2026-10-16

Status: **TERMINALLY CLOSED / ALL THREE CELLS / NO PRODUCTION CHANGE**

Model:
**PRISM — Player Role & Individual Situation Model**

Frozen contract:
`docs/research/PRISM_TEAM_HIGH_VALUE_CREATION_CONDITIONAL_PARENT_V1_FREEZE.md`

Frozen mechanism commit:
`b15573fb7e5e0662a2d6b4a894cc2a13814c3df5`

Canonical scored execution:
- run `38999998874` — **SUCCESS**;
- exact head `2f5ff029f017bfd7b25d667be9d56b3905f87c73`;
- artifact `12087739890`;
- digest `sha256:ebee1d07e0edf5ed92335818aa8bb5c2995ccdaa6bdb40be15c267987b0d2b20`.

Overall disposition:

`PRISM_TEAM_HIGH_VALUE_CREATION_CLOSED`

All three predeclared cells closed:
- `RB_RUSH_INSIDE10_PRISM_TEAM_HIGH_VALUE_CREATION_CLOSED`;
- `RB_RECEIVING_REDZONE_PRISM_TEAM_HIGH_VALUE_CREATION_CLOSED`;
- `TE_RECEIVING_REDZONE_PRISM_TEAM_HIGH_VALUE_CREATION_CLOSED`.

## Frozen question

Conditional on the already-persisted canonical PRISM parent opportunity quantity, did the team's own strictly-prior same-season high-value opportunity creation rate improve next-game high-value opportunity count?

Baseline:
- strictly-prior same-season **league** high-value / parent-opportunity rate;
- multiplied by the exact same canonical `authority_denominator_mean`.

Candidate:
- same team's frozen last-4 / minimum-3 same-season high-value / parent-opportunity rate;
- multiplied by the exact same canonical `authority_denominator_mean`.

Player allocation:
- identical baseline/candidate qualified Stage-1 participation + Stage-2 exact active-set expected share.

No player recent scoring-area share, opponent/environment, touchdown outcome, sportsbook, OddsAPI or 2026 outcome entered the candidate.

## Results

### RB rushing — inside-10

2024 development:
- team-games 416;
- positive-HV team-games 338;
- player-games 1,386;
- players 156;
- team CRPS gain **-0.082065**;
- team AE gain **-0.088291**;
- player CRPS gain **-0.013105**;
- player AE gain **-0.009146**;
- supported-player mean CRPS gain **-0.013764**;
- median supported-player gain **-0.009770**;
- positive supported-player share 0.3571.

2025 confirmation:
- team-games 416;
- positive-HV team-games 337;
- player-games 1,419;
- players 154;
- team CRPS gain **-0.091510**;
- team AE gain **-0.071513**;
- player CRPS gain **-0.008304**;
- player AE gain **-0.001932**;
- supported-player mean CRPS gain **-0.008882**;
- median supported-player gain **-0.006477**;
- positive supported-player share 0.3817;
- team-cluster CRPS-gain 95% CI **[-0.131038, -0.051043]**;
- game-cluster team CI **[-0.147623, -0.033158]**;
- player-cluster CI **[-0.015479, -0.001024]**;
- player game-cluster CI **[-0.016088, -0.000748]**;
- equal-player CI **[-0.016320, -0.001420]**.

Interpretation:
the candidate was not merely unsupported; it was directionally worse with confirmation intervals below zero at both team and player dependence-aware levels.

### RB receiving — red-zone targets

2024 development:
- team-games 416;
- positive-HV team-games 391;
- player-games 1,386;
- players 156;
- team CRPS gain **-0.160973**;
- team AE gain **-0.188864**;
- player CRPS gain **-0.002408**;
- player AE gain **-0.001578**;
- supported-player mean CRPS gain **-0.002394**;
- median supported-player gain **-0.000732**;
- positive supported-player share 0.4524.

2025 confirmation:
- team-games 416;
- positive-HV team-games 393;
- player-games 1,419;
- players 154;
- team CRPS gain **-0.155009**;
- team AE gain **-0.202835**;
- player CRPS gain **-0.000635**;
- player AE gain **-0.001830**;
- supported-player mean CRPS gain **-0.000452**;
- median supported-player gain **-0.001143**;
- positive supported-player share 0.4122;
- team-cluster CI **[-0.236401, -0.069188]**;
- game-cluster team CI **[-0.238247, -0.068987]**;
- player-cluster CI **[-0.002925, 0.001703]**;
- player game-cluster CI **[-0.002840, 0.001684]**;
- equal-player CI **[-0.002596, 0.001781]**.

### TE receiving — red-zone targets

2024 development:
- team-games 416;
- positive-HV team-games 391;
- player-games 1,334;
- players 138;
- team CRPS gain **-0.160973**;
- team AE gain **-0.188864**;
- player CRPS gain **-0.003073**;
- player AE gain **+0.000839**;
- supported-player mean CRPS gain **-0.002181**;
- median supported-player gain **-0.002928**;
- positive supported-player share 0.4215.

2025 confirmation:
- team-games 416;
- positive-HV team-games 393;
- player-games 1,305;
- players 141;
- team CRPS gain **-0.155009**;
- team AE gain **-0.202835**;
- player CRPS gain **-0.001554**;
- player AE gain **+0.001453**;
- supported-player mean CRPS gain **-0.001947**;
- median supported-player gain **-0.003028**;
- positive supported-player share 0.4583;
- team-cluster CI **[-0.235464, -0.069260]**;
- game-cluster team CI **[-0.242332, -0.065655]**;
- player-cluster CI **[-0.006884, 0.004245]**;
- player game-cluster CI **[-0.006399, 0.003484]**;
- equal-player CI **[-0.007412, 0.004024]**.

## Scientific interpretation

The team's own recent high-value event rate is **not** a useful persistence state for PRISM's next-game high-value opportunity pool under this frozen construction.

The strongest evidence is upstream at the team level:
- every 2024 development team CRPS gain was negative;
- every 2025 confirmation team CRPS gain was negative;
- every 2025 team- and game-cluster confidence interval was entirely below zero.

That means the failure is not caused by the player allocation bridge. The player allocation was held fixed between baseline and candidate, and the candidate already degraded the finite high-value pool before allocation.

This exact mechanism is terminally closed. Do not rescue it with:
- another rolling window;
- shrinkage/blending;
- lead-back / TE1 subgroups;
- another yardline band;
- end-zone targets;
- touchdowns;
- opponent variables inserted into this same formula;
- gate weakening.

## Concurrent duplicate-lane reconciliation

A later duplicate freeze at `02e25a29e26c0562f298fdd71a517f7602f424cf` and run `38999926446` were created after the first binding freeze.

They are **NONCANONICAL SENSITIVITY ONLY** because:
- the project instruction was to freeze one mechanism before scoring;
- `b15573fb...` is the first mechanism freeze in branch ancestry;
- the later duplicate changed the count law and player mapping after that binding freeze.

Its direction was also CLOSED across all three cells, so it does not contradict the canonical scientific conclusion, but it must not become a second authority.

## Integrity

Canonical run:
- strict repository audit: PASS;
- parent denominator drift: 0 within frozen tolerance;
- baseline/candidate Stage-2 player allocation drift: 0;
- target-game outcomes used as predictors: 0;
- fitted coefficients: 0;
- thresholds searched: 0;
- player recent high-value-share candidate: false;
- opponent/environment used: false;
- TD outcomes used: false;
- sportsbook inputs: 0;
- OddsAPI calls: 0;
- 2026 outcomes: 0;
- production changed: false.

## Next scientific frontier

Do **not** retest direct same-team high-value-rate persistence.

The next genuinely different source-ready PRISM opportunity-quality mechanism is **team scoring-territory drive creation**:
- RB rushing quality: strictly-prior team drives reaching opponent 10;
- RB/TE receiving quality: strictly-prior team drives reaching opponent 20.

That next mechanism must be frozen separately before any scoring and must remain conditional on the same canonical parent opportunity quantity and the same qualified player allocation.
