# M95Q — Expanded Historical Stable-Workhorse Backtest Reconstruction

## Purpose

M95P showed that the broad RB workload regime can be studied across 2018-2025, but the exact M95F/M95G/M95K-style stable-workhorse model trace remained concentrated in 2023-2025. M95Q expands exact comparable late-season rotations backward before any new dynamic-prior candidate is fit.

## Precommitted design

- New historical target seasons: **2020, 2021, 2022**.
- Rebuild M95A role + opponent-defense features from leakage-safe historical weekly player stats and nflverse PBP.
- Rebuild M95B offense-defense feature families with historical PBP, PFR advanced rushing where available, and NGS where available.
- Reuse the frozen M95F raw tail scorer family and frozen calibration families:
  - 20+ = Platt
  - 25+ = football
- Rotate the original temporal protocol backward:
  - prior season + target Weeks <=12 train raw scorer;
  - target Weeks 5-12 create temporal OOF calibration rows;
  - target Weeks 13-18 are scored holdout rows.
- Apply M95G pregame roster/injury/depth semantics and the exact M95K stable-workhorse rule:
  - workhorse pregame role;
  - prior team carry leader remains available;
  - target was prior team carry leader;
  - one-game vs five-game RB-share trend >= -0.10;
  - target not OUT or DOUBTFUL.
- No sportsbook input.
- No production change.
- No feature/model-family/coefficient search.

## Mechanical parity control

Before earlier seasons are treated as comparable, the same generalized reconstruction is run on **2024 W13-18** and compared with the authoritative frozen M95G 2024 trace.

Predeclared parity requirements:

- >=95% row overlap versus the authoritative 2024 trace;
- >=98% workhorse-role agreement;
- >=95% stable-workhorse mask agreement;
- reconstructed M95F 20+ probability correlation >=.90;
- reconstructed M95F 20+ MAE <=.05.

An earlier season is considered comparable only if the global 2024 parity check passes, its direct roster join rate is >=95%, its usable raw 20+ feature count is at least 70% of the 2024 control count (and at least 8 features), and it contains at least 15 late-season stable-workhorse observations.

If parity or source coverage fails, that is a reconstruction/source failure—not a scientific failure of the RB hypothesis. Fix only mechanical/source issues before interpreting earlier-season outcomes.
