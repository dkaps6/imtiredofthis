# M95D — RB Rushing Environment / Scheme / Personnel

## Purpose

M95C found that leakage-safe rushing-environment information was more stable for broad rushing-yard mean prediction than raw recent RB efficiency, while runner-created metrics were more useful for high-end discrimination. M95D therefore tests whether the rushing environment itself can be explained better with football structure instead of simply adding more player box-score efficiency.

## Frozen scientific question

Can pregame scheme/personnel/box/tackling information improve the stable M95C environment signal prospectively, without damaging carry prediction, when evaluated in 2024 and 2025?

## Inputs

All inputs are football-only and leakage-safe.

Frozen M95B trace provides role controls and existing M95C-relevant environment features, including player/team YBC and NGS expected-rushing context where available.

New M95D source families:

1. **FTN charting via nflverse (2022-2025)**
   - motion on rushing plays
   - RPO on rushing plays
   - shotgun / under-center rushing structure
   - offensive backfield count

2. **nflverse participation**
   - defenders in box on rushing plays
   - heavy/light box rates
   - offense formation where populated
   - 11/12 personnel rushing usage where populated

3. **PFR weekly advanced defense via nflverse, if schema/source is available**
   - missed tackles
   - missed-tackle rate proxy from tackles + missed tackles

4. **PFR weekly advanced rushing**
   - actual target-game YBC/attempt is attached only as an evaluation target for the environment-mechanism test; pregame rolling YBC remains the predictor.

No sportsbook or game-market data is permitted.

## Feature families

M95D compares four fixed families:

1. `role_baseline`
2. `role_plus_m95c_environment`
3. `role_plus_environment_scheme`
4. `full_environment_matchup`

The final family adds compact football interactions such as:

- rushing environment × defensive box
- rushing environment × defensive missed-tackle tendency
- motion × defensive box
- RPO × defensive box
- team YBC × defensive box
- player RYOE × defensive missed tackling
- heavy-box offense exposure × defensive heavy-box tendency
- 11-personnel rushing usage × defensive box

This is deliberately not a giant raw-feature soup.

## Temporal protocol

- Train 2023 → test 2024
- Train 2023+2024 → test 2025
- No post-hoc 2025 tuning

Pregame rolling structural features are shifted before each target game. 2022 source data can supply prior context for 2023 training rows.

## Targets

- carries (guardrail)
- rushing yards
- YPC on 8+ carry games
- target-game YBC/attempt on 5+ carry games, used as an environment-mechanism target
- 75+ rushing-yard AUC
- 100+ rushing-yard AUC
- 20+ explosive-run AUC

## Precommitted advancement logic

Advance M95D only if the carry guard passes and either:

1. the full environment matchup improves rushing-yard MAE over the M95C-environment family in both forward test years, or
2. it improves the YBC/attempt environment mechanism in both forward test years **and** produces meaningful tail support in 2025.

Otherwise retain the M95C environment finding and reject the added scheme/personnel layer.

## Production boundary

Research only. `production_change = 0` regardless of outcome. Any useful M95D signal must later be integrated with the still-unresolved M93/M94 opportunity engine; M95D is not allowed to masquerade as a solution to the known 25+ carry underprojection problem.
