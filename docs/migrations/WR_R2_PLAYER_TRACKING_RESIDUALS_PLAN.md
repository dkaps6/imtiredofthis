# WR-R2 — Player-Level Tracking Residuals — Frozen Plan

## Why this is next

WR-R1 confirmed exact M38 across 2020-2025 (`12,396` WR receiving-yard player-games) with no retuning. M38 beat M37 in receiving-yard MAE in all six target seasons. The simple post-M38 opportunity families (ND3/ND5) and simple explosive-history family (ND6) did not produce an actionable signal.

WR-R2 therefore tests **genuinely new player-level pregame receiving-tracking information**, not another target-share multiplier or another simple box-score window.

This is also the first WR branch to retain a durable individual-player error profile (MAE, bias, tail-miss rates) alongside population metrics. That profile is diagnostic only and cannot be used as a contemporaneous feature unless separately rebuilt walk-forward.

## Canonical parent

- WR-R1 result commit: `436c9a14d6ec820c88c2ff4367e7ac6b0938c98f`
- WR-R1 source run: `34058453941`
- WR-R1 artifact: `9997412312` (`wr-r1-multiseason-2020-2025`)
- M38 remains the canonical WR mean hierarchy.

## Source

NFL Next Gen Stats weekly receiving data through `nflreadpy`, using only rows from games strictly before the target game.

Attempt source window: 2019-2025 so 2020 Week 1 can use prior-season information if the source exists.

No target-game NGS row may be used as a feature. No future game, postseason participation, sportsbook input, or realized target-game tracking value may enter prediction diagnostics.

If historical NGS is unavailable for part of 2020-2025, the source window is reported mechanically. We do not fabricate/backfill tracking values or select a more favorable season window after results.

## Frozen history window

Last **8 eligible prior player games**, crossing seasons when available. No 3/4/6/10/12-game window search after results.

## Frozen player signals

A. `NGS_ADOT_PRIOR8` — average intended air yards / target depth

B. `NGS_YACOE_PRIOR8` — average YAC above expectation

C. `NGS_SEPARATION_PRIOR8` — average receiver separation

D. `NGS_IAY_SHARE_PRIOR8` — share of intended air yards

All hypotheses are directional: higher pregame player tracking value is expected to associate with a more positive M38 receiving-yard residual / more upper-tail outcomes.

No interactions in WR-R2. No post-result combinations with failed ND3/ND5/ND6 near-misses.

## Outcomes

Primary continuous outcome:

`rec_yards_residual = actual_receiving_yards - M38_receiving_yards_projection`

Tail outcomes:

- `UNDER25`: residual >= +25 yards
- `UNDER50`: residual >= +50 yards
- `ACTUAL100`: actual receiving yards >= 100

## Individual-player diagnostic profile

From the full WR-R1 M38 casebook, report per player:

- games
- M38 MAE
- signed bias (`projection - actual`)
- median absolute error
- 90th percentile absolute error
- 20+ / 30+ / 40+ yard miss rates

This profile is for diagnosis and prioritization only. It is not an upstream feature in WR-R2.

## Frozen source/integrity requirements

- exact WR-R1 M38 receiving-yard casebook parity
- target seasons remain 2020-2025
- no target/future NGS leakage
- no sportsbook input
- no model fitting
- source mapping and per-signal coverage reported before scientific interpretation
- a signal must have at least `3,000` eligible player-games and >= `0.60` coverage of the canonical M38 receiving-yard casebook to be scientifically scoreable

If those requirements fail, disposition is mechanical/source failure, not a scientific signal failure.

## Frozen actionable signal gate

A signal is actionable only if **all** are true:

- eligible rows >= `3,000`
- coverage >= `0.60`
- Spearman(signal, receiving-yard residual) >= `+0.08`
- top-vs-bottom quartile residual gap >= `+4.0 yards`
- UNDER25 enrichment >= `1.25x`
- at least one of UNDER50 or ACTUAL100 enrichment >= `1.25x`
- residual gap > 0 in at least 4 evaluable target seasons
- residual gap > 0 in 2024
- residual gap > 0 in 2025

No threshold lowering after results.

## Frozen dispositions

- one passing signal: `<SIGNAL>_TRACKING_SIGNAL`
- multiple passing signals: `MULTIPLE_PLAYER_TRACKING_SIGNALS`
- none: `NO_ACTIONABLE_PLAYER_TRACKING_SIGNAL`
- source/integrity failure: `PLAYER_TRACKING_SOURCE_OR_INTEGRITY_FAILURE`

A diagnostic pass does **not** change M38. A passing signal only authorizes a separately frozen full-stack integration test at its natural layer (mean, efficiency, variance/right-tail) using PlayerForm/TeamForm -> Bayesian -> rules/context -> `simulation_v2` Monte Carlo.
