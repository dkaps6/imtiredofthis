# QB 2026 Week 1 — No-Odds Projection + Manual Market Audit Plan

## Status

Frozen before the 2026 Week-1 QB projection output is inspected.

## Purpose

Produce the promoted M89/M90 2026 Week-1 QB passing-yard projections independently of sportsbook data, then compare the frozen football projections to the user-supplied sportsbook screenshots downstream.

This is a live production validation / market-audit lane. It is **not** a QB mean-feature research migration and does not reopen broad QB mean research.

## Canonical production parent

`754d0f4ed34a06a65d2db36f2fb2ca10c58264ed`

This is the production `main` SHA with promoted QB M89/M90 synthesis and RB P3 Week-1 production wiring.

## Football-only projection contract

The QB Week-1 projection must be produced with the same promoted architecture used by production pricing:

1. current Ourlads roles + authoritative 2026 Week-1 schedule;
2. TeamForm + promoted M89/M90 QB team context;
3. PlayerForm / stable identities;
4. ML v2;
5. State v2;
6. empirical Bayesian baseline;
7. canonical rules/context layer;
8. `simulation_v2` Monte Carlo;
9. M89 official-attempt conversion applied to the simulated QB pass-opportunity distribution;
10. evidence-weighted ensemble;
11. promoted `QB_PASS_SYNTHESIS_V1` residual synthesis;
12. rescale the Monte Carlo distribution to the promoted synthesis mean.

No sportsbook/player-prop line, odds, game market, or user screenshot may be read by the projection script.

## Target universe

2026 Week 1, one pregame-eligible primary QB per team when the canonical PlayerForm/QB-opportunity logic resolves one.

The projection output must report at minimum:

- player / canonical key
- team / opponent
- QB role source and projected pass-attempt share
- official-attempt conversion
- Monte Carlo mean before synthesis
- ML projection
- State projection
- ensemble projection and weights/status
- promoted M89/M90 synthesis projection
- synthesis correction/version
- predicted official pass attempts
- predicted YPA
- final distribution mean / median / SD
- simulation iterations
- `football_only_no_odds = 1`
- `sportsbook_inputs_used = 0`

## Integrity gates

Before market comparison:

- Week resolves to `1`.
- At least 30 teams have a resolved eligible QB; any unresolved team must be explicitly reported rather than silently filled.
- Every projected QB row must have `qb_synthesis_applied == 1`.
- Every projected QB row must use `QB_PASS_SYNTHESIS_V1` (or the exact promoted artifact version returned by the canonical artifact loader).
- attempt conversion must be finite and in `[0.50, 1.00]`.
- final football projection must equal the promoted synthesis mean within numerical tolerance.
- no sportsbook file is present in or read by the football-projection stage.

A failure of these gates is mechanical/production-integrity failure, not a scientific QB failure.

## Manual sportsbook comparison contract

Only **after** the football-only projection CSV exists and passes integrity may the manually transcribed sportsbook lines be joined.

The current screenshots provide 31 posted QB passing-yard O/U lines across the 16 Week-1 games; ATL currently has no second QB line visible in the supplied screenshots. The comparison must preserve missing-market rows rather than invent a line.

For each posted line report:

- sportsbook line
- football projection
- projection minus line
- model distribution `P(over line)` / `P(under line)`
- displayed sportsbook over/under price when supplied
- no-vig market probability when both prices are available
- model probability minus no-vig market probability

The current screenshots show symmetric `-114/-114` on the listed standard O/U lines. The screenshots are downstream comparison evidence only and cannot alter the football projection.

## Interpretation rules

This live comparison is descriptive until outcomes occur. Large disagreements are audit targets, not automatic bets and not authorization to retune M89/M90.

If the slate shows systematic or structurally suspicious disagreement, a separate frozen production-audit migration may investigate the responsible football component (starter/role, attempts, YPA, context, distribution calibration, etc.).

Do not alter M89/M90 from the Week-1 market comparison itself.

## Parallel-lane isolation

This QB lane is independent of:

- WR-R1 2020–2025 replication;
- RB P3 current-role/carry-allocation audit.

Results, thresholds, branches, and model changes must not be blended across positions.