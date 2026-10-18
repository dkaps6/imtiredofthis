# PRISM QB Passing-TD Market-Specific V1 — Source Result

Date: 2026-10-18

Status: **SOURCE READY / RESEARCH ONLY / NO TD PROJECTION SCORED**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_QB_PASS_TD_MARKET_SPECIFIC_V1_SOURCE_AUDIT`

Frozen source plan:
`af709a937f61c39240b73a1d6960ab47e60eb152`

Canonical certified run:
- run `39152776987` — SUCCESS;
- exact head `3240df850a7a663abd0063ecb49e53c61af99238`;
- artifact `12145852250`;
- artifact name `prism-qb-pass-td-market-specific-v1-source-39152776987`;
- digest `sha256:d4f5b9c41f0fa6f79b9ae8949f39a7df615ee0dadb3b43081d4cb1a58f780de8`;
- expired = false at certification;
- focused source tests passed;
- strict repository audit passed.

Disposition:

`PRISM_QB_PASS_TD_MARKET_SPECIFIC_SOURCE_READY`

## Why this source audit exists

PRISM has a promoted QB passing-yards architecture, but no dedicated certified
QB passing-touchdown projection layer.

The repository can acquire sportsbook `player_pass_tds` prices, but sportsbook
data are downstream only and are not projection authority.

The current joint simulation:
- projects QB passing yards;
- projects rushing/receiving anytime TDs;
- does not expose a QB passing-TD simulation market.

The current ATD science explicitly excludes passing touchdowns.

A prior QB research script computed recent pass-TD rate only as a contextual
feature for YPA. That did not certify a TD model.

Therefore the passing-TD gap is real and market-specific.

## Historical football source

Source:
nflverse / nflfastR play-by-play through the repository's
`nflreadpy.load_pbp` path.

Seasons:
- 2022 REG;
- 2023 REG;
- 2024 REG;
- 2025 REG.

No 2026 outcomes.

The source audit resolved exact event semantics for:
- regular-season type;
- offensive team;
- pass attempt;
- passing touchdown;
- stable passer ID;
- yardline from goal;
- game identity where available.

No sportsbook fields were used.

## Source coverage

| Season | Pass attempts | Distinct passers | Teams | Passer-ID coverage | Pass-TD coverage | Yardline coverage |
|---|---:|---:|---:|---:|---:|---:|
| 2022 | 19,455 | 114 | 32 | 100.0000% | 100.0000% | 100.0000% |
| 2023 | 19,807 | 119 | 32 | 100.0000% | 100.0000% | 100.0000% |
| 2024 | 19,224 | 110 | 32 | 100.0000% | 100.0000% | 100.0000% |
| 2025 | 18,822 | 102 | 32 | 100.0000% | 100.0000% | 100.0000% |

All frozen per-season source gates passed.

The persisted named-QB team-game source includes:
- pass attempts;
- passing TDs;
- pass-TD rate;
- pass attempts inside opponent 20 / 10 / 5;
- passing TDs from inside opponent 20 / 10 / 5;
- team pass attempts;
- team passing TDs;
- passers with attempts;
- top passer attempt share.

Count and scoring-area monotonicity integrity passed.

## Strictly-prior named-QB history feasibility

2024:
- 604 target QB-games have >=3 strictly-prior same-QB games;
- 72 distinct QBs represented;
- frozen feasibility gates passed.

2025:
- 596 target QB-games have >=3 strictly-prior same-QB games;
- 77 distinct QBs represented;
- frozen feasibility gates passed.

This is source support only.

No relationship between prior QB history and future TD outcomes was scored.

## Anti-retest / non-authorities

This result does not revive or redefine:
- M89/M90 passing-yards science;
- closed QB Stage-2 allocation;
- closed QB Stage-1 player-specific mechanism utility;
- QB game-script / YPA / chaos work as TD science;
- generic ATD;
- player high-value opportunity persistence;
- player high-value conversion identity;
- team direct high-value creation;
- team scoring-territory drive creation.

It also does not authorize:
- deriving passing TDs from passing yards with a league multiplier;
- using sportsbook pass-TD lines as an upstream parent;
- using ATD probability as a proxy;
- using 2026 outcomes to define the mechanism.

## Next authorized scientific question

The source is sufficient to freeze one player-first QB passing-TD mechanism.

The predictive architecture must keep separate:

`PASS OPPORTUNITY -> PASSING SCORING ACCESS -> NAMED-QB OWNERSHIP -> QB-SPECIFIC CONVERSION -> PASS-TD COUNT`

Before scoring, the next contract must:
- tie pass opportunity to the real PRISM QB opportunity lineage;
- use strictly-prior football state only;
- shrink named-QB evidence toward a broader prior;
- separate scoring-area access from TD conversion;
- preserve passing yards and passing TDs as separate markets;
- use named-player protections, not only pooled QB error;
- fit no candidate before the formula and qualification gates are frozen.

## Integrity

- predictive TD relationships scored: **0**;
- TD count/probability models fit: **0**;
- thresholds searched: **0**;
- QB passing-TD production outputs created: **0**;
- 2026 outcomes used: **0**;
- Week-6/Week-7 outcomes used: **0**;
- sportsbook inputs used: **0**;
- OddsAPI calls: **0**;
- production changed: **false**.
