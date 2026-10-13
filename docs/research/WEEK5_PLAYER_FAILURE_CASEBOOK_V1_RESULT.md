# WEEK 5 PLAYER FAILURE CASEBOOK V1 — RESULT

Date: 2026-10-13

Status: **TOP-40 PLAYER FAILURE DECOMPOSITION COMPLETE. NO WEEK-5 OUTCOME TUNING.**

Frozen protocol:
- `docs/research/WEEK5_PLAYER_FAILURE_CASEBOOK_V1_PROTOCOL.md`

Canonical authority:
- workflow run: `38397361891`
- exact head: `e21d801ffa4af7f830e56caab38288287868b14f`
- artifact: `11812403368`
- artifact digest: `sha256:4cdef29912f35368473d57e15b38efdc3dd0f75e897bbbdc3b97fe1ff8f67670`
- workflow conclusion: `SUCCESS`

## Case universe

- top 40 settled Week-5 player-market rows by absolute final-model error;
- canonical pregame full-slate authority unchanged;
- no sportsbook refetch;
- no model fitting;
- production unchanged.

## Attribution totals

Across the 40 largest misses:

- **EFFICIENCY_DOMINANT: 19**
- **OPPORTUNITY_DOMINANT: 14**
- **MIXED_OPPORTUNITY_EFFICIENCY: 7**

This confirms that Week-5 tail failures were not one-dimensional. Opportunity
remains a major source of error, but large player-level misses also include a
substantial efficiency tail.

## By market / position

QB pass yards:
- efficiency-dominant: **7**
- opportunity-dominant: **5**
- mixed: **2**

QB rush yards:
- efficiency-dominant: **2**
- mixed: **1**

RB rush yards:
- opportunity-dominant: **4**
- efficiency-dominant: **3**

RB rush+rec:
- one opportunity-dominant;
- one efficiency-dominant;
- one mixed.

WR receiving yards:
- opportunity-dominant: **4**
- efficiency-dominant: **4**
- mixed: **2**

TE receiving yards:
- efficiency-dominant: **1**
- mixed: **1**

## Watson case

Deshaun Watson, CLE vs NYJ, pass yards:

- final model projection: **238.23**
- actual: **155**
- absolute final-model error: **83.23**
- predicted attempts: **24.77**
- actual attempts: **33**
- predicted YPA: **7.28**
- actual YPA: **4.70**
- opportunity contribution: **+59.92 yards**
- efficiency contribution: **-85.33 yards**

Label:
`MIXED_OPPORTUNITY_EFFICIENCY`

Interpretation:
- the model underpredicted passing opportunity;
- realized passing efficiency collapsed enough to overwhelm the additional
  attempts;
- this was not an alt-line-selection failure;
- repeated Watson exposure across multiple tickets created concentration risk
  around one wrong player thesis.

Do not convert this single game into a QB retune.

## Examples

Large opportunity-dominant misses include:
- Matthew Stafford pass yards;
- Jameis Winston pass yards;
- George Pickens receiving yards;
- Braelon Allen rushing yards.

Large efficiency-dominant misses include:
- Drake Maye pass yards;
- Jordan Love pass yards;
- Tyler Shough pass yards;
- Bo Nix pass yards;
- Matthew Golden receiving yards;
- Christian Watson receiving yards.

Jahmyr Gibbs rush+rec was mixed, with substantial error from both rushing /
receiving opportunity and efficiency.

## Roadmap interpretation

The Week-5 casebook is consistent with the broader decomposition:

1. individual opportunity allocation remains a first-order problem;
2. opportunity uncertainty is real and player-specific;
3. extreme efficiency outcomes remain an independent tail problem;
4. generic matchup multipliers should not be used to explain either class;
5. large raw model-vs-line disagreement is not sufficient confidence.

The strongest independently replicated new evidence remains:
- player opportunity-share volatility across 2023-2025;
- beta-binomial allocation uncertainty historical PASS for RB carry share;
- beta-binomial allocation uncertainty historical PASS for WR target share.

Therefore the next model-development step should continue the preregistered
mean-preserving opportunity-uncertainty path rather than tune Week-5 failures.

## Scientific boundary

No Week-5 case may select:
- a coefficient;
- a threshold;
- a new source;
- a player subgroup;
- a matchup multiplier;
- a production change.

Any next candidate must be separately frozen and evaluated on independent
historical/prospective evidence.
