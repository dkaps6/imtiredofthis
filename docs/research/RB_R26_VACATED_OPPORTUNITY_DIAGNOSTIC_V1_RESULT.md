# RB R26 — Vacated Receiving Opportunity Diagnostic V1

Status: **DIAGNOSTIC / FORENSIC ONLY — NOT PROMOTION EVIDENCE**
Date: 2026-09-09
Production authority: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Research branch: `research-rb-r26-role-transition-entitlement-v1`

## Why this diagnostic exists

R25 failed its frozen receptions-specialist promotion gates. The failure was not discarded or retuned. R26 first audited whether leakage-safe RB role-transition state exists before attempting another receiving-entitlement model.

The corrected R26 source audit established a canonical transition-state population using only target-week `ACT` / `INA` roster membership plus strictly prior roster/depth state. No same-week historical depth, target-game participation, sportsbook input, or player outcome was used to create transition state.

This diagnostic then joins that already-frozen transition state to the already-exposed R25 2020-2022 prediction/outcome evidence to determine *where* the current production receiving baseline is weak. Because those outcomes were already exposed by R25, this atlas is explicitly forensic/mechanism evidence only. It cannot authorize promotion by itself.

## Authoritative lineage

### R25 prediction evidence
- Run: `34347752983`
- Head: `a74f8091edf12d3652345a945003509c58c9561c`
- Artifact: `rb-r25-receptions-specialist-v1`
- Artifact ID: `10102656324`
- Artifact SHA256: `2cc95d6dd718a40758ad3d706ae80dbeef67cb0469109ad9b228350f4e8851d3`

### Corrected R26 transition-state evidence
- Run: `34350644536`
- Head: `689943f2d121db4ee89eed88819b8168522c2c8f`
- Artifact: `rb-r26-role-transition-source-audit-v1`
- Artifact ID: `10103561555`
- Artifact SHA256: `7f71d55d2ec20175b79cd8d022f78c3ef3bd7c8099f8e62f2b717540ebfcf0b9`

## Population

R25 2020-2022 predictions joined exactly to R26 transition state:

- prediction rows: `6,732`
- rows with reception labels: `4,286`
- transition-state join coverage: `100%`
- fitted models in this diagnostic: `0`
- sportsbook inputs: `0`
- future outcomes: `0`

## Primary signal: vacated opportunity is the important transition

### Generic changed room vs stable room

Current production baseline receptions MAE:

- stable RB room: **1.2153**
- changed RB room: **1.2990**
- error enrichment: **+6.88%**

Current production baseline target MAE:

- stable RB room: **1.4319**
- changed RB room: **1.5434**
- error enrichment: **+7.79%**

This establishes that receiving entitlement is less reliable when RB-room membership changes, but generic turnover is not yet the most precise mechanism.

### Same-team incumbents when a competitor exits

Current production baseline receptions MAE:

- incumbent with one or more room exits: **1.4040**
- incumbent with no room exits: **1.2012**
- error enrichment: **+16.88%**

Current production baseline reception bias:

- incumbent with exits: **-0.6314 receptions/game**
- incumbent with no exits: **-0.3520 receptions/game**

Current production baseline target MAE:

- incumbent with exits: **1.6503**
- incumbent with no exits: **1.4250**

Current production baseline target bias:

- incumbent with exits: **-0.7520 targets/game**
- incumbent with no exits: **-0.4680 targets/game**

**Interpretation:** when a competitor leaves the RB room, the current baseline systematically fails to redistribute enough receiving opportunity to the backs who remain.

## The signal exists for both RB1 and RB2+

### RB1 incumbents

Receptions MAE:

- with exits: **1.8349**
- no exits: **1.6090**

Reception bias:

- with exits: **-1.0089**
- no exits: **-0.7990**

### RB2+ incumbents

Receptions MAE:

- with exits: **1.1810**
- no exits: **1.0023**

Reception bias:

- with exits: **-0.4361**
- no exits: **-0.1336**

Therefore this is not simply a 'lead back gets all vacated work' effect. Both lead and complementary backs can absorb receiving opportunity after a room departure.

## Generic entrant pressure is much weaker

For continuing same-team backs:

- receptions MAE with a new room entrant: **1.2549**
- receptions MAE with no entrant: **1.2468**
- enrichment: only about **+0.94%**

This is much smaller than the +16.88% exit/vacancy enrichment.

**Durable conclusion:** do not build a generic `new player entered -> suppress incumbents` rule from this evidence. The stronger state is **vacated competition**, not entrant count by itself.

## Week 1 is a distinct role-transition regime

Current production baseline receptions MAE:

- Week 1: **1.3853**
- Weeks 2+: **1.2394**
- Week-1 error enrichment: **+11.77%**

Current production baseline reception bias:

- Week 1: **-0.5794 receptions/game**
- Weeks 2+: **-0.3681 receptions/game**

The broad R23/R25 history-based redistribution does reduce mean underprediction but is not a safe Week-1 solution:

- Week-1 candidate receptions MAE: **1.4611**
- Week-1 baseline receptions MAE: **1.3853**
- candidate change: **-5.47%** (worse)

This supports the architectural distinction already adopted by the project:

- identity = **who**
- state = **when**

Week 1 is precisely where historical same-room identity should be trusted less unless current state confirms it.

## New-to-team and no-prior-history backs

The R23/R25 redistribution showed apparent MAE gains for new/low-history populations, but those gains were accompanied by harmful negative bias and worse tails/RMSE. Example: new-to-team veterans improved receptions MAE descriptively, but candidate reception bias moved to roughly **-0.80 receptions/game**. This is compression, not a safe specialist.

Therefore R26 must not treat 'new to team' or 'little history' as an instruction to suppress receiving opportunity.

A more plausible mechanism is to preserve portable player receiving identity while using current room hierarchy/state to determine whether and where that identity should express itself.

## Current hierarchy implication

The 2025+ nflverse depth-chart source is ESPN-based and date/timestamp-bearing, with `pos_slot` and `pos_rank`. That source can support a timestamp-safe current hierarchy authority for 2025+ / 2026 once an as-of-before-kickoff audit is passed.

Ourlads remains useful separately for alignment/formation labels such as LWR/RWR/SWR and matchup plumbing. R26 does not require replacing Ourlads.

## Durable R26 direction

The next candidate should **not** be generic turnover, blanket history shrinkage, or raw depth-order remapping.

The strongest supported mechanism to test is:

`finite production RB target pool`

→ detect **vacated receiving opportunity** from leakage-safe room exits

→ preserve persistent player receiving identity (`R9` is the scientifically supported identity family)

→ use current/timestamp-safe hierarchy as a reliability/state signal where available

→ redistribute only within the existing finite RB room

→ leave non-RB entitlement unchanged

→ leave receiving-yard production mean and R22 tail authority unchanged during a receptions-specialist test.

## Scientific status

This document does **not** promote R26 or any new production parameter.

It authorizes designing a separately frozen R26 candidate around vacated-opportunity redistribution and current hierarchy. Historical testing must be labeled retrospective because the historical RB-receiving outcome blocks have already been exposed across R8-R25. The strongest eventual confirmation is therefore prospective 2026 grading under a prediction lock.
