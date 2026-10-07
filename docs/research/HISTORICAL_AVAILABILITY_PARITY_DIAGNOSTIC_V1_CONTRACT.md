# HISTORICAL AVAILABILITY PARITY DIAGNOSTIC V1 — FROZEN CONTRACT

Date: 2026-10-07  
Branch: `research-historical-availability-parity-diagnostic-v1`

## Purpose

Quantify how much of the 2026 Weeks 1-4 individual-player opportunity
compression in the historical replay is caused by a known reconstruction
mismatch:

- historical replay universe currently admits nflverse weekly roster statuses
  `ACT` and `INA`;
- canonical live production already excludes definitive unavailable players
  before opportunity allocation through the promoted current-player-availability
  stack.

This is a reconstruction diagnostic only. It does not reopen availability
science and does not authorize a production change.

## Frozen populations

Run the exact same 2026 Weeks 1-4 player opportunity audit twice.

### A. Baseline historical universe

Current frozen historical replay semantics:
- same schedule;
- same nflverse weekly rosters;
- same positions;
- roster status may be `ACT` or `INA`;
- same protected football stack.

### B. Explicit-status parity universe

Derive only by filtering the exact baseline pregame universe with the same
nflverse weekly roster source:

- retain rows whose exact source roster status is `ACT`;
- remove rows whose exact source roster status is `INA`;
- do not use weekly stats, target-week outcomes, snaps, sportsbook information,
  or any postgame field to select rows;
- preserve all other baseline universe columns and football inputs unchanged.

Every baseline player row must resolve to exactly one `ACT` or `INA` source
status. Missing/ambiguous status fails closed.

This ACT-only reconstruction is a **minimal parity diagnostic**, not a claim that
it fully reproduces the richer live Ourlads/injury/official-inactives/T-75
availability hierarchy.

## Football stack

Both variants use the exact same:
- historical context construction;
- canonical MC;
- M38;
- TE-R5P;
- WR-R15;
- explicit entitlement simulation;
- QB expected-pass-attempt authority;
- seeds and iterations;
- no sportsbook inputs.

The only allowed difference is removal of explicit `INA` universe rows before
context/opportunity allocation.

## Required outputs

1. `historical_availability_status_map.csv`
2. `baseline_player_opportunity_rows.csv`
3. `act_only_player_opportunity_rows.csv`
4. `historical_availability_parity_group_comparison.csv`
5. `historical_availability_parity_mass_audit.csv`
6. `historical_availability_parity_summary.json`

## Required comparisons

On exact baseline `ACT` player identities that also exist in ACT-only output,
report by position + opportunity type:

- paired row count;
- baseline MAE / ACT-only MAE;
- MAE delta = baseline - ACT-only;
- baseline bias / ACT-only bias;
- baseline and ACT-only prediction-vs-actual correlation;
- mean predicted opportunity change;
- actual-zero rate;
- mean prediction at actual zero;
- active/nonzero actual opportunity mean;
- active/nonzero prediction bias;
- high-workload-bin baseline vs ACT-only bias.

Also report:
- number of baseline `INA` player rows;
- baseline predicted opportunity mass assigned to `INA` rows;
- share of total modeled player opportunity assigned to `INA` rows;
- team-week counts with positive inactive opportunity mass;
- QB identity changes between baseline and ACT-only;
- ACT-only player identities not present in baseline audit output;
- common-player linked yard/count error correlation where the frozen parent
  replay supplies it.

## Interpretation

Possible dispositions:

1. `HISTORICAL_AVAILABILITY_MISMATCH_EXPLAINS_MAJORITY_OF_COMPRESSION`
2. `HISTORICAL_AVAILABILITY_MISMATCH_EXPLAINS_PART_OF_COMPRESSION_RESIDUAL_PLAYER_ALLOCATION_REMAINS`
3. `HISTORICAL_AVAILABILITY_MISMATCH_NOT_MATERIAL_PLAYER_ALLOCATION_REMAINS`
4. `HISTORICAL_AVAILABILITY_PARITY_DIAGNOSTIC_INVALID`

No numerical cutoff is frozen for these labels. The result document must report
the raw paired changes and use the labels only as descriptive disposition, not
as an automatic promotion gate.

## Prohibited

- no parameter fitting;
- no threshold selection;
- no model/production promotion;
- no paid OddsAPI;
- no sportsbook input upstream;
- no target/future-week result in universe filtering;
- no change to promoted live availability semantics;
- no change to M38 / WR-R15 / TE-R5P / QB / RB science;
- no weakening of closed research decisions.

## Next-step rule

Only after this diagnostic may we decide whether another individual-player
opportunity mechanism is justified. If the ACT-only reconstruction removes most
of the compression, do not invent new science to solve an already-promoted
availability problem. If substantial compression survives among explicit active
players, localize that residual before proposing a new player-state mechanism.
