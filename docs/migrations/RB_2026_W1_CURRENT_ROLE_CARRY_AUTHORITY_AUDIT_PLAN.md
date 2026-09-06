# RB 2026 Week 1 — Current-Role / Carry-Authority Audit Plan

## Status

Frozen before any 2026 Week-1 audit output is inspected.

## Purpose

Audit whether the promoted Week-1 RB P3 production path is actually giving current pregame backfield state enough authority when allocating carries among RB/FB players.

This is a production-integration / mechanism audit. It is **not** a sportsbook-calibration exercise, not a new point model, and not authorization to alter P3.

## Canonical parent

Production `main` SHA:

`754d0f4ed34a06a65d2db36f2fb2ca10c58264ed`

Promoted Week-1 RB route:

- version `RB_P3_SYNTHESIS_V1`
- route `WEEK1_STACK_OVERRIDE`
- football-only no-odds context from `scripts/run_rb_week1_no_odds.py`

## Why this audit is materially new

Historical RB role work must not be repeated or re-labeled.

Prior durable evidence:

1. RB-ND2B proved timestamp-safe 2025 pregame depth state is reconstructable at high coverage, while warning that depth rank is a role signal rather than a deterministic carry rule.
2. STACK6 / STACK6B tested lagged situational and compact secondary-back role models. The bidirectional compact model failed frozen football-first retention gates; P3 remained champion.
3. STACK6B showed a directional clue: prior situational role information was more useful for identifying false-high secondary-back workload than for justifying carry expansion. That finding is diagnostic only and may not be converted into a new rule here.
4. The live 2026 Week-1 code path preserves Ourlads depth alignment in `depth_role`, but PlayerForm derives `model_role` from historical usage and writes `role = model_role`. The audit must quantify the live consequence rather than assume it is either correct or incorrect.

Therefore this audit asks a narrower current-production question: **what current depth/availability information reaches the final Week-1 carry allocation, where does historical usage override it, and how large is the live exposure?**

## No-fit / no-market contract

- Model fit: `0`.
- Hyperparameter search: `0`.
- Threshold search: `0`.
- Weight search: `0`.
- Sportsbook/player-prop inputs upstream: `0`.
- Manual FanDuel screenshots are not read by this workflow.
- No P3 projection is changed.
- No role redistribution is applied.

This audit can diagnose integration/authority gaps. It cannot declare a new RB projection model superior without a separately frozen historical/prospective test.

## Required live inputs

Build the same Week-1 football context used by production before P3:

1. current Ourlads roles/status;
2. authoritative 2026 Week-1 team-week map;
3. TeamForm / preseason priors;
4. weather, injuries and coverage context used by the canonical stack;
5. PlayerForm / identities;
6. Bayesian, ML, State and rules components;
7. `simulation_v2` Monte Carlo;
8. promoted P3 Week-1 no-odds output.

## Required player-level audit fields

For every current RB/HB/FB row on all 32 teams, report where available:

- player, team, opponent;
- Ourlads position / position group;
- Ourlads `depth_slot`, `depth_index`, `depth_chart_role`, normalized current `role`, and status;
- PlayerForm `depth_role`;
- PlayerForm `model_role` and effective `role`;
- historical/preseason `rush_share`;
- Bayesian rush share;
- rules rush share / rules role;
- P3 projected carries (`stack_att`);
- P3 rushing yards / synthesis projection;
- implied P3 YPC;
- team RB projected carry pool;
- player share of that projected RB carry pool;
- projected-carry rank within current backfield;
- current-depth rank versus model-role rank mismatch flag;
- current-depth rank versus projected-carry rank mismatch flag;
- active/inactive state and any positive projection for an inactive player.

## Required team-level diagnostics

For all 32 teams report:

- number of current active RB/HB/FB players;
- projected team RB carry pool;
- current RB1 projected carries/share;
- current RB2 projected carries/share;
- current RB3+ projected carries/share;
- whether the current depth-1 player is the projected carry leader;
- count of depth/model-role mismatches;
- count of depth/projected-carry-order mismatches;
- any inactive player with positive projected carries;
- concentration of projected carries (HHI).

## Static production-path audit

Record, without modifying code:

1. whether `run_player_form_v2.py` preserves Ourlads role in `depth_role`;
2. whether it replaces effective `role` with usage-derived `model_role`;
3. whether `simulation_v2.py` directly references `depth_role` in carry allocation;
4. whether `simulation_rules.py` directly references `depth_role` in rushing-share logic;
5. which share fields are present immediately before simulation (`rush_share`, `bayes_rush_share`, `rules_rush_share`, etc.).

This is code-path evidence, not a statistical model result.

## Frozen integrity gates

The audit is mechanically valid only if all are true:

- Week resolves to 1.
- P3 output covers all 32 teams.
- Ourlads-to-P3 identity coverage >= 95% of P3 RB/FB rows.
- Every P3 row has `sportsbook_inputs_used == 0`.
- Every P3 row uses `RB_P3_SYNTHESIS_V1` and `WEEK1_STACK_OVERRIDE`.
- P3 rushing-yard projection remains exactly equal to its frozen Week-1 full-stack parent as required by production.
- No audit script alters any model input or projection.

Failure is mechanical/source/integrity failure, not scientific evidence against P3.

## Frozen interpretation

The audit will classify the live production mechanism, not choose a replacement model.

Possible conclusions:

- `CURRENT_ROLE_AUTHORITY_PRESENT_AND_COHERENT` — current role is directly represented in the effective allocation path and live ordering is broadly coherent.
- `CURRENT_ROLE_PRESERVED_BUT_NOT_DIRECT_ALLOCATION_INPUT` — current role exists in the data but the effective carry allocator is driven by other usage/share fields.
- `CURRENT_ROLE_INTEGRATION_INCOMPLETE` — current-role identity/status coverage or path wiring is materially incomplete.
- `MECHANICAL_AUDIT_FAILURE` — integrity gates fail.

No conclusion authorizes matching Vegas, lowering/raising individual carries, or inventing a current-depth multiplier.

## Authorized next step

If the audit confirms a current-role authority gap, the next step is a **separate frozen historical allocation experiment** that explicitly accounts for the prior ND2B and STACK6/STACK6B evidence and tests only materially new architecture/information. The 2026 sportsbook discrepancies may be reported downstream but may not select the correction.
