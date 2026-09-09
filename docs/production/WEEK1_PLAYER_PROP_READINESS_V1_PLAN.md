# 2026 Week 1 Player-Prop Readiness V1 — Frozen Execution Plan

Date frozen: 2026-09-08 ET
Branch: `production-week1-player-prop-readiness-v1`
Base: current `main` at branch creation
Production authority: `.github/workflows/full-slate.yml`

## North star

The product goal is the most accurate possible individual pregame football projections/distributions for QB, RB, WR and TE, followed by downstream fair-probability/prop pricing. Sportsbook information is downstream only and must never alter the independent football projection.

The near-term deadline is 2026 Week 1. Research evidence must be converted into safe production gains before kickoff where the evidence already supports doing so. Prospective ledgers may run in parallel; they are not automatic blockers to using a validated production candidate.

Canonical architecture:

`GAME STATE -> TEAM OPPORTUNITY -> POSITION/ROOM POOL -> PLAYER ENTITLEMENT -> PLAYER EFFICIENCY -> JOINT MONTE CARLO DISTRIBUTION -> PROP PRICING`

## Current production/readiness audit

### QB

Production status: **WIRED** for passing-yard mean.

Authority: `QB_PASS_SYNTHESIS_V1` / M89-M90. Full Slate builds promoted QB context and production pricing fails closed unless every priced `pass_yards` row uses the promoted synthesis.

Historical winner is preserved. Do not launch a new broad generic QB mean hunt before Week 1.

Remaining Week-1 work:

1. run the already-authorized football-only Week-1 path audit on actual projected QBs;
2. audit `qb_pred_attempts`, `qb_pred_ypa`, current-team/player-history mapping, opponent/game environment, receiver ecosystem, cap/extrapolation and borrowed/missing 2026 context;
3. preserve M89/M90 unless a concrete mechanical/input defect is found;
4. repair/confirm C2 shared pass/receiving conservation for distribution shape if possible without changing the QB mean.

### WR

Production status: **WIRED baseline, not proven exhausted**.

Authority: M38 WR hierarchy target-share transform (`1.40 / 1.14 / .91 / .78`) inside canonical Monte Carlo. M38 improved receiving-yard MAE in all six 2020-2025 seasons and preserves team WR target mass.

WR-R11 additive NGS residual candidate is closed scientific failure and must not be rescued.

Remaining Week-1 work:

1. confirm M38 is active on the current Week-1 Full Slate player universe;
2. audit individual WR1/WR2/WR3 target/receiving projections and current role/injury/transition context;
3. do not promote an unvalidated new WR entitlement model merely to meet the deadline;
4. if a strict-prior relative-entitlement-around-M38 candidate can be frozen, run, and clear historical + full-stack gates before kickoff, it may be considered separately. Otherwise ship M38 and preserve the new architecture for post-Week-1 refinement.

### TE

Production status: **NOT OPTIMIZED / VALIDATED WINNER NOT YET WIRED**.

TE-R5 `TE_PARTICIPATION_ENTITLEMENT_V1` is a scientific PASS on 3,214 OOS player-games (2023-2025):

- target MAE `1.682530 -> 1.541270`;
- receiving-yard MAE `16.267971 -> 15.858042`;
- receiving-yard RMSE `23.498617 -> 21.620042`;
- receiving-yard bias `-6.610834 -> -0.350016`;
- receiving-yard p90 AE `37.227233 -> 33.284738`;
- 30+ miss `13.8768% -> 12.4144%`;
- 40+ miss `8.7430% -> 6.5028%`;
- receiving-yard MAE improved in 2023, 2024 and 2025;
- high-volume Q4 receiving-yard MAE `24.073076 -> 22.255704`.

TE-R5 preserves the finite TE pool and allocates individual entitlement using strict-prior participation/history. It changes neither sportsbook inputs nor production today.

Remaining Week-1 work has highest priority:

1. build a deployable train-through-2025 TE-R5 scorer/adapter using the exact frozen model semantics;
2. prove required 2026 Week-1 features are available strict-prior on current Full Slate;
3. prove exact TE-room target-pool conservation and non-TE protection;
4. run a full-stack integration confirmation against the current 2026 architecture;
5. if every frozen integration gate passes, promote TE-R5 for Week 1 through a separate explicit production commit.

### RB rushing

Production status: **WIRED** for 2026 Week-1 rushing yards.

Authority: `RB_P3_SYNTHESIS_V1`, route `WEEK1_STACK_OVERRIDE`. Preserve this winner and its fail-closed Week-1 boundary.

### RB receiving yards

Production status: **TECHNICALLY READY, SHADOW-ONLY**.

R16-R20 established predictable tail signal, mean-preserving distribution improvement, canonical MC adapter parity, deployable strict-prior 2026 scorer and real Week-1 Full Slate parity. R21 has now sealed pre-outcome Week-1 CONTROL and SHADOW forecasts for later prospective grading.

Remaining Week-1 work:

1. perform a governed Week-1 production-promotion review using already-completed R16-R20 evidence;
2. if promotion is approved, wire the exact R17/R18 adapter + R19 scorer into canonical Full Slate after the canonical football mean is produced and before sportsbook comparison;
3. preserve receiving-yard mean, target entitlement, receptions, rushing distributions, non-RB outputs and RB-room conservation;
4. keep R21 prospective CONTROL/SHADOW evidence intact in parallel.

### RB receptions / target entitlement

Production status: **NOT separately solved by R16-R21**.

R8-R12 research contains useful target/identity/state evidence, but the receiving-yard tail adapter does not change receptions. Audit the current canonical reception projections and determine whether the strongest validated R9/R11/R12 component qualifies for a separate Week-1 production integration. Do not conflate a receiving-yard promotion with a reception-model promotion.

## C2 shared pass/receiving conservation status

The canonical scientific C2 mechanism remains supported historically, especially for QB distribution calibration.

However, full-stack integration run `34139757238` is **NOT a valid production clearance**, despite GitHub Actions concluding success. Its result disposition is `MECHANICAL_OR_INTEGRITY_FAILURE` because the evaluator produced `receiver_rows=0`, making receiving-side metrics NaN. QB-side mechanics still showed CRPS `40.387988 -> 39.103434` with identical mean MAE `55.060118` and bootstrap probability `1.0`.

Before C2 production consideration, repair only the receiver-evaluation plumbing, rerun the frozen integration contract, and preserve all scientific gates/model logic.

## Week-1 execution order

Priority 0 — production truth:
- confirm current no-odds Full Slate runs end-to-end on main and inventory actual Week-1 player outputs/feature provenance.

Priority 1 — TE-R5:
- deployable scorer -> 2026 availability/parity -> full-stack confirmation -> promotion if passed.

Priority 2 — RB receiving yards:
- governed R16-R20 promotion review -> canonical Full Slate integration -> no-odds/live-odds confirmation if approved.

Priority 3 — QB:
- complete actual Week-1 path audit; repair only concrete input/path defects; retain M89/M90 mean otherwise.

Priority 4 — C2:
- repair zero-receiver integration evaluator and rerun frozen C2 full-stack test; promote only if valid full-stack receiving/player protection gates pass.

Priority 5 — WR:
- verify current M38 Week-1 projections/roles; ship M38 unless a new relative-entitlement candidate earns validation before kickoff.

Priority 6 — RB receptions:
- audit current reception distribution and strongest R8-R12 evidence; promote only if a distinct reception/target integration is scientifically justified.

## Final Week-1 release gate

Before calling the model Week-1 ready:

1. run canonical Full Slate no-live-odds successfully;
2. verify all promoted position adapters report exact version/provenance and zero sportsbook input to football projections;
3. inspect player-level Week-1 projection tables for QB/RB/WR/TE, with targets/carries/attempts/efficiency decomposition where applicable;
4. run a controlled live-odds Full Slate when active markets are available;
5. confirm `outputs/props_priced_clean.csv` contains the final football mean/distribution, model probabilities, sportsbook implied/fair probabilities and model edge without upstream market leakage;
6. preserve a pre-kickoff Week-1 forecast artifact for postgame grading.

---

# Downstream roadmap after player-prop readiness

## Anytime TD

Canonical simulation/pricing already contains an `anytime_td` scaffold using `offensive_td_rate`, `rz_share`, a team-script multiplier and shared scoring shock. This is not yet a validated elite TD model.

The dedicated TD model should be built on the same hierarchy:

`TEAM TD / DRIVE SCORING STATE -> RUSH/PASS TD OPPORTUNITY -> RED-ZONE/GOAL-LINE ROLE -> PLAYER TD ENTITLEMENT -> JOINT SIMULATION`

Use the mature player opportunity architecture (QB pass state, RB carry room, WR/TE/RB target shares, participation, injuries, role transitions) rather than fitting an isolated touchdown classifier.

## Game spread / total / moneyline

The player model has already accumulated much of the causal data required for a strong game model: plays/pace, pass/run tendency, QB attempt/YPA state, rushing opportunity/efficiency, receiver ecosystem, defensive context, weather, injuries, roster identity and correlated Monte Carlo outcomes.

The future game model should not simply regress final score on generic team stats. It should create a team/game scoring process consistent with player opportunity:

`DRIVES/PLAYS -> POSSESSION/SCORING OPPORTUNITIES -> PASS/RUSH EFFICIENCY -> POINTS / TD / FG STATE -> JOINT TEAM SCORE DISTRIBUTION`

Then derive:
- win probability / moneyline;
- point spread distribution;
- game total distribution;
- team totals;
- correlated player/game outcomes.

Historical sportsbook spreads/totals can be used as downstream evaluation/benchmarking and, only if intentionally versioned later, as a separately labeled market-assisted model. The independent football model remains market-free upstream.
