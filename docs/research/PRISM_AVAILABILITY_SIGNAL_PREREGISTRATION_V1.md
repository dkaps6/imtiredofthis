# PRISM — AVAILABILITY SIGNAL V1: PRIOR-WORK LEDGER + PREREGISTRATION

**Frozen:** 2026-11-03, **before any arm is computed.**
**Status:** PREREGISTRATION. No result in this document. Nothing promoted.

---

## 0. PRIOR-WORK LEDGER — read this before proposing matchup or defence work

Written because the same questions keep being re-asked. **Each line below is settled or
designed; do not re-run it.**

### SETTLED NULLS — do not re-test

| question | verdict | evidence |
|---|---|---|
| *"Some players are matchup-proof, others are fragile — tag sensitivity to the player"* | **NOT MEASURABLE** | `PRISM_OPPONENT_SIGNAL_NULLS_V1` Null 1. n=254 players, observed/permutation spread **1.014**, p=**0.395**. Cross-season carry +0.126 then −0.044. The wide per-player slope spread is estimation noise from 10–17 games. |
| *"This defence is soft deep — attack its shape"* | **NO PERSISTENT SHAPE** | Null 2. Mean r(shape) across five windows **+0.017**, one window −0.295. Defence *level* has weak persistence (r ≈ 0.2–0.37); *shape* has none. |
| *"Forecast game script / play volume"* | **NOT FORECASTABLE** | Null 3. |
| *"Player aDOT × opponent man/zone profile improves receiving yards"* | **H1 NOT SUPPORTED** | `PRISM_PLAYER_SPECIFIC_MATCHUP_RESULTS_V1`, preregistered, OOS. |

**Consequence for the common intuition** *"measure how big producers do against certain
defences and tag it to the player"*: **the per-player form of that idea is tested and dead.**
What survived is different — a **single global coefficient** on
`player_efficiency × defence_efficiency_allowed`, which says a player's swing is proportional
to **his own level** (a high-efficiency receiver has more absolute yards to lose). That is a
**scale effect, not differential responsiveness**, and Null 1 does not contradict it.

### QUALIFIED — reuse, do not rebuild

| mechanism | status |
|---|---|
| **Active-role participation probability** | **QUALIFIED, all positions** incl. WR/TE (`PLAYER_ACTIVE_ROLE_PARTICIPATION_V1_MECHANISM_RESULT`). Research-only; **not wired into production.** |
| Active-role → conditional allocation bridge | **QUALIFIED for RB rushing, RB receiving, TE receiving.** QB passing CLOSED. **WR receiving NOT SCORED** — the frozen source audit could not establish a legitimate 2025 WR-R15 OOS authority. |

### DESIGNED, NOT RUN

`prior_eff × def_eff_allowed`, one global coefficient per market. Confirmation design is already
frozen at `PRISM_PLAYER_BY_OPPONENT_QUALITY_EXPLORATORY_V1` §5 and requires **2023 + 2024 held
back** (2025 is spent as hypothesis-generating). **Do not redesign it; run it as written.**

### GENUINELY OPEN — no prior test found

1. **QB quality → receiver opportunity.** No preregistered test exists. A receiver's output is
   plausibly conditioned on who is throwing, and that is distinct from every null above because
   it is a *teammate* effect, not an opponent effect.
2. **NGS receiving advanced stats.** `load_nextgen_stats(2026, "receiving")` returns 695 rows
   with `avg_separation`, `avg_cushion`, `percent_share_of_intended_air_yards`,
   `avg_yac_above_expectation`. **Production uses none of it**, and separately `route_rate` and
   `yprr` are NaN for every player (`PRISM_TGT_SHARE_RECONCILIATION_V1` §5).

Both are logged here as leads. **Neither is bundled into this study.**

---

## 1. What this study does

`PRISM_MC_LEAK_FIX_RESULTS_V1` §7 found E1 — availability-weighted allocation — passes its
gates but **harms the 5–12% share tier** (31% improved, mean gain −0.0080), because its signal
`A_j = fraction of prior weeks with ≥1 target` **cannot separate a full-time tight end who draws
three targets from an alpha receiver**: both score `A_j = 1.0`.

The ledger above shows a **better signal already exists and is qualified**. This study replaces
the crude proxy with it, and scores the **WR receiving** case the allocation bridge left open.

**Research question:** does the qualified active-role participation probability, used as the
availability weight, beat E1's crude proxy — **without harming any share tier**?

---

## 2. Sample and protocol

Identical to `PRISM_MC_LEAK_FIX_PREREGISTRATION_V1`: W7 and W8 committed production universes,
769 player-weeks, allocation rule scored **in expectation** (no MC run, no production code
modified). Minimum **100 players with ≥2 evaluated weeks** (310 available).

The participation probability must be computed **strictly from weeks 1..W−1**. If the qualified
mechanism cannot be reproduced leakage-safely for W7/W8, **report that and stop** — do not
substitute an approximation and call it the qualified signal.

## 3. Arms

- **C1** production allocation, as shipped.
- **E1** the prior winner: `share × A_j`, `A_j` = fraction of prior weeks with ≥1 target.
- **H1** `share × P_active`, where `P_active` is the **qualified** active-role participation
  probability.
- **H2** `share × P_active × R_j`, where `R_j` = the player's prior-weeks **targets per
  opportunity** (target rate given participation). *This is the term that should separate the
  full-time low-target tight end from the alpha — the specific failure E1 exhibits.*

## 4. Primary metric — per player AND per tier

Per player with ≥2 evaluated weeks: `bias_p = mean(allocated − actual)`.

- **M1** mean over players of `|bias_p|`, **reported twice**: all players, and **excluding
  never-targeted players** — the §7a dilution must not recur.
- **M2** breadth, fraction improved versus C1.
- **M3 — tier table, mandatory.** Mean gain and % improved by actual-share tier
  (0 / <5% / 5–12% / 12–20% / 20%+).

Bootstrap resampled over players, 10,000 iterations.

## 5. Gates — all must pass

- **G1** arm beats **E1** on M1 computed **excluding never-targeted players**, CI excluding 0.
  *The comparator is E1, not C1 — beating production is no longer interesting.*
- **G2** breadth ≥60% on that same population, CI lower bound above 50%.
- **G3 — no starved starters:** ≤5% of players who drew ≥1 target receive zero allocation.
- **G4 — NO TIER HARMED:** **every** tier in M3 must have mean gain ≥ **−0.002** versus E1.
  *This is the gate E1 would have failed. A whole-sample win that degrades an identifiable
  group of real contributors is not an improvement.*
- **G5** same sign in W7 and W8 scored separately.

## 6. Kill conditions

**K1** no arm passes G1 · **K2** G2 fails · **K3** G3 breached · **K4** G4 breached — report the
harmed tier by name and promote nothing · **K5** W7/W8 disagree · **K6** the qualified
participation probability cannot be reproduced leakage-safely (§2).

## 7. Scope

A pass licenses one follow-up: flagged implementation (default off), full validation suite,
**prospective grading** before live use, plus a **separate** downstream test for receptions and
`rec_yards` — the allocation rule is scored in expectation only.

Out of scope and explicitly not bundled: QB-quality effects, NGS features, `rush_shares`, and
every settled null in §0. No sportsbook data in any arm.
