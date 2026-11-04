# PRISM — MONTE CARLO OPPORTUNITY LEAK, FIX PREREGISTRATION V1

**Frozen:** 2026-11-03, **before any arm is computed.**
**Status:** PREREGISTRATION. No result in this document. Nothing promoted.

---

## 0. Objective

This repository builds **individual-player projections**. Pooled or position-level metrics are
downstream diagnostics that improve *as a consequence*; they are never the target.

That framing is load-bearing here. `PRISM_MC_OPPORTUNITY_LEAK_V1.md` established, on two
independent weeks, that the simulator allocates **~30–34% of each team's target mass to players
who record zero targets**, leaving the players who actually play **−29.3% (W7) / −35.0% (W8)**
short. Week 8 further showed the per-player share **estimates are nearly unbiased (+2.9%)**
while the **allocated** values are 35% wrong — so the defect is in **universe composition**, not
estimation.

Three prior studies failed in this lane (prior weighting, ensemble reweighting, additive
offset). **All three were estimator corrections and none could reach a composition defect.**
This study tests the composition directly.

---

## 1. What is evaluated — the allocation rule, not the simulator

`_allocate_counts` is deterministic in expectation: a player's expected opportunity is
`team_total × allocated_share_j`. Arms are therefore scored on **allocated share versus actual
share**, with no Monte Carlo run and **no production code modified**.

This isolates the thing being changed. A pass licenses a simulator-level test; it does not
license a code change.

---

## 2. Sample

The two committed production universes:

| week | trace | rows |
|---|---|---|
| W7 | `data/research/week7_opportunity_trace/…football_simulation_universe.csv` | 396 |
| W8 | `data/market_track_record/explanatory/2026_wk08_trace__…_universe.csv` | 403 |

**799 player-weeks, 28 teams, 325 player-team keys present in both weeks.**
Minimum to proceed: **100 players with ≥2 evaluated weeks** — met (325). If a run reports
otherwise, declare underpowered and read no gate.

`actual_share_j` = player targets ÷ team targets that week (nflverse, REG).

**Availability signal, leakage-safe:**
`A_j = (weeks in 1..W−1 where player j recorded ≥1 target) ÷ (W−1)`.
W7 uses weeks 1–6; W8 uses weeks 1–7. **No week-W information enters any arm.**

---

## 3. Arms

`share_j` is the effective `tgt_share` read with the simulator's own lookup order
(`rules_tgt_share` → `bayes_tgt_share` → `target_share` → `tgt_share`).
"C1-normalise" means: clip to [0, 0.95]; if the team sum exceeds 0.95, scale to 0.95.

- **C1 — production.** C1-normalise `share_j`. Exactly as shipped.
- **F0 — naive control.** Drop players with `A_j = 0` (never targeted in any prior week), then
  C1-normalise the survivors. No weighting, no threshold, no tuning — *"just stop allocating to
  players who have never been targeted."*
  **This is the control that matters.** If F0 matches the sophisticated arms, the finding is
  that the universe is too big, not that availability needs modelling (**K4**).
- **E1 — soft availability weighting.** `share_j × A_j`, then C1-normalise.
- **E2 — hard availability filter.** Keep `A_j ≥ 0.5`, then C1-normalise survivors.
- **E3 — top-N filter.** Keep the team's top **N = 7** shares, then C1-normalise. The direct
  analogue of `_top_n_shares(..., 5)`, which rushing already applies and receiving does not.

**Fixed a priori:** E2 threshold 0.5; E3 **N = 7**, chosen because the diagnosis measured
6.7–7.0 targeted players per team. That basis is a structural count from the diagnosis, not an
arm result. Sensitivity grids (threshold ∈ {0.3, 0.5, 0.7}; N ∈ {5, 6, 7, 8, 9}) are reported
as robustness and **may not decide a gate**.

---

## 4. Primary metric — per player

For each player with **≥2 evaluated weeks**: `bias_p = mean(allocated_share − actual_share)`.

- **M1 — individual bias:** mean over players of `|bias_p|`. Each player counts once.
- **M2 — breadth:** fraction of players whose `|bias_p|` moves toward zero versus C1.

Bootstrap **resampled over players**, 10,000 iterations.
Reported as guardrails only, never deciding a gate: pooled bias, pooled MAE, team-level mass.

---

## 5. Gates — all must pass

- **G1 — individual bias:** arm beats C1 on **M1**, 95% CI excluding 0.
- **G2 — breadth:** improves **M2** for **≥60%** of players, CI lower bound above 50%.
- **G3 — no starved starters:** **≤5%** of players who actually drew ≥1 target that week may
  receive **zero** allocated share. *This is the specific risk of the hard filters (E2, E3): a
  player who is dropped but plays becomes a catastrophic individual error. A pooled-mass gain
  bought by zeroing real receivers is not an improvement.*
- **G4 — beats the naive control:** arm beats **F0** on M1, CI excluding 0.
- **G5 — both weeks:** the M1 gain carries the **same sign in W7 and W8** scored separately.

## 6. Kill conditions

- **K1** no arm passes G1.
- **K2** G1 passes but G2 fails — aggregate gain without player-level breadth.
- **K3** G3 breached — the arm starves players who actually play.
- **K4** **F0 matches the best arm within CI** — the finding is *"the universe is too large"*,
  not *"availability must be modelled"*. Report that plainly; promote no weighting scheme.
- **K5** W7 and W8 disagree in direction on G1 — report both, promote neither.

## 7. Scope

A pass licenses **one** follow-up: implementing the winning rule in `scripts/simulation_v2.py`
**behind a version flag, default off**, re-running the full production validation suite, and
**grading it prospectively** before any live use.

It does **not** license a code change on its own, and it does **not** establish that downstream
receptions, yards or pricing improve — the allocation step is evaluated in expectation only,
and a separate test is required for each.

`rush_shares` is **out of scope**. Rushing already applies `_top_n_shares(..., 5)` and shows a
different leak magnitude (−14.6%); any change there needs its own study. Truncating a tail is
not the same as modelling it, and that caveat applies to E3 here too.

No sportsbook data enters any arm.
