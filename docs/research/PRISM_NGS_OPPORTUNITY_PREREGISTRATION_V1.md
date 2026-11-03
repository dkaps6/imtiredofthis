# PRISM — NGS AS AN OPPORTUNITY SIGNAL, PREREGISTRATION V1

**Frozen:** 2026-11-03, **before any arm is computed or any outcome inspected.**
**Status:** PREREGISTRATION. Two studies, separate gates. No result here. Nothing promoted.

---

## 0. Why these two, and what the last study established

`PRISM_NGS_RECEIVING_RESULTS_V1` returned a null: NGS receiving metrics do not improve receiving
**yards** prediction, and the naive *targets × league-YPT* control beat every NGS arm. Its
stated limitation was that **only yards were tested** — and four independent findings today all
point at **opportunity**, not production, as the model's defect:

- the MC leak allocates ~30% of target mass to players who record none;
- the alpha-tier bias is an *allocation* compression;
- prior-anchoring and ensemble reweighting — estimator corrections — all failed;
- counting targets beat every efficiency-style metric.

**Study A therefore tests NGS against target share**, the opportunity quantity, which is a
materially different question from the one already answered.

**Study B tests rushing NGS**, never evaluated here. It is scoped honestly: rushing NGS is an
**efficiency** family (RYOE, efficiency, stacked-box rate), so it is tested on the efficiency
question and **cannot address the RB allocation defect** documented in
`PRISM_RB_RUSH_SHARE_MASS_BUDGET_V1`. A pass would not fix that lane.

**Carried forward from the last null:** stability is **not** a design criterion. The most stable
NGS metric produced the worst arm. No stability table is used to motivate anything below.

---

## PART A — NGS → target share

### A.1 Data and protocol
NGS receiving joins `load_player_stats` on `player_gsis_id` + week at **100%**. Walk-forward
within season, 2023/2024/2025 scored separately, coefficients fitted **leave-one-season-out**.
Outcome: week-W **target share** = player targets ÷ team targets. Predictors from weeks
`1..W−1` only; target-week NGS supplies nothing.
Eligibility: ≥4 prior NGS weeks; ≥1 target in week W; team ≥10 targets.
**Minimum: 100 player-seasons with ≥3 evaluated weeks**, else underpowered, no gate read.

### A.2 Arms
`SH` = prior mean `percent_share_of_intended_air_yards` (already a share — a depth-weighted
role measure). One **global** coefficient each; never per player.

- **B** prior mean target share. *(incumbent)*
- **C** naive control: prior mean targets ÷ 35 (league team-target constant). *No share
  normalisation — "just count his targets."*
- **A1** `β · SH` — replaces the baseline with the air-yards share.
- **A2** `B + β · (SH − B)` — the disagreement term: moves the baseline toward the air-share
  only where the two differ.

### A.3 Gates (all must pass)
- **G1** beats **B** on M1, CI excluding 0 · **G2** breadth ≥60%, CI lower >50% ·
- **G3** no tier harmed (tiers by prior target share: <8% / 8–15% / 15–22% / 22%+), each ≥ −0.005 ·
- **G4** same sign in all three seasons · **G5** beats **C**, CI excluding 0 ·
- **G6 mechanism:** gain concentrates in the top tercile of `|SH − B|`, ≥ full-sample gain, CI excluding 0.

### A.4 Kills
**K1** G1 fails · **K2** G2 fails · **K3** G3 breached (name the tier) · **K4** G4 fails ·
**K5** G5 fails — counting targets matches, NGS adds nothing · **K6** G6 fails — flat gain is a
rescaling, not information.

---

## PART B — rushing NGS → rush yards

### B.1 Data and protocol
`load_nextgen_stats(stat_type="rushing")`: ~85 players and ~575 rows per season, 2023–2025.
Same walk-forward and leave-one-season-out protocol. Outcome: week-W **rush yards**.
Eligibility: ≥4 prior NGS rushing weeks; ≥1 carry in week W.
**Minimum: 80 player-seasons with ≥3 evaluated weeks** — lower than Part A because the rushing
panel is genuinely smaller; if unmet, underpowered, no gate read.

### B.2 Arms
`ROE` = prior mean `rush_yards_over_expected_per_att`; `EFF` = prior mean `efficiency`.

- **B** prior mean rush yards. *(incumbent)*
- **C** naive control: prior mean carries × league prior yards-per-carry.
- **R1** `B + β · ROE`.
- **R2** decomposed: `prior mean carries × (league YPC + β · ROE)` — separates **opportunity**
  (carries) from **efficiency** (RYOE), the distinction the whole day turns on.

`EFF` and `percent_attempts_gte_eight_defenders` are **reported descriptively only** and carried
into no arm, to keep the arm count honest against the sample size.

### B.3 Gates (all must pass)
- **G1** beats **B**, CI excluding 0 · **G2** breadth ≥60%, CI lower >50% ·
- **G3** no tier harmed (tiers by prior rush yards/game: <25 / 25–50 / 50–75 / 75+), each ≥ −0.5 yd ·
- **G4** same sign in all three seasons · **G5** beats **C**, CI excluding 0 ·
- **G6 mechanism:** gain concentrates in the top tercile of `|ROE|`, ≥ full-sample gain, CI excluding 0.

### B.4 Kills
Same shape as A.4, with **K5** reading: the naive carries × YPC control matches, so RYOE adds
nothing beyond opportunity volume.

---

## 3. Shared method

Primary metric is **per player-season** (`err_p` = mean |prediction − actual|), each counted
once, bootstrap resampled **over players**, 10,000 iterations. A **tier table is mandatory** in
both studies. Pooled MAE is a guardrail and decides no gate.

## 4. Scope

A pass in either licenses **one** follow-up: a preregistered test inside the production
projection path, flagged and default-off, graded prospectively. Neither licenses a production
change.

**Coverage caveat for both:** NGS covers ~60 receivers and ~85 rushers per week. Any production
use must **fail closed** for players without NGS history rather than substituting a position
default — the exact defect `PRISM_MC_OPPORTUNITY_LEAK_V1` traced.

No sportsbook data enters any arm.
