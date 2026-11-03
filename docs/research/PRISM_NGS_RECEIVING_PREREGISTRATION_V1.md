# PRISM — NGS RECEIVING SIGNAL, PREREGISTRATION V1

**Frozen:** 2026-11-03, **before any arm is computed or any outcome inspected.**
**Status:** PREREGISTRATION. No predictive result in this document. Nothing promoted.

---

## 0. Prior work and the gates already set

**No prior NGS study exists** in `docs/research/` (grep hits are passing mentions only).

The V21 handoff set three conditions on any NGS work. All three are checked here **before**
freezing:

| condition | status |
|---|---|
| publication timing proven pregame-safe | **PASS** — `load_nextgen_stats(2026, "receiving")` carries **weeks 1–8 on 2026-11-03**, one day after the Monday night game. Week W−1 is in hand before week W. |
| strictly prior-week walk-forward inputs | enforced by design (§2) |
| target-week NGS never used upstream | enforced — target-week rows supply the **outcome only**, never a predictor |

---

## 1. Design input (descriptive, not a hypothesis test)

Split-half stability within player-season, 223 player-seasons with ≥8 weeks, 2023–2025:

| metric | r |
|---|---|
| `avg_intended_air_yards` | **+0.754** |
| `percent_share_of_intended_air_yards` | **+0.730** |
| `avg_separation` | **+0.545** |
| **`yards` — the current baseline** | **+0.457** |
| `catch_percentage` | +0.406 |
| `avg_cushion` | +0.348 |
| `avg_yac_above_expectation` | **+0.295** |

**Three metrics are more stable than receiving yards itself.** That is the entire motivation: a
more stable quantity is a better thing to build a baseline on. `avg_yac_above_expectation` is
the *least* stable and is therefore **not** carried into any arm — recorded so no future session
revisits it on intuition.

This is a property of the predictors, measured before any outcome was scored, and is design
input in the same way a sample-size check is.

---

## 2. Sample and protocol

Walk-forward within season; predict receiver *r*'s week-W receiving yards from weeks `1..W−1`
only. Seasons **2023, 2024, 2025** scored separately. Coefficients fitted
**leave-one-season-out** and applied unchanged to the held-out season.

- NGS receiving covers ~**60 players per week** (qualifying receivers), 313 distinct players,
  3,946 player-weeks across the three seasons.
- **Eligibility:** ≥4 prior NGS weeks; ≥1 target in week W.
- **Minimum to proceed: 100 player-seasons with ≥3 evaluated weeks.** Otherwise report
  underpowered and read no gate.

## 3. Arms

`AY` = prior-weeks mean `avg_intended_air_yards`; `SH` = prior-weeks mean
`percent_share_of_intended_air_yards`; `SEP` = prior-weeks mean `avg_separation`. All centred
on the training-season mean. One **global** coefficient each — never per player
(`PRISM_OPPONENT_SIGNAL_NULLS_V1` Null 1).

- **B — baseline.** Prior-weeks mean receiving yards. *The incumbent.*
- **C — naive opportunity control.** `prior mean targets × league prior yards-per-target`.
  *If C matches the NGS arms, the finding is "count his targets", not "use NGS."*
- **N1 — air-yards baseline.** `β · AY`, replacing the yards baseline entirely.
- **N2 — separation increment.** `B + β · SEP`.
- **N3 — air-share increment.** `B + β · SH`.

## 4. Primary metric

Per player-season with ≥3 evaluated weeks: `err_p = mean |prediction − actual|`.

- **M1** mean over player-seasons of `err_p`. Each counts once.
- **M2** breadth versus B.
- **M3 — mandatory tier table** by prior yards-per-game (<30 / 30–55 / 55–80 / 80+).

Bootstrap resampled **over players**, 10,000 iterations.

## 5. Gates — all must pass

- **G1** arm beats **B** on M1, 95% CI excluding 0.
- **G2** breadth ≥**60%**, CI lower bound above 50%.
- **G3 — no tier harmed:** every M3 tier has mean gain ≥ **−0.5 yards**.
- **G4 — all three seasons:** same sign in 2023, 2024 and 2025.
- **G5 — beats the naive control:** arm beats **C** on M1, CI excluding 0.
- **G6 — MECHANISM:** the gain must concentrate where the NGS metric most **disagrees** with the
  yards baseline — top tercile of `|z(AY) − z(B)|` must show gain ≥ the full-sample gain, CI
  excluding 0. *That is where the extra information is, if it exists. A flat gain is a scale
  artefact, not new signal.*

## 6. Kill conditions

**K1** no arm passes G1 · **K2** G2 fails · **K3** G3 breached — name the harmed tier, promote
nothing · **K4** G4 fails · **K5** G5 fails — the naive target count matches, so NGS adds
nothing beyond opportunity volume · **K6** G6 fails — gain is flat across disagreement, so it is
a rescaling and not new information.

## 7. Scope

A pass licenses **one** follow-up: a preregistered test inside the production projection path,
flagged and default-off, graded prospectively. It does **not** license a production change.

**Coverage caveat that must survive into any follow-up:** NGS covers ~60 receivers per week.
Any production use must fail closed for players without NGS history rather than silently
substituting a position default — which is the exact defect
`PRISM_MC_OPPORTUNITY_LEAK_V1` traced.

Out of scope: rushing NGS, the availability lane, and every settled null. No sportsbook data.
