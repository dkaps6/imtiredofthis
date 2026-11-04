# PRISM — QB QUALITY → RECEIVER PRODUCTION, PREREGISTRATION V1

**Frozen:** 2026-11-03, **before any arm is computed or any outcome inspected.**
**Status:** PREREGISTRATION. No result in this document. Nothing promoted.

---

## 0. Why this is not already settled

`PRISM_AVAILABILITY_SIGNAL_PREREGISTRATION_V1` §0 banked four opponent-side nulls. **None of
them touch this question**, because every one concerns the **opponent**. QB quality is a
**teammate** effect: it conditions the receiver's own offence, not the defence he faces.

No preregistered test of QB quality → receiver production was found in `docs/research/`.
**No season is spent** on this question, so all of 2023–2025 is available as confirmation data
and no exploratory pass is run first.

---

## 1. Research question

**Does the quality of the quarterback throwing to a receiver condition that receiver's
production, beyond the receiver's own recent baseline?**

And critically: **is any such effect attributable to the quarterback, or merely to team
quality** — of which the QB is one visible symptom?

---

## 2. Design

Walk-forward within each season: predict receiver *r*'s week-W receiving yards using **only**
weeks `1..W−1` of that season. Seasons **2023, 2024, 2025** scored separately.

- **Eligibility, fixed in advance:** position WR/TE/RB; ≥1 target in week W; ≥4 prior games with
  ≥1 target; the team's primary QB in week W (most attempts) must have ≥4 prior games with ≥10
  attempts so his quality is estimable.
- **Anomaly gate (AGENTS.md):** receiver-weeks with zero targets are excluded from the outcome
  but retained for baseline construction. No week-W information enters any predictor.

### QB quality metric — chosen a priori, with reasoning

**Primary: `passing_cpoe`** (completion percentage over expected), prior-weeks mean.
Chosen because CPOE is constructed to isolate **QB accuracy from receiver and scheme quality**,
which is the confound this study must survive.

**Secondary, reported only: `passing_epa` per attempt.** EPA is partly produced *by* the
receivers, so a result that appears on EPA but not CPOE is evidence of endogeneity, not of a QB
effect.

Neither metric is tuned; both are prior-weeks means.

## 3. Arms

- **B — baseline.** Receiver's prior-weeks mean receiving yards per game.
- **Q — QB-conditioned.** `B + β · (QB_cpoe_prior − league_prior_mean)`, with **β fitted on the
  other two seasons** and applied to the held-out season (leave-one-season-out). **One global
  coefficient**, never per player — `PRISM_OPPONENT_SIGNAL_NULLS_V1` Null 1 established that
  per-player sensitivity coefficients are estimation noise.
- **T — team-quality control.** Identical to Q but substituting the **team's prior points per
  game** for QB CPOE. *If T performs as well as Q, the effect is team quality wearing a QB
  label.*

## 4. Primary metric

Per receiver with ≥3 evaluated weeks in a season: `err_p = mean |prediction − actual|`.

- **M1** mean over players of `err_p`. Each player counts once.
- **M2** breadth: fraction of players improved by Q versus B.
- **M3 — mandatory tier table** by receiver's own prior yards-per-game
  (<20 / 20–40 / 40–65 / 65+), reporting mean gain and % improved per tier.

Bootstrap resampled **over players**, 10,000 iterations.

## 5. Gates — all must pass

- **G1** Q beats B on **M1**, 95% CI excluding 0.
- **G2** breadth ≥**60%**, CI lower bound above 50%.
- **G3 — no tier harmed:** every tier in M3 has mean gain ≥ **−0.5 yards**.
- **G4 — all three seasons:** the M1 gain has the **same sign in 2023, 2024 and 2025**.
- **G5 — IDENTIFICATION, the gate that matters:** restrict to receivers whose **primary QB
  changed** between the prior window and week W. Q must beat B **at least as much** on that
  subgroup as on the full sample, CI excluding 0.
  *If QB quality is the mechanism, it must be most visible where the QB actually changed. A
  gain that is flat or weaker there is not a QB effect.*
- **G6 — beats the team-quality control:** Q beats **T** on M1, CI excluding 0.

## 6. Kill conditions

- **K1** G1 fails — no effect.
- **K2** G3 breached — a receiver tier is harmed; report by name, promote nothing.
- **K3** G5 fails — **the effect does not concentrate where QBs changed.** Report as team
  quality or unmodelled offence strength, **not** as a QB effect.
- **K4** G6 fails — team points predict as well as QB CPOE; the label is wrong.
- **K5** the effect appears on `passing_epa` but not `passing_cpoe` — report as endogeneity
  (receivers producing EPA for their QB), promote nothing.
- **K6** seasons disagree in sign (G4) — single-season noise.

## 7. Scope

A pass licenses **one** follow-up: a preregistered test of whether the effect survives inside
the production projection path, behind a version flag, graded prospectively. It does **not**
license a production change, and it does **not** extend to receiver *opportunity* (target
share) — only to receiving yards as specified.

Explicitly out of scope and not bundled: NGS features, the availability-signal lane, the
`prior_eff × def_eff_allowed` confirmation (already designed at
`PRISM_PLAYER_BY_OPPONENT_QUALITY_EXPLORATORY_V1` §5), and every settled null.

No sportsbook data enters any arm.
