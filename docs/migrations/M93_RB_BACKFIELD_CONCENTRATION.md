# Migration 93 — RB Backfield Concentration

## Why this comes next

M91 showed that the RB model is generally competent in ordinary workloads but compresses true lead-back / bell-cow outcomes. M92 then decomposed the error and showed that correcting rushing opportunity materially reduces rushing-yard error. Before changing team game-script volume, M93 isolates the narrower backfield-allocation question:

> Are the current projected RB shares directionally right but too compressed toward committee usage?

M93 is intentionally smaller than the later role-state work. It first tests the simplest falsifiable version of the hypothesis: preserve the existing projected RB rushing-attempt pool and sharpen only the within-backfield share distribution.

## Frozen inputs

M93 reuses the exact frozen M91 2024/2025 predictions. It does not rebuild history or retrain the base MC/ML/State components.

For the primary ML test, M93 keeps fixed:

- total projected RB rushing-attempt pool for every team-week
- player rushing efficiency implied by the M91 ML prediction
- receiving projection
- defense and matchup context
- M30 top-five architecture
- every production coefficient

Only the share distribution inside the already projected RB carry pool can change.

## Concentration transform

For each team-week, the existing ML RB carry shares are converted with a concentration exponent and renormalized to the same total RB carry pool. An exponent of 1.0 is the unchanged M91 baseline; values above 1.0 sharpen the leader/committee separation without changing player ordering.

The exploratory grid is fixed in advance:

- 1.00
- 1.10
- 1.20
- 1.30
- 1.40
- 1.50

## Development / validation discipline

The exponent is selected **only on 2024** using all-RB rushing-attempt MAE, with rushing-yard MAE and the smaller exponent as deterministic tie-breakers.

The selected value is then frozen and applied to **2025**. No 2025 result may participate in selecting the exponent.

This is important because 2025 rushing results have already been inspected heavily in prior work.

## Scoreboards

M93 reports attempts, rushing yards, and rushing + receiving yards for:

- all RBs
- 0–5 actual carries
- 6–10 actual carries
- 11–14 actual carries
- 15+ actual carries
- 20+ actual carries
- 25+ actual carries
- 15+ carry / 60%+ team-rush-share bell-cow games

It also keeps the original all-player rushing-yard scoreboard as a regression guard so RB-specific gains cannot silently damage the old M30 aggregate behavior.

## Yardage isolation

Candidate rushing yards use the new carries with the **same M91 implied rushing efficiency**. No YPC feature is changed.

Candidate rushing + receiving yards replace only the rushing component; the M91 receiving projection remains unchanged.

Therefore any yardage improvement in M93 is attributable solely to a better within-backfield workload distribution.

## Interpretation

A successful M93 does not finish the carry problem. It means the existing model is directionally ranking backfields correctly but needs modest concentration sharpening. M94 would then attack the other major error source identified by M92: total team rushing opportunity / football-only game script.

If fixed concentration fails validation, the next backfield test should use richer pregame role-state information (recent carry-share trends, competing-back availability, depth changes, and team committee behavior) rather than forcing a generic sharpening rule into production.

M93 is research only. No production promotion occurs from this run by itself.
