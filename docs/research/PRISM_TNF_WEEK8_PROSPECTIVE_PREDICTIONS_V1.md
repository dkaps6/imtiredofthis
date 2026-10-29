# PRISM — CAR @ GB, 2026-10-29 20:15: frozen prospective predictions

**Frozen before kickoff.** Board rows locked at
`data/research/tnf_week8_prospective/frozen_board_rows_v1.csv`, taken from the committed
week-8 board (`data/market_track_record/boards/2026_wk08.csv`, Full Slate run
**40953584718**). Nothing in this game was fitted to anything.

**Betting is CLOSED.** This is a model-diagnosis test, not a card.

## Why this game is worth grading

GB's backfield is a textbook instance of the defect measured over three days:
production's within-RB lead share is **0.5173** against football's **0.6277**
(gap −0.1103 [−0.1453, −0.0741]), independently reproduced on a 12-season historical
panel as **0.5535 vs 0.6584** (gap −0.1049).

Tonight the model puts GB's lead share at **0.558** while the market implies **0.739**, and
CAR's at **0.782** versus **0.877**. Same direction, both teams.

## The frozen rows

| player | team | market | line | model | mc | ml | state |
|---|---|---|---|---|---|---|---|
| Josh Jacobs | GB | rush_yards | 59.5 | **39.48** | 33.47 | 47.03 | 39.64 |
| MarShawn Lloyd | GB | rush_yards | 17.0 | **28.06** | 20.14 | 38.02 | 39.64 |
| Chuba Hubbard | CAR | rush_yards | 67.5 | **61.91** | 48.67 | 78.56 | 39.64 |
| Trevor Etienne | CAR | rush_yards | 9.0 | **17.24** | 16.70 | 17.93 | 23.31 |
| Savion Williams | GB | rush_yards | 3.5 | 3.25 | 1.20 | 5.83 | 7.23 |
| Josh Jacobs | GB | rush_rec_yards | 82.0 | **55.96** | 55.96 | 61.94 | 75.67 |
| Chuba Hubbard | CAR | rush_rec_yards | 89.0 | 79.14 | 79.14 | 98.57 | 75.67 |
| MarShawn Lloyd | GB | rush_rec_yards | 25.5 | **38.53** | 38.53 | 50.62 | 47.77 |

Team RB rush totals (QBs excluded): **GB model 70.8 vs market 80.5**; **CAR model 79.2 vs
market 77.0**.

## Frozen predictions — what the diagnosis implies

| id | prediction | basis |
|---|---|---|
| **P1** | Jacobs actual `rush_yards` **>** 39.48, and Hubbard **>** 61.91 | lead backs are under-projected |
| **P2** | Lloyd actual `rush_yards` **<** 28.06, and Etienne **<** 17.24 | backups are over-projected |
| **P3** | actual within-RB lead share **>** model — GB **>** 0.558, CAR **>** 0.782 | the ~10.5pp flattening |
| **P4** | actual team RB rush yards **>** model — GB **>** 70.8, CAR **>** 79.2 | the −8.5 level bias |
| **P5** | `rush_rec_yards` absolute error **>** `rush_yards` absolute error for the same lead backs | measured −18.33 vs −8.46 |

P1 and P2 are **mechanically linked** to P3 — they are not independent evidence. The four
weakly-dependent signs are **P3-GB, P3-CAR, P4-GB, P4-CAR**.

## What this test can and cannot establish

**It cannot confirm or refute the diagnosis.** n = 1 game. A single game's variance dwarfs
a 10pp share effect, and the result can go either way for reasons that have nothing to do
with the model — an injury, a blowout script, one broken run, a goal-line substitution.

What it *is*: a **free directional observation on data nothing was fitted to**, graded
against predictions committed before kickoff. Under a coin-flip null the four
weakly-dependent signs give at best p ≈ 0.0625 if all four land, which is **suggestive and
nothing more**. If they split, the test is uninformative — not a refutation.

**Pre-committed interpretation rules, so the result cannot be spun after the fact:**

- 4/4 predicted directions → consistent with the diagnosis; **not** promotion evidence.
- 2–3/4 → uninformative; report as such and change nothing.
- 0–1/4 → a point **against** the flattening story on this game; record it, do not explain
  it away, and flag it for the prospective study.

Nothing about tonight changes any promotion state. The allocative prior remains unpromoted
pending its own preregistration.

## Separate finding recorded with this card

**`state_proj` is a position constant, not a player projection.** On the week-8 board's 66
`rush_yards` players: `mc_proj` has **66 distinct values**, `ml_proj` **65**, and
**`state_proj` only 7**. Hubbard, Jacobs and Lloyd all carry the identical **39.6412**;
Young and Love share **16.5986**; fifteen players share one value.

The State component — a promoted production model — emits role-bucket constants for this
market. This was never examined in three days of work because the column is labelled as a
player-level component. It is the project's first-principle violation sitting in plain
sight, and it is **not** explained by any finding in the RB lane. Logged as a new open item;
no change proposed here.
