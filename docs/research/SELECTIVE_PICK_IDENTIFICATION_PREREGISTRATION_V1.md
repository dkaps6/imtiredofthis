# Can we tell, before the game, which model picks will win? PREREGISTRATION V1

**Date:** 2026-11-20, frozen before any selector below is scored. **Research only.**

## The question, in the operator's words

> "Even if we are only hitting lets say 49% of all lines... how do we identify those 49% of
> lines before the game? If we can identify those lines we are getting correct, then we are
> going to be profitable."

In plain terms: the model will never win every bet. The question is whether there is a group of
its picks, **recognizable before kickoff**, that wins often enough to beat the price (52.4% at
-110).

## What was already tried (prior-work check)

| tried | where | result |
|---|---|---|
| bet bigger model edges only (STRONG/LEAN gate) | `data/backtests/full_stack_vegas_benchmark_v1/README.md` | helps a little, never reaches breakeven (best: receptions -0.6%) |
| model confidence bands, `edge_pct` deciles (2026) | V21 handoff §3 (`docs/handoffs/NFL_HANDOFF_2026-11-02_V21_PRE_V22_ROOT_ARCHIVE.md`) | flat: an "88%" pick hits 54% |
| passing yards as the profitable market (2026) | `PRISM_PASS_YARDS_EDGE_DURABILITY_RESULTS_V1.md` | retracted 2026-11-04. **Weeks 9-10, checked today: 49.1%, -7.2%** (n=112) |
| `component_sd` as an uncertainty filter | `DECISION_GATE_COMPONENT_SD_DIAGNOSIS_V1.md` | not outcome uncertainty |
| RB-under situational rule, role-change rule | `ROLE_CHANGE_AND_RB_UNDER_HISTORICAL_RESULTS_V1.md` | failed on 2024-25 |

**Not tried:**
- splitting the model's picks by **side**, which matters because blind overs lose 7-12% in every
  market and season while blind unders lose about the vig;
- the **price** paid;
- the **game spread and total**;
- the model's **own running record on that player**;
- a classifier trained on all of these together.
- None was trained on one season and scored on another.

**What I have already seen on this data:** model-pick win rates by market and season, pooled
(51.1% in 2024, 53.1% in 2025); the full-stack tier table above; blind over/under by market. I
have **not** seen model picks split by side, price, spread or player record.

## Data

- **Primary:** `data/backtests/full_stack_vegas_benchmark_v1/non_qb_detail.csv`.
  - 16,973 graded lines, 2024-2025 regular season.
  - Markets: rush_yards, rec_yards, receptions, rush_rec_yards.
  - The projection is the **full production ensemble**, built strictly pregame.
  - Lines are mostly DraftKings.
- **Secondary (reported):** `pass_yards` from
  `data/backtests/historical_market_vegas_benchmark_v1/historical_market_vegas_benchmark_detail.csv`.
  It is the base engine only, not the QB synthesis.
- **2026 check:** the PRISM boards for weeks 1, 2 and 6-10 (FanDuel + DraftKings, `model_proj`),
  graded.

**The model's pick:** OVER if projection > line, UNDER if below; ties dropped. A bet is settled
at the posted price for that side. Pushes are dropped.

## The selectors (every one is fixed before scoring)

Each selector is **chosen or fit on one season and scored on the other**, in both directions:
2024 → 2025, then 2025 → 2024. Nothing is tuned on the season it is scored on.

| id | rule | fitted? |
|---|---|---|
| **S1** | the model's **UNDER** picks only (all four markets) | no |
| **S2** | the model's UNDER picks whose relative edge, \|proj - line\| / line, is in the top third. The cut-off is taken from the fit season. | cut-off only |
| **S3** | market x side cells (8 cells) with ROI > 0 and n ≥ 150 in the fit season, applied to the other season. If no cell qualifies, S3 selects nothing. | cell choice |
| **S4** | the model's picks priced better than -110 (the book is paying more than standard on the model's side) | no |
| **S5** | the model's **own record on this player and market** so far this season: at least 4 earlier picks, at least 60% won. Strictly earlier weeks. | no |
| **S6** | logistic regression trained on the fit season to predict "the pick wins"; take the top 20% of the scored season by predicted probability | yes |

**S6 features, all pregame:**
- market x side;
- relative edge and its absolute value;
- the line;
- the implied probability of the price paid;
- `component_sd / line`;
- the team's spread and the game total (nflverse `spread_line`, `total_line`);
- the week;
- the player's earlier-pick count and win rate this season (S5's inputs; 0.5 when none).

**Comparators, reported beside every selector:**
- **C0:** every model pick.
- **C1:** blind UNDER on the same rows, model ignored. This is the comparator that matters for
  S1 and S2. If the model's unders are no better than every under, the model adds nothing; the
  side does all the work.

## How a selector passes

- **PASS** (worth betting): all three of
  - ROI > 0 on **both** scored seasons;
  - the pooled out-of-sample ROI has a 95% CI above 0, under **both** a game-clustered and a
    player-clustered bootstrap (the 2026-11-04 study showed player clustering is the one that
    bites);
  - above breakeven in at least half of the scored weeks.
- **TRACK** (record it, do not size bets): ROI > 0 on both scored seasons, but the CI includes 0.
- **FAIL:** anything else.

Six selectors are tested, so one false PASS at the 95% level is plausible by chance alone. A PASS
must then also hold on the 2026 boards (ROI > 0) before it is called real.

## Prediction, written before scoring

- **S1:** the model's unders beat its overs by a wide margin, as blind sides do. Against C1
  (every under) the model adds a point or two, not enough to clear the vig. **TRACK or FAIL.**
- **S2:** no better than S1. Bigger edges were mostly not better in earlier work.
- **S3:** picks receptions or under cells; holds weakly in the other season.
- **S4:** noise.
- **S5: FAIL.** Player-level misses reverse (`TEAM_PLAYER_IDENTITY_2026_V1.md` §4), so a player
  the model "reads well" so far should not keep being read well.
- **S6:** about 53-54% on its top 20%. **TRACK at best.**
- **Most likely overall answer:** no selector PASSES. The useful output is the size of the side
  effect, and whether the model adds anything once the side is fixed.

**Named risks:**
- the archive is mostly DraftKings, and 2026 is FanDuel plus DraftKings;
- the archive's line is the "latest captured", not a fixed pregame snapshot;
- rush_rec_yards is thin in 2025 (688 rows).
