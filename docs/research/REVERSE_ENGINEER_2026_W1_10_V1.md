# Reverse-engineering every 2026 game, weeks 1-10

**Date:** 2026-11-19, before week 11 TNF. **Status:** research finding. Not a ticket rule
until the operator says so.

**Questions asked:** where is PRISM consistent; are there patterns for certain teams against
certain matchups; do game plans show up before kickoff; what does that say about player
production.

**Reproduce:** `scripts/research/reverse_engineer_season_v1.py` (inputs and commands are in
its docstring). **Rows:** `data/research/reverse_engineer/`:
- `team_games_2026_v1.csv`: 300 team-games from play-by-play.
- `team_tendency_reliability_v1.csv`
- `team_volume_walkforward_v1.csv`
- `fanduel_board_by_script_v1.csv`: every FanDuel main-line offer, both sides, by the team's spread.
- `prism_side_by_market_pos_v1.csv`: PRISM's picked side by market, position and period.

Boards used: weeks 1, 2 and 6-10. The week 3-5 boards are permanently lost (AGENTS.md §5a).

## 1. Where PRISM is consistent: only in losing on overs

PRISM's picked side means one bet per offer, `edge_pct > 0`. The segments were found on weeks
6-10 and then checked on weeks 1-2, which had never been looked at.

| segment | weeks 6-10 | weeks 1-2 (out of sample) |
|---|---|---|
| all picks | -3.3% (n=4,123) | -6.1% (n=1,619) |
| OVER side | -9.1%, 0 of 5 weeks positive | -8.7% |
| RB rushing OVER | -14.9% | -21.6% |
| RB receptions OVER | -18.0% | -35.0% |
| WR receiving yards OVER | -9.4% | -8.7% |
| QB rushing UNDER | **+20.4%**, 4/5 weeks | **-5.9%**: did not hold up |
| model/line 0.8-0.9 | **+4.5%**, 4/5 weeks | **-0.8%**: did not hold up |

**No PRISM market/side made money in both periods.** The one consistent result is negative:
when PRISM says OVER on an RB or WR main line, it loses, in both samples.

## 2. What the games say: the score sets the game plan, and the spread only partly predicts the score

One row per team-game from nflverse play-by-play, with kneels and spikes removed.

| final result | rushes | dropbacks | pass rate |
|---|---|---|---|
| lost by 15+ | 20.2 | 40.7 | 0.67 |
| lost by 1-7 | 23.1 | 43.4 | 0.65 |
| won by 0-7 | 26.8 | 38.6 | 0.59 |
| won by 15+ | **30.8** | 32.7 | 0.51 |

| pregame spread | rushes | dropbacks | pass rate | average margin |
|---|---|---|---|---|
| underdog by 7+ | 24.2 | 38.8 | 0.61 | -8.1 |
| underdog by 3-6 | 23.5 | 40.8 | 0.63 | -5.4 |
| pick'em | 25.1 | 41.6 | 0.62 | +1.4 |
| favored by 3-6 | 26.2 | 38.8 | 0.60 | +5.9 |
| favored by 7+ | 27.1 | 35.7 | 0.57 | +8.2 |

- Rushing attempts correlate with the final margin at r = 0.54, but with the pregame spread
  at only r = 0.14. The game plan is mostly written by the scoreboard after kickoff.
- The spread is still informative. Underdogs run about 3 fewer times than favorites.

**What is stable from game to game.** The table below correlates each team's average in odd
weeks with its average in even weeks, across all 32 teams:

| measure | offense | what the defense allows |
|---|---|---|
| neutral-situation pass rate over expected | **0.54** | 0.00 |
| neutral pass rate | 0.43 | -0.06 |
| plays | 0.22 | 0.26 |
| rushes | 0.27 | 0.12 |

- **Offenses have identities.** How pass-heavy a team is in neutral situations repeats.
- **Defenses do not dictate an opponent's game plan.** What a defense "allows" in pass/run mix
  does not repeat between halves of the season.
- So "team X against defense type Y" game-plan patterns are not measurable this season. This
  agrees with the earlier opponent null in `PRISM_OPPONENT_SIGNAL_NULLS_V1.md`.

> **CORRECTED 2026-11-19 (see §10).** The walk-forward table below is **invalid as a test of
> the spread**. The "team season average" arm was unfit, while the "+ spread + total" arm was
> a fitted regression, so the fitted arm could win just by shrinking the team average. PR
> #558's review caught exactly this flaw in its own design. The "production" row also used
> league plays, not production's team-specific plays. The valid re-test is
> `team_volume_market_increment_v2.py`:
> - **the spread and total add nothing measurable**, which confirms #558;
> - **a team's own fitted run/pass tendency does beat production's constant split**.
>
> The original table is kept below as the record.

**Walk-forward test (superseded).** Each week from 4 to 10 was predicted only from earlier
weeks. The table shows team volume error (MAE):

| method | team rushes | team dropbacks |
|---|---|---|
| production: constant pass share 0.57 for every team | 6.02 | 7.27 |
| team's season average | 5.67 | 7.65 |
| **team average + spread + total** | **5.51 (better in 7 of 7 weeks)** | 7.21 |
| + what the opponent allows | 5.46 | 7.25 |

- Production's game script gives every team the same pass share.
  - A team-specific average plus the spread cuts the team rush-attempt error by about 8%.
  - Dropbacks are not predictable beyond a constant.
- **This is a production model improvement, not a betting edge.** It needs its own promotion
  test before it touches the pipeline.

## 3. What it says about player production: underdog running backs go under their rushing line

**Every FanDuel main-line offer, both sides, ignoring the model.** These are weeks 1, 2 and
6-10.

| RB rushing yards, team is... | n | UNDER hit | UNDER ROI | weeks above breakeven | line | median actual |
|---|---|---|---|---|---|---|
| **underdog by 3+** | 127 | **61.4%** | **+15.5%** | **6 of 7** | 39.9 | **30** |
| pick'em | 56 | 51.8% | -2.7% | | 45.7 | 42 |
| favored by 3+ | 129 | 48.1% | -9.6% | | 49.0 | 50 |

- **It replicates at DraftKings:** underdog by 3+, 61.3% hit, +16.0%.
- **Weekly UNDER hit rate (FanDuel):** 54%, 59%, 44%, 62%, 67%, 59%, 75%. Only week 6 (n=9)
  fell below the 52.4% breakeven.
- **Mechanism.** Take RBs who averaged 8+ carries before the game. As underdogs:
  - carries fell from 14.5 to 13.2;
  - yards per carry fell from 4.02 to 3.74.

  The book's line sits near the average outcome. A bet is decided by the median, and the
  median underdog RB ran for 10 yards less than his line.
- **The alt ladder shows the same thing from the other side.** In weeks 9-10, FanDuel RB
  rushing alt OVER rungs on underdogs returned **-30.4%** (523 rungs, 33 players). On
  favorites they returned -13.0%.

**Anomaly gate applied.** Games where the RB's snap share fell more than 40% below his own
prior were removed: injury exits and role cuts, 9 of the 127 underdog bets. The result:
- **59.3% hit, +11.6%, 6 of 7 weeks positive**, across 118 bets, 53 players and 70 games;
- game-clustered bootstrap 95% CI **[-6.0%, +28.1%]**, P(ROI > 0) = 90%.

**The line size matters:**

| underdog RB line | n | UNDER hit | ROI |
|---|---|---|---|
| under 40 | 70 | 62.9% | +18.2% |
| 40-60 | 28 | 71.4% | +34.3% |
| **60+ (lead stars)** | 29 | **48.3%** | **-9.2%** |

**Star backs still get theirs as underdogs.** Jonathan Taylor went under only 2 of 5
times as an underdog. For lines under 60 with normal snaps: 63.7% hit, +19.9%, n=91.

Other markets by script, from the same table:
- **QB rushing UNDER:** 57.9% overall, but only 4 of 7 weeks.
- **WR receiving yards:** no consistent split.
- **Passing yards:** no consistent split.

## 4. What this means, plainly

1. **PRISM has no market where its picked side made money in both samples.** Its consistent
   trait is that its RB/WR OVER calls lose.
2. **Matchup game plans are not readable pregame.**
   - Defenses don't impose a plan.
   - The score does, and the spread predicts about a quarter of the score.
   - Offense identity (neutral pass rate) is real and stable.
3. **The one pattern that held week after week is a game-script pattern on player
   production: underdog RBs with lines under 60 go under their rushing line.**
   - 7 weeks, two books.
   - The CI still includes zero, so it is strong evidence, not proof.
4. **Ticket construction implication (proposal for the operator, not a rule):** every card so
   far has been built from alt OVER ladders. On RB rushing, those lost on underdogs (-30%).
   The only consistently positive spot this season is the opposite side, a main-line UNDER.
5. ~~Production's constant 0.57 pass share should be replaced by team identity + spread +
   total.~~ **Corrected (§10):** replace it with the team's own fitted run/pass tendency. The
   spread and total add nothing beyond that. This is a separate promotion test.

---

# Part 2 (2026-11-19, afternoon): player production

**Scripts:**
- `scripts/research/player_volume_efficiency_v1.py`
- `scripts/research/rb_rush_situation_flags_v1.py`

**Rows** are in `data/research/reverse_engineer/`:
- `player_stat_reliability_v1.csv`
- `player_recency_weights_v1.csv`
- `rb_rush_flags_backtest_v1.csv`
- `rb_rush_flags_wk11_v1.csv`

## 5. What repeats: the role does, efficiency doesn't

Odd-week vs even-week correlation for the same player, 2026, players with 30%+ snaps:

| | WR | TE | RB |
|---|---|---|---|
| snap share | 0.82 | 0.93 | 0.83 |
| target share | 0.80 | 0.89 | 0.45 |
| carry share | | | **0.90** |
| targets / carries per game | 0.74 | 0.86 | 0.81 (carries) |
| **yards per target** | **0.25** | **0.31** | 0.19 |
| **catch rate** | **0.16** | **-0.12** | 0.14 |
| **yards per carry** | | | **0.21** |
| yards per game (the betting outcome) | 0.60 | 0.75 | 0.77 rushing / 0.36 receiving |

**A player's next game is his share of the work times close-to-average efficiency.**
- Hit history at a yardage line mixes the stable part (role) with the noise (efficiency).
- That is why hit history alone has not held up as validation.

## 6. How much recent games should count (three seasons, 2024-2026)

The projection is (1-w) x the season-to-date mean + w x the last-3 mean, with w chosen for the
lowest next-game error.

| stat | best weight on last 3 games | by season |
|---|---|---|
| WR targets | 0.35 | 0.25 / 0.55 / 0.00 |
| WR receiving yards | 0.20 | 0.10 / 0.40 / 0.00 |
| TE receiving yards | 0.15 | 0.10 / 0.25 / 0.00 |
| **RB carries** | **0.70** | **0.75 / 0.65 / 0.50** |
| **RB carry share** | **0.65** | **0.75 / 0.55 / 0.70** |
| RB rushing yards | 0.35 | 0.55 / 0.25 / 0.30 |

- **Backfield roles change and the change sticks.** Weight recent games heavily for RB carries.
- **Receiver roles mostly don't change.** The season average carries most of the information.

**Model implication: project RB carry share mostly from the last three games.** This addresses
the Emmett Johnson-type miss (PRISM 6.3, book 73.5, actual 101).

**What the book does, from a regression of the FanDuel line on season vs last-3 volume, weeks
6-10:**
- Receiving lines put about 20% on the last 3 games, which is about right.
- RB rushing lines put about 29% on the last 3 carries, against 0.70 optimal.

So the book under-reacts to backfield changes. On its own this did not produce a clean side in
weeks 6-10 (r = 0.08 between the recent-carry trend and the line's error).

## 7. Market tests on player production (FanDuel, every offer, model ignored)

| hypothesis | result | verdict |
|---|---|---|
| Book overprices receivers whose efficiency has been hot | hot (YPT > 1.2x league) UNDER 54.0%, +1.6% | no edge |
| Book chases recent receiver targets | 2026: falling targets -> OVER 60%; rising 10-25% -> UNDER 59%. Across 3 seasons the book's ~20% recent weight is about optimal | 2026-only, **unconfirmed** |
| **RB rushing UNDER, 2+ "bad situation" flags** | see below | **strongest pattern found** |

**The RB flags** (`rb_rush_situation_flags_v1.py`), all strictly prior:
1. Team is an underdog by 3+.
2. His yards per carry so far is under 95% of the league RB average.
3. **The book has already cut him:** carries per game x shrunk efficiency is more than 1.2x
   the line.

The flags overlap only partly (r 0.0-0.4).

| RB rushing | weeks 6-10 | weeks 1-2 (out of sample) |
|---|---|---|
| UNDER, 2+ flags | **70.3% / 63.6%** (2 / 3 flags), n=59 | **63.2% / 60.0%**, n=24 |
| UNDER, 0 flags | 39.1% | 51.7% |
| OVER, 0 flags | 60.9% | **48.3%: failed out of sample** |

**All weeks, UNDER with 2+ flags: 83 bets, 66.3% hit, +24.6%.**
- Above breakeven in **6 of 7 weeks**: 67, 60, 50, 79, 67, 64, 71%.
- With in-game exits removed (12 bets where snaps fell 40%+ below his norm): **60.6%,
  +13.9%**, 37 players, 53 games.
- Game-clustered 95% CI [-8.9%, +35.7%], P(ROI > 0) = 88%.
- **Lines under 30 hit only 51%** (backups). Lines of 30 and up: about 74% (n=34).

**Strong, consistent, not proven.**

## 8. Week 11 candidates (Thursday capture; re-check lines and state before Sunday)

RB rushing UNDER, 2+ flags, line 30 and up:

| RB | line | team spread | flags |
|---|---|---|---|
| Jonathan Taylor | 51.5 | IND +12.5 | dog, cut |
| Cam Skattebo | 52.5 | NYG +6 | dog, cut |
| Ashton Jeanty | 32.5 | LV +10 | dog, cold, cut |
| Aaron Jones | 30.5 | MIN +3 | dog, cut |

Full list, including sub-30 lines: `rb_rush_flags_wk11_v1.csv`. Rule 7 still applies: verify
injuries, depth chart and teammates before any of these is used.

## 9. What the model should take from this

1. **Project volume, shrink efficiency.** Efficiency repeats at r 0.2-0.3, so own-player
   efficiency should get about 20-30% weight against the position mean.
2. **RB carry share: about 0.65-0.70 weight on the last three games.** Receivers: about 0.2.
3. **Team rush attempts: the team's own fitted rolling rush tendency**, not a constant 0.57
   pass share. Corrected in §10: rush-attempt MAE 6.39 -> 5.54 on the 2025 holdout and 5.81
   -> 5.29 on 2026. The spread and total add nothing beyond it.
4. **The opponent's defense does not set the opponent's game plan** (r 0.00). Opponent
   adjustments belong in efficiency, if anywhere, not in volume.
5. Each of these is a promotion candidate. Each needs the normal research-to-production test
   (AGENTS.md §2) before it touches Full Slate.

---

## 10. Validity audit (2026-11-19, evening): prior work checked, flaws corrected

Before more research, each claim was checked against earlier studies and re-tested where the
test was weak.

| claim | prior work | validity problem found | status now |
|---|---|---|---|
| §2: spread + total improve team volume | PR #558 (`MARKET_IMPLIED_GAME_SCRIPT_V1_PLAN.md`): no incremental value for plays or pass rate | **Invalid comparator.** The unfit team-average arm was compared with a fitted market arm, the flaw #558's review caught. The "production" baseline used league plays. 7 weeks only. | **Re-tested** (`team_volume_market_increment_v2.py`, three arms, fit 2023-24 → test 2025 and fit 2023-25 → test 2026). **The spread and total add nothing** (rush gain -0.00 [-0.14, +0.14] and +0.06 [-0.09, +0.21]), which confirms #558. **The team's own fitted tendency does beat production's constant split** (rush MAE 6.39 → 5.54 and 5.81 → 5.29). |
| §5: efficiency does not repeat | `PRISM_ROLE_SHRINKAGE_FIX_RESULTS_V1.md` (efficiency shrinkage is correct; nine prior nulls) | None, but **not new**. It replicates prior work. | Replication. |
| §5-6: stability and recency weights | `PRISM_SNAP_SIGNAL_AND_NOISE_FLOOR_V1.md`: target-share headroom is only 0.0077 | **Name-join defect.** Snap counts were joined to stats by name, which zeroed players like Gainwell and Robinson Jr. Selecting on in-game snaps ≥ 30% also slightly inflates snap-share r. | Re-joined on IDs. Numbers moved by 0.02-0.07 and the conclusions hold. The receiver recency question is effectively closed by the noise-floor study. |
| §6: RB carries should weight recent games | Carry concentration and role shrinkage tested share level and shrinkage, not recency. **PlayerForm has no recent-games component.** | The original test had no production baseline and no holdout. | **Preregistered and tested:** `RB_CARRY_SHARE_RECENCY_RESULTS_V1.md`. **PASS:** a = 0.85; 2025 +0.0043 [+0.0016, +0.0070]; 2026 +0.0055 [+0.0022, +0.0088]. Candidate only. |
| §3 and §7: RB rushing UNDER patterns | `BACKTEST_2026_WEEKS_1_2_FINDINGS.md` warns that PRISM's board is UNDER-skewed and that "UNDER is not itself predictive" | That warning concerns PRISM's selected side. This test grades **both sides of every offer, model ignored**, so the warning does not apply. **Spread timing:** the capture-time FanDuel spread gives an identical result (n=59, 67.8%, weeks 6-10). **Discovery contamination:** the underdog component was viewed on weeks 1-2 before the composite was tested, and the cold/cut thresholds were picked on weeks 6-10, so weeks 1-2 are not a fully clean holdout. About 60 cells were scanned. | Strong and consistent, **not proven**. The CI includes 0. **Only prospective tracking from week 11 is a clean test.** |

**Prior studies dissected while doing this (all judged valid as designs):**
- **#558:** sound three-arm design. **Its numbers were never committed** and its input file
  (`data/team_weekly_history.csv`) is not in the repo, so it cannot be reproduced. The
  independent V2 re-test above confirms its conclusion.
- **Game-script trace:** a source-only fact, valid. Its open question ("does a non-constant
  split help?") is now answered: a team-specific tendency helps; the market adds nothing.
- **Carry concentration:** preregistered, retracted an off-production-path number itself, and
  its CIs exclude 0. The finding stands. The fix ships INERT.
- **Role shrinkage V1:** its specification errors are self-reported. ~~V2 was preregistered
  and never run.~~ **Corrected: V2 was run on 2026-11-03.** Its results are filed as
  `PRISM_ROLE_STATE_FRONTIER_RESULTS_V2.md`, a name this audit's search missed. It passed 8 of
  9 gates and was blocked by its own infeasible G7 threshold (K6).
- **Noise floor:** valid, including its own matched-rows correction.
- **Rush allocation:** a valid arithmetic decomposition. The 14% RB rushing gap is too few rush
  plays (0.959) x the 0.95 cap (0.941) x YPC (0.953).
