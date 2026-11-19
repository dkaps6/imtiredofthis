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

**Walk-forward test.** Each week from 4 to 10 was predicted only from earlier weeks. The table
shows team volume error (MAE):

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
5. Production's constant 0.57 pass share should be replaced by team identity + spread +
   total. That is a separate promotion test.
