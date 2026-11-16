# Week 10 2026 backtest — every card, every construction rule, every miss

**Graded 2026-11-16 09:00 ET** from nflverse week-10 player stats and snap counts.
TNF and the full Sunday slate are complete; **MNF (LAC@BAL) is not** and was on no card.
Rows: `data/market_track_record/graded/w10_2026_card_tickets_graded_v1.csv`,
`w10_2026_card_legs_graded_v1.csv`, `w10_2026_sunday_fanduel_board_graded_v1.csv`
(2,947 FanDuel over rungs for players who played). A leg is void if the player took no
offensive snap.

## 1. Every card shipped this week

| card (ET time) | result |
|---|---|
| TNF nine-gate builder cards (Nov 12) | **4 of 4 won** (see `TNF_2026_WK10_GRADED_V1.md`) |
| TNF ad hoc rebuilds | 1 of 9 |
| Sunday v1 10:31 / 11:20 / 11:25 (5 tickets each) | **0 of 15** |
| Sunday v2 11:43 / 11:54 (3 tickets each) | 0 of 6 |
| Sunday v3 11:57 (history edge basis) | **SAFE +152 WON (7/7)**; other 4 lost |
| Sunday v4 12:20 SAFE +161 / HIGH +684 | lost 8/10 (Rodgers 143, Warren 15) / lost 7/8 (McMillan 35 vs 39.5) |
| Sunday v4 LOTTERY 12:50 +29,179 | lost 2/6 |
| Sunday v5 12:36 (marked do-not-play) | 0 of 4 |
| SNF, five versions (19:47 to 20:17) | **0 of 14** |

## 2. Why the legs missed (anomaly gate applied)

| leg | actual | state | what happened |
|---|---|---|---|
| **Aaron Rodgers** (on 4 Sunday cards + every SNF ticket) | 143 pass yds, 29 att | **PARTIAL_GAME** (82% snaps) | PIT lost 38-18; Rudolph played the last 18% of snaps. The pregame PIT secondary injuries (S Elliott, CB Murphy-Bunting) pointed to a CIN-controlled game, and the card still leaned on Rodgers volume. |
| Jaylen Warren 30+/40+/50+ rush | 15 | **UNRESOLVED_CONTEXT** (38% snaps) | 7 carries; Lew Nichols III 6 for 43, Dowdle 5. Committee plus blowout script. |
| Germie Bernard 25+/30+ rec yds | 0 | HEALTHY_FULL (95% snaps) | 3 targets, 0 catches. A dud, not a role change. |
| Kayshon Boutte 15+/25+ | 0 | HEALTHY_FULL (70%) | 1 target. **Collins and Tank Dell back** — the teammate check flagged every HOU receiver (2 shared games) and the morning card had already shipped them. |
| Dalton Schultz 20+/30+ | 11 | HEALTHY_FULL (66%) | 2 targets. Same HOU teammate context. |
| Brenton Strange 15+ / 5+ rec | 9 / 2 | HEALTHY_FULL (78%) | 3 targets in a 40-23 JAX win. |
| Devaughn Vele 15+ | 14 | HEALTHY_FULL | Missed by one yard. |
| Bryce Young 250+/275+/325+ | 200 | HEALTHY_FULL | 30 attempts in a 20-17 game. |
| Tetairoa McMillan 40+ | 35 | HEALTHY_FULL | 7 targets. |
| Rico Dowdle 30+/40+ | 26 | 49% snaps | 5 carries. 40% snap share and a 5-game sample were flagged before kickoff. |

## 3. Which construction rule worked — the whole Sunday board

| rule | rungs | players | hit | needed | ROI |
|---|---|---|---|---|---|
| A blind, every FanDuel over rung | 2947 | 167 | 36.6% | 41.6% | **-16.5%** |
| B snaps + role + matchup + injury + context only | 775 | 50 | 38.2% | 43.4% | -17.1% |
| C nine gates (Nov 9 rule) | 100 | 40 | 78.0% | 86.3% | -8.9% |
| **D nine gates + own season beats the price (v4)** | 38 | 17 | 84.2% | 84.8% | **+1.1%** |
| E model probability beats the price (v2) | 87 | 23 | 62.1% | 68.9% | -11.3% |
| F model AND FanDuel line both >= 1.2x the rung | 176 | 50 | 72.7% | 83.8% | -13.5% |
| G F + own season beats the price | 67 | 26 | 79.1% | 83.2% | -4.6% |
| H FanDuel line >= 1.2x only | 232 | 50 | 71.1% | 81.0% | -12.4% |
| I model >= 1.5x only | 115 | 44 | 78.3% | 85.7% | -8.1% |

**Week 10 was a bad week for overs** (blind -16.5%). The only rule near breakeven was the
strictest one, D — the 16-17-player pool the operator rejected as too small at 12:20.
Widening the pool (C, F, H, I) lost money every time. **Pool size is not the target;
on this week the narrow pool was the only non-losing one.** One week, n=38: not proof.

Ex-ante parlays (ranked before kickoff, one rung per player): no rule cashed an 8- or
10-leg ticket. G went 6/6 at -125; F went 4/4 at +168. Rodgers 149.5 sat near the top of
every "safest" ranking and cost every one of those tickets.

## 4. Were the exclusions right?

| excluded group | rungs | hit | needed | ROI | verdict |
|---|---|---|---|---|---|
| QB-change teams (ARI/MIA/LV/MIN) | 484 | 26.7% | 41.0% | -48.1% | right |
| Opponent in toughest quartile | 839 | 29.6% | 41.8% | -43.2% | right |
| Teammate context (e.g. HOU, Hall back) | 221 | 35.3% | 43.5% | -24.4% | right |
| Competitor arrival (KC/SF) | 124 | 36.3% | 43.3% | -18.0% | right |
| Book role line too small | 814 | 33.5% | 37.0% | -7.1% | right |
| Snaps under 30% | 439 | 41.9% | 40.8% | +6.5% | roughly neutral |
| **Mahomes quarantine** (news: elbow; club report: full, no designation) | 17 | 64.7% | 40.6% | **+56.6%** | **wrong** |
| **Keenan Allen team change** (first SF game) | 48 | 64.6% | 41.0% | **+59.0%** | **wrong this week** |

Mahomes: the morning news about a sore elbow was weighed above the official club report
(full participation, no designation). The official report was right. Keenan Allen: one game,
SF won 46-40; the team-change rule is a fail-closed anomaly rule and one game does not
overturn it, but it cost money this week.

## 5. What this week says, without inventing new rules

1. **The single biggest loss was concentration, not selection.** Rodgers was on four
   Sunday tickets and every SNF ticket. One partial game (blowout pull) took them all.
   The operator's "no player on more than two tickets" rule exists for exactly this; the
   SNF laddered construction put the same five players on all five tickets.
2. **Game script was readable before kickoff and was not used.** The PIT secondary lost
   two starters pregame and CIN was favored; the card bet on PIT volume anyway.
3. **The teammate check worked on HOU and was applied too late** — the morning cards
   with Boutte/Schultz shipped before it existed.
4. **The card changed five times in two hours.** The only Sunday ticket that won (v3 SAFE)
   was replaced 23 minutes later.
5. **Official injury reports should outrank news headlines** (Mahomes).
