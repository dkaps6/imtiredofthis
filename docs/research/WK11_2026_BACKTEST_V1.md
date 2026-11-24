# Week 11 2026 backtest: every card, the whole board, every leg type

**Graded 2026-11-24** from nflverse week-11 player stats and snap counts. All 13 games are final,
TNF through MNF.

**Script:** `scripts/research/grade_week11_2026_v1.py`.

**Rows:** `data/market_track_record/graded/w11_2026_*.csv`. A leg is void if the player took
no offensive snap. There were no voids this week.

## 1. Every ticket

| card | ticket | est. odds | result | missed legs |
|---|---|---|---|---|
| TNF IND @ HOU | SAFE / MODERATE / HIGH | +158 / +170 / +610 | **0 of 3** | see `TNF_2026_WK11_IND_HOU_CARD_V1.md` |
| Sunday v2 | **SAFE** | **+158** | **WON 3/3** | none |
| Sunday v2 | MODERATE | +297 | lost 4/5 | **Purdy 10+ rush: 0** |
| Sunday v2 | MEDIUM | +417 | lost 3/5 | **Hubbard 10+ rec: 0**, **Warren 10+ rec: 1** |
| Sunday v2 | HIGH | +693 | lost 2/5 | Gibbs U84.5: 98, McCarthy U16.5: 33, Chase Brown 30+ rush: 6 |
| Sunday v2 | LOTTERY | +26,820 | lost 10/16 | the six above |
| SNF script (suggested) | McCaffrey 40+ / Black 15+ / Purdy 10+ | about +130 | lost 1/3 | Black 8, Purdy 0 |
| **MNF challenge: operator** | 6 legs | about +511 | **WON 6/6** | none |
| MNF challenge: card | 4 legs | about +145 | lost 2/4 | Ja'Marr Chase 60+: **31**, Chase Brown 30+ rush: 6 |
| RB-under tracker | 5 singles | | **3-2, +0.64u** | Jeanty 79, Henry 94 |

**What the misses say:**

1. **The fixed matchup check would have removed three losing legs.** Purdy 10+ rush, Hubbard 10+
   rec and Warren 10+ rec all passed G7 only because of the position bug fixed on 2026-11-22
   (commit 246d8f6). All three missed, with 0, 0 and 1 yards.
   - One week.
   - For QB rushing specifically, the 2,100-start test found the opponent measure does not
     predict (`OPPONENT_POSITION_DEFENSE_ADJUSTMENT_RESULTS_V1.md`). Purdy had 0 carries in a
     35-10 win.
2. **The two final-check removals split 1-1.**
   - Odunze (removed for Chicago rain and wind) went 45, which would have won.
   - Hubbard's rushing under (removed because Jonathon Brooks was out) went 120. Removing it
     was right.
3. **MNF: the operator's football read beat the card.**
   - Ja'Marr Chase drew the attention, as the operator said, and finished with 31.
   - Burrow threw for 300.
   - The tight ends went 66 (Okonkwo) and 77 (Gesicki).
   - Antonio Williams had 122.
   - Five of the operator's six legs would have been rejected by G3 (rung too close to the
     projection) (`data/research/cards/wk11_2026_mnf_challenge_entries.csv`).

## 2. Main lines, FanDuel + DraftKings (889 lines)

| market | blind OVER | blind UNDER | model OVER picks | model UNDER picks |
|---|---|---|---|---|
| rec_yards | -9.4% | -1.7% | -0.8% | **+7.9%** |
| receptions | -8.6% | -4.3% | -1.6% | -1.4% |
| rush_yards | -4.0% | -6.9% | -10.9% | **-12.3%** |
| pass_yards | +1.7% | -12.9% | -1.2% | -24.6% |
| **all** | **-7.3%** | -4.4% | -1.9% | -0.7% |

- Overs lost again.
- **Rushing unders had a bad week:** model rushing unders went -12.3%.

## 3. The six football screens, first forward week

**Screened rushing unders went 1-3:**
- Tuten U70.5: 33, a win.
- Hubbard U75.5: 120, the one quarantined.
- McCarthy U16.5: 33.
- Gibbs U84.5: 98.

**The screen-rejected rushing unders went 2-3:**
- Henry U63.5: 94;
- Barkley U55.5: 75;
- Hampton U62.5: 89;
- Mason U43.5: 38, a win;
- Skattebo U54.5: 34, a win.

Every other screened cell was negative. **The forward record of the one cell that held
historically is now 1-3.**

## 4. Alt rungs (FanDuel overs, 3,330 graded)

| cushion | n | hit | breakeven | ROI |
|---|---|---|---|---|
| < 1.0 (above the projection) | 2,217 | 20.7% | 25.5% | -27.4% |
| 1.0-1.25 | 323 | 57.3% | 60.9% | -5.5% |
| 1.25-1.5 | 195 | 63.1% | 71.2% | -11.1% |
| 1.5-2.0 | 235 | 73.2% | 77.7% | -5.3% |
| 2.0-3.0 | 192 | 75.0% | 81.0% | -6.6% |
| > 3.0 | 168 | 81.5% | 84.5% | -2.4% |

**The shortlist band** (1.5-3.0x, -150 to -400), by week:
- week 9: **+11.3%**;
- week 10: **-15.6%**;
- week 11: **-3.8%**.

## 5. Leg-type record for rule 23, weeks 9-11 (shortlist band)

`data/market_track_record/graded/w9_w11_2026_leg_type_record_v1.csv`

| leg type | n | hit | ROI | wk 9 / 10 / 11 | eligible |
|---|---|---|---|---|---|
| TE receptions alt | 23 | 78% | **+11.0%** | +47 / -22 / +15 | yes |
| QB rushing alt | 19 | 79% | **+14.9%** | +25 / +44 / **-30** | yes |
| RB receptions alt | 17 | 77% | +2.2% | +31 / +2 / -6 | yes |
| RB rushing alt | 62 | 73% | +0.9% | +32 / -29 / +4 | yes (barely) |
| WR receiving-yards alt | 157 | 69% | -4.6% | +2 / -16 / -1 | no |
| RB receiving-yards alt | 57 | 67% | -4.8% | | no |
| TE receiving-yards alt | 84 | 67% | -7.4% | | no |
| WR receptions alt | 31 | 61% | -15.6% | | no |

Every eligible type swings week to week. The samples are small, and eligibility is a floor, not
proof.

## 6. Line movement: betting the model's Thursday pick at Thursday's price

537 player lines priced on both Thursday and Sunday:

| Thursday-to-Sunday move | n | hit | ROI at Thursday's line and price |
|---|---|---|---|
| **toward the model's Thursday number** | 158 | **58.9%** | **+9.7%** |
| away from it | 123 | 51.2% | -2.9% |
| no move | 256 | 52.3% | -3.8% |
| **all Thursday picks** | 537 | 54.0% | **+0.4%** |

- **The model's picks taken on Thursday broke even (+0.4%).** The same picks taken at Sunday's
  numbers lost (-1.2% across all model picks).
- **When the market moved toward the model, the Thursday number won 58.9%.**
- **Caveat:** which lines will move is only known after the move. This supports betting
  earlier, but it is one week.
- **Week 10 cannot be checked.** Its Thursday board (run 43327080728) priced only the TNF game's
  main lines (76 rows); the whole-week Thursday capture was alt ladders only.
- **More weeks need a full-slate Thursday capture.** Week 12's Thursday pull is the next data
  point.
