# RB rushing UNDER: weekly forward record

**Started:** 2026-11-19, week 11. **Operator agreed:** "Sure why not."

**The point:** a clean forward record, on weeks nobody has looked at, of the one betting
pattern that held up through the reverse-engineering
(`REVERSE_ENGINEER_2026_W1_10_V1.md` §7-8, independently re-sighted in
`D1_RUSH_YARDS_VS_MARKET_RESULTS_V1.md`).

## The rule, frozen 2026-11-19 (no tuning after any graded week)

- FanDuel main-line **RB rushing yards UNDER**.
- At least **2 of 3** flags:
  - the RB's team is an **underdog by 3+**;
  - his **YPC is below 95%** of the league RB average this season;
  - the book has **already cut him**: carries per game x shrunk YPC is more than 1.2x the line.
- **Line ≥ 30.** Lines under 30 were backups and hit 51% in the backtest.
- **Void** if the player is Out or Doubtful, or takes no offensive snap.
- **Backtest record before tracking:**
  - 83 bets at 66.3%, +24.6%, 6 of 7 weeks positive;
  - about 74% on lines of 30+;
  - 60.6% after removing in-game exits;
  - the CI includes 0. **Not proven: that is why this exists.**

## How it runs

```
# after the Thursday paid capture is archived
PYTHONPATH=. python scripts/research/rb_under_tracker_v1.py log --week N \
    --board data/market_track_record/boards/2026_wkNN.csv --capture "<when / run id>"
# after the week's games (nflverse publishes the next day)
PYTHONPATH=. python scripts/research/rb_under_tracker_v1.py grade --week N
```

The ledger is `data/research/rb_under_tracker/ledger.csv`, committed every week. It is graded
at the captured price; the price you actually got may differ.

## Week 11 picks (Thursday capture, run 44580190319)

| RB | line | price | flags | note |
|---|---|---|---|---|
| Jonathan Taylor (IND) | 51.5 | -113 | dog, cut | TNF tonight. IND's starting tackles are out. |
| Cam Skattebo (NYG) | 52.5 | -114 | dog, cut | |
| Ashton Jeanty (LV) | 32.5 | -114 | dog, cold, cut | |
| Aaron Jones (MIN) | 30.5 | -114 | dog, cut | Committee with Jordan Mason. |
| Derrick Henry (BAL) | 67.5 | -114 | dog, cold | The spread moved from -2.5 to -3 on 2026-11-19, which added the dog flag. BAL's QB changed in week 10 (Henry: 10 carries, 10 yards). |

Sunday props were thin at the Thursday capture. RBs without a FanDuel line then are not in
the record.
