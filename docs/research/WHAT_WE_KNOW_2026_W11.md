# What is actually settled, as of week 11 2026

**Written 2026-11-20** in answer to the operator: *"is there ANYTHING that we have learned... or
are we just throwing shit at the wall each week?"* Only findings that held across weeks,
seasons or independent tests are listed here. Single-week results are not.

## 1. The market itself: FanDuel main lines, weeks 1, 2, 6-10, every offer, no model

| | n | hit | ROI | weeks above breakeven (52.4%) |
|---|---|---|---|---|
| **every OVER** | 2880 | 47.7% | **-9.5%** | **0 of 7** |
| every UNDER | 2880 | 52.3% | -2.4% | 4 of 7 |
| **rush_yards UNDER** | 502 | **56.2%** | **+5.2%** | **6 of 7** |
| receptions OVER | 991 | 46.9% | -9.8% | 0 of 7 |
| pass_yards OVER | 194 | 51.5% | -3.1% | 2 of 7 (the one market where overs are not bad) |

**Overs lost in every week we have.** Every ticket card built this season was made of FanDuel
alt **overs**. That is the side of the market that loses every week.

## 2. What repeats and what does not (three seasons, plus nine earlier null studies)

- **Role repeats:**
  - snap share week to week r 0.82-0.93;
  - target share r 0.80-0.89;
  - carry share r 0.86-0.90.
- **Efficiency does not:**
  - yards per target r 0.25-0.31;
  - yards per carry r 0.21;
  - catch rate about 0.

A yardage prop is role x a large dose of noise. Sources: `REVERSE_ENGINEER_2026_W1_10_V1.md` §5
and `PRISM_ROLE_SHRINKAGE_FIX_RESULTS_V1.md`.

## 3. How games go

- **The score sets the game plan.**
  - Winners by 15+ run 31 times; losers by 15+ run 20 times.
  - The pregame spread predicts only part of that.
- **Defenses do not dictate an opponent's game plan** (r 0.00).
- **Offenses have stable identities** (neutral pass rate over expected, r 0.54).
- **Big favorites build a lead and run.** TNF week 11: Montgomery had 26 carries; IND's passing
  game was dead.

## 4. Where the book is better than us

- On **yards**, FanDuel's line has a lower error than every projection we have, in every market
  tested:
  - rush yards: 21.3 vs PRISM 23.3;
  - rec yards: 21.1 vs 21.7.
- **Passing yards is the only market where PRISM matches the line.**
- Fixing PRISM's carry and target shares (`RB_CARRY_SHARE_*`, `ALLOCATIVE_PRIOR_*`,
  `TARGET_ALLOCATIVE_*`) made opportunity more accurate. It did **not** close the yards gap.

## 5. Why "model mean above the line" loses

Yardage outcomes are right-skewed: a few big games pull the average above the typical game.
- The line sits between the median and the mean.
- A projection that is right **on average** still says OVER too often.
- PRISM's OVER picks lost in 6 of 7 weeks.
- When our carries model was above the RB rushing line, the over hit **38%**.

## 6. ~~The one rule that held up everywhere it was tested~~: CORRECTED 2026-11-20, it did not hold on unseen seasons

**On 2024-2025 lines it went 60.2% and then 47.7%:** 54.0% and +1.5% across both, CI
[-11%, +14%] (`ROLE_CHANGE_AND_RB_UNDER_HISTORICAL_RESULTS_V1.md`). The 2026 result did not
generalize. **What does hold across three seasons is §1: blind overs lose 7-12% in every
market.**

Original text, kept for the record:

**RB rushing UNDER, with 2+ of: team underdog by 3+, below-average YPC, book already cut the
line; line ≥ 30.**
- 83 bets, 66%, 6 of 7 weeks positive;
- it held on weeks not used to build it;
- an independent re-sighting (model-above-line overs at 38%);
- forward record: 1-0 (Taylor UNDER 51.5, 21 yards).

It sits inside finding 1: rushing unders win even blind (+5.2%).

## 7. What did NOT hold up (do not reuse without new evidence)

- The cushion band (week 9 +13.5% → week 10 -15%).
- The PRISM side edge (weeks 6-8 +3.5 pts → week 10 -5.7).
- QB rushing unders: in-sample yes, out of sample no.
- The model/line 0.8-0.9 band.
- Receiver role-trend fades: 2026 only, reversed over 3 seasons.

## 8. Why it felt like a different approach every week

The method changed after each losing week. **One week is far too small to judge a method.** A
3-leg ticket with 70% legs hits 34% of the time, so a sound method still loses about two weeks
in three. Changing rules after each result chases noise. A method is judged on 50+ bets, under
rules frozen in advance.
