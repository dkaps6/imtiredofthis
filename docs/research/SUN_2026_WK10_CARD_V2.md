# Sunday 2026 week 10 card, v2 (one procedure, built by code)

Built 11:41 ET by `scripts/research/build_weekly_alt_card_v1.py` from the run
43730255663 board (FanDuel prices captured 10:31 ET), legs
`data/research/sun_2026_wk10_card_v2.csv`, every rung's checks
`data/research/sun_2026_wk10_leg_checks_v2.csv`. Supersedes v1.

```
python -m scripts.research.build_weekly_alt_card_v1 --legs <sun_legs.csv> --season 2026 --week 10 \
  --exclude-teams ARI,MIA,LV,MIN --quarantine "Patrick Mahomes" --out-card ... --out-checks ...
```

* `--exclude-teams`: first NFL start (ARI Beck, MIA McCord, LV Mendoza) or first
  game after a QB trade (MIN Cousins).
* `--quarantine Patrick Mahomes`: sore throwing elbow (UCL sprain) per this
  morning's news; the club report lists him full/no designation. Kept out.

| ticket | price | legs | model ticket % | breakeven |
|---|---|---|---|---|
| SAFE | +153 | 4 | 54.7% | 39.6% |
| MODERATE | +500 | 5 | 24.4% | 16.7% |
| MEDIUM | SHORT | - | - | - |
| HIGH | +3176 | 4 | 4.7% | 3.1% |
| LOTTERY | UNSOLVED | - | - | - |

SAFE: Shakir o29.5 (-270), Strange o14.5 (-650), Kupp o9.5 (-275), Deshaun Watson
pass o149.5 (-580).
MODERATE: Meyers o19.5 (-500), Dowdle rush o39.5 (+164), Kupp o9.5 (-275),
Judkins rush o19.5 (-900), Bryce Young pass o199.5 (-400).
HIGH: Strange receptions o4.5 (+250), Dowdle rush o39.5 (+164), Barner o4.5
(-1100), Deshaun Watson pass o249.5 (+225).

## Why only three tickets
3,016 FanDuel rungs, 170 players. After every check, 13 players in 6 games
remain (first-failing check: teammate context 867, role floor 614, snaps 487,
matchup 427, no model edge 212, teammate arrival 134, QB context 121,
questionable 82, team change 22). With one leg per game and two tickets per
player:
* MEDIUM (+1200): its only plus-money eligible rung is Dowdle o39.5; every
  combination reaching 13x needs players already used twice.
* LOTTERY: the highest passing price in every game multiplies to 166x < 250x.
No check was loosened to fill them.

## Checked after the build
* No card player has a week-10 designation. Brian Thomas Jr. and C.J. Stroud,
  named "out" in a stale live-blog snippet, are not on the week-10 report and
  are active (Stroud started every game).
* Official 1pm inactives could not be read directly (nfl.com and fantasypros
  are blocked in this session's network); confirm before placing.
* SAFE is +153 at 10:31 prices: if FanDuel now shows it at or under +150, it is
  not a SAFE ticket.
* Dowdle (SNF) is on MODERATE and HIGH: 40% snaps, 5 games, inactive risk
  until ~6:50pm ET.
