# Sunday 2026 week 10 card, v2 (one procedure, built by code)

Built 11:53 ET by `scripts/research/build_weekly_alt_card_v1.py` from the run
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
| SAFE | +152 | 4 | 55.2% | 39.6% |
| MODERATE | +500 | 5 | 24.4% | 16.7% |
| MEDIUM | SHORT | - | - | - |
| HIGH | +3176 | 4 | 4.7% | 3.1% |
| LOTTERY | UNSOLVED | - | - | - |

SAFE: Strange o14.5 (-650), Kupp o14.5 (-162), Deshaun Watson pass o149.5
(-580), Darren Waller o9.5 (-650).
MODERATE: Meyers o19.5 (-500), Dowdle rush o39.5 (+164), Kupp o9.5 (-275),
Judkins rush o19.5 (-900), Bryce Young pass o199.5 (-400).
HIGH: Strange receptions o4.5 (+250), Dowdle rush o39.5 (+164), Barner o4.5
(-1100), Deshaun Watson pass o249.5 (+225).

## Rebuild 11:53 ET: four defects in the checks, found when the operator
## challenged "13 players"
1. Teammate context required an exact match on every key teammate in both
   directions. Now directional: for receiving/rushing legs drop games a
   competitor who plays today missed (Breece Hall); for QBs drop games only
   when receivers out today held >= 30% of targets (Nacua + Adams 70%,
   Downs + Keenan Allen 49%, Mike Evans 32%). A competitor out today is not a
   reason to distrust history (Ayomanor with Carnell Tate out).
2. The context sample was held to "above breakeven" on every tier; it now uses
   the tier's own history rule (>= 60% SAFE/MOD/MED, > breakeven HIGH/LOT).
3. A key teammate now needs volume (>= 10 carries or 5 targets per game
   played), so one-game backups (DJ Giddens, Will Mallory) no longer sink a
   starter (Jonathan Taylor, Tyler Warren).
4. G7 ranked only today's opponents; it now ranks all 32 defenses per market.

Funnel (170 board players): snaps -32, book role line -22, QB-change team -15,
designation -4, traded -2, matchup (toughest 25% of the NFL) -23, competitor
arrival -6, teammate context -16, model edge -27, tier history rules -2 =
**21 players in 7 games**. Braelon Allen, Stafford, Daniel Jones and Keenan
Allen still fail.

**The model-edge rule removes 27 players**, including Jaxon Smith-Njigba (9/9
over 49.5; model 77 vs book 91), Ja'Marr Chase, Burrow, Dak Prescott and
McCaffrey. That is the model's documented ceiling compression for stars
(`MODEL_CEILING_COMPRESSION_2026_W10_V1.md`). Without the rule the pool is 45
players in 9 games. The rule is the operator's ("every leg backed by the model,
with an edge"); changing it is the operator's call.

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
