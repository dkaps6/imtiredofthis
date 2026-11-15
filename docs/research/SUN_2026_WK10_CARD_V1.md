# Sunday 2026 week 10 card (1pm, 4pm and SNF games)

Board: Full Slate run 43730255663 (paid pull 15:31Z, all steps green incl. strict
audits); legs `data/research/sun_2026_wk10_card_v1.csv`; builder
`scripts/research/build_sun_2026_wk10_card_v1.py`. NE@DET (9:30am) was missed.

| ticket | price | legs | model ticket % | breakeven |
|---|---|---|---|---|
| SAFE | +188 | 9 | 49.9% | 34.8% |
| MODERATE | +707 | 4 | 21.3% | 12.4% |
| MEDIUM | +1561 | 6 | 17.2% | 6.0% |
| HIGH | +3922 | 6 | 5.8% | 2.5% |
| LOTTERY | +28475 | 5 | 0.81% | 0.35% |

## How it was built (and what it was checked against)
* Quarterbacks verified per team before pricing: ARI Beck, TEN Ward, LV Mendoza
  written to `config/qb_starter_authority_v1.csv` from official announcements;
  CLE Watson, MIA McCord, ATL Penix confirmed by news. ARI, MIA, LV and MIN (first
  start / first game after trade) supply NO legs.
* Every leg passed the non-cushion gates (snaps, role floor, own history, matchup,
  availability); SAFE legs passed all nine. Every leg's model probability beats
  its price's breakeven, and HIGH/LOTTERY legs ALSO have own history above
  breakeven (leg standard does not fall with the tier).
* One leg per game per ticket, so each ticket is a standard cross-game parlay and
  FanDuel's payout is exactly the product of the leg prices at placement.
* No player on more than two tickets; no receptions-only ticket; SAFE over +150;
  LOTTERY over 250x. `validate_ticket_card_v1`: 0 errors.

## Rejected on football grounds
Daniel Jones beyond 99.5 (model 246 vs book 186 and median 180: gigantic
disagreement); Cam Ward 174.5 (own 5/8 below breakeven); Mahomes (sore throwing
elbow, UCL sprain); Juwan Johnson and Darren Waller (zero-yard games at the rung).

## Known risks
* "Model ticket %" uses the model's own probabilities, which V27 records as
  overconfident; read them as optimistic.
* MODERATE and HIGH share Kupp, Keenan Allen, Boutte and Young.
* SNF legs (Rodgers, Dowdle) carry inactive risk until ~6:50pm ET.
* Braelon Allen: NYJ may trail BUF. Dowdle: 40% snaps, 5-game sample.
* Prices captured 10:31 ET.
