# Sunday 2026 week 10 card (1pm, 4pm and SNF games)

Board: Full Slate run 43730255663 (paid pull 15:31Z, all steps green incl. strict
audits); legs `data/research/sun_2026_wk10_card_v1.csv`; builder
`scripts/research/build_sun_2026_wk10_card_v1.py`. NE@DET (9:30am) was missed.

| ticket | price | legs | model ticket % | breakeven |
|---|---|---|---|---|
| SAFE | +169 | 8 | 50.0% | 37.2% |
| MODERATE | +669 | 5 | 20.3% | 13.0% |
| MEDIUM | +1438 | 7 | 17.3% | 6.5% |
| HIGH | +3270 | 7 | 4.9% | 3.0% |
| LOTTERY | +28475 | 5 | 0.81% | 0.35% |

## Revision 11:30 ET: Keenan Allen removed (team change)
Keenan Allen (o19.5, on MODERATE and HIGH) was traded IND -> SF; all 8 of his 2026
games were with IND and today is his first SF game. His own history and the
model's projection describe his IND role, so the leg had no evidence behind it.
The card process stamped every leg HEALTHY_FULL without a team-change check; it
now derives `state_label` from each player's prior teams
(`build_slate_alt_legs_v1.team_change_state`: fewer than 3 games with today's
team after playing elsewhere = UNRESOLVED_CONTEXT) and `validate_ticket_card_v1`
refuses UNRESOLVED_CONTEXT and DNP_INACTIVE legs as errors. No other card or pool
player had a team change.
* MODERATE: Allen -> Javonte Williams rush o39.5 (-360) + Khalil Shakir o24.5 (-380).
* HIGH: Allen -> Javonte Williams rush o39.5 (-360).
* Considered and rejected: Mahomes (still the elbow rejection below; the club
  report lists him full/no designation, but the morning call stands), Tony Pollard
  (his wk7-8 surge came with Tyjae Spears out; Spears is questionable/full and
  likely back, and in the shared weeks Pollard cleared 49.5 only 3/5), Jakobi
  Meyers (limited again this week; snaps fell to 54-68% after the wk8 hamstring).

## Revision 11:55 ET: teammate-context check (Braelon Allen, Daniel Jones removed)
Extending the team-change check to teammates (every 2026 skill player whose team
changed, plus the week-10 injury report):
* Braelon Allen (MEDIUM, HIGH) removed: his 5/5 over 39.5 came in weeks 4-8 with
  Breece Hall out; with Hall active (wks 1-3, 9) he cleared 39.5 in 1/4, below
  the 40% breakeven. Hall returned in week 9 and is not on the report.
* Daniel Jones o99.5 (SAFE, MEDIUM) removed: IND traded its top two receivers by
  yards (Josh Downs -> KC, Keenan Allen -> SF); Jones threw 108 and 144 in weeks
  7-8 even with them, and -1450 needs 93.5%.
* Added (each with game log, snaps and injury history checked): Shakir o29.5,
  Elic Ayomanor o9.5 (Carnell Tate OUT again), Tommy Tremble o9.5, Chase Brown
  rush o49.5 (replaces Rodgers on HIGH; same game), Josh Allen o174.5.
* Rejected: Mason Taylor (missed wks 2-4, Garrett Wilson back), Dawson Knox
  (two of seven hits were 5 yards; model 21 vs median 6).
* Other trades touching card teams: HOU gained Marvin Harrison Jr. (OUT today),
  DAL gained RB Kendre Miller (first DAL game; Goodson OUT; Javonte flagged),
  PIT gained Valdes-Scantling (neutral for Rodgers).

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
* MODERATE and HIGH share Kupp, Javonte Williams, Boutte and Young.
* Javonte Williams: DAL added RB Kendre Miller this week (first DAL game).
* SNF legs (Rodgers, Dowdle) carry inactive risk until ~6:50pm ET.
* Dowdle: 40% snaps, 5-game sample.
* Prices captured 10:31 ET.
