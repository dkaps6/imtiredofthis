# TNF 2026 week 11: IND @ HOU (8:15 ET), single-game card

**Paid run:** 44580190319, FanDuel, captured about 12:45 ET on Nov 19. It priced the whole week,
Thursday through MNF, and was archived to git by run 44581177692:
- `boards/2026_wk11.csv`
- `alt_props/2026_wk11_thu.csv`
- `raw_props/2026_wk11.csv`
- `game_odds/2026_wk11.csv`

**Credits.** Two pulls were made:
- **Run 44578946345 failed** at the live-prop identity gate on Andrew Ogletree. Ourlads lists
  him as "Drew". 141 credits were spent and are not recoverable, because the gate stops before
  the replay seam.
- A verified roster row was added to `data/manual_roster_overrides.csv`, and the run was
  re-fired.
- **218 credits remain.**

**Rules used:** the CURRENT RULES block in `YOUR_RULES_PLAIN_ENGLISH.md`:
- never bet a rung above our own projection (rule 2);
- snaps and role (rule 5);
- matchup (rule 6);
- verify state (rule 7);
- SAFE strictly over +150 (rule 13);
- HIGH over +600 (rule 14);
- no player on more than two tickets (rule 18);
- same-game price is FanDuel's real quote (rule 17).

The nine gates apply to every tier, except G3 cushion ≥ 1.5, which applies to SAFE only. "Model
probability must beat the price" is **not** used as a filter: it is a session-invented rule. It
is reported below.

**Game:** HOU -12.5, total 43.5 (FanDuel).
- **IND:**
  - Josh Downs was traded to KC and Keenan Allen to SF; Alec Pierce is on IR.
  - Daniel Jones threw 22 passes for 108 yards in week 10.
  - LT Raimann and RT Travis are out; CB Ward is out; CB Gardner is questionable (no practice
    all week).
- **HOU:**
  - Marvin Harrison Jr. and Woody Marks are out.
  - David Montgomery returns from a 5-game absence.
  - DE Danielle Hunter is out.

## The card

| ticket | naive price | blended hit | price implies | legs |
|---|---|---|---|---|
| SAFE | **+158** | 37.4% | 38.8% | Schultz 19.5+ rec yds (-260), Boutte 14.5+ rec yds (-290), Treadwell 14.5+ rec yds (-260) |
| MODERATE | **+170** | 30.4% | 37.1% | Boutte 14.5+ rec yds (-290), Warren 3.5+ receptions (-300), Hutchinson 2.5+ receptions (-198) |
| HIGH | **+610** | 9.7% | 14.1% | Stroud 199.5+ pass yds (-440), Schultz 24.5+ rec yds (-174), Treadwell 14.5+ rec yds (-260), Collins 49.5+ rec yds (-310), Warren 3.5+ receptions (-300), Hutchinson 2.5+ receptions (-198) |
| MEDIUM | **not buildable** | | | see below |
| LOTTERY | **UNSOLVED** | | | see below |

- Prices are naive products of single-leg prices.
- **FanDuel's same-game quote is the real price.** SAFE must still be over +150 on the real
  quote.
- Rows: `data/research/tnf_2026_wk11_ind_hou_card_v1.csv`.
- The search script picked the set that maximizes blended expected return under every rule.
- Each player is on at most two tickets.
- The structural check (`validate_ticket_card_v1.py`) passes with 0 errors.

**Blended hit** averages two numbers:
- PRISM's model probability;
- the player's hit rate in games that match today's teammates (`audit_card_legs_v1.py`).

**Every ticket is priced below its blended chance.** Only Dalton Schultz's rungs have a positive
blended edge. FanDuel's alt prices tonight are steep. PRISM also under-projects stars (Collins
57 vs a 72.5 line), so Collins and Stroud may be better than their blend says.

**Why MEDIUM and LOTTERY can't be built under the rules:**
- Eight players qualify.
- With all eight on one ticket, each at his deepest rung that passes the gates and does not
  exceed his projection, the ticket reaches only **+1,638**. 250x is out of reach.
- With the two-ticket cap, four tickets can't satisfy SAFE > +150 < MODERATE < MEDIUM <
  HIGH > +600 (exhaustive search). Three can.

## Excluded, and why

| player | reason |
|---|---|
| Daniel Jones (passing) | Receivers out today held 49% of his targets. No games match today's receiving corps except week 10 (108 yds). |
| David Montgomery | RETURN_FROM_INJURY after 5 missed games. Model 31.5 vs line 57. Fails G6. |
| Jonathan Taylor rushing overs | IND is +12.5 and both starting tackles are out. Underdog-RB rushing alt overs lost 30.4% in weeks 9-10 (`REVERSE_ENGINEER_2026_W1_10_V1.md`). His only miss at 24.5+ was the week-8 blowout as a 10-point dog (23 yds). |
| Tank Dell | 3 games with HOU; fails G6. |
| McGowan, Dulin, Burks, Gould, Demercado | Under 30% snaps (G1). Gould is also questionable. |
| Moreau, QB rushing | Book role line under the G2 floor. |

## The research pattern play: separate, the operator's call

**Jonathan Taylor UNDER 51.5 rushing yards (FanDuel -113, breakeven 53.1%).** This is not on
the tickets. It bets against PRISM's 55 projection, and rule 2 forbids that on a ticket.

| for the under | against the under |
|---|---|
| The season pattern: RB on a 3+ point underdog, line under 60, normal snaps. **63.7% under, n=91, 6 of 7 weeks** | Taylor went over 51.5 in 7 of 10 games this season |
| IND is +12.5 and both starting tackles are out | PRISM projects 55 |
| Recent: 23, 46, 83 on 8, 13, 16 carries; McGowan's carries are rising | |
| The week-8 blowout as a 10-point dog: 23 yds | |
| PRISM's RB rushing OVER picks lost in both samples (-14.9%, -21.6%), so its 55 is the weaker evidence | |
