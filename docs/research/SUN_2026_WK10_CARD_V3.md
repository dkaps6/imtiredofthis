# Sunday 2026 week 10 card, v3 (edge basis: history)

Built 11:57 ET by `build_weekly_alt_card_v1 --edge-basis history` (same command
as v2 plus that flag). Legs `data/research/sun_2026_wk10_card_v3.csv`, checks
`data/research/sun_2026_wk10_leg_checks_v3.csv`. Supersedes v2.

Why: the v2 rule "model probability must beat the price" removed 27 players,
largely stars the model under-projects (MODEL_CEILING_COMPRESSION_2026_W10_V1).
The TNF-validated nine-gate builder never used that rule; it used the model as
a projection that must clear the line. v3: a leg is model-backed when the
projection clears the line or the model probability beats the price, and the
price edge must come from own history AND history with today's teammates.
All other checks unchanged. Pool: 23 players in 9 games.

| ticket | price | legs |
|---|---|---|
| SAFE | +152 | Ayomanor o9.5, Freiermuth o4.5, Barner o4.5, Higbee o9.5, Purdy rush o4.5, Judkins rush o19.5, Waller o19.5 |
| MODERATE | +510 | Meyers o19.5, Dowdle rush o39.5, Kupp o14.5, Dak pass o174.5, Judkins rush o19.5 |
| MEDIUM | +1207 | Cook rush o69.5, Strange o14.5, Dowdle rush o39.5, Kupp o14.5, Dak pass o174.5, J. Taylor rush o39.5, Waller o24.5 |
| HIGH | +3085 | Lawrence pass o174.5, Watson pass o274.5, Bryce Young pass o324.5 |
| LOTTERY | +25242 | Strange 5+ rec, Freiermuth o19.5, Barner o4.5, Kyren Williams o9.5 rec, J. Taylor rush o39.5, Watson pass o274.5, Young pass o324.5 |

Still out, by check: Chase (6/8 at -850 needs 89.5%), Burrow (7/8 at -1450),
Bijan / Lamb / Josh Allen (opponent in the NFL's toughest quartile for the
market), McCaffrey (Keenan Allen arrival). Smith-Njigba is eligible but the
search chose Kupp/Barner for the SEA game.
