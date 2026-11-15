# Sunday 2026 week 10 card, v5 — the Nov 9 integrated alt-line rule

> **DO NOT PLAY (12:40 ET).** Checked leg by leg against each player's own season,
> 28 of the 40 legs do not clear their own price (e.g. Mumpfield over 4.5
> receiving yards at -1000: 6/8 this season against 91% needed). v5 checked only
> the model cushion. The card where every leg passes the cushion AND the
> player's season beats the price is v4 (`SUN_2026_WK10_CARD_V4.md`). The chat
> message for v5 also wrote "rec" for both receptions and receiving yards.

Built 12:36 ET. The rule the operator and I settled on 2026-11-09
(`PRISM_ALT_CUSHION_STRATEGY_2026_W9.md` addenda, `PRISM_ALT_FOOTBALL_INTEGRATION_2026_W9.md`):
real role from the book's main line, rung >= 1.5x under the projection, toughest
quartile of defenses excluded, snaps >= 30%, own record >= 60% over >= 5 games,
no per-game quota; plus the operator's verify-the-state rule (QB-change teams,
traded players, teammates back/out). Per player, the highest-paying rung that
still holds a 1.5x cushion; one market per player; no player on more than two
tickets (players spread across tickets first); max 20 legs.

Pool: 40 players, 9 games. SAFE +163 (9), MODERATE +357 (8), MEDIUM +801 (12),
HIGH +1679 (11). LOTTERY UNSOLVED: the remaining legs reach +2033 in 20 legs.
Validator 0 errors (edge basis "cushion": every rung >= 1.5x under the
projection). Prices 10:31 ET; same-game legs mean FanDuel's quote is the price.
Card: `data/research/sun_2026_wk10_card_v5.csv`.

What went wrong earlier today (v1-v4): the session added filters that are not in
the Nov 9 rule (model probability above the price, own history above the price,
an exact-match teammate check, one leg per game, invented tier minimums).
