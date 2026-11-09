# The rules, in plain English — written down so they can be checked without reading code

This file exists because the person directing this work cannot read the Python
and should not have to. Every rule below came from him, usually after a bad
card. Each entry says what he said, what it means operationally, and where it
is enforced. **If any line here misrepresents his intent, the code is wrong,
not the idea.**

Enforced in `scripts/research/build_alt_cards_v1.py`, tested in
`tests/test_build_alt_cards_gates_v1.py`.

---

### 1. "He's third string. I know what his usage is."
*About Roschon Johnson, CHI, who I had on a card at +294.*

**Rule:** if a player isn't meaningfully on the field, nothing else about him
matters. Before any projection, price or ratio is considered, he must be
playing — at least 30% of his team's offensive snaps.

**Gate G1.** Roschon was 15.6%. Rejected.

---

### 2. "An alt line of 4.5 rec yards? That's literally telling us nobody thinks his production warrants that."

**Rule:** the book's main line is a statement about the player's role. A tiny
line means the book doesn't think he's part of the offense, and it's usually
right. A big ratio against a tiny line is not an edge — it's a bet that a
bench player gets involved.

**Gate G2.** Receiving lines under 18 yards and rushing lines under 15 are
excluded outright. In week 9, players with sub-10 receiving lines failed to
record a single yard 29.4% of the time.

---

### 3. "Everything else hit but Egbuka."
*He was on four of five cards. Cushion 0.92.*

**Rule A:** never bet a number higher than our own projection. If the model
says 4.1 receptions and the rung needs 5, that is betting against ourselves.

**Gate G3,** with the rejection printed as `BETTING AGAINST OUR OWN
PROJECTION`. 174 of week 9's legs failed this.

**Rule B:** the same leg must never appear on more than one card. Five cards
sharing legs aren't five bets, they're one bet five times — which is why a
single miss took out 80% of the board.

**Enforced in card construction.** Tested by
`test_cards_never_share_a_leg`.

---

### 4. "We are putting ourselves in a box. I just want to play the best plays each week. Not uniform."

**Rule:** no quotas. No "one per game". Take the best qualifying legs wherever
they are — three from one game and one from another is fine.

**Removed.** The one-leg-per-game rule was mine, not his, and it was dropped
once it was shown to cost real options.

---

### 5. "There's something similar there. All of those guys were super low projected by the books, and their usage backs it up."

**Rule:** own-record evidence must come from games the player actually played,
not games where he happened to record a stat. A 12-snap cameo with one catch
should not count as proof he clears a number.

**Gate G6.** Minimum 5 games played, 60% own hit rate, denominator from snap
counts.

---

### 6. "You're forgetting football logic and matchup logic. It all needs to be evaluated together. Nothing left out."

**Rule:** who he's playing matters as much as what he's priced at.

**Gate G7.** Legs against the toughest quarter of defenses are excluded. In
week 9 those returned −1.8% — no edge at all — while the softest quarter
returned +16.9%.

---

### 7. "I'm only using FanDuel."

**Rule:** only show lines that exist on his book, at prices he can actually
take.

**Gate G9.** One book, price range enforced. Three times I showed him
DraftKings rows he couldn't bet.

---

### 8. Not his words — the repository's, and I ignored them

`AGENTS.md` makes the **player-state anomaly gate non-negotiable**: a player
returning from injury, or coming off missed games, must be labelled before his
past production is used as evidence.

Emeka Egbuka was **Out** in week 7 (concussion), **Out** in week 8 (foot), and
**Questionable** in week 9 (foot). His 4/6 record was from weeks 1–6, before
the injury. I applied it to a player returning from a two-game absence, on four
cards.

**Gate G8.** Out, Doubtful and Questionable are all disqualifying.

---

## What he is owed going forward

1. **Every card comes from the script**, with its rejection log attached. No
   improvised picks, no "three methods clear this" from reading a table.
2. **If the script returns nothing, he is told nothing.** An empty slate is a
   real answer. A relaxed card is not.
3. **Prices are checked against his book before they are shown**, and the
   capture time is stated so he knows how stale they are.
4. **His ideas get tested, not acknowledged.** The pattern he correctly
   identified — agreeing and then exploring something else — is the failure
   underneath every other failure in this project.
