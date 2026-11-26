# The rules, in plain English — written down so they can be checked without reading code

> ## CURRENT RULES (re-read from the full transcript 2026-11-15). READ THIS FIRST.
>
> On 2026-11-15 the Sunday card was rebuilt five times in one morning because the
> session worked from a compressed summary instead of these rules. Every rule
> below is the operator's, with the date and his words. Where an older section
> further down conflicts, THIS block wins. Method doc:
> `MODEL_BACKED_ALT_LINE_PARLAY_CONSTRUCTION_V1.md`; binding override:
> `evidence/2026-11-01_week8_sunday_tickets/WEEK8_USER_BINDING_TICKET_PAYOUT_AND_CONSTRUCTION_OVERRIDE_20261101.md`;
> strongest measured result: `PRISM_ALT_CUSHION_STRATEGY_2026_W9.md`.
>
> ### What a leg is
> 1. **The right side of the line.** Start from the model's projection for the
>    named player against the book's main line. "The point is to hit more bets...
>    we were on the right side of the market." (Oct 19)
> 2. **Alt lines buy cushion under that projection.** "Isn't there value in
>    stacking alt lines under their projections?" (Nov 9). Measured: rungs where
>    the projection is 1.5-3.0x the rung, priced -150 to -400, were the profitable
>    band in week 9 (+13.5% singles, parlays positive). Never bet a rung above
>    our own projection (Egbuka).
> 3. **Blend everything, leave nothing out:** model + the Vegas line + football
>    knowledge + the individual player + matchup + environment. "They all need to
>    logically be applied together." (Nov 9, Nov 12). Hit-rate history alone is
>    not validation: "seems like no math or football logic was actually used."
>    (Nov 8)
> 4. **In-season usage and role matter most.** "In season metrics are
>    incredibly important. It's what's telling the story THIS season." (Nov 13)
> 5. **On the field and a real role.** >= 30% snaps (Roschon, "He's 3rd string");
>    a tiny book line means no role ("alt line of 4.5 rec yards? ... nobody thinks
>    his production warrants that"). (Nov 9)
> 6. **Matchup matters** (toughest defenses excluded). (Nov 9)
> 7. **Verify the state before picking — don't fire from the hip:** injuries and
>    returns from injury (Egbuka), the actual starting QB ("JJ McCarthy is the
>    QB not Winston"), Ourlads depth chart, trades/new team (Keenan Allen),
>    teammates back or out (Breece Hall back; Nacua/Adams on IR). "If a guy does
>    not have an active line tonight then that's indication he shouldn't be
>    playing." (Nov 12-15)
> 8. **Use the players the model has been on the right side of.** (Nov 1)
> 9. **FanDuel only**, prices he can take, capture time stated. (Nov 9)
> 10. **No anytime TD props.** (Nov 4)
> 11. **Every market is fair game** — rec yards, rush yards, pass yards,
>     receptions; main and alt lines mixed. "I never said receptions were bad
>     with alt lines, I just didn't want one entire parlay full of just alt
>     reception lines. It should be a true blend of everything, but the best
>     plays." (Nov 13). "That's where we are making our money" = yardage alts
>     on real roles (Nov 9).
>
> ### How tickets are built
> 12. **Five tickets: SAFE, MODERATE, MEDIUM, HIGH, LOTTERY.**
> 13. **SAFE strictly over +150** ("+250 can be reevaluated. As long as it's
>     over +150", Nov 13).
> 14. **HIGH and LOTTERY over +600** ("everything above medium should have odds
>     greater than +600", Nov 8). **LOTTERY at least 250x (+24,900)** or labeled
>     UNSOLVED. An optional "million-dollar" exhibition is allowed, built from
>     validated legs, never random shots. (Nov 1)
> 15. **No hard-coded leg count.** Stacking many high-survival alt lines is the
>     preferred way to earn the odds. (Nov 1)
> 16. **NO one-leg-per-game rule.** "Ok what if we didn't use one player per
>     game? I don't remember making that rule... Doesn't have to be uniform, just
>     the best plays. We are putting ourselves in a box." (Nov 9)
> 17. **Same-game legs are allowed; the real FanDuel quote is the price** — no
>     invented per-leg pricing or false independence. (Nov 1)
> 18. **No player on more than two tickets**; don't copy the same fragile legs
>     onto every ticket — "we are over covered and over exposed." (Nov 1)
> 19. **HIGH/LOTTERY are less likely, not worse legs.** "A parlay that still
>     has an edge but is way less likely to hit... smart football decisions...
>     not just random shots." (Nov 13)
> 20. **Best plays of the whole day**, any kickoff time. (Nov 1)
> 21. **Single-game slates (TNF/SNF/MNF) are built differently** from the full
>     slate; the Sunday full slate comes first. (Nov 13)
> 22. **His ideas get tested, not just acknowledged** (see bottom of this file).
> 23. **Nothing goes in a parlay unless that kind of leg has been graded before.**
>     "Stop putting shit in parlays that haven't been fucking graded before."
>     (Nov 22, SNF). A leg type is market x side x line type (alt / main) x
>     position, e.g. "QB rushing alt over" or "RB rushing main-line under, all
>     six screens".
>     - It is eligible only if our graded record for that type, taken from the
>       committed graded boards, is **positive at the price actually paid** and
>       has **at least 10 graded bets**.
>     - Every week's grading is added to the record, so the pool can grow or
>       shrink.
>     - First application (weeks 9-10 alt grading, band 1.5-3.0x cushion, -150
>       to -400):
>       - QB rushing alt over: 12/13, +35%;
>       - RB receptions alt: +8% (n=10);
>       - TE receptions alt: +7% (n=12).
>     - These were **negative** and are not eligible until they turn positive:
>       - WR receiving-yards alt: -6.6% (n=100);
>       - TE receiving-yards alt: -4.2% (n=55);
>       - RB receiving-yards alt: -4.3% (n=34);
>       - WR receptions alt: -22.7% (n=20);
>       - RB rushing alt: -0.9% (n=39).
>     - Main-line screened RB rushing unders are eligible (2026: 43 bets,
>       +31%; 2024-25: 246 bets, +2.2%). First forward week (wk 11): 1-3.
>     - **Current record:**
>       `data/market_track_record/graded/w9_w11_2026_leg_type_record_v1.csv`,
>       updated every week. After week 11 the eligible types are:
>       - TE receptions alt: +11%, n=23;
>       - QB rushing alt: +15%, n=19;
>       - RB receptions alt: +2%, n=17;
>       - RB rushing alt: +1%, n=62.
>     - Every receiving-yards alt type and WR receptions alts are still
>       negative.
> 24. **Think about each game like a football person first; the model comes
>     second. This is NOT a rule or a checklist.** (Nov 23, after his MNF
>     CIN @ WAS ticket went 6/6 at about +511, while the gated card missed.)
>     "I used football logic first then asked what the model thought.
>     FOOTBALL DECISIONS ARE DIFFERENT FROM A MATH PROBLEM." And, when it was
>     first written down as a four-question method: "It's not a rule. Every
>     game is different. It's fluid. So not every rule is going to fit every
>     game."
>
>     **Do not turn this into a template, a gate or a fixed list of
>     questions.** Each game has its own story, and what matters changes from
>     game to game. His CIN @ WAS read is one example of that thinking, not a
>     formula:
>     - CIN's two star receivers would draw double coverage, so Burrow's safe
>       throws go to the tight ends and backs;
>     - the defense keys on Chase, which leaves Higgins some shots;
>     - WAS had to throw to keep up, and Daniels spreads it around.
>
>     The next game might turn on something else entirely: a weather game, a
>     backup QB, a coach's tendency, an injured line. The model's projection
>     and the guardrails are there to check and size a read, never to stand in
>     for one.
>
>     **Context from testing his Monday read (2026-11-26,
>     `MNF_READS_SINGLE_GAME_TEST_RESULTS_V1.md`). This is information for his
>     reads, not a rule:**
>     - A high total or a missing top receiver is, on average, already in the
>       book's line. Pass-catchers' overs still lost about 3% in shootouts, and
>       backs' and tight ends' overs did no better when the top receiver sat.
>     - So the edge is in reading the specific game: who the defense keys on,
>       and who is forced to throw. The situation alone is not enough.
>     - In primetime single games, rungs near the projection came out about even
>       (-2%, 6 games). On Sunday afternoons the same rungs lost 13%.
>
> ### Rules a session invented that are NOT his (do not reapply)
> * one leg per game per ticket (removed Nov 9; reapplied by mistake Nov 15);
> * "model probability must beat the price" as a hard filter (Nov 15) — it
>   removed every star the model under-projects; the method uses the model as
>   the projection the rung must sit under (rule 2);
> * MODERATE >= +500 / MEDIUM >= +1200 / HIGH >= +3000 minimums (Nov 15);
> * an all-teammates exact-match context check (Nov 15) — the anomaly gate is
>   his, that implementation was not.
>
> ---


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

**Rule B (superseded Nov 1 by "no player on more than two tickets"):** the same leg must never appear on more than one card. Five cards
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
