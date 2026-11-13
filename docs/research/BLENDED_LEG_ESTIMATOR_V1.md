# One estimator per leg, not a sequence of alternative ones

## The failure this replaces

Across one session this leg selection was done four separate ways, each
discarding the previous instead of combining with it:

1. **floor test** -- worst game of the season clears the line. Ignored the
   model, the opponent and the game environment.
2. **nine gates** -- used `model_proj` only as a ratio (`proj/threshold >= 1.50`),
   which is a probability proxy and a poor one, and threw away the model's own
   dispersion.
3. **model distribution** -- gamma on (`model_proj`, `model_sd`). Ignored the
   player's own production, which is what exposed the ceiling compression.
4. **own history** -- empirical rates. Ignored the model, the opponent and the
   environment entirely.

Every one of those is a partial view, and each conclusion was presented as if it
superseded the last. The operator's objection was that the sources have to be
applied together. They do.

## The estimator

For each (player, market, threshold), P(over) is a weighted blend of three
sources, each of which knows something the others do not:

| source | weight | knows | blind to |
|---|---|---|---|
| de-vigged **market ladder** | 0.45 | opponent, environment, role, injury news, sharp money | carries the book's margin; is the thing we must beat, so cannot be the only input |
| **model** gamma(`model_proj`,`model_sd`) | 0.25 | our football projection: team context, coverage, weather, allocation | measured ceiling compression (see `MODEL_CEILING_COMPRESSION_2026_W10_V1.md`) |
| **own history**, matchup- and environment-adjusted | 0.30 | current role, real variance, the player's actual ceiling | opponent- and environment-naive before adjustment; 6-8 games |

The market ladder is read as a **whole implied distribution**, not one price at a
time: every rung's price is de-vigged by the two-sided hold taken from that
player's main line, then forced monotone in the threshold.

Own history is the only source that is opponent- and environment-naive, so it
is the only one adjusted, which avoids double-counting what the model and the
market already price:

    adjusted values = own values x (game total / league mean total) x (opponent allowed / league mean allowed)

Football gates still apply on top and are not part of the blend: excluded
markets (`receptions` -30.7% blind, `rush_rec_yards`, `anytime_td`), a snap
floor, availability, and the role-disagreement rule -- a book line at or under
10 yards is a statement that the player has no role, and books are reliably
right about roles, so our model projecting 2x that line is a model error rather
than an edge (2026 week 10: Theo Johnson 4.5 vs 17.9, Van Jefferson 7.5 vs 22.0,
both excluded).

## What blending changed on the 2026 week 10 TNF board

Environment: total **40.5** against a week-10 mean of **44.9** (ratio 0.901),
spread NYG -1.5. Matchup: NYG allow **0.837x** league average to WR/TE, WAS
allow **1.103x**. So WAS receivers take a combined **0.754x** haircut and NYG
receivers are roughly neutral at **0.994x**.

| leg | single-lens verdict | blended verdict |
|---|---|---|
| Diggs rec o29.5 (-182) | floor test: best bet on the board, 7/7, EV +35 pts | **-2.7% EV, dropped** |
| Antonio Williams rec o14.5 (-587) | nine gates: carded in MODERATE | **-1.4% EV, dropped** |
| Nabers rec o109.5 (+430) | model: 3.8%, a terrible bet | **+27.2% EV, best single** |
| Nabers rec o69.5 (+100) | own history: 50%, coin flip at even money | **-14.3% EV, dropped** |

The low total plus a tough NYG pass defense is what kills the Diggs rung that
four hours of single-lens analysis had as the headline bet.

## Sensitivity -- which conclusions are load-bearing

Weights are a judgment call, so every leg was re-priced under five schemes
(market-heavy, equal, model-heavy, own-heavy, base):

* **Robust**: Diggs o24.5 (+6.7% to +15.5%), Fields o14.5 (+7.8% to +16.6%).
  Positive under all five.
* **Model-dependent**: Nabers o99.5, Nabers o109.5, Daniels rush o89.5. Positive
  under four schemes, **negative under model-heavy** (-16.1%, -9.6%, -17.5%).
  These bets rest on the judgment that the model's tail is wrong. That judgment
  is measured, not convenient -- the model's p90 for Nabers is 90.7 against his
  own p90 of 134.2 -- but it is the load-bearing assumption and must be stated
  as such every time one of these legs is offered.

A second sensitivity covers a real double-count risk: a 40.5 total already
embeds both defenses, so multiplying the environment and matchup factors may
haircut twice. Re-running at full scale, `sqrt(scale)` and no adjustment moves
**only** Diggs o29.5 (-2.7% / +10.5% / +17.2%) and leaves every other leg
unchanged.

The reason is structural and worth knowing: **the own-history component is a
step function on 6-8 points.** It is insensitive to modest rescaling except when
a threshold sits between two observations, where it flips hard. Diggs o29.5 is
exactly such a boundary. That fragility is the reason to drop the leg rather
than to pick whichever haircut produces the preferred answer.

## Honest limits

* The weights are reasoned, not fitted. Fitting them needs graded weeks.
* Ticket probabilities are independent products. Same-game legs are positively
  correlated (Diggs o29.5 and Fields o14.5 cleared together in 6 of 6 shared
  games against a 48.8% independent product), so a ticket probability is a
  **floor** -- and a book's SGP quote claws back part of that, so a naive
  product is never a takeable price.
* A LOTTERY ticket at 0.11% is a 1-in-900 shot. A positive EV on it is a
  statement about the price, not a prediction, and it should be staked that way.

---

## ADDENDUM 2026-11-13 00:05Z -- own history must be per-opportunity, and two claims are withdrawn

The operator doubted that Jayden Daniels had rushed for 103 yards or that Malik
Nabers had a 160-yard receiving game. Both are real (nflverse rows, box-score
arithmetic closing exactly, PFR snap counts, and outside game reports agree):

* Week 5, 2026-10-11, NYG @ WAS, 27-27 tie: Daniels 17/25 199, **9 carries 103**;
  Nabers **11-121 on 20 targets**; Jameis Winston 33/**54** for 326.
* Week 6, 2026-10-18, NO @ NYG: Nabers **12-165 on 15 targets**; NYG **47**
  pass attempts (Winston 26 + McCarthy 21).

But the doubt was pointing at a real defect in the estimator above. Those
outcomes came in opportunity contexts that tonight does not project:

* Nabers' 121 and 165 came on **54 and 47** team pass attempts. The model
  projects **29.8** for McCarthy tonight. Scaled per attempt, those are 67- and
  105-yard games.
* Daniels' 103 came on 9 carries in his first game back from an elbow injury.
  His three games since: **4, 2, 4 carries** for 10, 23, 16 yards.

This is the AGENTS.md anomaly gate -- "anomalies are questions, not trends" --
and the version above broke it: own history was treated as eight exchangeable
games with a single 0.901 total-ratio scalar. That scalar is replaced. Own
history is now **per team pass attempt, scaled to tonight's projected attempts,
times the matchup factor** -- opportunity x efficiency, which is the repo's own
allocation principle.

### Withdrawn

| claim (made ~23:48Z) | corrected |
|---|---|
| Nabers o109.5 +430 is "+27.2% EV, best single on the board" | **+7.3%, and the sign flips with the weights (-19.5% to +34.1%)**. A fair-priced high-risk leg, not an edge. |
| Nabers o99.5 +310 "+10.7% EV" | **-4.7%** (-23.7% to +14.4%). Fair-priced. |
| Nabers o159.5 +2300 "+33.8% EV" lottery leg | **-56.2%**. It happened once, on 47 attempts; 0 of 8 at tonight's volume. Removed. |
| Daniels rush o89.5 +1480 "+21.8% EV" lottery leg | **Unresolved context, fail closed.** The 9-carry usage has not existed for three games. Removed. |
| SAFE ticket "+26.7% EV", LOTTERY "+91.2% EV" | SAFE **57.5% vs 52.7% breakeven**. Edges on this board are single digits. |

### What survived every version of the estimator

Diggs rec o24.5 (-270): positive under the floor test, the cushion gate
(1.81x), all weight schemes, both own-history adjustments. Fields rec o9.5
(-420) likewise. Fields o14.5 (-260) is positive but thin (+0.9% to +4.8%): he
is 8 for 8 in reality, across three different quarterbacks, but 6 of 8 once the
two 47+ attempt games are scaled to tonight's volume.

Final legs: `data/research/tnf_2026_wk10_final_legs.csv`.

### Also corrected in MODEL_CEILING_COMPRESSION_2026_W10_V1.md

The Daniels rushing row (-21.0 mean gap, -47.4 p90 gap) is **not** ceiling
compression. The model's 21.0 matches his last three games (10, 23, 16); it was
tracking a usage change that the season mean hides. Nabers' mean gap shrinks
from -25.1 to about -14 once his games are put on a per-attempt basis. The
compression finding stands for Nabers and Likely at reduced size; it was
overstated by comparing a current projection against opportunity-unadjusted
season aggregates.

---

## ADDENDUM 2026-11-13 00:15Z -- condition receiver history on tonight's quarterback

The operator's objection: J.J. McCarthy is the Giants' quarterback, the depth
chart, practice report and model all know it, so why was receiver history still
being read from Winston games?

The system did have it right. Depth chart: McCarthy QB1, Winston QB2, Dart QB3.
Practice report: McCarthy full participation, no designation. Model:
`qb_pass_att_share = 1.0`, 29.8 projected attempts, 213.9 yards. The defect was
in the own-history component of this estimator, which split by GAME when
play-by-play can split by PASSER. Three of McCarthy's appearances were shared or
relief games, so game-level history mixed two quarterbacks.

Every McCarthy attempt this season (65 attempts, weeks 6, 7, 9), against
Winston's 167:

| receiver | share of McCarthy targets | share of Winston targets | exp. targets tonight | exp. yards | book line | model |
|---|---|---|---|---|---|---|
| Malik Nabers | 21/65 = **32.3%** | 29.3% | 9.6 | ~79 | 68.5 | 55.9 |
| Malachi Fields | 13/65 = **20.0%** | 10.2% | 6.0 | ~49 | 27.5 | 33.1 |
| Isaiah Likely | 12/65 = **18.5%** | 29.3% | 5.5 | ~35 | 46.5 | 43.7 |
| Darnell Mooney | 6/65 = **9.2%** | 10.8% | 2.8 | ~18 | 13.5 | 20.1 |

Expected yards = McCarthy target share x model-projected attempts x the
player's season yards per target. Shares carry a binomial standard error of
about five points on 65 attempts; these are estimates, not measurements.

Consequences:

* **Likely** was Winston's co-primary target (29.3%) and is McCarthy's third
  (18.5%, exactly 4 targets in each of McCarthy's three appearances). ~35
  expected yards against a 46.5 line: no over, and the under is not far enough
  through the line to bet on 12 targets of evidence.
* **Fields** doubled his share under McCarthy. That is the football reason his
  floor leg holds, and it is a better reason than "8 for 8".
* **Nabers** draws a third of McCarthy's throws. The model's 55.9 is low against
  current-quarterback usage; the earlier per-attempt walk-back used Winston-era
  volume and over-corrected.
* **Mooney** has 6 targets on 65 McCarthy attempts. Fair-priced at best.

The same split for Washington: Stefon Diggs with Jayden Daniels at quarterback
is 55, 47, 44, 38, 51 -- lowest 38. His two lowest games of the season (33, 35)
came with Mariota and Kaliakmanis starting. With tonight's quarterback he has
not been under 38.

**Rule going forward:** own history enters this estimator per target on the
CURRENT quarterback's attempts, from play-by-play, not per game. This is the
AGENTS.md current-role/state principle ("coaching/QB context") and the anomaly
gate's quarterback-change trigger, applied where the data supports it.
