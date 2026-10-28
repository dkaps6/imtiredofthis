# PRISM — Three opponent-side nulls, settled (V1)

**Script:** `scripts/research/measure_opponent_signal_nulls_v1.py`
**Record:** `data/research/opponent_nulls/opponent_signal_nulls_v1.json`
**Tests:** `tests/test_measure_opponent_signal_nulls_v1.py`
**Seasons:** 2023–2025.

Banked so no future session re-runs them. Each settles a question that was raised
directly and is reasonable to ask.

---

## Null 1 — "tough matchups are worse for some players than others" is NOT measurable

Per-player matchup sensitivity = the slope of a receiver's yards on his opponent's
strictly-prior pass-defense EPA allowed, for players with ≥12 qualifying games.

| | value |
|---|---|
| players | 254 |
| observed spread of per-player sensitivity | sd **51.26** |
| permutation null spread | sd **50.58** (95% range 43.64–57.00) |
| observed / null | **1.014** |
| permutation p | **0.395** |

The permutation reassigns each player's opponents **among his own games**, preserving
his outcome distribution and the opponent pool while destroying any genuine
player–opponent link. The observed spread is what random reassignment produces.

Cross-season carry: 2023→2024 **+0.126** (n=44), 2024→2025 **−0.044** (n=50).

So "some players are matchup-proof and others are fragile" is not supported. The wide
spread of per-player slopes (earlier observed as −42.9 to +34.5) is estimation noise
from fitting a slope on 10–17 games.

**How this squares with the quality × quality result that did help.** The exploratory
`prior_ypt × def_ypa_allowed` term carries **one global coefficient**, so it says a
player's swing is proportional to his **own level** — a high-efficiency receiver has
more absolute yards to lose. That is a scale effect, not differential responsiveness,
and it is not contradicted by this null.

## Null 2 — defenses do NOT have a persistent exploitable shape

The scouting intuition: "this defense is soft deep, attack it there." Tested as a
defense's deep-minus-short yards-allowed profile (`air_yards` ≥ 10 vs < 10), separating
**shape** from overall **level**.

| window | r(LEVEL) | r(SHAPE) |
|---|---|---|
| 2023 disjoint halves | +0.371 | **+0.041** |
| 2024 disjoint halves | −0.061 | **−0.295** |
| 2025 disjoint halves | +0.222 | **+0.034** |
| 2023 → 2024 | −0.156 | +0.045 |
| 2024 → 2025 | +0.191 | +0.257 |

Mean r(shape) = **+0.017**, and one window is materially **negative**. A defense soft
deep in the first half of a season is not soft deep in the second half, so there is no
stable weakness to match a player's depth profile against.

## Null 3 — game script is not forecastable, and neither are plays

Already established for pass/rush **rate**: predicting team-week rush rate out of
sample, own prior 0.0814, adding a football-only strength gap 0.0817 (**worse**),
production's constant 0.43 **0.0831** — within 2% of the best available model.

This adds total **plays**, the other half of volume:

| predictor | holdout MAE |
|---|---|
| league constant | **6.74 plays** |
| team's own prior | 6.85 plays *(worse)* |

Realized plays sd 8.55; disjoint-half persistence of team plays/game r = **+0.130**.

**Why.** Game script is not a property of the teams — it is a consequence of plays that
have not happened yet. Play-calling responds to score state, and score state is
generated during the game. Pre-game strength predicts the winner only weakly and the
within-game trajectory far more weakly, so there is little to forecast from. Team pass
volume as a whole is therefore close to unforecastable pre-game.

---

## What is NOT nulled: defensive quality

Defense is not irrelevant and these nulls must not be cited as if it were. Measured in
`PRISM_PLAYER_BY_OPPONENT_QUALITY_EXPLORATORY_V1`, disjoint halves:

| property | r |
|---|---|
| def stuff rate allowed | **+0.411** |
| def rush YPC allowed | **+0.381** |
| def pass EPA allowed | **+0.368** |
| def pass YPA allowed | +0.270 |

"This is a good or bad defense" is genuinely knowable and carries a real if small
effect, applied in proportion to each player's own level. What fails is everything
finer-grained than that: per-player sensitivity, exploitable shape, and script.

## Known limitation, stated not buried

Every test here treats a defense as a stable **team-season** object. It is not —
personnel changes weekly with injury and in-season scheme change. That is a plausible
mechanism for the shape null specifically, and it leaves **defender-level availability**
as a genuinely untested axis rather than a refuted one. The same caveat applies to
alignment (LWR/RWR/SWR), which `run_player_form_v2.py` collapses before any artifact
sees it.

## Why this matters for what comes next

Ranked by measured strength, across everything tested in this line of work:

| signal | strength |
|---|---|
| role centrality → floor as × of own mean | **r = +0.620** |
| player's own mean, persistence | **r = +0.59 … +0.74** |
| defensive quality | r = +0.27 … +0.41 |
| per-player volatility | r = +0.15 … +0.57 (weak, 5/5 positive) |
| player-specific matchup sensitivity | **noise** (p = 0.395) |
| defense exploitable shape | **noise** (mean r = +0.017) |
| game script: rate and plays | **noise** |

The two strongest signals are both properties of **who the player is**, and the
centrality one predicts his **distribution shape** — precisely what a single
`sigma = 26.0` for every receiver discards. The opponent side is largely irreducible;
the player side has large unexploited signal. That is the basis for making the
dispersion lane the next build rather than any further opponent work.

## Scope

No production code changed. No promoted model, weight or pricing path touched. No
sportsbook variable read. Betting remains CLOSED.
