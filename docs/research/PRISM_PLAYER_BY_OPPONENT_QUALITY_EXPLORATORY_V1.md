# PRISM — Player × opponent-QUALITY interaction: EXPLORATORY (V1)

> **STATUS: EXPLORATORY. NOT VALIDATED. NOT PROMOTABLE.**
> These specifications were run after inspecting 2025 holdout results, across several
> forms. 2025 is therefore a **hypothesis-generating** sample, not a confirming one.
> Nothing here may be cited as a validated effect or merged into production. The
> confirmation design is in section 5 and must run on seasons never inspected.

## 1. Why this exists

`PRISM_PLAYER_SPECIFIC_MATCHUP_RESULTS_V1` returned a null for H1 and I drew too wide
a conclusion from it — that the opponent layer is a weak lever generally. Two user
challenges corrected that:

1. the null covered **receiving only**; Kenneth Walker is a rushing case;
2. whatever is tested for backs should be tested for receivers too.

Both were right. The null tested one interaction form — player **profile** (aDOT) ×
defense **scheme** (man/zone rate). It never tested player **quality** × defense
**quality**, which is the form that moves.

## 2. Which opponent properties are forecastable at all

Disjoint halves, weeks 1–9 vs 10–18, 2025, 32 teams. Disjoint because overlapping
as-of windows previously manufactured persistence
(`PRISM_STRANDED_SIGNAL_DISPERSION_RESULTS_V1`).

| opponent property | disjoint-half r | forecastable? |
|---|---|---|
| def stuff rate allowed | **+0.411** | yes |
| def rush YPC allowed | **+0.381** | yes |
| def pass EPA allowed | **+0.368** | yes |
| def pass YPA allowed | **+0.270** | weakly |
| **team rush rate (game script)** | **not forecastable** | **no** |

This is the first useful split. Defensive **quality** is a stable team trait.
Play-calling **script** is not.

## 3. The game-script channel is dead at its source

Predicting team-week rush rate out of sample (fit weeks 4–11, score 12–18, 448
team-weeks), football-only inputs, no sportsbook:

| predictor | holdout MAE |
|---|---|
| team's own prior rush rate | **0.0814** |
| + football-only strength gap | 0.0817 *(worse)* |
| **production's constant 0.43** | **0.0831** |

Production's constant sits within **2%** of the best available model, and the
opponent-strength term *degrades* out-of-sample prediction despite a directionally
sensible coefficient (`+0.0338`).

Realized rush rate spans 0.162–0.759 (sd 0.102). That spread is **in-game
randomness** — score state evolving during play — not pre-game forecastable signal.
I initially read the spread as evidence of a large unexpressed lever. That reading
was wrong.

**Consequence:** the volume/game-script channel is dead for backs *and* receivers,
because no reliable script input exists for a player term to interact with. The
constant `rules_pass_rate = 0.57` is close to optimal, not a defect. That retracts
part of how I characterized the transmission findings.

## 4. The player × opponent-quality interaction (exploratory)

Strictly-prior player evidence, strictly-prior opponent evidence, outcome week-*W*.
Fit weeks 4–11, score 12–18. Baseline is the player's own `prior_eff × prior_volume`.

**Backs — rushing yards.** 529 games, 68 backs.

| model | holdout MAE |
|---|---|
| own baseline only | 25.797 |
| + opp run defense, **positional** | 25.791 |
| + **`prior_ypc × def_ypc_allowed`** | **25.680** |

**Receivers — receiving yards.** 1,471 games, 182 WR/TE.

| model | holdout MAE |
|---|---|
| own baseline only | 24.596 |
| + opp pass defense, **positional** | 24.612 |
| + **`prior_ypt × def_ypa_allowed`** | **24.377** |

Consistent across both groups:

- the **positional** matchup term adds nothing (RB `+0.006`) or hurts (WR/TE `−0.016`);
- the **player × opponent-quality** term helps (RB `−0.117`, WR/TE `−0.219`).

**Mechanism.** How much a strong defense costs depends on how good the player is: a
high-efficiency player has more to give up than a low-efficiency one. A multiplier
keyed to position is identical for every player sharing the label and therefore cannot
represent this term at all. That is a structural reason the positional form fails while
the interaction form does not, rather than a fitted coincidence.

**Honest magnitude.** 0.45% (RB) and 0.89% (WR/TE). Small. Directionally consistent,
structurally motivated, and unvalidated.

## 5. Required confirmation before any claim or mechanism

The confirmation must be preregistered and scored on data never inspected:

1. **Confirmation seasons 2023 and 2024**, held back entirely. 2025 is spent.
2. Frozen candidate: `prior_eff × def_eff_allowed` only, for RB rushing yards and
   WR/TE receiving yards. One interaction term per market. No feature search.
3. Must beat **both** the own-baseline model and the positional model on
   **player-level** error, not position-aggregate MAE.
4. **Disjoint-window persistence** of the interaction coefficient within each
   confirmation season — sign agreement required.
5. Concentration test: the gain must sit with players whose own efficiency is far from
   their position mean, since that is where the mechanism predicts it.
6. Anomaly gate per AGENTS.md: `DNP_INACTIVE` / `PARTIAL_GAME` / `UNRESOLVED_CONTEXT`
   state-labelled before any observation becomes evidence.
7. Sportsbook separation: no market variable at any stage.

If the interaction fails to replicate on 2023–2024, it was 2025 noise and will be
reported as such.

## 6. Scope

No production code changed. No promoted model, weight, or pricing path touched. No
sportsbook variable read. Betting remains CLOSED. The `PERCENTY_COLS` scale bug and
SLOT unreachability remain separate open items, neither resolved here.
