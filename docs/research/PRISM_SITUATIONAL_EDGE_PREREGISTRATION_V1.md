# PRISM — SITUATIONAL EDGE: VACANCY UNDERREACTION AND THE ALPHA-TIER SIDE, PREREGISTRATION V1

**Frozen:** 2026-11-04, **before any statistic is computed.**
**Status:** PREREGISTRATION. No result. **No bet is authorized by this study.**

---

## 0. A correction to my own claim, and what is genuinely closed

`PRISM_PASS_YARDS_EDGE_DURABILITY_RESULTS_V1` §5 said *"there is no currently exploitable edge."*
**That was overstated.** What was measured is narrower: **the model's own side-pick on MAIN lines,
bet FLAT, across the whole board.** Four structures were never tested — **alt lines, situational
spots, correlated parlays, live.**

**Genuinely closed, not re-tested here:**

| | why | source |
|---|---|---|
| model confidence as a key | flat 49–57% realized; claims 88%, delivers 53.7% | V21 §3 |
| `edge_pct` as a key | non-monotone deciles, top decile CI [47.3, 62.6] | V21 §3 |
| **per-player skill** | **4 of 88 clear a 95% bound where ~2 clear by chance** | V21 §5 |
| flat main-line side-picking | fails all four durability gates under clustering | today |

**Genuinely open and partially positive:** the **empirical alt-line curve**, trained W1/2/6 and
tested W7, transfers directionally with a **6–10pp haircut** — ~50%-of-projection thresholds clear
**69–79%**, while the market's popular 75–80% tier clears only **51–60%** (V21 §4). **Alt prices
are not on the committed boards, so that lane needs a paid fetch and is out of scope here.**

**This study tests the one open structure that needs no new data: situation-based selection.**

## 1. The two hypotheses

**H1 — VACANCY UNDERREACTION.** When a team's primary receiver is **ruled OUT**, his
**0.14–0.25 of team target share** redistributes. If the market underreacts to that, his
teammates beat their lines. **The model is structurally blind to this** — D1 showed OUT players
retain full share — so any such edge is currently **unexploited rather than already harvested.**

**H2 — THE ALPHA-TIER SIDE.** The model under-rates each team's alpha by **16%**
(`IRREDUCIBLE_FLOOR_AND_ALPHA_HAIRCUT_V1`), so its side-pick on those players is **biased by a
known mechanism rather than by a read**. Earlier measurement gave contradictory answers —
**46.80%** under outcome-contaminated tiering, **56.62%** under pregame tiering — and **neither
used EV or clustering.** H2 resolves that contradiction properly.

## 2. Method — the corrections from today are mandatory

- **EV per $1 at the board's posted odds.** Never win rate (the `anytime_td` trap).
- **Cluster bootstrap over PLAYERS.** Never over rows (the `pass_yards` artifact: 1.89× too narrow).
- **All tiers and filters computed from PRIOR WEEKS ONLY.** (the board-check lookahead error).
- **Market data is permitted** — selection is explicitly market-assisted under AGENTS.md §3;
  what is forbidden is market data inside the *football projection*.

## 3. Tests

**H1:** for each board row, flag whether a **teammate at the same position group** carried an
`OUT`/`DOUBTFUL` designation that week (nflverse, pregame report).
- **V1** EV of rows **with** a vacancy versus **without**, clustered over players.
- **V2** EV restricted to the model's **OVER** side in vacancy rows — the directional form of the
  hypothesis.
- **V3** the naive control: EV of **all** vacancy rows regardless of side. **If V3 ≈ V1, the
  market is simply mispricing vacancies and the model adds nothing** — still usable, but it is a
  market fact, not model skill, and must be reported that way.

**H2:** pregame volume quartile × model side, EV, clustered.
- **A1** top-quartile + model says UNDER.
- **A2** top-quartile + model says OVER.
- **A3** the same two cells for the bottom quartile, as a shape check.

## 4. Gates

- **G1** the cell's EV **cluster** 95% CI lower bound is **above 0**.
- **G2** EV positive in **≥4 of 5 weeks**.
- **G3** dropping the single best player leaves the CI lower bound **above 0**.
- **G4** the cell holds **n ≥ 200** rows and **≥ 25 distinct players**.

**All four, for any cell to be called an edge.** These are the gates `pass_yards` failed; applying
anything weaker here would be choosing a bar to fit a hope.

## 5. Kill conditions

**K1** no cell passes all four → **situational selection is not established on this data either,
and the honest statement becomes "no edge has been identified in any structure we can currently
measure"** · **K2** a cell passes G1 only → small-sample, report as such · **K3** V3 ≈ V1 → the
effect is market mispricing of vacancies, not model skill (§3) · **K4** any cell is defined after
inspecting outcomes → **void**; the four cells above are fixed now · **K5** fewer than 3 vacancy
weeks have usable rows → underpowered.

## 6. Prediction

**H1 passes G4 and fails G1.** Injury news is the most public information in the sport and
books move props hard on it; the plausible inefficiency is in *degree*, which is small relative to
prop variance. **But it is the best remaining shot** precisely because the model is blind to it.

**H2's contradiction resolves toward the pregame figure (56.62% win), and still fails on EV once
clustered** — 56.62% at a −114 median is roughly +6% raw, but clustering over players will widen
the interval through zero, exactly as it did for `pass_yards`.

**Named risk — the one that matters:** I have found eleven negatives today and the user has
explicitly asked for an angle. **The temptation is to accept a cell that passes two gates and call
it promising.** The gates are set at four for that reason. **A cell that passes G1 and G4 but
fails G2 and G3 is `pass_yards` again, and must be reported as a negative.**

My record: **10 prediction hits, ~19 misses**, named risk landing nearly every time.

## 7. Scope

2026 W1/W2/W6/W7/W8 committed boards, all markets with posted odds. **No odds fetched.** **No bet
authorized regardless of outcome** — a pass licenses one prospective W9+ test. No production code
modified. Alt lines, parlays and live betting remain **untested and out of scope**, and must not
be described as closed.
