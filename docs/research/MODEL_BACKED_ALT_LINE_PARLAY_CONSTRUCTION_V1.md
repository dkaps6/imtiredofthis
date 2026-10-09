# Model-Backed Alt-Line Parlay Construction V1

Status: **DOCUMENTED WORKING PROCESS — NOT A VALIDATED DOWNSTREAM BET SELECTOR**  
Date: 2026-10-08

## Purpose

Preserve the manual selection style that produced several of the cleaner recent
tickets without confusing successful bets with statistical proof.

The goal is not to sort the board by raw edge.

The goal is to identify a small set of props where:
1. the football projection has a coherent reason to disagree with the market;
2. the underlying player/team role is trustworthy enough to support that mean;
3. current injury/starter context does not invalidate the frozen model state;
4. known distorted/uncertified lanes are excluded;
5. the entry line still gives enough room for the football thesis to survive;
6. parlay construction does not let one player or one fragile assumption kill
   every ticket.

This document preserves the process. It does **not** claim that the process is a
historically validated winner selector. The existing market-relative and
market-residual selector studies remain terminal NULL.

---

## 1. Start from the football model, not from the sportsbook board

Candidate discovery begins with the canonical football-only projection artifact.

Do not:
- sort by raw EV and take the largest numbers;
- use sportsbook movement to construct the football projection;
- promote a bet merely because the model-market gap is large;
- rescue a known bad family with a post-hoc threshold.

The initial question is:

> "Where does the football model have a real, explainable opinion?"

That requires inspecting the projection lineage and football state behind the
number, not only the final edge.

---

## 2. Require a coherent football reason

A candidate is stronger when the model disagreement is supported by the
appropriate football layer for that player/market.

Examples:
- QB passing: promoted QB synthesis / team pass opportunity / starter authority.
- RB: role, carry share, receiving share, vacancy state and other validated
  opportunity evidence; do not blindly trust the known systematic RB
  rush/rush+receiving under disagreement.
- WR/TE receiving: target-share / entitlement / specialist lineage and role
  evidence.
- Availability: the projection must still be compatible with the current
  starter/inactive state.
- Matchup: use only matchup information that has validated or source-qualified
  transmission into the model. Do not invent generic defense-vs-position
  boosts after seeing results.

The selected leg should have a football explanation that can be stated before
kickoff without referring to the eventual outcome.

---

## 3. Prefer clean disagreement over gigantic disagreement

The recent successful primetime selections were not chosen because they were
the largest raw edges on the slate.

A cleaner candidate can be preferred over a larger raw edge when:
- the model lineage is stronger;
- the player role is more stable;
- the market is one where production science is more trustworthy;
- the current injury/starter state is clean;
- the model-market gap is still meaningful after current-line reconciliation.

This is why known systematic RB under clusters were often excluded even when
their raw EV was enormous.

---

## 4. Reconcile the current line and current availability before entry

The player name is not the bet. The **line is the bet**.

Before using a candidate:
1. reconcile official/current starter and availability state;
2. compare the current sportsbook number with the frozen model mean;
3. determine whether the model thesis still exists at the actual entry line.

If the line moves materially toward the model, the leg must be downgraded or
removed rather than taken automatically because it appeared on an earlier list.

---

## 5. Why alt lines are used

Alt lines are used for **ticket construction**, not because lowering a threshold
creates new football information.

For an OVER:
- if the model mean is comfortably above the main line;
- and the football thesis is coherent;
- an alt threshold below the main line may preserve most of the model's thesis
  while materially increasing the chance that the parlay leg survives ordinary
  game variance.

For an UNDER, the analogous construction is an alt threshold above the main
line when available and sensibly priced.

The principle is:

> Trade some payout for more distance between the entry threshold and the model's
> expected outcome when the purpose is to build a multi-leg ticket.

This is particularly attractive when:
- the full main-line edge is real but not enormous;
- the player has a stable role;
- the alt still leaves a meaningful cushion versus the model mean;
- the leg is being used as a parlay building block rather than a standalone
  maximum-EV wager.

This process does **not** currently contain a scientifically validated rule for
the optimal number of yards to buy. Do not convert "25+" or "250+" into a fixed
universal threshold. The alt must be evaluated against that player's projection,
role and current book price.

---

## 6. Recent examples of the style

### Week 4 MNF — ATL/NO

The preferred core construction was:

- Tyler Shough **250+ passing**
  - model: 266.17
  - full/main area: ~254.5
  - rationale: promoted QB passing authority plus a modest buy-down for parlay
    survival.

- Juwan Johnson **40+ receiving**
  - model: 44.48
  - full/main area: ~40.5
  - rationale: model and receiving-role evidence aligned; alt threshold stayed
    close to the main market while giving a clean integer target.

- Kyle Pitts **25+ receiving**
  - model: 30.40
  - full/main area: ~28.5
  - rationale: the model edge at the main line was modest, so the alt line was
    preferred instead of forcing the full over.

The important feature was not "three overs." It was one promoted QB lane plus
two role-backed TE receiving lanes with thresholds chosen below the football
means.

### Week 5 TNF — DAL/TB

The same construction philosophy produced:

- Jalon Daniels **175+ passing**
  - model: 220.71
  - current public main area at selection: ~180.5
  - QB synthesis active / QB C2 selected.

- Cade Otton **25+ receiving**
  - model: 44.24
  - current main area: ~29.5
  - TE entitlement target share 0.1772, specialist delta +0.0433, prior-week
    offensive participation 95%, TE matchup target multiplier 1.265.

- Jake Ferguson **20+ receiving**
  - model: 29.53
  - current main area: ~25.5-26.5
  - TE entitlement target share 0.1482, specialist delta +0.0185, prior-week
    offensive participation 71%, TE matchup target multiplier 1.265.

Why the alt lines:
- Daniels already had a large model cushion; 175+ retained the football thesis
  while lowering the failure threshold.
- Otton's model/role evidence was strong enough that 25+ kept substantial room
  below the projection.
- Ferguson had a smaller raw model gap than Otton, so 20+ was preferable to
  forcing the full 25.5/26.5 main line in a parlay.

This is the intended use of alt lines: **preserve the football thesis while
reducing fragility**.

---

## 7. Ticket tiers

When multiple tickets are built, use the same qualified candidate pool but vary
the amount of accumulated variance.

### Core / "lock-style" ticket
- 2-3 legs
- strongest football lineage
- cleanest current availability
- meaningful projection cushion
- alt lines favored when they materially improve survival

"Lock" is informal language only. It never means guaranteed.

### Core + juice
- 4-5 legs
- same quality requirement
- more independent football hypotheses
- do not simply duplicate every core leg

### Aggressive
- 5-7+ legs
- may include more contrarian model opinions
- every leg still needs a football/model reason

### High-ceiling / lottery
- probability becomes intentionally small through leg count
- individual leg quality standard does **not** fall
- do not add unsupported ATDs, arbitrary longshots or known-distorted props just
  to raise payout.

---

## 8. Diversification rule

Avoid using one player as the universal anchor across every ticket.

Preferred construction:
- distribute players across tickets;
- diversify games and football hypotheses where possible;
- avoid having one injury-sensitive or role-sensitive player kill the entire
  portfolio;
- correlated same-game legs are allowed when the football story is coherent,
  but correlation must be intentional rather than accidental.

For multi-ticket slates, a useful working constraint is:
- no player on more than two tickets unless there is an explicit reason.

This is a portfolio construction rule, not a claim of statistical independence.

---

## 9. Explicit exclusions

Do not use this method to justify:
- largest-edge = best-bet logic;
- top-N raw edge rules;
- automatic UNDER selection;
- RB rushing/rush+receiving under clusters solely because the model is low;
- uncertified ATD probabilities as core bets;
- stale projections after material injury/starter changes;
- sportsbook-derived inputs feeding upstream football projections;
- postgame rationalization.

---

## 10. What should be recorded prospectively

For every selected leg, preserve:
- player / market / side;
- sportsbook main line at review;
- chosen entry or alt line;
- football model mean;
- relevant specialist/role lineage;
- current availability/starter reconciliation;
- reason for using main vs alt;
- reason the leg survived known model caveats;
- ticket assignment;
- timestamp / source artifact authority.

That allows later postmortem work to grade not just whether the bet won, but
whether the **selection reasoning itself** is replicable.

---

## 11. Scientific boundary

This process has produced some very good individual tickets, including recent
primetime examples, but success on a handful of tickets is not enough to label
it a validated downstream selector.

The correct next scientific step, if desired, is to prospectively freeze this
selection contract across future slates and score:
- main-line candidate performance;
- alt-line survival rate;
- ticket-level hit rate;
- calibration of model cushion versus realized outcome;
- whether lineage/role-quality tags add predictive information beyond raw edge.

Until that prospective sample exists:

**use this as a disciplined model-backed construction method, not as proof that
we have solved bet selection.**
