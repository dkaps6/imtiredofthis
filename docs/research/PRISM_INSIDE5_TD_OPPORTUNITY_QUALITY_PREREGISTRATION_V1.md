# PRISM — Inside-5 Opportunity Quality for Anytime TD — PRE-REGISTRATION V1

**Status:** `PRE-REGISTERED — NOT SCORED — NOT PROMOTED`
**Date:** 2026-10-27
**Lane:** Issue #535 (Claude executor / GPT-5.6 auditor)

This document freezes a candidate mechanism **before** it is scored, as the
anti-retest rules in `6420198257` and AGENTS.md require. It contains no
scoring, no fitted coefficient, and no promotion request.

---

## 1. The gap, stated precisely

Production's anytime-TD mechanism (`scripts/simulation_v2.py:157`):

```
rz        = rz_share                                    # share of team RED ZONE work
rz_mult   = clip(0.75 + rz, 0.75, 1.35)
script    = 1 + 0.08 * (team_wp - 0.5)
lam       = offensive_td_rate * rz_mult * script
p(TD)     = 1 - exp(-lam * team_scoring_shock)
```

`rz_share` is built from `yardline_100 <= 20` (`enrich_player_scoring_v2.py:170,182`).

**Red zone is one bucket twenty yards deep.** A carry from the 19 and a carry
from the 1 contribute identically to `rz_share`, so two backs with the same
red-zone share but different goal-line roles receive the same TD multiplier.

`inside_5` / `goal_to_go` appear **nowhere** in production. A repository-wide
search finds them only in `scripts/research/build_player_identity_card_v1.py`.

### What is NOT the problem

I initially suspected the `clip(..., 1.35)` ceiling was compressing the signal.
**Measured: it is not.** Of 362 Week-7 players with an `rz_share`, only **7
(1.9%)** reach the ceiling and 54 sit on the floor; 301 are in the interior
where the signal varies continuously. The clip is not the issue and this
proposal does not touch it.

---

## 2. Is the finer signal actually distinguishable? Mostly no — and that matters

Measured on the Week-7 identity card, 394 carry cards with both shares:

| | |
|---|---|
| correlation(red-zone share, inside-5 share) | **0.890** |
| mean difference | 0.000 |
| sd of difference | 0.098 |

**Inside-5 share is largely redundant with red-zone share.** Any claim that
production is broadly blind to goal-line role would be false, and this
pre-registration does not make it.

The case for the mechanism is narrower: the measures **diverge materially for
a minority of players**, and that minority is identifiable in advance.

Divergence shrinks under a support threshold, as small-sample noise should:

| min inside-5 carries | players | diverge ≥ 0.20 | corr |
|---|---|---|---|
| 0 | 394 | 24 | 0.890 |
| 3 | 71 | 15 | 0.849 |
| 5 | 45 | 10 | 0.875 |
| 8 | 39 | 6 | 0.894 |
| **10** | **31** | **5** | 0.890 |
| 15 | 17 | 3 | 0.873 |

At ≥10 inside-5 carries, 5 of 31 well-supported players (16%) diverge by ≥0.20:

| player | team | red-zone share | inside-5 share | inside-5 carries | direction |
|---|---|---|---|---|---|
| Sean Tucker | TB | 0.100 | **0.500** | 13 | underrated |
| Chase Brown | CIN | 0.440 | **0.800** | 17 | underrated |
| Christian McCaffrey | SF | 0.577 | **0.875** | 32 | underrated |
| Kenneth Walker III | KC | 0.794 | **1.000** | 20 | underrated |
| Woody Marks | HOU | 0.400 | **0.200** | 11 | overrated |

De'Von Achane (0.583 → 1.000) and Kyler Murray (0.364 → 1.000) appear at
support 0 and **drop out** by support 10, on 6 and 3 inside-5 carries
respectively. That is the threshold working, not a result being hidden.

---

## 3. Frozen candidate mechanism

Replace the red-zone term with a **support-gated** goal-line term. Nothing
else in the TD path changes.

```
i5_n     = player's strictly-prior inside-5 opportunities (target team, gated weeks)
i5_share = player's strictly-prior share of team inside-5 work
rz_share = existing red-zone share

if i5_n >= MIN_I5_SUPPORT:
    quality = i5_share
    source  = "INSIDE_5"
else:
    quality = rz_share                       # unchanged from production
    source  = "RED_ZONE_FALLBACK"

rz_mult = clip(0.75 + quality, 0.75, 1.35)   # functional form UNCHANGED
```

**`MIN_I5_SUPPORT = 10`**, frozen here. It is chosen from the support sweep in
§2 as the point where divergence stabilises, **not** tuned against any outcome.
It is not to be re-tuned after scoring; if 10 is wrong, that is a finding to
report, not a dial to turn.

Design constraints, all deliberate:

- the multiplier's functional form and clip bounds are **unchanged**, so any
  measured difference is attributable to the quality input alone;
- players below support are **untouched** — the mechanism is a no-op for ~92%
  of the population (363 of 394 Week-7 carry cards);
- `i5_share` comes from the **gated** card (contaminated weeks excluded) and is
  strictly prior, so no look-ahead;
- `source` is emitted per row so every projection states which input it used.

---

## 4. Anti-retest declaration — this is the binding part

**Week 7 generated this hypothesis and therefore cannot test it.** The
divergence table in §2 was produced by inspecting Week-7 cards. Scoring the
mechanism on Week 7 would be grading on the data that suggested it, which is
the exact failure that killed earlier lineages.

Permissible tests, declared in advance:

1. **Prospective** — Week 8 onward, frozen before kickoff, graded after.
2. **Historical weeks not inspected for this purpose** — permissible only with
   an explicit statement of which weeks were examined when, and only if the
   examination did not include outcome data.

Forbidden, explicitly:

- re-tuning `MIN_I5_SUPPORT` after seeing scores;
- reporting only the subset of players where the mechanism helped;
- adding a carve-out for a named player after the fact;
- declaring success on aggregate TD calibration while per-player direction is
  unexamined.

### Pre-declared success criteria

The mechanism is a **candidate for promotion only if all** of:

1. it changes the projection for ≥3 players in the test window (otherwise it is
   untestable, not successful);
2. among players whose `source == INSIDE_5` **and** whose share diverged ≥0.20
   from `rz_share`, the TD-probability direction moves the correct way more
   often than not;
3. pooled TD calibration over the full population does **not** materially
   degrade — this is a guardrail, not the objective, per the
   PLAYER PROJECTION FIRST PRINCIPLE;
4. no player below support has a changed projection (a correctness check: the
   mechanism must be a no-op there).

### What would falsify it

- divergent-player direction is no better than chance → the finer signal is
  noise at this sample size;
- pooled calibration degrades materially → the quality input is mis-scaled;
- the no-op check fails → implementation bug, not a finding.

---

## 5. What this does not claim

- It does **not** claim production ignores goal-line role. `rz_share` is
  populated for 362 of 396 Week-7 players and correlates 0.890 with inside-5.
- It does **not** claim a large effect. It is a no-op for ~92% of players.
- It does **not** address steps 2 or 4 of the reasoning chain (current
  teammate/QB state; opponent and environment transmission). Those remain open,
  and the opponent/environment gap — `def_rush_epa` unconsumed, `pass_share`
  hardcoded at 0.57 in `rules_v2.py:98` — is a separate and larger problem.
- It is **not** a betting recommendation. Betting remains CLOSED.

## 6. Dependencies

Requires the Identity Card's `inside_5_opp` and `team_inside_5_share_current`
at projection time, which means the card must be built in the Full Slate path.
It is not today. Wiring it is a separate production change and is not
requested here.

---

**Nothing in this document has been scored. No promotion is requested.**
Audit of the mechanism's design is requested before any test is run.
