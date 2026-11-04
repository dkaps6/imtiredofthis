# PRISM — AVAILABILITY SIGNAL V1 — **K6 FIRED: QUALIFIED MECHANISM IS NOT REPRODUCIBLE**

**Date:** 2026-11-03
**Preregistration:** `PRISM_AVAILABILITY_SIGNAL_PREREGISTRATION_V1.md`, frozen at `d7866e7c`.
**Status:** STOPPED AT THE §2 REPRODUCIBILITY CHECK. **No arm computed. No gate read.**

---

## What the preregistration required

> §2 — *"The participation probability must be computed **strictly from weeks 1..W−1**. If the
> qualified mechanism cannot be reproduced leakage-safely for W7/W8, **report that and stop** —
> do not substitute an approximation and call it the qualified signal."*
> **K6** — kill if the qualified participation probability cannot be reproduced leakage-safely.

## What was found

`PLAYER_ACTIVE_ROLE_PARTICIPATION_V1_MECHANISM_RESULT.md` carries the repository's **highest
research status — QUALIFIED, all positions including WR and TE.** Searching for the means to
re-run it:

| | result |
|---|---|
| committed script producing the mechanism | **none** (`grep -rln "ACTIVE_ROLE_PARTICIPATION_MECHANISM" --include=*.py` → empty) |
| workflow producing it | **none** in `.github/workflows/` |
| committed data artifact | **none** under `data/` |
| what the document cites instead | **GitHub Actions artifacts `12049120604` and `12044155086`** |

The result document records the scoring run's integrity (`target-game snap used as predictor: 0`,
`sportsbook inputs used: 0`, **`2026 outcomes read: 0`**) — so the mechanism was scored on
historical seasons only. **Even if those artifacts have not yet expired, they contain no 2026
W7/W8 probabilities**, and without a script there is no way to generate them.

**H1 and H2 are therefore not computable as specified. K6 fires.**

---

## This is an AGENTS.md §5a failure, and a severe one

> *"Anything that must survive the season belongs in git. A result that only exists as an
> artifact is a result with an expiry date on it."*
> *"cite a run id for provenance, never as the place the evidence lives."*

A **qualified** mechanism — the strongest status this repository awards, carried forward in
handoffs as reusable science — **exists only as a run id.** It cannot be re-run, re-scored,
extended to a new season, or wired into production, because the code that produced it was never
committed.

This is worse than the Weeks 3–5 board loss already recorded in §5a. That cost gradable rows.
This costs a **mechanism**: the conclusion survives in prose while the thing that justified it
does not.

**The qualification should be treated as unverifiable until the mechanism is reconstructed and
re-scored.** It should not be cited as available science, and no downstream work should assume
it can be reused.

---

## What was NOT done

No arm was computed. No M1, M2, M3, breadth, tier table or gate figure from this design exists.
Nothing here qualifies or disqualifies H1 or H2 on their merits — they were never scored.

E1 and its §7 qualification are untouched and remain the current state of that lane.

---

## Next

Two options, and the choice is the user's:

1. **Reconstruct the mechanism from its result document** and re-score it from scratch. The
   document describes the design well enough to attempt this, but a reconstruction is a **new
   mechanism requiring its own preregistration and qualification** — it does not inherit the
   old status.
2. **Proceed without it**, using a leakage-safe proxy that is honestly labelled as a proxy — in
   which case the study's comparator is E1's crude `A_j` versus a slightly better proxy, which
   is a much weaker question than the one preregistered.

Recommended alongside either: an audit of **every other "qualified" result in `docs/research/`
for the same defect** — a cited run id with no committed script. If this one is not unique, the
repository's qualification ledger is partly unverifiable, and that is worth knowing before more
work is built on it.
