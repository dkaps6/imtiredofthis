# PRISM — D1: INJURY STATUS NEVER REACHES THE MODEL, PREREGISTRATION V1

**Frozen:** 2026-11-03, **before any arm is computed.**
**Status:** PREREGISTRATION. No result here. Nothing promoted.
**Supersedes the framing of** `PRISM_RULES_SIM_BAYES_CODE_AUDIT_V1` **D1**, which was wrong in the
optimistic direction — see §0.

---

## 0. The audit's D1 was wrong, and the truth is worse

The audit reported *"a player ruled OUT keeps 50% of his opportunity."* **That overstated how much
the model does.** It keeps **100%**, and can be boosted **above** it.

**Root cause, traced to the artifact.** Production `injuries.csv` (archived for W7 and W8) ships
with:

```
['player', 'team', 'season', 'week', 'source']
```

`build_injuries_weekly.SCHEMA` declares **eleven** columns —
`player, team, season, week, status, practice_status, body_part, designation, report_date,
source, report_available`. **The shipped artifact has five. Every status field is missing.**

The chain, each link verified in the live code:

1. `injuries.csv` has no `status`;
2. `context_bridge.py:193` — `inj_keep = [c for c in [...,"status",...] if c in inj.columns]` —
   silently drops it;
3. the merge then joins on `team` + `player_key_bridge` and contributes **nothing**;
4. `context_bridge.py:225` — `_text(row, ["status"])` — needs an exact column match, finds none,
   returns `""`;
5. `simulation_rules._injury_limited` upper-cases `""`, matches no token, and **returns False for
   every player in every week.**

**The entire injury path in the rules layer is dead code.** The column names are *correct* — my
first hypothesis, a `status`/`injury_status` mismatch, was wrong. The artifact is the fault.

This is precisely the failure AGENTS.md §4 exists to prevent: *"Never treat file existence alone
as proof that an artifact is valid. Validate schema, rows, season/week and provenance."* The file
exists, carries 274 and 320 rows, the right season and week, and a named source — and is missing
the only fields it is consumed for.

**Measured consequence (W7+W8, nflverse designations joined to the production universe):**

| | |
|---|---|
| players nflverse ruled **Out/Doubtful** who appear in the universe | **29** |
| mean target share the model gave them | **0.1392** |
| mean targets they actually recorded | **0.00** |
| share of them recording **zero** targets | **100%** |
| recovered multipliers on them | {1.0, **1.2**, **1.265**, **1.32**} |

Tyler Warren (IND TE, **Out**) was *boosted* from 0.1957 to **0.2476**. DeVonta Smith (**Out**)
held **0.2516**; Rashee Rice (**Out**) held **0.2315** — both alpha-sized shares, belonging to
players who did not dress.

## 1. Research question

The defect is **established fact, not hypothesis** — there is nothing to test about whether
status arrives; it does not. What needs preregistering is **the fix's value and its calibration**:

**What is the correct opportunity retention by injury designation, and what does applying it do to
projections — with and without redistributing the freed mass?**

## 2. Part 1 — calibration (the number the fix needs)

Using `nflreadpy.load_injuries(2026)` W1–8 (2,459 rows; **Out 513, Doubtful 37,
Questionable 517**) joined to realized weekly stats, measure **empirical opportunity retention**
by designation: realized target/rush share divided by the player's **prior-weeks** baseline.

Reported per designation with n. `Questionable` is included because
`_injury_limited` **does not match it at all**, so production's implied retention for a
Questionable player is **1.00** — and that needs checking too, not assuming.

**This is a pregame designation predicting a realized outcome, so there is no lookahead** — but
the baseline must come from **prior weeks only**, and that is G5.

## 3. Part 2 — arms, applied to the production universes

| arm | Out / IR / PUP | Doubtful | Questionable | freed mass |
|---|---|---|---|---|
| **C0** production | **×1.00** (and boosted) | ×1.00 | ×1.00 | — |
| **A1** the *intended* rule | ×0.50 | ×0.50 | ×1.00 | discarded |
| **A2** calibrated, no redistribution | ×0.00 | Part-1 value | Part-1 value | discarded |
| **A3** calibrated + redistribution | ×0.00 | Part-1 value | Part-1 value | **redistributed proportionally to the team's remaining players** |

**A1 is included deliberately** so the result separates *"the artifact is broken"* from *"the rule
was also wrong."* If A1 beats C0 but A2 beats A1, **both** defects are real and both need fixing.

**A2 vs A3 is the question that matters for the alpha.** Zeroing an inactive alpha frees
0.14–0.25 of team share; whether the remaining players should absorb it decides whether this fix
helps the one unsolved tier or merely deflates the team.

**Scope:** proportional redistribution only. The production 60/30/10 bucket rule is defects
**D4/D5** and is **out of scope** — it must not be bundled in.

## 4. Gates

- **G1 (mechanical)** A3 must preserve team mass exactly: team sum after zeroing and
  redistributing equals the team sum before, to **1e-12**. **Failure voids the run.**
- **G2 (mechanical)** every arm must leave players with **no** designation bit-identical to C0, to
  **1e-12**. **Failure voids the run** — the fix must touch only designated players.
- **G3 (projection)** A2 and A3 each beat C0 on per-player bias against realized share, 95% CI
  excluding 0, bootstrapped **over players**.
- **G4 (no tier harmed)** no pregame tier degrades by more than 0.002, **top tier reported first.**
- **G5 (no lookahead)** designations are the pregame report and baselines use **prior weeks only**.
  **Failure voids the run.**
- **G6 (mass direction)** team sums must move **toward** the realized 0.9536 and must **not
  undershoot it on average.**

## 5. Kill conditions

**K1** G1, G2 or G5 breached → **run void** · **K2** G3 fails → removing inactive players does not
improve projections, which would be extraordinary and must be published · **K3** G4 breached —
name the tier · **K4** G6 undershoots → zeroing without redistribution over-corrects, and **A3
becomes the only admissible arm** · **K5** A1 ≈ A2 → the intended ×0.50 was adequate and the
**artifact is the whole defect** · **K6** Part-1 retention for `Out` is materially above 0 → the
premise is wrong and the entire framing must be revisited.

## 6. Prediction

**Part 1:** `Out` retention ≈ **0.00–0.02**; `Doubtful` low but non-zero; `Questionable`
**high, 0.85–0.95** — most Questionable players play, so production's implicit ×1.00 for them is
roughly right and only `Out` is badly wrong. **A2 and A3 beat C0 decisively** (0.1392 × 29 of
phantom mass is large and deterministic). **A1 beats C0 but loses to A2** — both defects real.
**A3 > A2**, because the freed share belongs to the teammates who actually played.

**Named risk:** **n = 29** designated players in two weeks. The per-player bias CI may be wide
even though the per-row effect is large and near-deterministic, and **a team with two inactive
starters could undershoot** under A2 (K4). A second risk: nflverse designations are the *final*
report, and production consumes an **earlier** snapshot — so Part 1's retention is an upper bound
on what a live pipeline could act on, and **this study cannot measure the timing loss.**

My record: **four hits, six misses**, with the named risk landing nearly every time. This
prediction carries no evidentiary weight.

## 7. Scope and constraints

`tgt_share` and `rush_share` only. D2, D3, D4, D5 explicitly out of scope — **D2 must be OFF**
when D1 is measured, per the audit's interaction warning that D1, D2, D3 and the
`ROLE_SHRINKAGE` arms all compete to remove the same finite +0.5110 of excess mass. No sportsbook
variable enters any arm; market lines are not used at all in this study. No production code is
modified.

**Separately, and not part of this study:** the artifact defect itself should be fixed upstream in
`build_injuries_weekly`, and `scripts/audit_2026_production_readiness.py` should gain a schema
assertion so an injuries file missing `status` **fails closed** rather than shipping. That is a
production fix, not a research result, and needs no preregistration — but it must not be
conflated with the arms above.
