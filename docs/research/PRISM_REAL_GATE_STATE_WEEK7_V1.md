# PRISM — Real Production Gate State, 2026 Week 7

**Status:** direct observation of committed production artifacts. **No proxy, no
inference, no fitting, no scoring, no alternative proposed.**
**Supersedes, on the coverage and box gates:** the 2025 participation proxy in
`PRISM_GATE_PREVALENCE_RESULTS_V1.md` — and **reverses it on the light-box gate.**
**Source:** `data/research/week7_opportunity_trace/`, recovered by the widened
preservation filter (`e7264023`) re-running against artifact `run_39849371821`.

Outcomes inspected: **none**. Sportsbook inputs: **0**. These are pregame
context and rule-layer artifacts only.

## The values were recoverable after all

I previously recorded the Week-7 opponent values as **gone and unrecoverable**.
That was wrong. The source artifact does not expire until 2027-01-20, so once
the preservation filter was widened, a re-run recovered them. What was lost was
never the data — only my filtered copy of it.

Now committed with values: `cb_coverage_team.csv` (the canonical coverage
artifact) with `man_rate`, `zone_rate`, `coverage_available`, `coverage_source`;
`team_form.csv` with `light_box_rate`, `heavy_box_rate`, defensive EPA and
`proe`; the raw `sharp_def_tend_2026.csv`; and — decisively — the **realized**
multipliers in `model_rule_diagnostics.csv` and `model_rule_simulation_inputs.csv`.

## Sharp's coverage denominator — RESOLVED

Marked UNESTABLISHED through three audits. The test: sum `man_rate + zone_rate`
per team.

| | value |
|---|---|
| teams | 28 |
| min | 0.8970 |
| **median** | **0.9365** |
| max | 0.9680 |

**The denominator is coverage-classified plays, not all plays.** An all-plays
denominator would sum to roughly 0.50; these cluster just under 1.00. So the
`COVERAGE_CLASSIFIED` arm of the proxy study was the correct comparison, and
`coverage_available = 1` with `coverage_source = sharp_team_form` for **all 28**
teams — Coverage-v2 produced finite inputs for every team on the slate.

## Real gate state, and it reverses the proxy on light box

Team-level, from the real Sharp-fed values:

| gate | real Week-7 | 2025 proxy said | verdict |
|---|---|---|---|
| `coverage_zone_rate >= 0.60` | **27 / 28 = 96.4%** | DISCRIMINATING 71–84% | fires almost always |
| `coverage_man_rate >= 0.50` | **0 / 28 = 0.0%** (max 0.4170) | NEAR_DEAD | **confirmed dead** |
| `light_box_rate >= 0.60` | **1 / 32 = 3.1%** (med 0.4525) | NEAR_CONSTANT 90–99% | **REVERSED — near-dead** |
| `heavy_box_rate >= 0.60` | **0 / 32 = 0.0%** (max 0.3420) | NEAR_DEAD 0% | confirmed dead |

**The light-box reversal matters.** My proxy computed light box as
`defenders_in_box <= 6` from participation and got a league median of ~0.70.
Sharp's `light_box_rate` has a median of **0.4525** — a different definition on a
different scale. The proxy's `NEAR_CONSTANT` label was an artifact of my own
construction, not a property of production. On real values the gate is
**near-dead**, which is the opposite conclusion.

## Realized multipliers — direct, no decoding

396 player rows, 2026 Week 7. These are production's own emitted values:

| multiplier | observed values |
|---|---|
| `wr1_target_mult` | **1.0 × 396** |
| `wr1_5_target_mult` | **1.0 × 396** |
| `slot_target_mult` | 1.1 × 396 |
| `te_target_mult` | 1.265 × 383, 1.1 × 13 |
| `rb_rec_target_mult` | 1.2 × 368, 1.32 × 15, 1.0 × 13 |
| `rush_eff_mult` | **1.0 × 381**, 1.07 × 15 |
| `pass_eff_mult` | 1.0 × 350, 1.01 × 31, 0.98 × 15 |
| `volatility_mult` | 1.0 × 381, 1.05 × 15 |

Three things follow directly, with no inference:

**1. Perimeter wide receivers received no opponent coverage adjustment at all.**
`wr1_target_mult` and `wr1_5_target_mult` were exactly **1.0 in every one of 396
rows** — because they key only on `coverage_man_rate`, which no team reached.
Meanwhile every tight end got 1.1–1.265 and nearly every back got 1.2–1.32.

A refinement to the earlier role-routing write-up: **slot** receivers did get a
uniform ×1.1, via `middle_open_rate`. So the shut-out role is specifically the
**perimeter WR**, not every receiver.

**2. The RB rushing channel is inert.** `rush_eff_mult` = 1.0 in **381 of 396**
rows (96.2%). The only opponent→RB-rushing-efficiency path in production did
nothing for almost every back — now confirmed on real values rather than
decoded.

**3. Near-universal multipliers are level shifts, not matchup signals.**
`slot_target_mult` is 1.1 for **all** 396 rows and `te_target_mult` is 1.265 for
383 of 396. A multiplier applied to essentially everyone does not discriminate
between matchups; it rescales a whole position group.

## The game-script constant, confirmed on real data

`rules_pass_rate` is **exactly 0.57 in all 1,412 rows.** The
`PRISM_GAME_SCRIPT_TRANSMISSION_TRACE_V1` finding was derived from source
arithmetic; this is the same result observed in production output.

## Scope and limits

- **One week, one slate.** 28 teams had coverage rows (14 games); box rates
  cover 32. A single week cannot establish season-long prevalence, and the
  2025 proxy's cross-basis caveats do not transfer to it.
- **Player rows are not independent.** A team's multiplier applies to all its
  players, so the 396-row counts reflect ~28 team states. Team-level counts are
  reported separately above for that reason.
- **No predictive claim.** Nothing here shows that changing any gate, threshold
  or constant would improve a projection. No alternative is proposed.
- **Sharp's box definition is still not documented** — only that it differs
  materially in scale from `defenders_in_box <= 6`. Why is unestablished.

## What this changes in the ledger

- Sharp's coverage denominator: **UNESTABLISHED → coverage-classified.**
- `light_box_rate` gate: **NEAR_CONSTANT (proxy) → near-dead (real).**
- The WR-versus-TE/RB coverage asymmetry: **proxy-only → observed in
  production**, and stronger than the proxy suggested — 96.4% versus 0.0%.
- "Week-7 opponent values are unrecoverable": **withdrawn.**
