# RB carry-share recency, end to end: RESULTS V2 (the corrected test)

**Preregistration:** `RB_CARRY_SHARE_RECENCY_END_TO_END_PREREGISTRATION_V2.md`, committed at
`d00cf3f` before any arm ran.

**Code:**
- `scripts/research/evaluate_rb_recency_end_to_end_v2.py`;
- the shared module `scripts/modeling/rb_carry_recency_v2.py`, which production PlayerForm
  also calls.

**Rows:** `data/research/rb_recency_end_to_end_v2_rows.csv`. **Summary:**
`data/research/rb_recency_end_to_end_v2.csv`.

## Verdict: PASS. `RB_CARRY_SHARE_RECENCY_V2` replaces V1 in production.

## Primary (carries MAE, C0 minus arm; 95% CI clustered by team-game)

| arm | season | n | MAE C0 | MAE arm | gain | 95% CI | lead-back gain | lead 95% CI |
|---|---|---|---|---|---|---|---|---|
| **C1g (production)** | 2025 | 1,347 | 3.732 | 3.686 | **+0.046** | [+0.022, +0.070] | +0.059 | [+0.014, +0.104] |
| **C1g (production)** | 2026 wk 4-10 | 600 | 3.905 | 3.865 | **+0.040** | [+0.020, +0.060] | +0.066 | [+0.018, +0.113] |
| C1u (ungated, reference) | 2025 | 1,347 | 3.732 | 3.658 | +0.074 | [+0.042, +0.108] | +0.088 | [+0.024, +0.152] |
| C1u (ungated, reference) | 2026 wk 4-10 | 600 | 3.905 | 3.873 | +0.033 | [+0.007, +0.060] | +0.048 | [-0.010, +0.107] |

- **K1 (CI includes 0):** not triggered for C1g in either holdout.
- **K2 (lead backs worse):** not triggered. Lead backs improve in both, with CIs above 0.

**The gate's cost and benefit:**
- **2025:** the gate gives up some of the raw gain (+0.046 vs +0.074).
- **2026:** the gated version is better (+0.040 vs +0.033), and its lead-back CI clears 0 where
  the ungated one does not.

The ungated arm is reference only. It breaks the AGENTS.md anomaly gate, so it is never a
production option.

## How often the gate holds recency back (scored RB rows)

| season | applied | fewer than 3 games | withheld, anomaly in window |
|---|---|---|---|
| 2025 | 840 | 277 | 230 (17%) |
| 2026 | 335 | 193 | 72 (12%) |

## What changed from V1

1. **RB/FB only.** V1's test changed every position's share. QB, WR and TE shares compete in
   the same top 5 and 0.95 cap, so V1 measured a different intervention.
2. **Snap-game window.** Production's V1 window skipped games an RB played without a touch.
   V2 counts every offensive-snap game, carries filled as 0.
3. **Anomaly gate.** Each window game is labelled from nflverse snap counts and the official
   injury report. Recency needs all three games `HEALTHY_FULL`. The thresholds were fixed in
   the preregistration and were not tuned.
4. **C0 is production's own code.** V1's C0 counted season-to-date with the current team only.
   V2 builds it with `player_form_v2._normalize_weekly` and `_season_totals`, the same as
   production.
5. **Team codes** are canonicalised in the module, so LA becomes LAR as in PlayerForm. This
   plumbing fix landed after the preregistration commit and before any result was read.

## Production check on real data (2026 week 12, local)

`player_form_v2.rb_recency_windows` against live nflverse sources:
- 72 RBs applied;
- 33 withheld for an anomaly in the window;
- 41 with fewer than 3 games.

Withheld examples:

| player | why |
|---|---|
| Saquon Barkley | week 8 `INJURY_LIMITED` |
| David Montgomery | week 5 `PARTIAL_GAME` |
| Breece Hall | week 9 `RETURN_FROM_INJURY` |
| Emmett Johnson | weeks 10-11 `UNRESOLVED_CONTEXT` |

Johnson's surge is a snap spike that no injury or roster record explains, so the gate keeps his
season-to-date share. That is the rule working as written. His role is the D1 allocative-prior
problem, not this one.
