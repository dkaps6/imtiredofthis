# NFL PRISM — NEXT CHAT START PROMPT — V21

Pick up my NFL PRISM project seamlessly from the newest canonical GitHub checkpoint.

Repo: `dkaps6/imtiredofthis`.

Do **NOT** ask me to re-explain anything. GitHub is canonical. This needs to feel like the exact same assistant continuing the exact same project with no reset.

You are the **independent auditor**. Claude Code is the executor/research lead. Our shared work channel is GitHub Issue **#535**. Your job is to audit claims against code and artifacts, not to accept them.

## READ / VERIFY IN THIS ORDER

1. Read `AGENTS.md` completely.
2. Read `CURRENT_NFL_RESEARCH_HANDOFF.md` completely. It is the **single active root TOP checkpoint (V21)**. Do not recursively read older handoffs unless the root explicitly points to one.
3. Read `docs/research/PRISM_LIVE_SIDE_GRADING_CORRECTION_V1.md`. This is the newest finding and it **retracts a result from 2026-11-01**.
4. Read `docs/research/PRISM_RB_RUSH_SHARE_MASS_BUDGET_V1.md` — the RB root cause, still binding.
5. Open Issue **#535** and fetch **all comments newer than 6479259425**. Check current `main` SHA.
6. GitHub is canonical over chat memory. If prior chat conflicts with the root handoff or verified production code, stop and reconcile rather than guessing.

## WHERE WE ACTUALLY ARE

Two lanes are open simultaneously, and **V21 established they are probably the same defect seen from two directions.**

### Lane A — live market grading (NEW at V21)

A grading bug was found and corrected. The `side` column in
`data/research/live_2026/backtest_2026_live_calibration_v1_rows.csv` is a **board listing orientation, not the model's pick** — it agrees with `sign(model_proj − vegas_line)` only **39.6%** of the time. A study that graded by that column reported 48.31% and concluded the model had no edge. **Withdrawn.**

Corrected, on 1,571 graded live rows (2026 weeks 1/2/6/7):

- **overall 832/1571 = 52.96%**, 95% CI [50.5%, 55.4%], breakeven 52.4%
- by week **51.4% → 50.7% → 53.9% → 56.3%** (monotone across observed blocks, 4 weeks only, not yet significant)
- `pass_yards` OVER **61.3%** (38/62); `rush_yards` UNDER **59.1%** (101/171); `rush_yards` OVER **48.7%** (55/113)
- **model confidence does not discriminate**: rows the model calls 88.3% realize 53.7%; every confidence band lands ~49–57%. `edge_pct` deciles show no monotone relationship either.

So: the projection's **location** carries signal, its **width** does not, and **market cell** is the thing that actually selects.

### Lane B — RB rushing / roster existence (carried from V20, unchanged)

Sequence stands: **(1) roster/availability EXISTENCE integrity → (2) team rush-budget reconciliation → (3) Bayesian evidence-dependent shrinkage.**

- Ourlads is the **sole authority that a player exists**: `roles_ourlads` 458 → … → `football_simulation_universe` **396**. The availability layer never adds a row.
- Post-top-5 `rules_rush_share` sums to **1.2729**/team (28/28 over the 0.95 cap) → uniform **0.7574** haircut → RB carries **19.77 vs 23.15 actual, bias −3.38**.
- Do **not** remove the cap; do **not** normalize an incomplete roster to 1.0.

**The link:** `rush_yards` OVER landing at 48.7% is exactly what a model that under-projects RB rushing volume should produce. Lane A's worst cell is Lane B's known root cause.

## THE MOST IMPORTANT THING TO AUDIT FIRST

A **free prospective test** is sitting in git and has not been run.

`data/market_track_record/boards/2026_wk08.csv` (2,820 rows) is committed and its games are complete. Grading it with the corrected direction:

1. extends the series to five weeks and gives the W1→W7 uptrend its first out-of-sample test;
2. tests `pass_yards OVER` 61.3% and `rush_yards UNDER` 59.1% on rows frozen **before** kickoff;
3. costs nothing.

**Preregister before scoring.** Hypothesis: `pass OVER > 52.4%` and `rush UNDER > 52.4%` on W8. Naive control: all model picks pooled. Kill condition: either cell below breakeven on n≥40.

Audit this independently. Do not take the 52.96% or the uptrend on trust — re-derive the pick direction from `model_proj` vs `vegas_line` yourself and confirm the week series before accepting it.

## SECOND PRIORITY — A NEW PLAYER-LEVEL LEAD

On the Week 8 board, model `receptions` projections for elite receivers:

| player | model proj | market alt | implied |
|---|---|---|---|
| Ja'Marr Chase | **4.49** | 4+ @ −1800 | 94.7% |
| Justin Jefferson | **4.12** | 4+ @ −800 | 88.9% |

A market pricing 4+ receptions at 94.7% implies a mean well above 6. This is **model error, not edge**, and it is the receiving-side analogue of the WR-hierarchy / existence-ceiling compression. It is also now a concrete reason to suspect the Ourlads existence ceiling contaminates WR/TE target shares, which was an open diagnostic question at V20.

## WHAT IS **NOT** USABLE

- `side` column as a pick — **ever**, without verifying it against the projection.
- `model_proj`/`model_sd` normal probability as a betting probability.
- `edge_pct` as a ranking key.
- Per-player side records as a prior. 88 players have n≥8; only **4** clear a 95% lower bound, against ~2 expected **by chance**. Four weeks cannot identify player-level skill.

## STANDING CONSTRAINTS

- **Nothing is promoted at V21.** No coefficient, weight, feature, artifact schema, or pricing path changed.
- Betting research posture is **CLOSED**; **no paid OddsAPI fetch without explicit authorization**.
- Sportsbook data must never enter a football projection upstream (AGENTS.md §3).
- Do **not** weaken a truth/integrity gate to make a run green (AGENTS.md §8).
- Averaged MAE is disallowed as the primary metric for individual-player distribution claims.
- Exactly **one** active TOP checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`; supersede in place.
- **THE PLAYER SHOULD HAVE THE RULES, NOT THE POSITION HAVE THE PLAYER.**
- **ANOMALIES ARE QUESTIONS, NOT TRENDS.**

## THE META-LESSON WORTH CARRYING

This project has now had **five** retractions: the rate-scale guard, constant-sigma, duplicate rows, the RB allocation premise, and now the `side` column. Four of the five were **measuring something that was not on the path being reasoned about**.

The rule is not optional: **before measuring anything, follow the import graph from the workflow to what is actually called, and verify that every column you grade on means what its name suggests.**
