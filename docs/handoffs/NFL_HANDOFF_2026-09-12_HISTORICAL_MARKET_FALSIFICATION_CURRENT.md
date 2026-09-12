# NFL HANDOFF — 2026-09-12 — HISTORICAL MARKET FALSIFICATION CURRENT

## Status

This is the active research checkpoint on branch `research-historical-market-falsification-v1`.

Branch base / canonical `main` at creation:
- `22057ba9571d7976fdc4f7e6c5ddc9d6d6f1bd7e`
- latest merged PR at that SHA: #533, situational edge hunt v1

GitHub is canonical; chat memory is secondary.

## Priority override

A newly surfaced historical Vegas benchmark is now the highest-priority issue.

The project objective is not merely to produce mechanically valid football projections. The player-prop stack must demonstrate a reproducible, prospectively usable betting edge after sportsbook pricing is attached downstream.

Until this falsification audit is closed:

- do not present the player-prop stack as historically profitable overall;
- do not treat large current-week `edge_pct` values as validated wagering edges merely because the production pipeline runs;
- do not change upstream football science in reaction to sportsbook results until the failure is decomposed;
- do not continue season-continuity plumbing, QB/WR public-intent research, game ML/spread/total work, or ATD work ahead of this audit unless the user explicitly changes priority;
- do not use sportsbook data as an upstream football feature.

## Canonical benchmark evidence already on `main`

The repository already contains a real full-stack Vegas benchmark at:
- `data/backtests/full_stack_vegas_benchmark_v1/README.md`
- `data/backtests/full_stack_vegas_benchmark_v1/non_qb_detail.csv`
- `data/backtests/full_stack_vegas_benchmark_v1/non_qb_summary.csv`
- QB benchmark summaries in the same directory
- grader: `scripts/backtest/grade_full_stack_vegas_benchmark_v1.py`

The committed disposition is:
- `FULL_STACK_EDGE_THRESHOLD_HISTORICAL_BENCHMARK_QB_SYNTHESIS_GATES_CONFIRMED_NO_FOOTBALL_ONLY_EDGE`

### QB pass-yards benchmark

Using real 2024-2025 DraftKings/FanDuel lines with the production-style PLAY/LEAN threshold logic:

- base ensemble PLAY win rate: 50.3%
- base ensemble PLAY ROI/unit: -5.3%
- football-only M89/M90 synthesis reconstruction PLAY win rate: 52.5%
- football-only synthesis PLAY ROI/unit: -1.4%
- football-only synthesis improves football accuracy and materially narrows the loss, but does not reach breakeven
- market-assisted QB candidate reaches +2.6% ROI, but it uses Vegas market context as input and therefore is NOT evidence of a football-only edge and cannot be promoted upstream under this project's architecture

The M89/M90 reconstruction still passed its football-science improvement gates versus its own base candidate. The market benchmark therefore does not by itself prove M89/M90 is bad football science; it proves that the currently reconstructed football-only QB betting stack has not demonstrated positive ROI against the tested market.

### Non-QB benchmark

Committed PLAY-tier results across 2024-2025:

- `rush_yards`: 51.5% win rate, -2.5% ROI/unit
- `rec_yards`: 51.6% win rate, -2.9% ROI/unit
- `rush_rec_yards`: 50.7% win rate, -4.9% ROI/unit
- `receptions`: 54.1% win rate, -0.6% ROI/unit

The tighter edge filter improves ROI monotonically in all four markets, but none crosses breakeven in the committed full-market summary.

## Critical fidelity limitations that must be resolved before final verdicts

The existing benchmark is strong enough to trigger this audit, but it is not a perfect literal replay of 2026 production.

1. Historical fair probability uses a Normal approximation around the ensemble projection with `component_sd` from MC/ML/State disagreement. It does NOT use the literal historical Monte Carlo outcome distribution that production uses for `fair_prob`.
2. Historical free-market lines are the latest captured prices, not a frozen fixed pre-kickoff snapshot such as 30 minutes before kickoff.
3. `WR_R15_PRODUCTION_MODEL_V1` and `TE_R5P_PRODUCTION_MODEL_V1` are not yet applied to the historical `rec_yards` / `receptions` benchmark even though their scientific validation spans multiple seasons. Therefore the existing WR/TE market result is not yet a final grade of the actual promoted WR/TE stack.
4. RB P3/R26 were excluded from the 2024-2025 benchmark because their current promoted production routes are 2026 Week-1 scoped. Therefore the committed historical non-QB benchmark is not a clean historical grade of the exact current 2026 RB stack.
5. The user reports Claude separately tested the code against historical Vegas odds and found no positive season overall. Treat that as an external falsification signal until its exact methodology/artifacts are reconciled against the committed benchmark.

## Situational edge-hunt result already on `main`

PR #533 re-sliced 16,973 already-graded 2024-2025 non-QB bets across 146 situational slices.

One candidate survived a both-seasons-positive screen:
- market: receptions
- model highest-confidence quartile (`prob_edge` approximately >= 0.44)
- side: UNDER only
- n = 1,258
- win rate = 52.5%
- ROI/unit = +2.66%
- 2024 ROI = +2.43%
- 2025 ROI = +2.90%

This is a research lead, NOT a validated production edge, because:
- 146 slices were tested (multiple-comparisons exposure);
- the receptions benchmark does not yet apply WR-R15/TE-R5P;
- a third independent season or a prospectively frozen confirmation set is preferred if data permits.

Do not promote this cohort yet.

## Frozen falsification protocol

Before changing any model formula, freeze and execute the following decomposition.

### A. Reproduce the committed benchmark exactly

- reproduce the current full-stack 2024-2025 benchmark from committed inputs;
- report year-by-year and combined results;
- report by market, position, side, edge tier and book/source;
- verify unit-return arithmetic, push treatment, vig handling, line selection and player/game identity joins;
- preserve all negative results.

### B. Close benchmark-to-production fidelity gaps

1. Apply historical WR-R15 and TE-R5P adapters using only historically valid pregame inputs and their frozen scientific contracts.
2. Where technically reproducible, use the actual historical Monte Carlo distribution/fair-probability path rather than the component-SD Normal approximation. If exact reconstruction is impossible, quantify sensitivity to plausible distribution widths and do not hide the limitation.
3. Freeze a defensible line/price timestamp policy. Do not select favorable books or timestamps after seeing outcomes.
4. Preserve sportsbook information strictly downstream of football projections.
5. Do not silently retrofit 2026 Week-1-only RB refinements into 2024-2025 data.

### C. Separate football-model failure from betting-translation failure

For every market and season, compare:
- model MAE / bias / calibration versus actual outcomes;
- Vegas line MAE / bias versus actual outcomes;
- directional accuracy relative to the line;
- probability calibration and Brier/log-loss where available;
- ROI after price/juice;
- results by OVER/UNDER, edge size, role and uncertainty.

Interpretation rule:
- if football projections systematically lose to the Vegas line on error/ordering, upstream model science may need reopening;
- if football projections are competitive or better on football error but ROI is negative, prioritize distribution calibration, fair-probability conversion, line/price timing and downstream bet-selection before touching football science.

### D. Prospectively freeze any attempted repair

Any candidate repair must be written before evaluating its final test set and must state:
- exact hypothesis;
- training/calibration years;
- untouched confirmation years or folds;
- exact metrics and promotion gates;
- market/position scope;
- no sportsbook-upstream rule;
- fallback behavior if it fails.

No post-hoc threshold hunting may be called a production edge.

## Relationship to season-long production continuity

The season-long Weeks 1-18 continuity problem remains real and preserved, including the P3 W2-18 `enriched_att` bridge and R26/R22 future-week routing issue.

However, continuity is temporarily SECONDARY to proving what portions of the betting stack, if any, have a real historical edge. Building an all-season pipeline that simply repeats a negative-expectation betting decision is not sufficient.

Once this falsification audit tells us which layers are sound versus broken, resume season-long production work using only the authorities that remain defensible.

## Exact next step

Reproduce and audit the committed full-stack Vegas benchmark at current `main`, with year-by-year decomposition and benchmark-fidelity checks, before changing production science or resuming season-continuity implementation.
