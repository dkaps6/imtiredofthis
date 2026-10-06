# Distribution Right-Tail Asymmetry V1 — Frozen Diagnostic Plan

Status: **FROZEN BEFORE HISTORICAL SCORING — RESEARCH ONLY — NO PRODUCTION CHANGE**

Branch:
`research-distribution-right-tail-asymmetry-v1`

## Motivation

The canonical 2026 Weeks 1-4 production postmortem found that catastrophic
projection misses are highly asymmetric.

Among 1,667 decided production bets:
- 270 realized outcomes were more than 2 model SD from the final projection;
- 93.3% of those extreme outcomes were **above** the model;
- receiving yards, receptions, rush+receiving yards and rushing yards all showed
  far more >+2SD realized misses than <-2SD misses.

This does **not** authorize a 2026-derived correction. It motivates one
historical football-distribution diagnostic.

Prior generic Distribution Widening V1 tested symmetric mean-neutral widening
and improved calibration but did not solve betting profitability. This V1 does
not retest symmetric width. It asks whether historical production-aligned
simulations specifically underrepresent the **right/upper tail**.

## Data authority

Rebuild the existing leakage-safe 2024/2025 historical football distributions
with the already-tested reconstruction path:

- `scripts/backtest/build_historical_inputs.py`
- `scripts/backtest/walk_forward.py`
- `scripts/research/persist_historical_simulated_outcomes_v1.py`
- `scripts/backtest/build_full_stack_vegas_projection_trace_v2.py`

No sportsbook props, lines, odds, market probabilities or betting outcomes are
used in this diagnostic.

Historical targets:
- 2024 Weeks 1-18, prior season 2023
- 2025 Weeks 1-18, prior season 2024

Simulation contract:
- 2,000 draws per player-market row
- existing deterministic seed policy
- raw MC array must reproduce stored `mc_proj` within 1e-8
- array is then rescaled to the frozen final ensemble mean with the same
  production-style mean alignment already used by empirical fair-probability
  research

No football mean is changed by the audit.

## Markets

Frozen scope:
1. `pass_yards`
2. `rush_yards`
3. `rec_yards`
4. `receptions`
5. `rush_rec_yards`

No market may be added after results.

## Primary tail definition

For every row, from the mean-aligned simulated outcome array compute:

- q10 = empirical 10th percentile
- q90 = empirical 90th percentile

Then define:
- `upper_break = 1(actual > q90)`
- `lower_break = 1(actual < q10)`

Primary asymmetry statistic:

`upper_break_rate - lower_break_rate`

A calibrated symmetric tail would not require these rates to be exactly 10%
in finite samples, but persistent positive imbalance means realized error is
escaping the model more often on the upper side than the lower side.

## Secondary descriptive tail

Also report q05/q95 break rates and their upper-minus-lower difference.

The q05/q95 result is descriptive only and cannot rescue a failed primary
q10/q90 test.

## Cluster inference

Rows within the same NFL game are dependent.

For every season/market:
- cluster = authoritative `game_id`
- 10,000 bootstrap replicates
- seed = 20261006
- resample games with replacement
- statistic = upper-break rate minus lower-break rate

## Support floor

A season/market cell is scoreable only when:
- >= 200 player-market rows;
- >= 50 distinct games;
- all rows have finite actual and final football mean;
- every reconstructed distribution has exactly 2,000 finite draws.

Otherwise:
`INSUFFICIENT_SUPPORT`.

## Frozen per-market disposition

A market is:

`RIGHT_TAIL_ASYMMETRY_REPLICATED`

only if in **both 2024 and 2025**:
1. support passes;
2. q90 upper-break rate > q10 lower-break rate;
3. 95% game-cluster bootstrap CI for the difference has lower bound > 0.

Otherwise:

`RIGHT_TAIL_ASYMMETRY_NOT_REPLICATED`.

Overall:
- >=1 replicated market:
  `DISTRIBUTION_RIGHT_TAIL_ASYMMETRY_SIGNAL_CONFIRMED`
- none:
  `DISTRIBUTION_RIGHT_TAIL_ASYMMETRY_NULL`
- any lineage/mean-alignment/draw-count failure:
  `DISTRIBUTION_RIGHT_TAIL_ASYMMETRY_INTEGRITY_FAILURE`

Per-market classification is frozen before scoring. A result that appears only
in receiving yards, for example, may be reported without turning the other
markets into hidden failures or post-hoc exclusions.

## Anti-retest / no-rescue rules

This diagnostic MUST NOT:
- alter projection means;
- fit or search a widening factor;
- fit skew parameters;
- use sportsbook lines/odds;
- create an UNDER-only rule;
- create an edge threshold/top-N rule;
- alter fair probabilities;
- reopen generic symmetric Distribution Widening V1;
- use q05/q95 to rescue failure at q10/q90;
- change quantile levels after results;
- add position/depth/role subgroups after results.

## What a positive result authorizes

A replicated market-specific right-tail signal authorizes only a **separately
frozen** asymmetric-distribution candidate.

That future candidate must:
- keep the football mean exactly invariant;
- define its transformation before scoring;
- use genuine holdout or prospective confirmation;
- show better distribution/probability scoring, not merely better retrospective
  wager results;
- preserve existing specialist authorities or explicitly prove superiority.

No production change is authorized by this diagnostic itself.

## What a null result means

If no market replicates in both seasons, the general historical right-tail
asymmetry hypothesis closes. The Week-4 2026 tail pattern remains an observed
live-season phenomenon but cannot justify an asymmetric-distribution model
without genuinely new evidence.
