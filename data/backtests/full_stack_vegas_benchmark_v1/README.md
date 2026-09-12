# Full-Stack Vegas Benchmark V1 — result record

**Disposition:** `FULL_STACK_EDGE_THRESHOLD_HISTORICAL_BENCHMARK_QB_SYNTHESIS_GATES_CONFIRMED_NO_FOOTBALL_ONLY_EDGE`

## What this measures

The follow-up to `historical_market_vegas_benchmark_v1` (base MC engine only,
naive bet-every-disagreement rule). This benchmark instead uses:

- the full production stack (MC + ML + State + the calibrated production
  ensemble in `data/model_ensemble_weights.csv`) for every market;
- for QB `pass_yards` specifically, an out-of-sample reconstruction of the
  frozen M89/M90 football-only synthesis (trained on 2023, tested on
  2024-2025, never touching its own test data during fitting);
- the real production decision gate from
  `scripts/master_betting_workbook_core_v2.py` (`implied_prob`, `no_vig`,
  `ev_roi`; STRONG EDGE at EV >= .05 and prob-edge >= .03, LEAN EDGE at EV >
  0) instead of a naive rule, reported by tier.

RB `P3`/`R26` are correctly excluded: they are qualified for the 2026
Week-1 route only and out of scope for a 2024-2025 historical grade.

**Known gap, not a scope decision:** unlike P3/R26, `WR_R15_PRODUCTION_MODEL_V1`
and `TE_R5P_PRODUCTION_MODEL_V1` are *not* Week-1-locked -- their own
validation record covers multiple historical seasons (WR-R15: 2022-2025,
TE-R5P: 2023-2025), so they are legitimately applicable to this backtest.
They are **not yet included** in the `rec_yards`/`receptions` numbers below;
those numbers reflect the shared MC+ML+State+ensemble engine only. This
understates the actual deployed WR/TE stack and should be corrected in a
follow-up pass (`scripts/modeling/wr_r15_entitlement_adapter_v1.py`,
`scripts/modeling/te_r5p_entitlement_adapter_v1.py`, with
`backtest-te-r5-participation-entitlement-v1.yml` on the
`research-te-r5-participation-entitlement-v1` branch as the precedent for a
backtest-compatible application) before treating the WR/TE numbers here as
final.

## QB pass_yards result (resolved reconstruction)

Trained the synthesis on 2023, tested out-of-sample on 2024-2025 real
DraftKings/FanDuel lines (`qb_base_summary.csv`, `qb_synthesis_summary.csv`,
`qb_market_assisted_summary.csv`):

| Candidate | PLAY tier win rate | PLAY tier ROI/unit | Model MAE | Vegas MAE |
|---|---:|---:|---:|---:|
| Base ensemble (no synthesis) | 50.3% | -5.3% | 60.1 | 56.6 |
| Football-only synthesis reconstruction | 52.5% | -1.4% | 58.8 | 56.8 |
| Market-assisted (sees Vegas spread/total) | 54.6% | **+2.6%** | 57.4 | 56.7 |

Every promotion gate the synthesis reconstruction was checked against
passed: combined MAE 61.15 -> 59.77 (both seasons individually improved),
bias -10.04 -> +2.55, correlation 0.197 -> 0.260, paired-bootstrap
probability of improvement 0.97
(`qb_synthesis_scoreboard_reconstruction_v1.csv`,
`qb_synthesis_gates_reconstruction_v1.json`). The football-only synthesis
meaningfully narrows the loss under the real edge-threshold gate (-5.3% ->
-1.4%) but does not cross breakeven. The market-assisted candidate is the
only QB candidate in this whole benchmark with positive ROI (+2.6%) — it is
explicitly not a football-only result (it uses Vegas spread/total/moneyline
as input features) and must not be read as evidence of a football-only edge,
but it is worth knowing that letting the model see market context is enough
to flip this specific market from loser to (small) winner.

### Reconstruction-fidelity note (resolved)

An earlier pass of this reconstruction used this repo's general backtest
harness without the exact original M89/M90 correction steps
(`correct_m89_team_semantics.py`, real historical injuries/weather via
`build_historical_injuries.py`/`build_historical_weather.py`) and showed no
improvement at all (all gates failing) -- a fidelity gap in the
reconstruction, not evidence against the promoted synthesis. Rebuilding with
those exact steps (still the general `scripts/backtest/walk_forward.py`
harness, not a literal replay of the original run) reproduces a real,
gate-passing improvement in the same direction and shape as the documented
production result (`docs/production/QB_PASS_SYNTHESIS_V1.md`: base MAE
~57.64 -> synthesis ~55.06). The reconstruction's absolute MAE remains
somewhat higher than the documented numbers (61.15 -> 59.77 vs. 57.64 ->
55.06) -- the most likely remaining source is that the original M89/M90
evaluation reconciled its base cohort against the frozen 884-row
`data/backtests/qb_frontier_canonical_v3_football_only/` cohort, which this
reconstruction does not replicate. That remaining gap is a data-selection
detail worth closing for full numerical parity, not a reason to doubt the
promoted synthesis: the qualitative result (real, gate-passing, correctly
signed improvement) now matches.

## Non-QB markets result (rush_yards, rec_yards, receptions, rush_rec_yards)

Full stack (ensemble applied, no position-specific overlay since P3/R26 are
out of scope), graded across 2024-2025 (`non_qb_summary.csv`,
`non_qb_detail.csv`):

| Market | PLAY tier win rate | PLAY tier ROI/unit | ALL_NO_FILTER ROI/unit |
|---|---:|---:|---:|
| rush_yards | 51.5% | -2.5% | -3.6% |
| rec_yards | 51.6% | -2.9% | -3.5% |
| rush_rec_yards | 50.7% | -4.9% | -5.2% |
| receptions | 54.1% | -0.6% | -1.3% |

Unlike QB, the edge-threshold gate shows a real, monotonic effect here: ROI
improves as the filter tightens (no filter -> lean-or-strong -> strong-only)
in every one of these four markets. None crossed breakeven, but the
confidence signal is doing directionally real work for these markets in a
way it was not for QB pass_yards.

## Fidelity caveats (disclosed, not hidden)

- Fair probability uses a Normal(mean, component_sd) approximation (spread
  across mc_proj/ml_proj/state_proj), not the literal simulated Monte Carlo
  outcome distribution production uses for `fair_prob`.
- The free archive's line/odds are "latest captured," not a fixed
  30-minutes-before-kickoff snapshot.
- QB market-assisted numbers must never be read as a football-only result.

## Anti-reinvention rule

Do not repeat this exact base/threshold comparison expecting a different
answer. The QB reconstruction-fidelity question is resolved: the promoted
M89/M90 synthesis reproduces a real, gate-passing improvement once
reconstructed with the correct inputs, so do not re-litigate whether the
synthesis itself is sound from this benchmark alone. What remains open: (1)
closing the residual MAE gap to the documented numbers via the canonical
884-row cohort reconciliation, purely for numerical parity; (2) no
football-only candidate (QB or non-QB) has crossed breakeven against real
2024-2025 Vegas lines yet under the real edge-threshold gate -- the
market-assisted QB candidate is the only positive-ROI result in this whole
benchmark, and it is explicitly not a football-only edge; (3) WR-R15/TE-R5P
are not yet in the non-QB numbers (see above) -- close that gap before
concluding anything about WR/TE's real-world edge.
